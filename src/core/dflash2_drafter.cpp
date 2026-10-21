#include "strata/core/dflash2_drafter.hpp"
#include "strata/core/dflash2_forward.hpp"
#include "strata/core/dflash2_selector.hpp"
#include "strata/core/dflash2_weights.hpp"
#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/elementwise.hpp"
#include "conversation_store_sha256.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <exception>

namespace strata::core::dflash2 {
namespace {
class NativeFailure final : public std::exception {
public:
    NativeFailure(const char* message, cudaError_t status = cudaSuccess) noexcept : message_(message), status_(status) {}
    const char* what() const noexcept override { return message_; }
    cudaError_t status() const noexcept { return status_; }
private:
    const char* message_;
    cudaError_t status_;
};
void require(bool condition, const char* message) { if (!condition) throw NativeFailure(message); }
void cuda_require(cudaError_t status, const char* message) {
    if (status != cudaSuccess) throw NativeFailure(message, status);
}
bool refuse(std::string& error, const char* message) { error = std::string("dflash2 drafter: ") + message; return false; }
constexpr size_t raw_float_bytes = block_size * hidden_size * sizeof(float);
constexpr size_t raw_bf16_bytes = block_size * hidden_size * sizeof(uint16_t);
constexpr size_t hidden_three_bytes = (block_size - 1) * hidden_size * sizeof(uint16_t);
constexpr size_t head_row_bytes = hidden_size * sizeof(float);
constexpr size_t head_three_rows = block_size - 1;   // proposal rows, batched into one exact head call
constexpr size_t head_three_row_bytes = head_three_rows * head_row_bytes;
constexpr size_t vocabulary_row_bytes = vocabulary_size * sizeof(float);
constexpr size_t feature_tags_bytes = context_rows * sizeof(int64_t);
constexpr size_t feature_owner_bytes = max_feature_bytes + feature_tags_bytes;

struct Request {
    Request(std::span<uint16_t> staging, OperationLease& lease) : producer(staging, lease) {}
    DraftSlotState state;
    std::unique_ptr<DeviceContext> context;
    CaptureProducer producer;
    int64_t last_capture = 0;
    DraftOwner uniform_owner;
    int uniform_frontier = -1;
    std::array<float, 3> uniforms{};
    bool uniforms_valid = false;
};

}

struct DflashNativeDrafter::Impl {
    const RuntimeWeights& weights;
    VerifiedDailyTarget verified;
    const NativeEmbed& embedding;
    const NativeHead& head;
    int device;
    std::unique_ptr<DeviceForward> forward;
    std::unique_ptr<DeviceSelector> selector;
    std::unique_ptr<DeviceContextSnapshot> snapshot;
    std::array<std::unique_ptr<Request>, 9> requests;
    std::unique_ptr<uint16_t[]> tentative{
        std::make_unique_for_overwrite<uint16_t[]>(size_t(context_rows) * feature_width)};
    ContextSnapshotMetadata import_metadata;
    int32_t* tokens = nullptr;
    float* raw = nullptr;
    uint16_t *raw_bf16 = nullptr, *embedding_bf16 = nullptr, *hidden = nullptr, *head_hidden = nullptr;
    float *head_float = nullptr, *head_logits = nullptr, *logits = nullptr;
    void* head_scratch = nullptr;
    cudaStream_t embedding_stream = nullptr, head_stream = nullptr;
    cudaStream_t operation_stream = nullptr;
    size_t head_scratch_bytes = 0;
    ProposalObserver observer = nullptr;
    void* observer_user = nullptr;
    Selection selection;
    int32_t proposal_anchor = -1;

    Impl(const RuntimeWeights& weights, VerifiedDailyTarget verified,
         const NativeEmbed& embedding, const NativeHead& head, int device)
        : weights(weights), verified(std::move(verified)), embedding(embedding), head(head), device(device),
          head_scratch_bytes(head.projection_scratch_bytes(int(head_three_rows))) {}
    Request* request(int slot) { return slot >= 0 && slot <= 8 ? requests[size_t(slot)].get() : nullptr; }
    const Request* request(int slot) const { return slot >= 0 && slot <= 8 ? requests[size_t(slot)].get() : nullptr; }
    void check_target() const {
        require(embedding.loaded() && head.loaded() && embedding.identity() == verified.embedding() &&
                head.identity() == verified.head() && weights.target_identity() == verified.binding().identity,
                "dflash2 borrowed target was unloaded/replaced or has foreign verified receipts");
    }
    void drain_or_terminate() const noexcept {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) std::terminate();
        int current = previous;
        const std::array streams{std::pair{device, operation_stream},
            std::pair{embedding.device(), embedding_stream}, std::pair{head.device(), head_stream}};
        for (size_t i = 0; i < streams.size(); ++i) {
            if (!streams[i].second) continue;
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j) if (streams[j] == streams[i]) duplicate = true;
            if (duplicate) continue;
            if (current != streams[i].first) {
                if (cudaSetDevice(streams[i].first) != cudaSuccess) std::terminate();
                current = streams[i].first;
            }
            if (cudaStreamSynchronize(streams[i].second) != cudaSuccess) std::terminate();
        }
        if (current != previous && cudaSetDevice(previous) != cudaSuccess) std::terminate();
        std::string error;
        if (selector && !selector->drain(error)) std::terminate();
    }
    bool drain(std::string&) const { drain_or_terminate(); return true; }
    ~Impl() {
        drain_or_terminate();
        {
            const OnDevice on(embedding.device());
            if (raw_bf16 && embedding.device() != device && cudaFree(raw_bf16) != cudaSuccess) std::terminate();
            if (raw && cudaFree(raw) != cudaSuccess) std::terminate();
            if (tokens && cudaFree(tokens) != cudaSuccess) std::terminate();
            if (embedding_stream && embedding_stream != operation_stream && cudaStreamDestroy(embedding_stream) != cudaSuccess)
                std::terminate();
        }
        {
            const OnDevice on(head.device());
            if (head_hidden && head.device() != device && cudaFree(head_hidden) != cudaSuccess) std::terminate();
            if (head_float && cudaFree(head_float) != cudaSuccess) std::terminate();
            if (head_logits && head.device() != device && cudaFree(head_logits) != cudaSuccess) std::terminate();
            if (head_scratch && cudaFree(head_scratch) != cudaSuccess) std::terminate();
            if (head_stream && head_stream != operation_stream && cudaStreamDestroy(head_stream) != cudaSuccess)
                std::terminate();
        }
        {
            const OnDevice on(device);
            if (embedding_bf16 && cudaFree(embedding_bf16) != cudaSuccess) std::terminate();
            if (hidden && cudaFree(hidden) != cudaSuccess) std::terminate();
            if (logits && cudaFree(logits) != cudaSuccess) std::terminate();
        }
    }
};

DflashNativeDrafter::DflashNativeDrafter(const RuntimeWeights& weights, VerifiedDailyTarget verified,
        const NativeEmbed& embedding, const NativeHead& head, int device) {
    require(device >= 0 && verified.binding().identity == weights.target_identity() && embedding.loaded() &&
            head.loaded() && embedding.identity() == verified.embedding() && head.identity() == verified.head() &&
            head.projection_scratch_bytes() == size_t(hidden_size / 32 * 36),
            "dflash2 native adapter requires the verified loaded daily target and exact scratch geometry");
    auto state = std::make_unique<Impl>(weights, std::move(verified), embedding, head, device);
    device_ = device; window_ = context_rows;
    identity_.kind = DraftKind::dflash2; identity_.format = 1;
    identity_.geometry = {5, hidden_size, 8, 128, context_rows, block_size, max_position, 16};
    require(conversation_store_detail::sha256_hex_parse(weights.manifest_sha256(), identity_.weights),
            "dflash2 runtime manifest digest is not canonical SHA256");
    static constexpr char configuration[] = "dflash2:07ebd93db9f472af339b644bb70221ad8428328a:bf16:5:2560:7680:32x128:8x128:2047:block4:rope10000:rms1e-6:conv2g16:selector256top16";
    identity_.configuration = conversation_store_detail::sha256(configuration, sizeof(configuration) - 1);
    conversation_store_detail::Sha256 target;
    for (const auto* digest : {&weights.target_identity().pack_sha256, &weights.target_identity().embedding_sha256,
                              &weights.target_identity().head_sha256, &weights.target_identity().tokenizer_sha256}) {
        require(digest->size() == 64, "dflash2 borrowed target digest is absent");
        target.update(digest->data(), digest->size());
    }
    identity_.borrowed_target = target.digest(); identity_.load_generation = state->verified.binding().load_generation;
    state->import_metadata.manifest_sha256 = weights.manifest_sha256();
    state->import_metadata.target = weights.target_identity();
    {
        const OnDevice on(device);
        cuda_require(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking), "dflash2 operation stream");
        state->operation_stream = cs_;
        std::string error;
        require(canary_.init(error), "dflash2 operation canary allocation failed");
        state->forward = std::make_unique<DeviceForward>(weights, device);
        state->selector = std::make_unique<DeviceSelector>(device,
            state->forward->weight("candidate_selector.hidden_projection.weight"),
            state->forward->weight("candidate_selector.predecessor_codebook"),
            state->forward->weight("candidate_selector.successor_codebook"));
        state->snapshot = state->forward->create_snapshot();
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->embedding_bf16), raw_bf16_bytes), "draft raw embeddings");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->hidden), raw_bf16_bytes), "draft hidden rows");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->logits), logits_bytes), "draft three full logit rows");
        state->requests[8] = std::make_unique<Request>(
            std::span<uint16_t>{state->tentative.get(), size_t(context_rows) * feature_width}, scratch_);
        state->requests[8]->context = state->forward->create_context(0);
    }
    {
        const OnDevice on(embedding.device());
        state->embedding_stream = cs_;
        if (embedding.device() != device)
            cuda_require(cudaStreamCreateWithFlags(&state->embedding_stream, cudaStreamNonBlocking), "native embedding stream");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->tokens), block_size * sizeof(int32_t)), "native embedding tokens");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->raw), raw_float_bytes), "native raw embedding rows");
        state->raw_bf16 = state->embedding_bf16;
        if (embedding.device() != device)
            cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->raw_bf16), raw_bf16_bytes), "native embedding transfer rows");
    }
    {
        const OnDevice on(head.device());
        state->head_stream = cs_;
        if (head.device() != device)
            cuda_require(cudaStreamCreateWithFlags(&state->head_stream, cudaStreamNonBlocking), "native head stream");
        state->head_hidden = state->hidden + hidden_size; state->head_logits = state->logits;
        if (head.device() != device) {
            cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->head_hidden), hidden_three_bytes), "native head BF16 transfer");
            cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->head_logits), head_three_rows * vocabulary_row_bytes),
                         "native head transfer logits");
        }
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->head_float), head_three_row_bytes), "native head float input");
        cuda_require(cudaMalloc(&state->head_scratch, state->head_scratch_bytes), "native dedicated head scratch");
    }
    impl_ = std::move(state);
}

DflashNativeDrafter::~DflashNativeDrafter() {
    if (impl_) {
        std::string error;
        if (!impl_->drain(error)) std::terminate();
        impl_.reset();
    }
}

DraftAllocationPlan DflashNativeDrafter::planned_allocation(int device, size_t dense_bytes,
        int embedding_device, int head_device, size_t head_scratch_bytes) {
    DraftAllocationPlan plan; plan.device = device;
    plan.shared_device = dense_bytes + DeviceForward::scratch_bytes() + DeviceSelector::scratch_bytes() +
        2 * raw_bf16_bytes + logits_bytes + kOperationCanaryDeviceBytes;
    if (embedding_device == device) plan.shared_device += block_size * sizeof(int32_t) + raw_float_bytes;
    if (head_device == device) plan.shared_device += head_three_row_bytes + head_scratch_bytes;
    plan.shared_host = DeviceSelector::pinned_bytes() + kOperationCanaryHostBytes + max_feature_bytes +
                       sizeof(DflashNativeDrafter) + sizeof(Impl) + 65536;
    plan.owner_device = DeviceForward::context_bytes();
    // Include conservative opaque numerical/metadata-control admission allowances,
    // in addition to the exact payloads and three reserved request-id strings.
    plan.owner_host = feature_owner_bytes + sizeof(Request) + 3 * 129 + 16384;
    plan.transfer_host = DeviceContextSnapshot::storage_bytes(); // preallocated, charged for the whole owner lifetime
    ConversationBuffer buffer;
    const size_t part = buffer.allocation_peak(size_t(context_rows) * 8 * 128 * sizeof(uint16_t));
    plan.snapshot_host = 10 * part + 5 * sizeof(ConversationKv);
    return plan;
}
DraftAllocationPlan DflashNativeDrafter::allocation_plan() const {
    return planned_allocation(device_, impl_->weights.dense().size_bytes(), impl_->embedding.device(),
                              impl_->head.device(), impl_->head_scratch_bytes);
}
DraftAllocationPlan DflashNativeDrafter::planned_target_allocation(int device, int drafter_device,
        int embedding_device, int head_device, size_t head_scratch_bytes) {
    DraftAllocationPlan plan; plan.device = device;
    if (device != drafter_device && device == embedding_device)
        plan.shared_device += block_size * sizeof(int32_t) + raw_float_bytes + raw_bf16_bytes;
    if (device != drafter_device && device == head_device)
        plan.shared_device += hidden_three_bytes + head_three_row_bytes + head_three_rows * vocabulary_row_bytes +
                              head_scratch_bytes;
    return plan;
}
DraftAllocationPlan DflashNativeDrafter::target_allocation_plan(int device) const {
    return planned_target_allocation(device, device_, impl_->embedding.device(), impl_->head.device(),
                                     impl_->head_scratch_bytes);
}
void DflashNativeDrafter::invalidate_coverage() {
    NeuralDrafter::invalidate_coverage();
    if (impl_) for (auto& request : impl_->requests)
        if (request && request->producer.active()) request->producer.abort();
}
uint64_t DflashNativeDrafter::vram_bytes() const {
    auto plan = allocation_plan();
    size_t owners = 0;
    for (const auto& request : impl_->requests) owners += request != nullptr;
    return plan.shared_device + plan.owner_device * owners;
}
DraftSlotState* DflashNativeDrafter::slot_metadata(int slot) {
    auto* request = impl_ ? impl_->request(slot) : nullptr; return request ? &request->state : nullptr;
}
const DraftSlotState* DflashNativeDrafter::slot_metadata(int slot) const {
    const auto* request = impl_ ? impl_->request(slot) : nullptr; return request ? &request->state : nullptr;
}
bool DflashNativeDrafter::drain_adapter(std::string& error) { return !impl_ || impl_->drain(error); }

bool DflashNativeDrafter::init_slots(const std::vector<SessionState*>& sessions, std::string& error,
                                    const plan::QsaPlacement*) {
    if (sessions.empty() || sessions.size() > 8 || impl_->requests[0])
        return refuse(error, "invalid or repeated resident context initialization");
    if (!idle(error)) return false;
    for (const auto* session : sessions)
        if (!session || session->max_cells < 1 || session->max_cells > max_position)
            return refuse(error, "resident target context exceeds native absolute-position limit");
    for (size_t slot = 0; slot < sessions.size(); ++slot) {
        impl_->requests[slot] = std::make_unique<Request>(
            std::span<uint16_t>{impl_->tentative.get(), size_t(context_rows) * feature_width}, scratch_);
        impl_->requests[slot]->context = impl_->forward->create_context(0);
    }
    return true;
}
CaptureProducer& DflashNativeDrafter::capture_producer(int slot) {
    auto* request = impl_->request(slot);
    require(request != nullptr, "dflash2 capture seat was not allocated before admission");
    return request->producer;
}
bool DflashNativeDrafter::begin_capture(NeuralDrafter::Operation& operation, CapturePhase phase,
        int64_t start, int64_t count, std::string& error) {
    const auto expected = operation.identity();
    auto* request = impl_->request(expected.seat);
    if (!operation || !request || !owns(expected) || !expected.request || !expected.incarnation ||
        request->producer.owner().request != expected.request ||
        request->producer.owner().incarnation != expected.incarnation ||
        start != request->context->end())
        return refuse(error, "stale capture owner/incarnation/frontier");
    return request->producer.begin(phase, start, count, error, &operation.guard());
}
int64_t DflashNativeDrafter::snapshot_origin(int slot) const {
    const auto* request = impl_->request(slot);
    return request ? request->context->origin() : -1;
}
bool DflashNativeDrafter::reset_owner(int slot, const DraftOwner& expected, int origin, std::string& error) {
    if (!owns(expected) || expected.seat != slot || origin < 0 || origin > max_position)
        return refuse(error, "stale owner or invalid absolute reset origin");
    Operation operation(*this, error, slot, nullptr, 0, origin, &expected);
    if (!operation) return false;
    auto& request = *impl_->request(slot);
    if (!impl_->forward->reset(*request.context, origin, error)) return false;
    if (!request.producer.resume(expected.request, expected.incarnation, origin, origin, error)) return false;
    request.last_capture = 0; request.uniforms_valid = false;
    request.state.cells = origin; request.state.completed_cells = request.state.begin = -1;
    request.state.certified = false; request.state.tail_token = -1;
    if (slot == 8) { coverage_.clear(); coverage_owner_ = 0; }
    return operation.finish(error);
}
bool DflashNativeDrafter::prepare_slot(int slot, uint64_t request_id, int64_t prompt_length,
        int64_t cells, bool reset, std::string& error) {
    const auto expected = owner(slot);
    auto* request = impl_->request(slot);
    if (!request || expected.request != request_id || !request_id || prompt_length < 1 ||
        prompt_length > max_position || cells < 0 || cells >= prompt_length ||
        (reset ? cells != 0 : !slot_covers(slot, cells, prompt_length)))
        return refuse(error, "invalid direct preparation or absent retained context certificate");
    if (reset) {
        if (!reset_owner(slot, expected, 0, error)) return false;
    } else {
        Operation operation(*this, error, slot, nullptr, 0, int32_t(cells), &expected);
        if (!operation || !impl_->forward->rewind(*request->context, int(cells), error)) return false;
        operation.writes(request->context->begin(), cells, -1);
        if (!operation.finish(error)) return false;
        if (!request->producer.resume(expected.request, expected.incarnation, request->context->origin(),
                                       int(cells), error)) return false;
        request->uniforms_valid = false;
    }
    request->state.prompt_len = prompt_length;
    if (slot == 8) prompt_len_ = prompt_length;
    return true;
}

bool DflashNativeDrafter::ingest_committed(Operation& operation, const FeatureView& features, std::string& error) {
    const auto expected = operation.identity();
    if (!operation || !owns(expected) || !expected.request || !expected.incarnation)
        return refuse(error, "stale feature completion owner");
    auto* request = impl_->request(expected.seat);
    if (!request || features.start < 0 || features.count < 1 || features.count > context_rows ||
        features.start > max_position - features.count || features.incarnation != int64_t(expected.incarnation))
        return refuse(error, "invalid feature owner or committed interval");
    std::array<char, 21> id{};
    const auto end = std::to_chars(id.data(), id.data() + id.size(), expected.request);
    if (features.request_id != std::string_view(id.data(), size_t(end.ptr - id.data())))
        return refuse(error, "foreign feature request identity");
    const int frontier = features.start + features.count;
    const auto& context = *request->context;
    const bool replaces_context = context.initialized() && features.count == context_rows &&
                                  features.start > context.end();
    if ((context.initialized() && !replaces_context) ? features.start != context.end() :
        features.start != std::max(context.origin(), frontier - context_rows))
        return refuse(error, "feature completion is duplicate, skipped or noncontiguous");
    size_t rows = 0;
    int64_t generation = request->last_capture;
    for (size_t part = 0; part < features.segments.size(); ++part) {
        const auto values = features.segments[part]; const auto tags = features.generations[part];
        if (values.size() != tags.size() * feature_width) return refuse(error, "feature values/tags disagree");
        for (auto tag : tags) {
            if (tag < 1 || tag < generation || (rows == 0 && tag <= request->last_capture))
                return refuse(error, "stale feature generation");
            generation = tag; ++rows;
        }
    }
    if (rows != size_t(features.count)) return refuse(error, "feature segment count disagrees");
    const auto actual = request->producer.store().committed_since(features.start);
    if (actual.start != features.start || actual.count != features.count)
        return refuse(error, "feature source is not its actual completed capture producer");
    for (size_t part = 0; part < features.segments.size(); ++part)
        if (features.segments[part].data() != actual.segments[part].data() ||
            features.segments[part].size() != actual.segments[part].size() ||
            features.generations[part].data() != actual.generations[part].data() ||
            features.generations[part].size() != actual.generations[part].size())
            return refuse(error, "foreign borrowed feature ring or generation tags");
    if (replaces_context && !impl_->forward->reset(*request->context, context.origin(), error)) return false;
    if (!impl_->forward->ingest(*request->context, actual, frontier, cs_, error)) return false;
    request->last_capture = generation; request->uniforms_valid = false;
    operation.writes(request->context->begin(), frontier, -1);
    return true;
}
bool DflashNativeDrafter::ingest_committed(int slot, const DraftOwner& expected,
        const FeatureView& features, std::string& error) {
    Operation operation(*this, error, slot, nullptr, 0, features.start, &expected);
    return operation && ingest_committed(operation, features, error) && operation.finish(error);
}
bool DflashNativeDrafter::prefill_slot(int slot, const float*, const int32_t*, int64_t rows,
                                      int64_t position, std::string& error) {
    // A final target residual is not a five-tap feature. This acknowledgement
    // succeeds only after the real capture producer's context projection drains.
    if (!inputs_completed(slot, position, rows))
        return refuse(error, "prefill requires its actual completed five-tap transfer/context projection");
    return true;
}
bool DflashNativeDrafter::advance_slot(int slot, const float*, int32_t, int64_t cell, std::string& error) {
    const auto coverage = slot_coverage(slot);
    if (!coverage.present || !coverage.coherent || cell < coverage.certified_begin ||
        coverage.certified_end < cell + 1)
        return refuse(error, "target commit lacks its completed five-tap context projection");
    return true;
}

bool DflashNativeDrafter::set_proposal_uniforms(int slot, const DraftOwner& expected, int frontier,
        const std::array<float, 3>& uniforms, std::string& error) {
    auto* request = impl_->request(slot);
    if (!request || !owns(expected) || expected.seat != slot || !expected.request ||
        frontier != request->context->end() || frontier > max_position - block_size)
        return refuse(error, "stale proposal random owner/frontier");
    for (float value : uniforms)
        if (!std::isfinite(value) || value < 0 || value >= 1)
            return refuse(error, "proposal uniforms must be engine-owned [0,1) draws");
    request->uniform_owner = expected; request->uniform_frontier = frontier;
    request->uniforms = uniforms; request->uniforms_valid = true;
    return true;
}
void DflashNativeDrafter::set_draft_sampling(const kernels::SamplerParams& sampling) { sampling_[8] = sampling; }
bool DflashNativeDrafter::propose_at(int slot, const DraftOwner& expected, int frontier,
        int32_t anchor, const std::array<float, 3>& uniforms, DraftProposal& result, std::string& error) {
    return set_proposal_uniforms(slot, expected, frontier, uniforms, error) &&
        NeuralDrafter::propose_slot(slot, nullptr, anchor, frontier - 1, result, error);
}
bool DflashNativeDrafter::proposal_preflight(int slot, const float*, int32_t anchor, int64_t cell,
                                           std::string& error) const {
    auto* request = impl_->request(slot);
    const auto expected = owner(slot);
    const int frontier = int(cell + 1);
    const auto coverage = slot_coverage(slot);
    const auto& sampling = sampling_[size_t(slot)];
    const bool sampled = !sampling.greedy && sampling.temperature > 0;
    if (!request || !coverage.present || !coverage.coherent || cell < 0 ||
        cell + 1 != request->context->end() || cell + 1 > max_position - block_size ||
        anchor < 0 || anchor >= vocabulary_size || !std::isfinite(sampling.temperature) ||
        sampling.temperature < 0 || (sampled && (!request->uniforms_valid ||
        request->uniform_owner != expected || request->uniform_frontier != frontier)))
        return refuse(error, "proposal has no certified context or exact engine random owner");
    return true;
}
bool DflashNativeDrafter::propose_impl(Operation& operation, const float*, int32_t anchor, int64_t cell,
                                       DraftProposal& result, std::string& error) {
    const auto expected = operation.identity();
    auto* request = impl_->request(expected.seat);
    const int frontier = int(cell + 1);
    const auto& sampling = sampling_[size_t(expected.seat)];
    const bool sampled = !sampling.greedy && sampling.temperature > 0;
    const auto uniforms = request->uniforms;
    const int rows = std::max(1, std::min(draft_positions(), int(block_size) - 1));
    std::array<int32_t, block_size> ids{};
    ids[0] = anchor;
    for (int row = 1; row < rows + 1; ++row) ids[size_t(row)] = mask_token;
    request->uniforms_valid = false;
    try {
        impl_->check_target();
        {
            const OnDevice on(impl_->embedding.device());
            cuda_require(cudaMemcpyAsync(impl_->tokens, ids.data(), sizeof(int32_t) * size_t(rows + 1), cudaMemcpyHostToDevice,
                                          impl_->embedding_stream), "native embedding token upload");
            impl_->embedding.gather_dev(impl_->tokens, rows + 1, impl_->raw, impl_->embedding_stream);
            kernels::f32_to_bf16_bulk(impl_->raw, impl_->raw_bf16, block_size * hidden_size, impl_->embedding_stream);
            cuda_require(cudaStreamSynchronize(impl_->embedding_stream), "native raw embedding completion");
        }
        const OnDevice on(device_);
        if (impl_->embedding.device() != device_)
            cuda_require(cudaMemcpyPeerAsync(impl_->embedding_bf16, device_, impl_->raw_bf16,
                impl_->embedding.device(), raw_bf16_bytes, cs_), "raw BF16 embedding device transfer");
        FeatureView empty;
        empty.start = frontier; empty.count = 0; empty.incarnation = int64_t(expected.incarnation);
        std::array<char, 21> id{};
        const auto end = std::to_chars(id.data(), id.data() + id.size(), expected.request);
        empty.request_id = {id.data(), size_t(end.ptr - id.data())};
        if (!impl_->forward->forward(*request->context, empty, impl_->embedding_bf16,
                                     frontier, impl_->hidden, cs_, error)) return false;
        if (impl_->head.device() != device_) {
            const OnDevice head_device(impl_->head.device());
            cuda_require(cudaMemcpyPeerAsync(impl_->head_hidden, impl_->head.device(), impl_->hidden + hidden_size,
                device_, hidden_three_bytes, impl_->head_stream), "draft hidden device transfer");
        }
        {
            const OnDevice head_device(impl_->head.device());
            // One batched projection for all proposed rows: same weights, same exact-mode arithmetic per column
            // as single-column calls, one drain instead of per-row host round trips.
            for (int row = 0; row < rows; ++row) {
                require(kernels::dflash2::widen_bf16(impl_->head_hidden + row * hidden_size,
                        impl_->head_float + row * hidden_size, hidden_size, impl_->head_stream) == 0,
                        "draft hidden BF16 widening launch failed");
            }
            float* logits = impl_->head.device() == device_ ? impl_->logits : impl_->head_logits;
            if (!impl_->head.run_rows_with_scratch(impl_->head_float, logits, rows,
                    impl_->head_scratch, impl_->head_scratch_bytes, impl_->head_stream, error)) return false;
            cuda_require(cudaStreamSynchronize(impl_->head_stream), "native draft head completion");
        }
        if (impl_->head.device() != device_) {
            cuda_require(cudaMemcpyPeerAsync(impl_->logits, device_, impl_->head_logits, impl_->head.device(),
                size_t(rows) * vocabulary_row_bytes, cs_), "native logit device transfer");
            cuda_require(cudaStreamSynchronize(cs_), "native logit transfer completion");
        }
        auto& selected = impl_->selection;
        if (!impl_->selector->select(rows, impl_->hidden + hidden_size, impl_->logits, anchor,
                sampled ? sampling.temperature : 0.f, uniforms, selected, cs_, error)) return false;
        impl_->check_target();
        result.count = rows;
        for (int row = 0; row < rows; ++row) {
            result.tokens[row] = selected.tokens[row]; result.confidence[row] = selected.confidence[row];
        }
        result.owner = operation.identity(); result.frontier = frontier;
        result.distribution = selected.sampled ? DraftProposal::Distribution::conditional_q :
                                                DraftProposal::Distribution::point_mass;
        if (selected.sampled) selected.fill_probability_rows(result.q.data());
        impl_->proposal_anchor = anchor;
        return true; // common owner validates, inspects, drains, then publishes
    } catch (const std::exception& failure) {
        impl_->drain_or_terminate(); // before formatting, unwinding host input ids, or releasing borrowed buffers
        error = failure.what();
        if (const auto* cuda = dynamic_cast<const NativeFailure*>(&failure); cuda && cuda->status() != cudaSuccess)
            error += std::string(": ") + cudaGetErrorString(cuda->status());
        return false;
    } catch (...) {
        impl_->drain_or_terminate();
        throw;
    }
}
bool DflashNativeDrafter::inspect_proposal(Operation& operation, const DraftProposal& result, std::string& error) {
    if (!impl_->observer) return true;
    auto* request = impl_->request(operation.identity().seat);
    ProposalCapture capture;
    capture.owner = operation.identity(); capture.origin = request->context->origin();
    capture.frontier = result.frontier; capture.anchor = impl_->proposal_anchor;
    capture.context = request->producer.store().committed_since(request->context->begin());
    capture.raw_embeddings_device = impl_->embedding_bf16; capture.hidden_device = impl_->hidden;
    capture.logits_device = impl_->logits; capture.proposal = result;
    capture.candidates = impl_->selection.candidates;
    capture.conditional_probabilities = impl_->selection.conditional_probabilities;
    const auto& sampling = sampling_[size_t(operation.identity().seat)];
    capture.temperature = !sampling.greedy && sampling.temperature > 0 ? sampling.temperature : 0.f;
    capture.uniforms = request->uniforms;
    capture.device = device_; capture.stream = cs_;
    return impl_->observer(impl_->observer_user, capture, error);
}
void DflashNativeDrafter::set_proposal_observer(ProposalObserver observer, void* user) {
    std::string error;
    if (!context_idle(error)) throw Refusal(error);
    impl_->observer = observer; impl_->observer_user = user;
}

bool DflashNativeDrafter::save_slot(int slot, int64_t cells, std::string& error) {
    auto* destination = impl_->request(slot); auto& source = *impl_->request(8);
    const auto coverage = slot_coverage(8);
    if (!destination || !owner(slot).request || !coverage.present || !coverage.coherent ||
        cells != coverage.certified_end) return refuse(error, "save requires its actual drained working context");
    const auto destination_owner_before = owner(slot), source_owner = owner(8);
    Operation operation(*this, error, slot, nullptr, 0, int32_t(cells), &destination_owner_before, &source_owner);
    if (!operation || !impl_->forward->copy_context(*source.context, *destination->context, cs_, error)) return false;
    operation.writes(source.context->begin(), cells, -1);
    if (!operation.finish(error)) return false;
    destination->state.prompt_len = prompt_len_;
    const auto destination_owner = owner(slot);
    if (!destination->producer.resume(destination_owner.request, destination_owner.incarnation,
            destination->context->origin(), int(cells), error)) return false;
    destination->last_capture = 0; destination->uniforms_valid = false;
    return true;
}
bool DflashNativeDrafter::restore_slot(int slot, int64_t cells, std::string& error) {
    const auto* source = impl_->request(slot); auto& destination = *impl_->request(8);
    if (!source || !slot_covers(slot, cells) || std::max(source->context->origin(), int(cells) - context_rows) !=
        source->context->begin()) return refuse(error, "resume requires its retained certified native interval");
    const auto destination_owner_before = owner(8), source_owner = owner(slot);
    Operation operation(*this, error, 8, nullptr, 0, int32_t(cells), &destination_owner_before, &source_owner);
    if (!operation || !impl_->forward->reset(*destination.context, source->context->origin(), error) ||
        !impl_->forward->copy_context(*source->context, *destination.context, cs_, error) ||
        !impl_->forward->rewind(*destination.context, int(cells), error)) return false;
    operation.writes(destination.context->begin(), cells, -1);
    if (!operation.finish(error)) return false;
    const auto destination_owner = owner(8);
    if (!destination.producer.resume(destination_owner.request, destination_owner.incarnation,
            destination.context->origin(), int(cells), error)) return false;
    destination.last_capture = 0; destination.uniforms_valid = false;
    prompt_len_ = source->state.prompt_len;
    return true;
}
uint64_t DflashNativeDrafter::copy_bytes(int slot, int64_t cells, bool, std::string& error) const {
    const auto coverage = slot_coverage(slot);
    if (!coverage.present || !coverage.coherent || cells < coverage.certified_begin || cells > coverage.certified_end) {
        refuse(error, "copy quote requires an actual certified context interval"); return UINT64_MAX;
    }
    return uint64_t(cells - coverage.certified_begin) * 5 * 2 * 8 * 128 * sizeof(uint16_t);
}
bool DflashNativeDrafter::discard_inputs(const DraftOwner& ticket, std::string& error) {
    OperationLease::Guard lease(scratch_);
    if (!lease) return refuse(error, lease.failure() == OperationLease::Failure::timeout
        ? "input discard scratch wait exceeded 2000 ms"
        : "input discard scratch is poisoned");
    auto* request = impl_->request(ticket.seat);
    if (!request || !owns(ticket) || !inputs_pending(ticket.seat) || !request->producer.cancellable() ||
        request->producer.owner().request != ticket.request ||
        request->producer.owner().incarnation != ticket.incarnation) {
        lease.complete(); return refuse(error, "stale/partially consumed request cannot discard pending capture");
    }
    if (!discard_inputs_owned(ticket, error)) return false;
    if (!request->producer.cancel(error)) std::terminate();
    lease.complete(); return true;
}
bool DflashNativeDrafter::state_fingerprint(int slot, int64_t cells, uint64_t& hash, std::string& error) {
    const auto* request = impl_->request(slot);
    const auto coverage = slot_coverage(slot);
    if (!request || !coverage.present || !coverage.coherent || cells != request->context->end())
        return refuse(error, "fingerprint requires its actual drained native context frontier");
    const auto expected = owner(slot);
    Operation operation(*this, error, slot, nullptr, 0, 0, &expected, nullptr, true);
    if (!operation || !impl_->forward->snapshot(*request->context, *impl_->snapshot, cs_, error)) return false;
    hash = 14695981039346656037ull;
    const auto payload = std::as_bytes(impl_->snapshot->payload());
    for (auto byte : payload) hash = (hash ^ uint8_t(byte)) * 1099511628211ull;
    return operation.finish(error);
}

bool DflashNativeDrafter::snapshot_capture(SavedConversation& image, Operation& operation, std::string& error) {
    const int slot = operation.identity().seat;
    if (!operation || !operation.guard().holds(scratch_) || !owns(operation.identity()))
        return refuse(error, "stale or foreign snapshot capture operation");
    impl_->check_target();
    const auto coverage = slot_coverage(slot);
    const auto* request = impl_->request(slot);
    const int64_t live = int64_t(image.live.ids.size());
    if (!request || !coverage.present || coverage.certified_begin < 0 || coverage.certified_end != live ||
        image.draft_cells != live || image.draft_begin != coverage.certified_begin ||
        !identity_.content_matches(image.draft_identity))
        return refuse(error, "snapshot lacks actual committed features/projected context certificate");
    if (image.kv.size() < 5)
        return refuse(error, "snapshot lacks its five-layer native directory");
    const size_t main = image.kv.size() - 5; // existing target directory, never inferred from target QSA geometry
    const size_t bytes = size_t(coverage.certified_end - coverage.certified_begin) * 8 * 128 * sizeof(uint16_t);
    // Allocate/COW detach every host destination before device readback. The
    // common read-only caller handles allocation refusal without poisoning.
    for (size_t layer = main; layer < image.kv.size(); ++layer) {
        auto& kv = image.kv[layer];
        kv.format = 0; kv.cells = live; kv.heads = 8; kv.head_dim = 128; kv.page_size = 1;
        kv.pooled_rows = kv.idx_dim = 0;
        kv.k_scale = {}; kv.v_scale = {}; kv.pooled = {};
        for (auto* buffer : {&kv.k, &kv.v}) {
            buffer->resize(bytes);
            if (!buffer->visit(0, bytes, [](uint8_t*, size_t, size_t) { return true; }))
                return refuse(error, "snapshot destination buffer is malformed");
        }
    }
    const OnDevice on(device_);
    if (!impl_->forward->snapshot(*request->context, *impl_->snapshot, cs_, error)) return false;
    const auto& metadata = impl_->snapshot->metadata();
    if (metadata.end != live || metadata.begin != coverage.certified_begin)
        return refuse(error, "snapshot readback does not match its actual producer certificate");
    image.draft_identity = identity_; image.draft_origin = metadata.origin;
    for (int layer = 0; layer < 5; ++layer)
        for (int values = 0; values < 2; ++values) {
            const auto part = impl_->snapshot->part(layer, values != 0);
            auto& kv = image.kv[main + size_t(layer)];
            auto& buffer = values ? kv.v : kv.k;
            if (!buffer.visit(0, buffer.size(), [&](uint8_t* p, size_t n, size_t at) {
                std::memcpy(p, reinterpret_cast<const uint8_t*>(part.data()) + at, n); return true;
            })) return refuse(error, "snapshot buffer segmentation failed");
        }
    return true; // common caller drains and publishes the entire image
}
bool DflashNativeDrafter::snapshot_preflight(const SavedConversation& image, int slot, int64_t cells,
        ConversationSnapshotExpectation, std::string& error) const {
    try { impl_->check_target(); }
    catch (const std::exception& failure) { error = failure.what(); return false; }
    if (!impl_->request(slot) || !identity_.content_matches(image.draft_identity) || image.draft_origin < 0 ||
        (image.draft_identity.load_generation &&
         image.draft_identity.load_generation != identity_.load_generation) ||
        image.draft_begin < image.draft_origin || image.draft_cells < image.draft_begin ||
        image.draft_cells > max_position || image.draft_cells != int64_t(image.live.ids.size()) ||
        image.draft_cells - image.draft_begin > context_rows || cells < image.draft_begin ||
        cells > image.draft_cells || std::max(image.draft_origin, cells - context_rows) != image.draft_begin ||
        image.draft_next_token != -1 || !image.draft_boundary_residual.empty() || image.kv.size() < 5)
        return refuse(error, "foreign kind/identity or absent/evicted native context certificate");
    const uint64_t bytes = uint64_t(image.draft_cells - image.draft_begin) * 8 * 128 * sizeof(uint16_t);
    for (size_t layer = image.kv.size() - 5; layer < image.kv.size(); ++layer) {
        const auto& kv = image.kv[layer];
        if (kv.format != 0 || kv.cells != image.draft_cells || kv.heads != 8 || kv.head_dim != 128 ||
            kv.page_size != 1 || kv.pooled_rows || kv.idx_dim || kv.k.size() != bytes || kv.v.size() != bytes ||
            !kv.k_scale.empty() || !kv.v_scale.empty() || !kv.pooled.empty())
            return refuse(error, "native snapshot is not five compact BF16 8x128 K/V layers");
        for (const auto* buffer : {&kv.k, &kv.v})
            if (!buffer->visit(0, buffer->size(), [](const uint8_t* p, size_t n, size_t) {
                if (n % sizeof(uint16_t)) return false;
                for (size_t offset = 0; offset < n; offset += sizeof(uint16_t)) {
                    uint16_t bits;
                    std::memcpy(&bits, p + offset, sizeof(bits));
                    if ((bits & 0x7f80u) == 0x7f80u) return false;
                }
                return true;
            })) return refuse(error, "native snapshot contains nonfinite BF16 context K/V");
    }
    return true;
}
bool DflashNativeDrafter::snapshot_restore(const SavedConversation& image, Operation& operation, int64_t cells,
        int64_t prompt_length, int32_t, ConversationSnapshotExpectation, std::string& error) {
    const int slot = operation.identity().seat;
    if (!operation || !operation.guard().holds(scratch_) || !owns(operation.identity()) || image.kv.size() < 5 ||
        !identity_.content_matches(image.draft_identity))
        return refuse(error, "stale or foreign snapshot restore operation/identity");
    const OnDevice on(device_);
    auto& metadata = impl_->import_metadata;
    metadata.origin = int(image.draft_origin); metadata.begin = int(image.draft_begin);
    metadata.end = int(image.draft_cells); metadata.initialized = true;
    std::array<const ConversationBuffer*, 10> parts{};
    const size_t main = image.kv.size() - 5;
    for (size_t layer = 0; layer < 5; ++layer) {
        parts[2 * layer] = &image.kv[main + layer].k; parts[2 * layer + 1] = &image.kv[main + layer].v;
    }
    auto& request = *impl_->request(slot);
    if (!impl_->snapshot->assign(metadata, parts, error) ||
        !impl_->forward->reset(*request.context, metadata.origin, error) ||
        !impl_->forward->validate_restore(*request.context, *impl_->snapshot, error) ||
        !impl_->forward->restore(*request.context, *impl_->snapshot, cs_, error) ||
        !impl_->forward->rewind(*request.context, int(cells), error)) return false;
    operation.writes(request.context->begin(), cells, -1);
    request.state.prompt_len = prompt_length;
    if (slot == 8) prompt_len_ = prompt_length;
    const auto destination_owner = owner(slot);
    if (!request.producer.resume(destination_owner.request, destination_owner.incarnation,
            request.context->origin(), int(cells), error)) return false;
    request.last_capture = 0; request.uniforms_valid = false;
    return true;
}
} // namespace strata::core::dflash2
