#include "strata/core/dflash2_bridge.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <poll.h>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::core::dflash2 {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little,
              "DFlash2 wire tensors require a little-endian host");
constexpr unsigned all_taps = (1u << tap_layers.size()) - 1;
constexpr size_t hidden_values = (block_size - 1) * hidden_size;
constexpr size_t embedding_values = block_size * hidden_size;
constexpr size_t proposal_values = (block_size - 1) * 17;

// No borrowed work or exchange bytes have started when this refusal is raised.
struct ScratchUnavailable : Refusal { using Refusal::Refusal; };

void require(bool condition, const char* message) {
    if (!condition) throw Refusal(message);
}
void validate_operation(const Operation& op) {
    require(!op.request_id.empty() && op.request_id.size() <= 128, "invalid request identity");
    for (char c : op.request_id)
        require((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ':' || c == '-',
                "invalid request identity");
    require(op.incarnation >= 0 && op.generation > 0 && op.frontier >= 0 &&
            op.frontier <= max_position - block_size, "invalid operation extent/generation");
}
void validate_identity(const TargetIdentity& id) {
    for (const auto* digest : {&id.pack_sha256, &id.embedding_sha256, &id.head_sha256, &id.tokenizer_sha256}) {
        require(digest->size() == 64, "invalid borrowed target digest");
        for (char c : *digest) require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'),
                                     "invalid borrowed target digest");
    }
}
void require_fp32_environment() {
    require(std::fegetround() == FE_TONEAREST, "capture requires FP32 round-to-nearest-even");
    // Refuse an inherited FTZ/DAZ mode rather than silently changing the CPU
    // reference reduction for small finite target residuals.
    volatile float tiny = std::numeric_limits<float>::denorm_min();
    volatile float one = 1.0f;
    volatile float preserved = tiny * one;
    require(std::bit_cast<uint32_t>(float(preserved)) == 1, "capture requires gradual FP32 underflow");
}
Json identity_json(const TargetIdentity& id) {
    return {{"pack_sha256", id.pack_sha256}, {"embedding_sha256", id.embedding_sha256},
            {"head_sha256", id.head_sha256}, {"tokenizer_sha256", id.tokenizer_sha256}};
}
Json operation_json(const Operation& op) {
    return {{"request_id", op.request_id}, {"incarnation", op.incarnation},
            {"generation", op.generation}, {"frontier", op.frontier}};
}
Json tensor(const char* name, const char* dtype, std::initializer_list<int> shape,
            size_t bytes, size_t offset = 0) {
    return {{"name", name}, {"dtype", dtype}, {"shape", shape}, {"offset", offset}, {"bytes", bytes}};
}
Json header(const char* phase, const Operation& op, const TargetIdentity& id,
            Json details, Json tensors, size_t payload_bytes) {
    return {{"version", 1}, {"phase", phase}, {"operation", operation_json(op)},
            {"target", identity_json(id)}, {"details", std::move(details)},
            {"tensors", std::move(tensors)}, {"payload_bytes", payload_bytes}};
}

// Refuse duplicate keys (json's default is last-key-wins), excess nesting and
// floating header numbers (json equality otherwise equates 1.0 and integer 1).
Json strict_header(std::string_view wire) {
    std::array<std::vector<std::string>, 8> keys;
    size_t objects = 0;
    auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
        require(depth < 8, "exchange header nesting limit exceeded");
        if (event == Json::parse_event_t::object_start) {
            require(objects < keys.size(), "exchange object nesting limit exceeded");
            keys[objects++].clear();
        } else if (event == Json::parse_event_t::key) {
            require(objects != 0, "unowned exchange JSON key");
            auto& names = keys[objects - 1];
            const auto& key = parsed.get_ref<const std::string&>();
            require(std::find(names.begin(), names.end(), key) == names.end(), "duplicate exchange JSON key");
            names.push_back(key);
        } else if (event == Json::parse_event_t::object_end) {
            require(objects != 0, "invalid exchange JSON object");
            --objects;
        } else if (event == Json::parse_event_t::value) {
            require(!parsed.is_number_float(), "exchange header numbers must be integers");
        }
        return true;
    };
    try {
        return Json::parse(wire.begin(), wire.end(), callback);
    } catch (const Json::exception&) {
        throw Refusal("malformed exchange JSON header");
    }
}

struct OwnedFd {
    explicit OwnedFd(int value) : value(value) {}
    ~OwnedFd() { if (value >= 0) ::close(value); }
    void close() noexcept { if (value >= 0) ::close(std::exchange(value, -1)); }
    int value;
};
class Channel {
public:
    Channel(int socket, std::chrono::milliseconds timeout) : fd_(socket), timeout_(timeout) {
        require(fd_.value >= 0 && timeout.count() > 0 && timeout <= std::chrono::seconds(300),
                "invalid inherited exchange fd/deadline");
        int type = 0, listening = 0;
        socklen_t size = sizeof(type);
        require(getsockopt(fd_.value, SOL_SOCKET, SO_TYPE, &type, &size) == 0 && type == SOCK_STREAM,
                "exchange requires an AF_UNIX stream socketpair");
        size = sizeof(listening);
        require(getsockopt(fd_.value, SOL_SOCKET, SO_ACCEPTCONN, &listening, &size) == 0 && !listening,
                "exchange cannot adopt a listening socket");
        sockaddr_un address{};
        size = sizeof(address);
        require(getsockname(fd_.value, reinterpret_cast<sockaddr*>(&address), &size) == 0 &&
                address.sun_family == AF_UNIX && size == offsetof(sockaddr_un, sun_path),
                "exchange requires an anonymous local socketpair");
        size = sizeof(address);
        require(getpeername(fd_.value, reinterpret_cast<sockaddr*>(&address), &size) == 0 &&
                address.sun_family == AF_UNIX && size == offsetof(sockaddr_un, sun_path),
                "exchange requires an anonymous connected peer");
    }
    std::chrono::milliseconds lease_wait_limit() const {
        return std::min(timeout_, OperationLease::wait_limit);
    }
    void begin(Clock::time_point started) {
        require(!poisoned_ && !active_, "poisoned/busy exchange");
        deadline_ = started + timeout_;
        active_ = true;
    }
    // receive() already validated the complete ack against the deadline. Once
    // acknowledged, publication must not acquire a new failure point.
    void finish() noexcept { active_ = false; }
    void abort() noexcept { poisoned_ = true; active_ = false; fd_.close(); }
    void remaining() const {
        require(!poisoned_ && active_ && Clock::now() < deadline_, "exchange transaction deadline/ownership failure");
    }
    void send(const Json& document, std::initializer_list<std::span<const std::byte>> payloads) {
        const std::string wire = document.dump();
        size_t payload_size = 0;
        for (auto payload : payloads) payload_size += payload.size();
        require(wire.size() >= 2 && wire.size() <= max_header_bytes &&
                payload_size <= max_feature_bytes + embedding_values * sizeof(uint16_t),
                "oversized exchange frame");
        const std::array<uint32_t, 3> prefix{0x324c4644u, uint32_t(wire.size()), uint32_t(payload_size)};
        transfer(std::as_bytes(std::span{prefix}));
        transfer(std::as_bytes(std::span{wire.data(), wire.size()}));
        for (auto payload : payloads) transfer(payload);
    }
    void receive(const Json& expected, std::span<std::byte> output) {
        std::array<uint32_t, 3> prefix;
        transfer(std::as_writable_bytes(std::span{prefix}));
        require(prefix[0] == 0x324c4644u && prefix[1] >= 2 && prefix[1] <= max_header_bytes &&
                prefix[2] == output.size(), "invalid/oversized exchange frame prefix");
        std::array<char, max_header_bytes> wire;
        transfer(std::as_writable_bytes(std::span{wire.data(), prefix[1]}));
        require(strict_header({wire.data(), prefix[1]}) == expected,
                "stale identity or exchange phase/dtype/shape/extent mismatch");
        transfer(output);
    }
private:
    template<size_t Extent> void transfer(std::span<const std::byte, Extent> bytes) {
        transfer(bytes.data(), nullptr, bytes.size(), true);
    }
    template<size_t Extent> void transfer(std::span<std::byte, Extent> bytes) {
        transfer(nullptr, bytes.data(), bytes.size(), false);
    }
    void transfer(const std::byte* source, std::byte* destination, size_t size, bool sending) {
        while (size != 0) {
            remaining();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline_ - Clock::now()).count();
            const int wait_ms = int(std::clamp<int64_t>((ns + 999999) / 1000000, 1, INT_MAX));
            pollfd wait{fd_.value, short(sending ? POLLOUT : POLLIN), 0};
            const int ready = ::poll(&wait, 1, wait_ms);
            if (ready < 0 && errno == EINTR) continue;
            require(ready > 0, "exchange poll timeout/failure");
            require(!(wait.revents & POLLNVAL), "invalid exchange fd");
            const ssize_t count = sending
                ? ::send(fd_.value, source, size, MSG_DONTWAIT | MSG_NOSIGNAL)
                : ::recv(fd_.value, destination, size, MSG_DONTWAIT);
            if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            require(count > 0, "EOF or I/O failure in owned exchange");
            if (sending) source += count;
            else destination += count;
            size -= size_t(count);
        }
        remaining();
    }
    OwnedFd fd_;
    std::chrono::milliseconds timeout_;
    Clock::time_point deadline_{};
    bool active_ = false, poisoned_ = false;
};
} // namespace

CaptureStore::CaptureStore(std::string request_id, int64_t incarnation, int origin, int capacity)
    : request_id_(std::move(request_id)), incarnation_(incarnation), origin_(origin), capacity_(capacity), end_(origin) {
    validate_operation({request_id_, incarnation_, 1, origin});
    require(capacity > 0 && capacity <= context_rows, "capture capacity must be in [1,2047]");
    ring_ = std::make_unique_for_overwrite<uint16_t[]>(size_t(context_rows) * feature_width);
    tentative_ = std::make_unique_for_overwrite<uint16_t[]>(size_t(capacity) * feature_width);
}
CaptureStore::~CaptureStore() = default;
size_t CaptureStore::storage_bytes() const noexcept {
    return (size_t(context_rows) + size_t(capacity_)) * feature_row_bytes;
}
bool CaptureStore::owns(const Operation& op) const {
    return op.request_id == request_id_ && op.incarnation == incarnation_;
}
void CaptureStore::abort() noexcept { poisoned_ = true; active_ = false; seen_ = 0; }
[[noreturn]] void CaptureStore::refuse(const char* message) { abort(); throw Refusal(message); }
void CaptureStore::begin(const Operation& op, int start, int count, CapturePhase phase) {
    try {
        validate_operation(op);
        require(!poisoned_ && !active_ && owns(op) && op.generation > generation_,
                "stale/unowned or busy capture");
        require((phase == CapturePhase::prefill || phase == CapturePhase::verify) &&
                count > 0 && count <= capacity_ && start == end_ && start <= max_position - count,
                "noncontiguous or oversized capture extent");
        require(phase == CapturePhase::prefill ? start + count == op.frontier :
                (start == op.frontier && count <= block_size), "capture phase/frontier mismatch");
        operation_ = op;
        start_ = start; count_ = count; phase_ = phase; generation_ = op.generation;
        seen_ = 0; active_ = true;
    } catch (...) { abort(); throw; }
}
void CaptureStore::capture(const Operation& op, int layer, int materialized_layer,
                          std::span<const int> positions, CapturePoint point,
                          std::span<const float> residuals) {
    try {
        require(!poisoned_ && active_ && op == operation_, "stale/unowned capture completion");
        const auto found = std::find(tap_layers.begin(), tap_layers.end(), layer);
        require(found != tap_layers.end(), "unknown capture tap layer");
        const size_t tap = size_t(found - tap_layers.begin());
        require(!(seen_ & (1u << tap)), "duplicate capture tap layer");
        require(materialized_layer == layer + (phase_ == CapturePhase::verify ? 1 : 0) &&
                point == (phase_ == CapturePhase::verify ? CapturePoint::next_layer_first_gr_read_group :
                          CapturePoint::completed_second_residual_write),
                "capture requires the completed unmixed residual at its declared seam");
        require(positions.size() == size_t(count_) && residuals.size() == size_t(count_) * 4 * hidden_size,
                "capture position/residual shape mismatch");
        for (int row = 0; row < count_; ++row)
            require(positions[size_t(row)] == start_ + row, "capture positions contain gaps/duplicates/reordering");
        require_fp32_environment();
        for (int row = 0; row < count_; ++row) {
            const float* streams = residuals.data() + size_t(row) * 4 * hidden_size;
            uint16_t* output = tentative_.get() + size_t(row) * feature_width + tap * hidden_size;
            for (int column = 0; column < hidden_size; ++column) {
                const float s0 = streams[column], s1 = streams[hidden_size + column];
                const float s2 = streams[2 * hidden_size + column], s3 = streams[3 * hidden_size + column];
                require(std::isfinite(s0) && std::isfinite(s1) && std::isfinite(s2) && std::isfinite(s3),
                        "nonfinite target residual");
                // Volatile stores make the four FP32 rounding points explicit,
                // even with an enclosing engine built with contraction enabled.
                volatile float sum01 = s0 + s1;
                volatile float sum012 = sum01 + s2;
                volatile float sum0123 = sum012 + s3;
                const float mean = sum0123 * 0.25f;
                const uint16_t bits = kernels::bf16_from_f32(mean);
                require((bits & 0x7f80u) != 0x7f80u, "nonfinite reduced target feature");
                output[column] = bits;
            }
        }
        seen_ |= 1u << tap;
    } catch (...) { abort(); throw; }
}
void CaptureStore::check_commit(const Operation& op, int keep) {
    if (poisoned_ || !active_ || op != operation_ || seen_ != all_taps ||
        keep < 1 || keep > count_ || (phase_ == CapturePhase::prefill && keep != count_))
        refuse("cannot publish incomplete/stale capture or invalid committed prefix");
}
void CaptureStore::commit(const Operation& op, int keep) {
    check_commit(op, keep);
    for (int row = 0; row < keep; ++row) {
        const int slot = (head_ + retained_) % context_rows;
        std::memcpy(ring_.get() + size_t(slot) * feature_width,
                    tentative_.get() + size_t(row) * feature_width, feature_row_bytes);
        if (retained_ == context_rows) head_ = (head_ + 1) % context_rows;
        else ++retained_;
    }
    end_ = start_ + keep;
    active_ = false; seen_ = 0;
}
void CaptureStore::discard(const Operation& op) {
    if (poisoned_ || !active_ || op != operation_) refuse("cannot discard an unowned capture");
    active_ = false; seen_ = 0;
}
FeatureView CaptureStore::committed_since(int start) const {
    require(!poisoned_ && !active_ && start >= retained_start() && start <= end_,
            "committed feature coverage is unavailable or capture is still tentative");
    FeatureView result;
    result.start = start; result.count = end_ - start;
    const int slot = (head_ + start - retained_start()) % context_rows;
    const int first = std::min(result.count, context_rows - slot);
    result.segments[0] = {ring_.get() + size_t(slot) * feature_width, size_t(first) * feature_width};
    result.segments[1] = {ring_.get(), size_t(result.count - first) * feature_width};
    return result;
}

struct Bridge::Impl {
    // Channel takes fd ownership before any validation or storage allocation.
    Channel channel;
    CaptureStore& store;
    BorrowedTarget& target;
    BorrowedBinding binding;
    std::array<float, embedding_values> raw;
    std::array<uint16_t, embedding_values> embeddings;
    std::array<uint16_t, hidden_values> hidden;
    std::array<float, hidden_size> head_input;
    std::unique_ptr<float[]> logits;
    std::array<int64_t, proposal_values> proposals;
    Operation operation;
    int reference_end = -1;
    int64_t generation = 0;
    bool awaiting_commit = false, poisoned = false;

    Impl(CaptureStore& store, BorrowedTarget& target, const TargetIdentity& expected,
         OwnedFd& fd, std::chrono::milliseconds timeout)
        : channel(std::exchange(fd.value, -1), timeout), store(store), target(target), binding(target.binding()) {
        validate_identity(expected);
        require(binding.identity == expected && binding.hidden == hidden_size &&
                binding.vocabulary == vocabulary_size && binding.load_generation > 0,
                "borrowed daily target identity/geometry/load mismatch");
        require(!store.poisoned() && !store.capturing() && !store.bridge_attached_,
                "capture store is poisoned/busy or already has a bridge");
        logits = std::make_unique_for_overwrite<float[]>(size_t(block_size - 1) * vocabulary_size);
        store.bridge_attached_ = true;
    }
    ~Impl() {
        if (awaiting_commit) abort();
        store.bridge_attached_ = false;
    }
    void check_binding() {
        require(target.binding() == binding, "borrowed target was reloaded/replaced during an owned operation");
    }
    void abort() noexcept { poisoned = true; awaiting_commit = false; channel.abort(); store.abort(); }
};

Bridge::Bridge(CaptureStore& store, BorrowedTarget& target, const TargetIdentity& expected,
               int fd, std::chrono::milliseconds timeout) {
    // The stack owner also closes fd if allocating Impl itself fails.
    OwnedFd incoming(fd);
    impl_ = std::make_unique<Impl>(store, target, expected, incoming, timeout);
}
Bridge::~Bridge() = default;
bool Bridge::poisoned() const noexcept { return impl_->poisoned; }
void Bridge::abort() noexcept { impl_->abort(); }
Proposals Bridge::propose(const Operation& op, int32_t anchor) {
    auto& state = *impl_;
    try {
        validate_operation(op);
        require(!state.poisoned && !state.awaiting_commit && state.store.owns(op) &&
                op.generation > state.generation && op.generation > state.store.generation_ &&
                op.frontier == state.store.frontier(),
                "stale/unowned or outstanding proposal operation");
        require(anchor >= 0 && anchor < vocabulary_size, "anchor outside borrowed vocabulary");
        state.check_binding();
        const int start = state.reference_end < 0 ? std::max(state.store.origin(), op.frontier - context_rows)
                                                  : state.reference_end;
        const auto features = state.store.committed_since(start);
        const auto started = Clock::now();
        {
            OperationLease::Guard lease(state.target.scratch_lease(), state.channel.lease_wait_limit());
            if (!lease) {
                const auto failure = lease.failure();
                if (failure == OperationLease::Failure::busy || failure == OperationLease::Failure::timeout)
                    throw ScratchUnavailable("borrowed native-head scratch wait timed out/busy");
                throw Refusal("borrowed native-head scratch is poisoned");
            }
            state.channel.begin(started);
            const std::array<int32_t, block_size> ids{anchor, mask_token, mask_token, mask_token};
            std::fill(state.raw.begin(), state.raw.end(), std::numeric_limits<float>::quiet_NaN());
            state.target.raw_embeddings(op, ids, state.raw);
            state.check_binding();
            state.channel.remaining();
            for (size_t index = 0; index < state.raw.size(); ++index) {
                const uint16_t bits = kernels::bf16_from_f32(state.raw[index]);
                require(std::isfinite(state.raw[index]) && (bits & 0x7f80u) != 0x7f80u,
                        "nonfinite/incomplete borrowed raw embedding");
                state.embeddings[index] = bits;
            }
            const size_t feature_bytes = size_t(features.count) * feature_row_bytes;
            state.channel.send(header("begin", op, state.binding.identity,
                {{"origin", state.store.origin()}, {"feature_start", features.start},
                 {"feature_count", features.count}, {"embedding_ids", ids}},
                Json::array({tensor("features", "bf16", {1, features.count, feature_width}, feature_bytes),
                             tensor("embeddings", "bf16", {1, block_size, hidden_size},
                                    sizeof(state.embeddings), feature_bytes)}), feature_bytes + sizeof(state.embeddings)),
                {std::as_bytes(features.segments[0]), std::as_bytes(features.segments[1]),
                 std::as_bytes(std::span{state.embeddings})});
            state.channel.receive(header("hidden", op, state.binding.identity, {{"origin", state.store.origin()}},
                Json::array({tensor("hidden", "bf16", {1, block_size - 1, hidden_size}, sizeof(state.hidden))}),
                sizeof(state.hidden)), std::as_writable_bytes(std::span{state.hidden}));
            for (int row = 0; row < block_size - 1; ++row) {
                state.check_binding();
                for (int column = 0; column < hidden_size; ++column) {
                    const uint16_t bits = state.hidden[size_t(row) * hidden_size + column];
                    require((bits & 0x7f80u) != 0x7f80u, "nonfinite CPU draft hidden row");
                    state.head_input[size_t(column)] = kernels::f32_from_bf16(bits);
                }
                std::span<float> output{state.logits.get() + size_t(row) * vocabulary_size, size_t(vocabulary_size)};
                std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
                state.target.native_head(op, state.head_input, output);
                state.check_binding();
                state.channel.remaining();
                for (float value : output) require(std::isfinite(value), "nonfinite/incomplete native-head logits");
            }
            lease.complete(); // all borrowed target consumers are host-complete
        } // release shared scratch before sending logits or waiting on the selector
        state.channel.send(header("logits", op, state.binding.identity, Json::object(),
            Json::array({tensor("logits", "f32", {1, block_size - 1, vocabulary_size}, logits_bytes)}), logits_bytes),
            {std::as_bytes(std::span{state.logits.get(), size_t(block_size - 1) * vocabulary_size})});
        state.channel.receive(header("proposals", op, state.binding.identity, Json::object(),
            Json::array({tensor("tokens", "i64", {1, block_size - 1}, (block_size - 1) * sizeof(int64_t)),
                         tensor("candidates", "i64", {1, block_size - 1, 16},
                                (block_size - 1) * 16 * sizeof(int64_t), (block_size - 1) * sizeof(int64_t))}),
            sizeof(state.proposals)), std::as_writable_bytes(std::span{state.proposals}));
        Proposals result;
        for (int row = 0; row < block_size - 1; ++row) {
            const int64_t token = state.proposals[size_t(row)];
            auto& candidates = result.candidates[size_t(row)];
            for (size_t column = 0; column < candidates.size(); ++column) {
                const int64_t candidate = state.proposals[block_size - 1 + size_t(row) * 16 + column];
                require(candidate >= 0 && candidate < vocabulary_size, "selector candidate outside vocabulary");
                require(std::find(candidates.begin(), candidates.begin() + column, candidate) == candidates.begin() + column,
                        "duplicate selector candidate");
                candidates[column] = candidate;
            }
            require(std::find(candidates.begin(), candidates.end(), token) != candidates.end(),
                    "selector proposal is not in its candidate row");
            result.tokens[size_t(row)] = token;
        }
        state.operation = op; state.generation = op.generation;
        state.reference_end = op.frontier; state.awaiting_commit = true;
        return result;
    } catch (const ScratchUnavailable&) { throw; }
    catch (...) { state.abort(); throw; }
}
void Bridge::commit_verified(const Operation& op, int keep) {
    auto& state = *impl_;
    try {
        require(!state.poisoned && state.awaiting_commit && op == state.operation,
                "stale/unowned verifier acknowledgement");
        state.check_binding();
        require(state.store.phase_ == CapturePhase::verify, "verifier acknowledgement requires verification capture");
        state.store.check_commit(op, keep);
        state.channel.send(header("commit", op, state.binding.identity, {{"keep", keep}}, Json::array(), 0), {});
        state.channel.receive(header("ack", op, state.binding.identity, {{"keep", keep}}, Json::array(), 0), {});
        state.store.commit(op, keep);
        state.channel.finish();
        state.awaiting_commit = false;
    } catch (...) { state.abort(); throw; }
}
} // namespace strata::core::dflash2
