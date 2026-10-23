#include "strata/core/dflash2_bridge.hpp"
// Host-only lifecycle regression. No CUDA initialization or neural arithmetic:
// the fixture writes five explicit host feature markers while the REAL common
// Operation owns them. The production owner, pending tickets, q validation,
// certificate publication, refusal and continuation functions execute unchanged.
#include "strata/core/neural_drafter.hpp"
#include "strata/program/continuation.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* reason) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", reason); std::exit(1); }
}
class HostRows final : public NeuralDrafter {
public:
    explicit HostRows(DraftKind kind = DraftKind::dflash2) {
        identity_.kind = kind; identity_.format = 1;
        identity_.weights[0] = 1; identity_.configuration[0] = 2; identity_.borrowed_target[0] = 3;
        identity_.geometry = dflash2::native_geometry;
        origins_.fill(kind == DraftKind::dflash2 ? 7 : 0);
        origins_[1] = kind == DraftKind::dflash2 ? 19 : 0;
        for (size_t seat = 0; seat < states_.size(); ++seat) states_[seat].cells = origins_[seat];
        window_ = 2047;
    }
    bool enabled() const override { return true; }
    uint64_t vram_bytes() const override { return 0; }
    DraftAllocationPlan allocation_plan() const override { return {}; }
    void set_draft_sampling(const strata::kernels::SamplerParams& sampling) override { sampling_[8] = sampling; }
    bool complete(const DraftOwner& ticket, int start, int accepted, std::array<bool, 5> ready,
                  std::string& error, int32_t next = -1) {
        if (!std::all_of(ready.begin(), ready.end(), [](bool done) { return done; })) {
            error = "host feature producer has not completed all five layers"; return false;
        }
        Operation operation(*this, error, ticket.seat, nullptr, next, start, &ticket);
        if (!operation) return false;
        for (int row = start; row < start + accepted; ++row)
            for (size_t tap = 0; tap < ready.size(); ++tap)
                rows_[size_t(ticket.seat)][size_t(row)][tap] = uint16_t(1 + tap + row % 31);
        const int end = start + accepted;
        operation.writes(kind() == DraftKind::dflash2 ? std::max(origins_[size_t(ticket.seat)], end - 2047) : 0,
                         end, next);
        return operation.finish(error);
    }
    bool reset_origin(int seat, int origin, std::string& error) {
        const auto expected = owner(seat);
        Operation operation(*this, error, seat, nullptr, 0, origin, &expected);
        if (!operation) return false;
        rows_[size_t(seat)] = {};
        origins_[size_t(seat)] = origin; states_[size_t(seat)].cells = origin;
        invalidate_slot_coverage(seat);
        return operation.finish(error); // no fabricated empty native certificate
    }
    uint64_t fingerprint(int seat) const {
        uint64_t hash = 14695981039346656037ull;
        for (const auto& row : rows_[size_t(seat)]) for (auto value : row) hash = (hash ^ value) * 1099511628211ull;
        return hash;
    }
    bool bad_q = false;
    bool init_slots(const std::vector<SessionState*>&, std::string& error, const strata::plan::QsaPlacement*) override {
        return unsupported(error);
    }
    bool save_slot(int, int64_t, std::string& error) override { return unsupported(error); }
    bool restore_slot(int, int64_t, std::string& error) override { return unsupported(error); }
    bool prepare_slot(int, uint64_t, int64_t, int64_t, bool, std::string& error) override { return unsupported(error); }
    bool prefill_slot(int, const float*, const int32_t*, int64_t, int64_t, std::string& error) override {
        return unsupported(error);
    }
    bool advance_slot(int, const float*, int32_t, int64_t, std::string& error) override { return unsupported(error); }
    const QsaState* legacy_state(int) const override { return nullptr; }
    uint64_t copy_bytes(int, int64_t, bool, std::string&) const override { return sizeof(rows_[0]); }
    bool snapshot_capture(SavedConversation&, Operation&, std::string& error) override { return unsupported(error); }
    bool snapshot_preflight(const SavedConversation&, int, int64_t, ConversationSnapshotExpectation,
                            std::string& error) const override {
        return unsupported(error);
    }
    bool snapshot_restore(const SavedConversation&, Operation&, int64_t, int64_t, int32_t,
                          ConversationSnapshotExpectation, std::string& error) override { return unsupported(error); }
private:
    static bool unsupported(std::string& error) {
        error = "host lifecycle fixture does not execute GPU/storage adapters"; return false;
    }
    DraftSlotState* slot_metadata(int seat) override { return seat >= 0 && seat <= 8 ? &states_[size_t(seat)] : nullptr; }
    const DraftSlotState* slot_metadata(int seat) const override {
        return seat >= 0 && seat <= 8 ? &states_[size_t(seat)] : nullptr;
    }
    int64_t state_capacity(int) const override { return 4096; }
    int64_t snapshot_origin(int seat) const override { return seat >= 0 && seat <= 8 ? origins_[size_t(seat)] : -1; }
    bool proposal_preflight(int seat, const float*, int32_t, int64_t cell, std::string& error) const override {
        if (cell + 1 == slot_cells(seat)) return true;
        error = "host proposal frontier mismatch"; return false;
    }
    bool propose_impl(Operation&, const float*, int32_t, int64_t, DraftProposal& result, std::string&) override {
        result.count = kind() == DraftKind::dflash2 ? 3 : kNeuralDraftMax;
        for (int row = 0; row < result.count; ++row) {
            result.tokens[row] = 100 + row; result.confidence[row] = .5f;
        }
        if (kind() == DraftKind::dflash2) {
            result.distribution = DraftProposal::Distribution::conditional_q;
            constexpr int half = kSpecQStride / 2;
            for (int row = 0; row < result.count; ++row)
                for (int item = 0; item < half; ++item) {
                    result.q[size_t(row) * kSpecQStride + item] = item < 16 ? 100 + item : -1;
                    result.q[size_t(row) * kSpecQStride + half + item] =
                        std::bit_cast<int32_t>(item < 16 ? 1.f / 16 : 0.f);
                }
            if (bad_q) result.q[half] = std::bit_cast<int32_t>(-.125f);
        }
        return true;
    }
    bool drain_adapter(std::string&) override { return true; } // all fixture writes are synchronous host writes
    std::array<DraftSlotState, 9> states_{};
    std::array<int, 9> origins_{};
    std::array<std::array<std::array<uint16_t, 5>, 4096>, 9> rows_{};
};
constexpr std::array<bool, 5> all_layers{true, true, true, true, true};
void lifecycle() {
    HostRows rows;
    std::string error;
    check(rows.bind_request(0, 101, error, 1) && rows.bind_request(1, 202, error, 1), "bind two independent request owners");
    const auto initial = rows.owner(0);
    DraftOwner first;
    check(rows.reserve_inputs(0, initial, 7, 4, first, error), "reserve actual target feature interval at nonzero origin");
    check(!rows.slot_coverage(0).present && !rows.idle(error), "pending target work is not completed context readiness");
    check(!rows.bind_request(0, 303, error, 2) && rows.owner(0) == first, "pending features block cancellation/reuse binding");
    check(!rows.complete(first, 7, 3, {true, true, false, true, true}, error) && rows.inputs_pending(0),
          "four completed layers do not publish a five-layer certificate");
    const auto peer_before = rows.fingerprint(1);
    check(rows.complete(first, 7, 3, all_layers, error), "commit accepted three-row prefix and drain actual owned writes");
    check(rows.slot_cells(0) == 10 && rows.slot_begin(0) == 7 && !rows.inputs_pending(0) && rows.idle(error),
          "only actual accepted feature prefix becomes ready");
    check(rows.fingerprint(1) == peer_before, "one owner's completion leaves peer features untouched");
    const auto once = rows.fingerprint(0);
    const auto completed = rows.owner(0);
    check(!rows.complete(first, 7, 3, all_layers, error) && rows.owner(0) == completed && rows.fingerprint(0) == once,
          "suspended/completed ticket transfers exactly once; stale completion is read-only refusal");
    DraftOwner second;
    check(rows.reserve_inputs(1, rows.owner(1), 19, 3, second, error), "second request reserves an independent context frontier");
    check(rows.slot_idle(0, error) && !rows.slot_idle(1, error) && !rows.context_idle(error),
          "per-seat drain permits ready decode while another seat owns target raw work; global fences still refuse");
    DraftProposal proposal;
    check(rows.propose_slot(0, nullptr, 11, 9, proposal, error) &&
          proposal.owner == rows.owner(0) && proposal.owner.request == 101 &&
          proposal.owner.incarnation == 1 && proposal.frontier == rows.slot_cells(0) &&
          rows.owner(1) == second && rows.inputs_pending(1) && rows.fingerprint(0) == once,
          "proposal binds the current producer while preserving a peer's pending capture and committed rows");
    {
        // Retain the real common owner's request/incarnation/generation and completed frontier.
        DraftProposal rounded = proposal;
        for (int row = 0; row < rounded.count; ++row) {
            rounded.confidence[row] = float(1.0 / 3.0);
            for (int item = 0; item < kSpecQEntries; ++item) {
                rounded.q[size_t(row) * kSpecQStride + item] = item < 16 ? 100 + item : -1;
                rounded.q[size_t(row) * kSpecQStride + kSpecQEntries + item] =
                    std::bit_cast<int32_t>(item < 3 ? float(1.0 / 3.0) : 0.0f);
            }
        }
        check(rounded.owner.request && rounded.owner.incarnation && rounded.owner.generation &&
              rounded.frontier == rows.slot_cells(0) && rounded.count == 3 &&
              rounded.validate(1024, error, DraftKind::dflash2),
              "owned sixteen-candidate DFlash proposal admits three rounded positive masses and thirteen zeros");
        DraftProposal three = rounded;
        for (int row = 0; row < three.count; ++row)
            std::fill_n(three.q.begin() + size_t(row) * kSpecQStride + 3, kSpecQEntries - 3, -1);
        check(three.validate(1024, error) && !three.validate(1024, error, DraftKind::dflash2),
              "numerically valid three-candidate rows cannot substitute for sixteen-candidate DFlash rows");
    }
    check(rows.complete(second, 19, 3, all_layers, error),
          "second request independently completes its own nonzero-origin context after the peer's proposal");
    const auto observed = rows.owner(0);
    {
        NeuralDrafter::Operation read(rows, error, 0, nullptr, 0, 0, &observed, nullptr, true);
        check(bool(read) && read.finish(error), "read-only lease still drains the actual common owner");
    }
    check(rows.owner(0) == observed && rows.owns(proposal.owner), "fingerprint/readback does not invalidate a live proposal generation");
    DraftOwner cancelled;
    check(rows.reserve_inputs(0, rows.owner(0), 10, 4, cancelled, error) && rows.discard_inputs(cancelled, error),
          "drained unconsumed target capture can be discarded without inventing coverage");
    check(!rows.slot_coverage(0).present && rows.release_identity(0, 101, 1, error) && rows.bind_request(0, 303, error, 2),
          "cancelled physical seat rebinds with an increased incarnation");
    check(rows.reset_origin(0, 31, error), "reuse resets actual storage without manufacturing a certificate");
    const auto reused = rows.owner(0); const auto reused_bytes = rows.fingerprint(0);
    check(!rows.complete(cancelled, 10, 4, all_layers, error) && rows.owner(0) == reused && rows.fingerprint(0) == reused_bytes,
          "late cancelled completion cannot mutate successor storage or generation");
    const std::vector<int32_t> ids(31, 7);
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> checkpoints;
    ConversationView view{ids, images, checkpoints, true};
    rows.certify_view(view, 0);
    check(view.draft_tokens == -1 && view.draft_owner == 0, "token count and origin cannot reconstruct an absent draft certificate");
}
void rolling_and_displaced() {
    HostRows rows;
    std::string error;
    check(rows.bind_request(8, 401, error, 1), "bind working owner");
    DraftOwner ticket;
    check(rows.reserve_inputs(8, rows.owner(), 7, 2048, ticket, error) && rows.complete(ticket, 7, 2048, all_layers, error),
          "project the real retained tail of a 2048-row target chunk");
    check(rows.inputs_completed(8, 7, 2048) && rows.slot_begin(8) == 8,
          "completed 2048-row chunk acknowledges its actual start without claiming its evicted first row");
    check(rows.reserve_inputs(8, rows.owner(), 2055, 1, ticket, error) && rows.complete(ticket, 2055, 1, all_layers, error),
          "append and evict actual context rows");
    const auto coverage = rows.slot_coverage(8);
    check(coverage.coherent && coverage.certified_begin == 9 && coverage.certified_end == 2056,
          "DFlash coverage names exact retained ring, not a union including evicted features");
    strata::program::ContinuationInput in;
    in.request = 401; in.incarnation = 1; in.committed_tokens = 2056;
    in.draft_present = true; in.draft_request = coverage.owner.request; in.draft_incarnation = coverage.owner.incarnation;
    in.draft_kind = DraftKind::dflash2; in.draft_origin = 7;
    in.draft_begin = 9; in.draft_end = in.draft_frontier = 2056;
    const auto good = strata::program::certify_continuation(in);
    check(good.certified && good.draft_ready, "nonzero-origin native N context is continuation-ready without an MTP witness");
    in.draft_begin = 8;
    check(!strata::program::certify_continuation(in).certified, "evicted native history cannot be advertised as ready");
    const auto old = rows.owner();
    check(rows.bind_request(8, 402, error, 2), "reserve successor before outgoing read-only capture");
    const std::vector<int32_t> ids(2056, 7);
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> checkpoints;
    ConversationView view{ids, images, checkpoints, true};
    rows.certify_view(view);
    check(view.draft_tokens == -1, "successor binding alone cannot retag predecessor certificate");
    rows.certify_view(view, 8, &old);
    check(view.draft_owner == 401 && view.draft_begin == 9 && view.draft_tokens == 2056,
          "unchanged-generation displaced source keeps its real producer certificate");
    check(rows.reset_origin(8, 7, error), "new owner begins its actual storage reset");
    rows.certify_view(view, 8, &old);
    check(view.draft_tokens == -1, "displaced source capability expires on actual reuse");
}
void proposal_and_identity() {
    HostRows rows;
    std::string error;
    DraftOwner ticket;
    check(rows.bind_request(0, 501, error, 1) && rows.reserve_inputs(0, rows.owner(0), 7, 3, ticket, error) &&
          rows.complete(ticket, 7, 3, all_layers, error), "prepare sampled q delivery fixture");
    rows.bad_q = true;
    DraftProposal output;
    const auto bytes = rows.fingerprint(0);
    check(!rows.propose_slot(0, nullptr, 1, 9, output, error) && output.count == 0 && !rows.context_idle(error),
          "malformed conditional q poisons common owner before any partial proposal publication");
    check(rows.fingerprint(0) == bytes && !rows.bind_request(0, 502, error, 2), "failed proposal cannot mutate KV or be reused as a fallback owner");
    HostRows mtp(DraftKind::mtp);
    check(mtp.bind_request(0, 601, error, 1) && mtp.reserve_inputs(0, mtp.owner(0), 0, 12, ticket, error) &&
          mtp.complete(ticket, 0, 12, all_layers, error, 42), "prepare MTP private witness");
    check(mtp.propose_slot(0, nullptr, 42, 11, output, error) && output.count == kNeuralDraftMax && output.count >= 7,
          "common bounded proposal does not narrow existing MTP widths to DFlash's three");
    const std::vector<int32_t> ids(12, 42);
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> checkpoints;
    ConversationView view{ids, images, checkpoints, true};
    mtp.certify_view(view, 0);
    check(view.draft_tokens == 12 && view.draft_next_token == 42, "private MTP N retains its actual next-token witness");
    view.exact_cut = true; mtp.certify_view(view, 0);
    check(view.draft_tokens == 11 && view.draft_next_token == -1, "portable MTP cut remains N-1, never a reconstructed N");
    ConversationBuffer boundary;
    boundary.resize(40960); boundary.freeze();
    view.draft_boundary_residual = &boundary; view.draft_boundary_position = 11; view.draft_boundary_owner = 601;
    mtp.certify_view(view, 0);
    check(view.draft_boundary_residual == &boundary && view.draft_boundary_position == 11 &&
          view.draft_boundary_owner == 601 && view.draft_tokens == 11,
          "common recertification preserves an existing immutable position/producer-bound resident R witness");
    const auto expected = rows.identity();
    for (int field = 0; field < 6; ++field) {
        auto foreign = expected;
        if (field == 0) foreign.kind = DraftKind::mtp;
        if (field == 1) foreign.weights[0] ^= 1;
        if (field == 2) foreign.borrowed_target[0] ^= 1;
        if (field == 3) ++foreign.geometry[2];
        if (field == 4) foreign.configuration[0] ^= 1;
        if (field == 5) ++foreign.format;
        check(!expected.content_matches(foreign),
              "persistent draft compatibility refuses kind, content, geometry and format changes");
    }
    auto rebound = expected;
    ++rebound.load_generation;
    check(expected.content_matches(rebound) && expected != rebound,
          "persistent content may rebind after reload without equating distinct live allocations");
}
}
int main() {
    lifecycle(); rolling_and_displaced(); proposal_and_identity();
    std::printf("neural lifecycle: %d checks passed\n", checks);
}
