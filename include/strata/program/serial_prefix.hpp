#pragma once

#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/on_device.hpp"
#include <array>
#include <cstring>

namespace strata::program {

// One call owns one exact prefix. Positional resident K/V stays in place; the
// retained image is an independent byte-comparison oracle, never a cache match.
// Streamed owners restore their authoritative image and rebuild residency too.
// Every restore is verified before any suffix may read it. No cross-call reuse.
class SerialPrefixBranch {
public:
    bool capture(const std::vector<int32_t>& prefix, const core::ModelGeometry& geometry,
                 const std::vector<core::SessionState*>& sessions, std::string& error) {
        if (!stages_.empty() || prefix.empty() || sessions.empty()) {
            error = "prefix branch: empty prefix/stages or already captured"; return false;
        }
        const std::vector<core::ConversationImageKey> images;
        const std::vector<core::ConversationCheckpoint> checkpoints;
        const core::ConversationView view{prefix, images, checkpoints, true};
        size_t bytes = 0;
        int64_t next = 0;
        for (const auto* session : sessions) {
            size_t stage_bytes = 0;
            if (!session || session->layer_lo != next || session->layer_hi <= next ||
                !core::conversation_snapshot_bytes(view, *session, geometry, nullptr, stage_bytes, error) ||
                stage_bytes > SIZE_MAX - bytes) {
                if (error.empty()) error = "prefix branch: invalid stage carve or byte count";
                return false;
            }
            const int32_t previous0 = prefix.size() >= 2 ? prefix[prefix.size() - 2] : -1;
            if (session->ple_prev[0] != previous0 || session->ple_prev[1] != prefix.back()) {
                error = "prefix branch: PLE token window does not describe the prefix"; return false;
            }
            bytes += stage_bytes; next = session->layer_hi;
        }
        const auto memory = core::sample_host_memory();
        if (next != geometry.n_layers || !memory || memory->mem_available < (8ull << 30) ||
            !core::conversation_memory_admit(memory->available, bytes, 0) ||
            bytes > memory->mem_available - (8ull << 30)) {
            error = "prefix branch: incomplete model carve, allocation cap or 8 GiB host reserve"; return false;
        }
        geometry_ = geometry;
        stages_.reserve(sessions.size());
        for (auto* session : sessions) {
            Stage stage;
            stage.session = session;
            stage.session_id = session->allocation_id;
            stage.device = session->allocation_device;
            stage.ple_token = session->ple_token;
            const core::OnDevice on(stage.device);
            if (!core::conversation_snapshot_save(stage.image, view, *session, geometry_, nullptr, error)) {
                release(); return false;
            }
            stages_.push_back(std::move(stage));
        }
        return verify(error);
    }

    bool restore(std::string& error) {
        if (!validate(error)) return false; // all stages checked before the first write
        for (auto& stage : stages_) {
            const core::OnDevice on(stage.device);
            auto& session = *stage.session;
            bool resident = true;
            for (int64_t q = 0; q < session.qsa_alloc; ++q)
                resident = resident && session.qsa_states[session.qsa_ord0 + q].kv_mode == 0;
            if (resident) {
                if (!core::conversation_checkpoint_restore(stage.image.live, session, geometry_, error)) return false;
            } else if (core::conversation_snapshot_restore(stage.image, session, geometry_, nullptr, error) !=
                       core::ConversationRestore::restored) return false;
            session.ple_token = stage.ple_token;
        }
        return verify(error);
    }

    // Public diagnostic seam: an omitted restore of even one byte is an error,
    // independently of whether that particular prompt's selected logits change.
    bool verify(std::string& error) const {
        if (!validate(error)) return false;
        std::array<uint8_t, 65536> buffer;
        auto equal = [&](const void* live, const uint8_t* saved, size_t bytes, const char* name) {
            for (size_t offset = 0; offset < bytes;) {
                const size_t count = std::min(buffer.size(), bytes - offset);
                if (cudaMemcpy(buffer.data(), static_cast<const uint8_t*>(live) + offset, count,
                               cudaMemcpyDefault) != cudaSuccess || std::memcmp(buffer.data(), saved + offset, count)) {
                    error = std::string("prefix branch: restored ") + name + " differs"; return false;
                }
                offset += count;
            }
            return true;
        };
        for (const auto& stage : stages_) {
            const core::OnDevice on(stage.device);
            const auto& session = *stage.session;
            const auto& checkpoint = stage.image.live;
            core::ConversationStateSizes sizes;
            if (cudaDeviceSynchronize() != cudaSuccess ||
                !core::conversation_session_sizes(geometry_, session, sizes, error)) {
                if (error.empty()) error = "prefix branch: state drain failed";
                return false;
            }
            if (!equal(session.gdn_state, checkpoint.gdn.data(), checkpoint.gdn.size(), "GDN recurrence/convolution") ||
                !equal(session.ple_hist, checkpoint.ple.data(), checkpoint.ple.size(), "PLE history")) return false;
            const size_t tokens = checkpoint.ids.size();
            if (session.ple_token != stage.ple_token || session.ple_prev[0] != (tokens >= 2 ? checkpoint.ids[tokens - 2] : -1) ||
                session.ple_prev[1] != checkpoint.ids.back()) {
                error = "prefix branch: restored PLE host tokens differ"; return false;
            }
            for (int64_t q = 0; q < session.qsa_alloc; ++q) {
                const auto& state = session.qsa_states[session.qsa_ord0 + q];
                if (!equal(state.idx_tail, checkpoint.tails.data() + q * sizes.tail, sizes.tail, "indexer tail") ||
                    !equal(state.idx_dead, checkpoint.dead.data() + q * sizes.dead, sizes.dead, "indexer dead key") ||
                    !equal(state.idx_block_pos, checkpoint.block_pos.data() + q * sizes.block_pos, sizes.block_pos,
                           "indexer block position")) return false;
                uint64_t fingerprint = 0;
                if (!core::conversation_kv_verify(stage.image.kv[size_t(q)], state, geometry_, int64_t(tokens),
                                                  true, fingerprint, error, true)) return false;
            }
        }
        return true;
    }

    int64_t tokens() const { return stages_.empty() ? 0 : int64_t(stages_.front().image.live.ids.size()); }
    void release() { std::vector<Stage>().swap(stages_); }

private:
    struct Stage {
        core::SessionState* session = nullptr;
        core::SavedConversation image;
        uint64_t session_id = 0;
        int device = -1;
        int32_t ple_token = -1;
    };
    core::ModelGeometry geometry_;
    std::vector<Stage> stages_;
    bool validate(std::string& error) const {
        if (stages_.empty()) { error = "prefix branch: no captured prefix"; return false; }
        for (const auto& stage : stages_) {
            if (stage.session->allocation_id != stage.session_id || stage.session->allocation_device != stage.device ||
                !core::conversation_snapshot_validate(stage.image, *stage.session, geometry_, nullptr, error)) {
                if (error.empty()) error = "prefix branch: allocation identity changed";
                return false;
            }
        }
        return true;
    }
};

} // namespace strata::program
