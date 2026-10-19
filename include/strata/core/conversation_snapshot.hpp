#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_reuse.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <array>
#include <string>

namespace strata::core {

// Process-immutable native resident handoff selection, resolved before admission.
// Only exact "1" enables it; unsupported layouts fail without a host-image fallback.
bool resident_direct_copy_enabled();
// Copy between existing same-device owners, using their authoritative pools (which
// may already be mapped host pools). No intermediate image or CPU fallback.
// Caller joins all consumers and synchronizes before copying. Success completes
// map/ring repair, including on nonblocking consumers. With `queued`, copies and
// repairs are only ordered; the caller must synchronize before either owner is reused.
// False after a write invalidates the destination. Validation performs no CUDA calls.
bool conversation_kv_copy_validate(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                                   int64_t upto, bool include_index, std::string& error);
bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error, bool queued = false);
// The 18-field runtime geometry key every captured image carries.  The RAM conversation cache and the persistent
// conversation store bucket on exactly this key, so a caller can build a store selector before it owns an image.
std::array<int64_t, 18> conversation_geometry_key(const ModelGeometry& g);
// KV first, then live running state or the selected host checkpoint, so a cut's
// spare pooled row is repaired after copying rows from the longer-lived source.
// Caller joins consumers and keeps owner metadata stable; this function owns
// the entry device fence, after all-layer validation, and completion fences.
bool conversation_session_copy(const SessionState& from, SessionState& to, const ModelGeometry& g,
                               const std::vector<int32_t>& ids, const ConversationCheckpoint* at,
                               std::string& error);

// Caller synchronizes the device before saving or restoring into an active owner.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
// `reused_bytes` sums the bytes this capture kept instead of copying; `retained`, when given, reports them per part
// (k, v, k_scale, v_scale, pooled) so a caller can name the exact intervals an already-stored image may donate. Both
// describe the same bytes: the leading bytes of the part that came from the retained image, never a byte the capture
// wrote, and never anything established by comparing or hashing an old payload.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr,
                          std::array<uint64_t, 5>* retained = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// Native shared mounts must select exact_cut explicitly. Record metadata never
// selects this contract: its owning draft must claim N-1 cells, and all excluded
// next-token/page-padding bytes must be zero. Ordinary/private APIs below keep
// their existing validation and backend signatures.
enum class ConversationSnapshotExpectation { ordinary, exact_cut };
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, ConversationSnapshotExpectation expectation,
                              std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, ConversationSnapshotExpectation expectation,
                             std::string& error, int64_t captured_tokens = -1);
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, ConversationSnapshotExpectation expectation,
                            uint64_t& fingerprint, std::string& error, int64_t captured_tokens = -1);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
// Success completes authoritative copies and any streaming-map/draft-ring repair;
// no caller completion fence is needed, even for nonblocking compute streams.
// Repair submission or synchronization failure returns false and may leave partial
// state: the destination MUST NOT resume inference.
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error,
                             int64_t captured_tokens = -1);
// The first `upto` tokens of an image captured at `captured` tokens. The image stays untouched;
// rows beyond the prefix in its last page are overwritten before a request can read them.
bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a successful restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error,
                            int64_t captured_tokens = -1);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Clear rows outside [begin,end) in a captured, page-rounded cut. No device read
// or allocation; layout is [page][head][cell][row bytes], including scale runs.
bool conversation_kv_clear_inactive(ConversationKv& image, int64_t tokens, int64_t begin, int64_t end,
                                    std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
// A null draft has no source/limit entry, matching the ordinary optional-draft snapshot contract.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState* draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);
// conversation_checkpoint_save(from) + conversation_checkpoint_restore(to) of `ids` without a host image, for two
// sessions carved alike on the same device; the caller synchronizes the device before (as for the restore).
// `queued`: the copies are only queued (the private copy stream), and the caller synchronizes the device after too.
bool conversation_checkpoint_copy(const SessionState& from, SessionState& to, const std::vector<int32_t>& ids,
                                  const ModelGeometry& g, std::string& error, bool queued = false);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
    // Opt-in export of exactly this token-zero cut. Private captures keep their
    // existing rounded payloads. A draft row i consumes token i+1, so the owning
    // draft must claim exactly N-1 cells. Skipped rows before draft_begin and
    // every inactive/page-padding byte are cleared before publication.
    bool exact_cut = false;
    // Actual completed interval (unknown: -1/-1/-1), never prompt arithmetic.
    int64_t draft_tokens = -1;
    int64_t draft_begin = -1;
    int32_t draft_next_token = -1;
    // Nonzero producer request owning the certification. Zero permits only a
    // legacy shared-cut sanitizing extent; it exports no persistent witness.
    uint64_t draft_owner = 0;
    // Immutable host row, explicitly position/owner bound by the producer after
    // its source operation drains. A bare device pointer cannot certify a row.
    const ConversationBuffer* draft_boundary_residual = nullptr;
    int64_t draft_boundary_position = -1;
    uint64_t draft_boundary_owner = 0;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
// `retained`, when given, receives the K/V prefix each layer and every stage part kept from `reuse` instead of
// copying: the exact bytes an already-stored image (the one this reuse was retained from) can donate, with the
// caller's own stage tag left at 0. Nothing here reads, hashes or compares the donor's payload.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr,
                                std::vector<ConversationRetainedPrefix>* retained = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft, when present.
// Caller has synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState* draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Success includes complete
// map/ring residency repair, even for nonblocking compute streams. Transfer failure
// (including repair synchronization) may leave partial state: caller MUST NOT
// continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);
// The K/V of the image's first `upto` tokens (main layers and the draft layer), not its running state: the caller
// mounts a checkpoint taken at `upto`.  For a checkpoint of a parked conversation that a new conversation starts
// with, while the parked one stays where it is (ConversationCache::borrows).  Same failure contract as above.
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr,
                                std::vector<ConversationRetainedPrefix>* retained = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error,
                                                   const ConversationCheckpoint* selected = nullptr);

// Native-only explicit contract for shared cuts. Pass it for every stage, even
// those without a draft owner, and for subsequent restore/diagnostic read-back.
// Ordinary private callers deliberately use the signatures above.
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft,
                                    ConversationSnapshotExpectation expectation, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft,
                                    ConversationSnapshotExpectation expectation, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft,
                                                   ConversationSnapshotExpectation expectation, std::string& error,
                                                   const ConversationCheckpoint* selected = nullptr);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   ConversationSnapshotExpectation expectation, std::string& error);
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState* draft, std::string& error);

} // namespace strata::core
