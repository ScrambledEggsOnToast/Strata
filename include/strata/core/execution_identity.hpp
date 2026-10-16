// strata/core/execution_identity.hpp - the anonymous arithmetic selection of a run: every setting that can change the
// bits a stored conversation image holds, in one canonical, auditable byte sequence.
//
// Persistent images are bound to their execution settings and state interpretation. This does not assert that
// independent executions produce identical floating-point values. Unknown STRATA_* controls are included
// conservatively; only explicitly reviewed non-arithmetic names are excluded.
//
// MEMBERSHIP RULE.  A control is bound when it can change the bits this image holds.  The binding is deliberately
// *finer* than the effective value where no accessor reports it: the variable's exact bytes are bound (unset, set
// empty and every distinct value are three different bindings), because a bind that is finer than the truth only
// costs a cache miss, while one that is coarser lets a wrong image be restored.  Where an accessor *does* report the
// effective value (the route tail rank, the draft layer's formats, the CPU ISA the dispatch resolved) the caller
// passes that value as a "resolved" entry, so a clamped or per-type rule is bound as the engine applies it.
//
// Exclusions are exact names, never family prefixes: a future STRATA_BATCH_* or STRATA_TEST_* control can change
// arithmetic even when today's similarly named controls do not. Resource/transport settings and cache mechanics
// below do not change state interpretation; resolved geometry and K/V layout remain bound by the caller.
//
// PATH-BACKED OVERRIDES.  Hashing an override's *pathname* binds nothing that matters, and its contents are what the
// engine reads: arithmetic_path_override_refusal() names the ones whose contents decide the arithmetic, so a caller
// that has not bound those contents must decline persistence instead of authorizing records it cannot identify
// (docs/agents/nvme-conversation-cache-research.md: "If a stable authorized identity or verified compatibility
// identity is unavailable, decline persistent reuse rather than inventing one").
//
// No cryptography lives here: a caller that needs a content digest passes one in (the store module owns SHA-256),
// so this module is a pure function of the environment plus the caller's resolved values and is unit-testable on the
// CPU without a GPU, a model or a store.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if !defined(_WIN32)
// POSIX: the process environment (declared exactly as src/platform/mps_ceiling.cpp declares it).  On Windows the
// collection below reads the CRT's `_environ`, which <cstdlib> already declares.
extern "C" char** environ;
#endif

namespace strata::core {

/// How a resolved arithmetic value is encoded.  `number` carries a u64 (an integer, or the exact IEEE-754 bits of a
/// float - never a rounded value), `digest` 32 bytes of a content digest the caller computed, `text` exact bytes.
enum class ArithmeticValueKind : uint8_t { number = 0, digest = 1, text = 2 };

/// One arithmetic selection the caller resolved (a CLI policy, an accessor's effective value, a content digest).
struct ArithmeticIdentityValue {
    std::string name;
    ArithmeticValueKind kind = ArithmeticValueKind::number;
    int64_t number = 0;
    std::array<uint8_t, 32> digest{};
    std::string text;
};

inline ArithmeticIdentityValue arithmetic_number(std::string name, int64_t value) {
    ArithmeticIdentityValue v;
    v.name = std::move(name);
    v.number = value;
    return v;
}
/// The exact bits, so a different rounding of the same decimal cannot collide with it.
inline ArithmeticIdentityValue arithmetic_double(std::string name, double value) {
    uint64_t bits = 0;
    static_assert(sizeof bits == sizeof value, "a double binds as its 64 bits");
    std::memcpy(&bits, &value, sizeof bits);
    return arithmetic_number(std::move(name), (int64_t) bits);
}

// With the execution split armed, the foreground target and background credit
// select recurrent arithmetic boundaries. Bind both resolved values exactly,
// including disabled budget=0; default-off adds no fields or identity changes.
inline void bind_prefill_split_policy(std::vector<ArithmeticIdentityValue>& resolved,
                                     const char* setting, double foreground_budget_ms, double background_share) {
    if (!setting || std::strcmp(setting, "1024") != 0) return;
    resolved.push_back(arithmetic_number("prompt.split_over_budget.tokens", 1024));
    resolved.push_back(arithmetic_double("prompt.split_over_budget.foreground_budget_ms", foreground_budget_ms));
    resolved.push_back(arithmetic_double("prompt.split_over_budget.background_share", background_share));
}
inline ArithmeticIdentityValue arithmetic_digest(std::string name, const std::array<uint8_t, 32>& value) {
    ArithmeticIdentityValue v;
    v.name = std::move(name);
    v.kind = ArithmeticValueKind::digest;
    v.digest = value;
    return v;
}
inline ArithmeticIdentityValue arithmetic_text(std::string name, std::string value) {
    ArithmeticIdentityValue v;
    v.name = std::move(name);
    v.kind = ArithmeticValueKind::text;
    v.text = std::move(value);
    return v;
}

namespace detail {

/// Canonical order is by name, then kind and payload: a caller cannot make two runs disagree by passing the same
/// values in a different order, and a duplicate name still binds both entries (it can only over-fragment).
inline bool arithmetic_value_less(const ArithmeticIdentityValue& a, const ArithmeticIdentityValue& b) {
    if (a.name != b.name) return a.name < b.name;
    if (a.kind != b.kind) return a.kind < b.kind;
    if (a.kind == ArithmeticValueKind::text) return a.text < b.text;
    if (a.kind == ArithmeticValueKind::digest) return a.digest < b.digest;
    return a.number < b.number;
}

inline void arithmetic_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back((uint8_t) (v >> (8 * i)));
}
inline void arithmetic_bytes(std::vector<uint8_t>& out, const char* data, size_t size) {
    arithmetic_u64(out, (uint64_t) size);
    out.insert(out.end(), data, data + size);
}

}  // namespace detail


/// Exact names excluded for the same reasons where a prefix would be too broad (a name in the same family *does*
/// change arithmetic, e.g. STRATA_PREFILL_MMQ beside STRATA_PREFILL_RING).
inline std::span<const std::string_view> arithmetic_environment_excluded_names() {
    static constexpr std::string_view names[] = {
        "STRATA_PREFIX_BACKING",
        "STRATA_PREFIX_BACKING_PROBE",
        "STRATA_PREFIX_EXTENT_PROBE",
        "STRATA_SNAPSHOT_DIRECT_CUT",
        "STRATA_SNAPSHOT_FULL_CAPTURE",
        "STRATA_SNAPSHOT_VERIFY",
        "STRATA_PREFILL_RING",
        "STRATA_PREFILL_ATTRIBUTION",
        "STRATA_PREFILL_TIMING",
        "STRATA_BATCH_TRACE",
        "STRATA_STATE_HASH",
        "STRATA_STATE_HASH_GDN",
        "STRATA_DUMP_FIRST_LOGITS",
        "STRATA_DUMP_COMMITTED_LOGITS",
        "STRATA_DUMP_TEACHER_LOGITS",
        "STRATA_MEMORY_LIMIT_MIB",     // process cap
        "STRATA_NO_LARGEPAGES",
        "STRATA_NO_ARENA_THP",
        "STRATA_STAGE_PIN",
        "STRATA_STAGE_TRIM",
        "STRATA_UMA_HEADROOM_GIB",
        "STRATA_RESIDENT_HEADROOM_GIB",
        "STRATA_RESIDENT_PIN",
        "STRATA_RESIDENT_DIRECT_COPY",
        "STRATA_FETCH_THREADS",
        "STRATA_READ_AHEAD",
        "STRATA_UNBUFFERED_LOAD",
        "STRATA_STARTUP_UNBUFFERED",
        "STRATA_HOST_CORE",
        "STRATA_DISJOINT_ADAPT",
        "STRATA_SEL_OVERLAP",
        "STRATA_KEEP_MAPPING",
        "STRATA_GROUP_COPY",
        "STRATA_FILE_RELEASE",
        "STRATA_DOORBELL_STORE",
        "STRATA_DEFERRED_REGISTER",
        "STRATA_FILL_AHEAD",
        "STRATA_OWNED_PRICE",
        "STRATA_KV_PREFETCH",
        "STRATA_KV_HOST_DMA",
        "STRATA_KV_STAGE_OWN",
        "STRATA_WDDM_BUDGET",
        "STRATA_MTP_CANARY_PROBE",
        "STRATA_MTP_TOP2",
        "STRATA_SESSION_GOLDEN_PRINT",
        "STRATA_SESSION_ROWS",
        "STRATA_SESSION_ROWS_MAX",
        "STRATA_SESSION_BUFFERED",
        "STRATA_IMMUTABLE_CHECKPOINTS",
        "STRATA_CONVERSATION_OFFER_FD",  // an inherited descriptor number: different every launch
        "STRATA_CKPT_REREAD",
        "STRATA_CHECKPOINT_COPY_TRACE",
        "STRATA_CACHE_MESSAGE_BOUNDARY",
        "STRATA_ADMISSION_COPY_BYTES",
        "STRATA_ADMISSION_SNAPSHOT",
        "STRATA_ADMISSION_TIMING",
        "STRATA_BATCH_DIAGNOSTICS",
        "STRATA_DECODE_TIMING",
        "STRATA_DRAFT_TIMING",
        "STRATA_COMMIT_TIMING",
        "STRATA_IO_STATS",
        "STRATA_SUPERVISED",
        "STRATA_PARITY_DUMP",
        "STRATA_Q6P_SELFTEST",
        "STRATA_SOURCE_RECEIPT",
        "STRATA_RSS_TRACE",
        "STRATA_LOGPOS_EXTRA",
        "STRATA_LOGPOS_TOPK",
        "STRATA_OLD_SAMPLER",
        "STRATA_SAMPLER_ONE_BLOCK",
        "STRATA_MTP_PREFILL_SYNC",
        "STRATA_REFILL_BLOCKING",
        "STRATA_REFILL_SERIAL",
        "STRATA_KV_RING_OFF",          // role placement; QsaState::kv_mode/n_slots bind the resolved result
        "STRATA_KV_MAIN_OFF",
        "STRATA_HIPBLASLT_VERBOSE",
        "STRATA_MPS_QUERY_FIXTURE",
    };
    return names;
}

/// True only for an explicitly reviewed non-arithmetic control.
inline bool arithmetic_environment_excluded(std::string_view name) {
    for (const std::string_view excluded : arithmetic_environment_excluded_names())
        if (name == excluded) return true;
    return false;
}

/// The reason a caller must decline persistence, or nullptr when it may proceed: a path-backed override whose
/// *contents* change the arithmetic and which this module does not read.  STRATA_DBG_ROUTE_OVERRIDE replaces the
/// routing of the layers its file lists with the file's own ids and weights (prefill.cpp), so binding its pathname
/// would authorize records computed under routing the identity cannot describe.
inline const char* arithmetic_path_override_refusal() {
    const char* v = std::getenv("STRATA_DBG_ROUTE_OVERRIDE");
    if (v != nullptr && v[0] != '\0')
        return "STRATA_DBG_ROUTE_OVERRIDE is set: a path-backed routing override whose contents, not its path, "
               "decide the arithmetic";
    v = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (v != nullptr && v[0] != '\0')
        return "STRATA_HIPBLASLT_TUNING is set without a verified tuning-file identity";
    return nullptr;
}

/// The canonical bytes of the run's arithmetic identity:
///   1. every present STRATA_* environment variable that is not excluded, in name order, as name and exact value
///      bytes (a variable that is absent is simply not listed; set-but-empty is a zero-length value);
///   2. the caller's resolved arithmetic values, in canonical (name, kind, payload) order.
/// Same bytes in, same bytes out: this is a pure function of the environment and the argument, so a test can assert
/// exactly which settings move the identity, and the caller appends the result to the store's domain digest.
inline std::vector<uint8_t> arithmetic_execution_identity(
    std::vector<ArithmeticIdentityValue> resolved = {}) {
    std::vector<std::pair<std::string_view, std::string_view>> env;   // name without '=', value
#if defined(_WIN32)
    char** const entries = _environ;
#else
    char** const entries = environ;
#endif
    for (char** entry = entries; entry != nullptr && *entry != nullptr; ++entry) {
        const char* text = *entry;
        if (text == nullptr) continue;
        const char* eq = std::strchr(text, '=');
        if (eq == nullptr) continue;
        const std::string_view name(text, (size_t) (eq - text));
        if (name.size() <= 7 || name.compare(0, 7, "STRATA_") != 0) continue;   // engine settings only, never the shell
        if (arithmetic_environment_excluded(name)) continue;
        env.emplace_back(name, std::string_view(eq + 1));
    }
    std::sort(env.begin(), env.end(),
              [](const auto& a, const auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
    std::sort(resolved.begin(), resolved.end(), detail::arithmetic_value_less);

    std::vector<uint8_t> out;
    static constexpr std::string_view kTag = "strata-arithmetic-identity";
    detail::arithmetic_bytes(out, kTag.data(), kTag.size());
    detail::arithmetic_u64(out, 1);            // encoding version
    detail::arithmetic_u64(out, (uint64_t) env.size());
    for (const auto& [name, value] : env) {
        detail::arithmetic_bytes(out, name.data(), name.size());
        detail::arithmetic_bytes(out, value.data(), value.size());
    }
    detail::arithmetic_u64(out, (uint64_t) resolved.size());
    for (const ArithmeticIdentityValue& value : resolved) {
        detail::arithmetic_bytes(out, value.name.data(), value.name.size());
        detail::arithmetic_u64(out, (uint64_t) value.kind);
        switch (value.kind) {
            case ArithmeticValueKind::number: detail::arithmetic_u64(out, (uint64_t) value.number); break;
            case ArithmeticValueKind::digest: detail::arithmetic_u64(out, value.digest.size());
                                              out.insert(out.end(), value.digest.begin(), value.digest.end()); break;
            case ArithmeticValueKind::text: detail::arithmetic_bytes(out, value.text.data(), value.text.size()); break;
        }
    }
    return out;
}

}  // namespace strata::core
