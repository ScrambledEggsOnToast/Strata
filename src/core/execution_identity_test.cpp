// The arithmetic execution identity's own test: which settings move the bytes the persistent store's domain digest is
// built from, and which deliberately do not.  CPU-only, no GPU, no model, no store: the module under test is a pure
// function of the environment plus the caller's resolved values, so every assertion here is deterministic.
//
// The test scrubs every STRATA_* variable from its own environment first, so a stale setting in the shell that runs it
// cannot make an assertion pass or fail by accident, and then drives the two directions that matter: a control that
// can change the arithmetic must change the identity, and a control that only moves work, stores the cache or reports
// a diagnostic must not.
#include "strata/core/execution_identity.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>   // environ, setenv/unsetenv
#include <vector>

using namespace strata::core;

#if defined(_WIN32)
int main() {
    std::fprintf(stderr, "execution identity test: POSIX environment only\n");
    return 77;   // the same skip convention as conversation_store_test
}
#else

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

void set_env(const char* name, const char* value) {
    if (value == nullptr) unsetenv(name);
    else check(setenv(name, value, 1) == 0, "setenv");
}

/// Everything the digest is built from, so one call per assertion stays cheap to read.
std::vector<uint8_t> identity(std::vector<ArithmeticIdentityValue> resolved = {}) {
    return arithmetic_execution_identity(std::move(resolved));
}

/// A control that can change what a stored image holds: setting it must change the identity, and clearing it must
/// restore the previous bytes exactly.
void check_bound(const char* name) {
    check(!arithmetic_environment_excluded(name), "an arithmetic control is not excluded");
    const std::vector<uint8_t> before = identity();
    set_env(name, "1");
    const std::vector<uint8_t> after = identity();
    check(after != before, "setting an arithmetic control changes the identity");
    set_env(name, nullptr);
    check(identity() == before, "clearing it restores the identity");
}

/// A control that only moves work, stores the cache, counts bytes or reports a diagnostic.
void check_excluded(const char* name) {
    check(arithmetic_environment_excluded(name), "a non-arithmetic control is excluded");
    const std::vector<uint8_t> before = identity();
    set_env(name, "1");
    check(identity() == before, "an excluded control leaves the identity alone");
    set_env(name, nullptr);
    check(identity() == before, "and clearing it does too");
}

}  // namespace

int main() {
    // The test owns the environment: no STRATA_* variable survives from the caller's shell.
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        const char* eq = std::strchr(*entry, '=');
        if (eq == nullptr || eq - *entry < 7) continue;
        if (std::strncmp(*entry, "STRATA_", 7) == 0) {
            const std::string name(*entry, (size_t) (eq - *entry));
            unsetenv(name.c_str());
        }
    }

    {
        // 1. The arithmetic controls the engine reads for itself are bound, one by one.
        const char* bound[] = {
            // the hyper-connection read and its weights
            "STRATA_GR_V3", "STRATA_GR_SPLIT", "STRATA_HC_Q8", "STRATA_HC_Q8_INJECT",
            // routing and its overrides
            "STRATA_ROUTE_RESIDENT", "STRATA_ROUTE_RESIDENT_RANKS", "STRATA_ROUTE_RESIDENT_MTP",
            "STRATA_ROUTE_PRIOR", "STRATA_ROUTE_PER_TOKEN", "STRATA_ROUTE_PROJ_PER_TOKEN", "STRATA_ROUTER_BLOCK",
            // the prompt path
            "STRATA_ROPE_TABLE", "STRATA_BF16_TC", "STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_PF_FUSED_KQ",
            "STRATA_PF_FUSED_NATIVE", "STRATA_PA_FAST", "STRATA_WMMA_GEMM", "STRATA_SELECT_WMMA",
            "STRATA_SELECT_SIMT", "STRATA_SELECT_OLD", "STRATA_QSA_WARP", "STRATA_PROMPT_ATTN_OLD",
            "STRATA_PROMPT_ATTN_V1", "STRATA_PROMPT_ATTN_Q4", "STRATA_HIP_WMMA", "STRATA_PF_SWITCH_MIN_T",
            "STRATA_PF_HCDOWN", "STRATA_HCD_EXACT", "STRATA_EMB_REUSE_ACCOUNT", "STRATA_PREFILL_BF16X2",
            "STRATA_HC_UPMIX", "STRATA_PREFILL_CPU_SHARE", "STRATA_PREFILL_CPU_SHARE_MAX", "STRATA_DENSE_MMQ",
            "STRATA_PREFILL_MMQ", "STRATA_FUSE_CONVAB", "STRATA_NORM_ROPE", "STRATA_INDEXER_PER_TOKEN",
            "STRATA_PREFILL_SPLIT_OVER_BUDGET",
            // the recurrence
            "STRATA_GDN_HEAD", "STRATA_GDN_PP", "STRATA_GDN_CONVL2", "STRATA_GDN_NOY", "STRATA_GDN_CHUNKED",
            "STRATA_GDN_KEYHEAD", "STRATA_GDN_PIPELINE", "STRATA_GDN_REC_HEADS", "STRATA_GDN_CONV_SERIAL",
            "STRATA_FAST_GDN", "STRATA_ONE_TOKEN_COMMIT",
            // the expert and quantized kernels
            "STRATA_EXP_MODE", "STRATA_EXPERT_V2K", "STRATA_IQ3S_MT1", "STRATA_IQ_MT_MIN", "STRATA_CPU_YMM",
            "STRATA_KQ256", "STRATA_DEC_BATCH",
            // the draft layer's own selection
            "STRATA_MTP_CATCHUP_ALL", "STRATA_MTP_FULL_HEAD",
            // host dispatch, and the diagnostics that change which path runs
            "STRATA_EMULATE_CC", "STRATA_DBG_GDN", "STRATA_DBG_FORCE_SLOW_LAYER", "STRATA_DBG_ROUTE_OVERRIDE",
            // Unknown controls must not inherit exclusions from a similarly named family.
            "STRATA_BATCH_NEW_KERNEL", "STRATA_TEST_NEW_ARITHMETIC", "STRATA_PLE_NEW_FORMAT",
        };
        for (const char* name : bound) check_bound(name);
    }
    {
        // Cache transport, storage mechanics and observational flags cannot invalidate a reusable history.
        const char* excluded[] = {
            "STRATA_CONVERSATION_OFFER_FD", "STRATA_PREFIX_BACKING", "STRATA_IMMUTABLE_CHECKPOINTS",
            "STRATA_SNAPSHOT_DIRECT_CUT", "STRATA_MEMORY_LIMIT_MIB", "STRATA_PREFILL_RING",
            "STRATA_STATE_HASH", "STRATA_DUMP_FIRST_LOGITS", "STRATA_BATCH_TRACE",
        };
        for (const char* name : excluded) check_excluded(name);
    }
    {
        // 3. Only the engine's own prefix is read: the shell's variables are never part of the identity.
        const std::vector<uint8_t> before = identity();
        set_env("PATH", "/nowhere:/bin");
        set_env("HOME", "/nowhere");
        set_env("STRATA", "1");             // not a STRATA_ setting: the underscore is what delimits the prefix
        set_env("STRATAGR_V3", "1");
        set_env("NOT_STRATA_GR_V3", "1");
        check(identity() == before, "non-STRATA variables are not bound");
        set_env("PATH", nullptr);
        set_env("HOME", nullptr);
        set_env("STRATA", nullptr);
        set_env("STRATAGR_V3", nullptr);
        set_env("NOT_STRATA_GR_V3", nullptr);
    }
    {
        // 4. Absent, set-empty and two different values are four different bindings: the shape of every env control,
        //    including the ones a kernel tests for presence alone.
        const std::vector<uint8_t> absent = identity();
        set_env("STRATA_GR_V3", "");
        const std::vector<uint8_t> empty = identity();
        set_env("STRATA_GR_V3", "0");
        const std::vector<uint8_t> zero = identity();
        set_env("STRATA_GR_V3", "1");
        const std::vector<uint8_t> one = identity();
        check(absent != empty && empty != zero && zero != one && absent != one,
              "present, empty and every distinct value are distinct bindings");
        // A binding finer than the parse is deliberate: "01" and "1" mean the same to the kernel's atoi(), and both
        // are bound, so a parse this module does not own can never hide a difference from the identity.
        set_env("STRATA_GR_V3", "01");
        check(identity() != one, "text that parses the same is still bound separately");
        set_env("STRATA_GR_V3", nullptr);
        check(identity() == absent, "and the control empties back to absent");
    }
    {
        // 5. Values are bound as their exact bytes: whitespace and embedded separators cannot collide.
        set_env("STRATA_PF_SWITCH_MIN_T", "1024");
        const std::vector<uint8_t> plain = identity();
        set_env("STRATA_PF_SWITCH_MIN_T", "1024 ");
        const std::vector<uint8_t> trailing = identity();
        set_env("STRATA_PF_SWITCH_MIN_T", " 1024");
        const std::vector<uint8_t> leading = identity();
        check(plain != trailing && plain != leading && trailing != leading, "exact value bytes are bound");
        set_env("STRATA_PF_SWITCH_MIN_T", nullptr);
    }
    {
        // 6. The caller's resolved values: canonical order, distinct kinds, and every payload bit bound.
        const auto number = [] { return arithmetic_number("prompt.chunk", 4096); };
        const auto other_number = [] { return arithmetic_number("prompt.chunk", 4097); };
        const auto as_text = [] { return arithmetic_text("prompt.chunk", "4096"); };
        std::array<uint8_t, 32> digest{};
        digest[0] = 1;
        std::array<uint8_t, 32> other_digest = digest;
        other_digest[31] = 2;
        check(identity({number(), arithmetic_number("draft.vocab_size", 106291)}) ==
                  identity({arithmetic_number("draft.vocab_size", 106291), number()}),
              "resolved values are canonical in order");
        check(identity({number()}) != identity({other_number()}), "a resolved number is bound exactly");
        check(identity({number()}) != identity({as_text()}), "a number and its decimal text differ");
        check(identity({arithmetic_digest("draft.vocab_bytes", digest)}) !=
                  identity({arithmetic_digest("draft.vocab_bytes", other_digest)}),
              "a content digest is bound byte for byte");
        check(identity({arithmetic_digest("draft.vocab_bytes", digest)}) !=
                  identity({arithmetic_text("draft.vocab_bytes", std::string(32, '\0'))}),
              "a digest is not the same binding as the same length of text");
        // A double is bound as its bits, so two values that print the same but differ in the last bit do not collide.
        double margin = 0.5;
        check(identity({arithmetic_double("draft.route_margin", margin)}) !=
                  identity({arithmetic_double("draft.route_margin", std::nextafter(margin, 1.0))}),
              "a float is bound as its exact bits");
        // A text value with an embedded NUL is still length-delimited, not NUL-terminated.
        check(identity({arithmetic_text("t", std::string("a\0b", 3))}) !=
                  identity({arithmetic_text("t", std::string("a", 1))}),
              "text is length-delimited");
        // The resolved values add to the environment rather than replacing it.
        set_env("STRATA_GR_V3", "1");
        check(identity({number()}) != identity(), "resolved values are appended to the environment's");
        set_env("STRATA_GR_V3", nullptr);
    }
    {
        // 7. Only the armed split policy makes the CLI budget part of the
        // arithmetic: zero disables splitting and must not alias a positive budget.
        const auto default_identity = identity();
        const auto split_identity = [](double budget_ms) {
            std::vector<ArithmeticIdentityValue> resolved;
            bind_prefill_split_policy(resolved, std::getenv("STRATA_PREFILL_SPLIT_OVER_BUDGET"), budget_ms);
            return identity(std::move(resolved));
        };
        check(split_identity(0) == default_identity && split_identity(1000) == default_identity,
              "an absent split gate leaves budget changes outside the arithmetic identity");
        set_env("STRATA_PREFILL_SPLIT_OVER_BUDGET", "0");
        const auto explicit_off = identity();
        check(split_identity(0) == explicit_off && split_identity(1200) == explicit_off,
              "the explicit off gate retains its existing identity independently of budget");
        set_env("STRATA_PREFILL_SPLIT_OVER_BUDGET", "1024");
        const auto disabled_budget = split_identity(0);
        const auto budget_1000 = split_identity(1000);
        const auto budget_1200 = split_identity(1200);
        check(disabled_budget != identity(), "an armed but disabled budget is explicitly bound");
        check(disabled_budget != budget_1000 && disabled_budget != budget_1200 && budget_1000 != budget_1200,
              "different split budgets and budget-disabled state cannot restore each other's recurrent image");
        check(budget_1000 != split_identity(std::nextafter(1000.0, 1200.0)),
              "split budget identity binds exact resolved double bits");
        set_env("STRATA_PREFILL_SPLIT_OVER_BUDGET", nullptr);
        check(split_identity(1200) == default_identity, "clearing the split gate restores default-off identity bytes");
    }
    {
        // 8. The path-backed routing override: unset or empty is not an override, anything else must refuse
        //    persistence (the contents decide the arithmetic and this module never reads them).
        check(arithmetic_path_override_refusal() == nullptr, "no override is not a refusal");
        set_env("STRATA_DBG_ROUTE_OVERRIDE", "");
        check(arithmetic_path_override_refusal() == nullptr, "an empty path is not an override");
        set_env("STRATA_DBG_ROUTE_OVERRIDE", "/tmp/routing.bin");
        check(arithmetic_path_override_refusal() != nullptr, "a set override refuses persistence");
        set_env("STRATA_DBG_ROUTE_OVERRIDE", nullptr);
        set_env("STRATA_HIPBLASLT_TUNING", "/tmp/blas-tuning.tsv");
        check(arithmetic_path_override_refusal() != nullptr, "an unverified BLAS tuning file refuses persistence");
        set_env("STRATA_HIPBLASLT_TUNING", nullptr);
    }

    std::fprintf(stderr, "execution_identity_test: %d checks passed\n", checks);
    return 0;
}
#endif
