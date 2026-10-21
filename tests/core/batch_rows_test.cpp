// The batch window's row layout (include/strata/core/batch_rows.hpp): plain one-row slots, a slot's contiguous group of
// proposed rows (--batch-mtp), the hand-off capacity of a layer split's stages, and the complete native DFlash2
// graph-cache key/host-node envelope. Header-only, CPU; no CUDA graph-storage claim.
#include "strata/core/batch_rows.hpp"
#include "strata/core/verify_ownership.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
const char* run(const int* rows, const long long* p, int S, int n_slots, int max_t, int hbase, bool handoff) {
    int64_t pos[8];
    for (int t = 0; t < S; ++t) pos[t] = p[t];
    return strata::core::batch_rows_error(rows, S, pos, n_slots, max_t, hbase, handoff, 8);
}

void dflash2_graph_capacity() {
    using namespace strata::core;
    using Layout = std::vector<int>;
    constexpr int slots = 4, max_rows = 8, handoff_rows = 16;
    constexpr size_t key_variants = 2 * 2; // two physical handoff bases and both ar_off_ variants
    std::set<Layout> layouts;
    std::set<VerifyBatchKey> keys;
    const auto positions = [](const Layout& rows) {
        std::array<int64_t, max_rows> pos{};
        for (size_t t = 0; t < rows.size(); ++t)
            pos[t] = t && rows[t] == rows[t - 1] ? pos[t - 1] + 1 : 100 * (rows[t] + 1);
        return pos;
    };
    const auto observe = [&](const Layout& rows) {
        check(dflash2_batch_rows_supported(rows.data(), int(rows.size())),
              "every enumerated native production layout passes its actual family guard");
        layouts.insert(rows);
        const auto pos = positions(rows);
        for (const int base : {0, 8}) {
            check(batch_rows_error(rows.data(), int(rows.size()), pos.data(), slots, max_rows,
                                   base, true, handoff_rows) == nullptr,
                  "native legal schedule passes the actual batch row validator");
            for (const bool ar_off : {false, true})
                keys.insert(verify_batch_graph_key(rows.data(), int(rows.size()), base, ar_off));
        }
    };

    // Decode independently chooses one/two distinct owners and a current-token-plus-proposals
    // depth of 1..4 for each. Context/output limits or partial proposal readiness can shorten
    // either group; cancellation can remove either owner. Do not assume canonical slot order.
    for (int first = 0; first < slots; ++first) {
        for (int first_depth = 1; first_depth <= 4; ++first_depth) {
            observe(Layout(size_t(first_depth), first));
            for (int second = 0; second < slots; ++second) {
                if (second == first) continue;
                for (int second_depth = 1; second_depth <= 4; ++second_depth) {
                    Layout rows(size_t(first_depth), first);
                    rows.insert(rows.end(), size_t(second_depth), second);
                    observe(rows);
                }
            }
        }
    }
    constexpr size_t decode_layouts = slots * 4 + slots * (slots - 1) * 4 * 4;
    check(layouts.size() == decode_layouts, "all 208 ordered native decode layouts are distinct");
    check(keys.size() == key_variants * decode_layouts, "decode keys do not collide across rows, bases or variants");

    // Retained known-input target windows may use all eight target rows for ONE owner.
    // This deliberately covers target lookup width eight, not only native forward block four.
    for (int owner = 0; owner < slots; ++owner)
        for (int depth = 1; depth <= max_rows; ++depth) observe(Layout(size_t(depth), owner));
    constexpr size_t retained_added = slots * (max_rows - 4);
    check(layouts.size() == decode_layouts + retained_added, "retained depths five to eight add 16 layouts");
    check(keys.size() == key_variants * (decode_layouts + retained_added),
          "retained target keys preserve every owner and depth");

    // Prompt completion issues one row per ready owner, up to four. Enumerate every slot
    // permutation and each ready prefix: no prompt-order restriction is needed by the cap.
    std::array<int, slots> order{0, 1, 2, 3};
    do {
        for (int ready = 1; ready <= slots; ++ready) observe(Layout(order.begin(), order.begin() + ready));
    } while (std::next_permutation(order.begin(), order.end()));
    constexpr size_t prompt_added = slots * (slots - 1) * (slots - 2) +
                                   slots * (slots - 1) * (slots - 2) * (slots - 3);
    constexpr size_t expected_layouts = decode_layouts + retained_added + prompt_added;
    check(layouts.size() == expected_layouts, "all prompt orders add exactly 48 three/four-owner layouts");
    check(keys.size() == key_variants * expected_layouts,
          "all 272 native layouts have distinct keys for both bases and residency variants");
    check(keys.size() <= kDflash2BatchGraphKeys, "complete native schedule fits its consumer-visible graph cache cap");
    const uint64_t host_nodes = 2ull * keys.size() + 1; // window + commit, including null failed-key reservations
    check(host_nodes * kDflash2BatchGraphNodeBytes <= kDflash2BatchGraphHostBytes,
          "host quote covers BOTH complete native key maps plus one insertion temporary");

    // Exhaust the entire four-slot row-string universe, not just the chosen legal examples.
    // This independently proves that the guard before native graph capture accepts exactly
    // the enumerated production-family union, so no generic-validator-only shape can escape.
    size_t supported_layouts = 0;
    for (int S = 1; S <= max_rows; ++S) {
        Layout rows(size_t(S), 0);
        const unsigned strings = 1u << (2 * S);
        for (unsigned encoded = 0; encoded < strings; ++encoded) {
            unsigned value = encoded;
            for (int t = 0; t < S; ++t) {
                rows[size_t(t)] = int(value & 3u);
                value >>= 2;
            }
            const bool supported = dflash2_batch_rows_supported(rows.data(), S);
            check(supported == (layouts.count(rows) == 1),
                  "native family guard accepts exactly the independently enumerated finite row language");
            supported_layouts += supported;
        }
    }
    check(supported_layouts == expected_layouts, "native family guard admits exactly 272 row layouts");
    check(!dflash2_batch_rows_supported(nullptr, 1) && !dflash2_batch_rows_supported(nullptr, 0),
          "native family guard refuses missing and empty row windows");

    // Whole-owner cancellation and any partial-ready owner subset may occur in every
    // family. Removing all owners is inert (Verifier does not capture an empty key).
    for (const auto& rows : layouts) {
        for (unsigned ready = 0; ready < (1u << slots); ++ready) {
            Layout active;
            for (const int owner : rows)
                if (ready & (1u << owner)) active.push_back(owner);
            if (active.empty()) continue;
            const auto pos = positions(active);
            check(dflash2_batch_rows_supported(active.data(), int(active.size())),
                  "cancelled and partial-ready subsets stay inside the enforced native family union");
            for (const int base : {0, 8}) {
                check(batch_rows_error(active.data(), int(active.size()), pos.data(), slots, max_rows,
                                       base, true, handoff_rows) == nullptr,
                      "cancelled and partial-ready subsets retain legal contiguous rows");
                for (const bool ar_off : {false, true})
                    check(keys.count(verify_batch_graph_key(active.data(), int(active.size()), base, ar_off)) == 1,
                          "cancelled and partial-ready subsets need no key outside the priced superset");
            }
        }

        // Reject each out-of-range owner at every row of every otherwise legal layout,
        // and reject every broken consecutive-position link in a proposed/known-input group.
        Layout invalid = rows;
        auto pos = positions(rows);
        for (size_t t = 0; t < rows.size(); ++t) {
            for (const int bad_owner : {-1, slots}) {
                invalid[t] = bad_owner;
                check(!dflash2_batch_rows_supported(invalid.data(), int(invalid.size())),
                      "native family guard refuses each out-of-range owner");
                for (const int base : {0, 8})
                    check(batch_rows_error(invalid.data(), int(invalid.size()), pos.data(), slots, max_rows,
                                           base, true, handoff_rows) != nullptr,
                          "native layout with an out-of-range owner is refused before key construction");
            }
            invalid[t] = rows[t];
            if (t && rows[t] == rows[t - 1]) {
                ++pos[t];
                for (const int base : {0, 8})
                    check(batch_rows_error(rows.data(), int(rows.size()), pos.data(), slots, max_rows,
                                           base, true, handoff_rows) != nullptr,
                          "every nonconsecutive native row group is refused");
                --pos[t];
            }
        }
        check(batch_rows_error(rows.data(), int(rows.size()), pos.data(), slots, max_rows,
                               handoff_rows - int(rows.size()) + 1, true, handoff_rows) != nullptr,
              "a native window past the physical sixteen-row handoff is refused");
    }
    for (int first = 0; first < slots; ++first) {
        for (int second = 0; second < slots; ++second) {
            if (first == second) continue;
            const int rows[] = {first, second, first};
            const int64_t pos[] = {100, 200, 101};
            check(!dflash2_batch_rows_supported(rows, 3), "native family guard refuses noncontiguous owners");
            for (const int base : {0, 8})
                check(batch_rows_error(rows, 3, pos, slots, max_rows, base, true, handoff_rows) != nullptr,
                      "all noncontiguous repeated-owner orders are refused");
        }
    }
    // The validator is intentionally more general than native production scheduling.
    // A three-owner multi-depth frame stays valid for other modes but must be refused
    // before native graph-cache reservation; no general-validator behavior is narrowed.
    const int other_mode_rows[] = {0, 0, 1, 1, 2, 2};
    const int64_t other_mode_pos[] = {10, 11, 20, 21, 30, 31};
    check(batch_rows_error(other_mode_rows, 6, other_mode_pos, slots, max_rows, 0, true, handoff_rows) == nullptr,
          "the shared row validator keeps legal non-native scheduling families");
    check(!dflash2_batch_rows_supported(other_mode_rows, 6),
          "qualified native scheduling refuses the generic-valid three-owner multi-depth counterexample");
    check(keys.count(verify_batch_graph_key(other_mode_rows, 6, 0, false)) == 0,
          "native capacity proof is not misrepresented as all generic validator-legal layouts");
    const int too_many[9] = {};
    const int64_t too_many_pos[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    check(!dflash2_batch_rows_supported(too_many, 9),
          "native family guard refuses more than eight physical verifier rows");
    check(batch_rows_error(too_many, 9, too_many_pos, slots, max_rows, 0, true, handoff_rows) != nullptr,
          "sixteen-row transport does not permit a nine-row physical verifier window");
    const int example_rows[] = {2, 2, 0};
    check(verify_batch_graph_key(example_rows, 3, 8, true) ==
              VerifyBatchKey{8, 3, 1, 2, 2, 0, 0, 0, 0, 0, 0},
          "shared actual key preserves field order, residency bit and zero-initialized tail");
    std::printf("DFLASH2_BATCH_GRAPH_BOUND decode_layouts=%zu retained_added=%zu prompt_added=%zu "
                "row_layouts=%zu supported_layouts=%zu bases=2 residency_variants=2 keys=%zu cap=%u map_nodes=%llu "
                "node_allowance_bytes=%llu host_accounting_allowance_bytes=%llu cuda_graph_storage_priced=0\n",
                decode_layouts, retained_added, prompt_added, layouts.size(), supported_layouts, keys.size(), kDflash2BatchGraphKeys,
                (unsigned long long) host_nodes, (unsigned long long) kDflash2BatchGraphNodeBytes,
                (unsigned long long) kDflash2BatchGraphHostBytes);
}
}  // namespace

int main() {
    using strata::core::batch_rows_error;
    // plain batch: one row per slot, any positions
    {
        const int rows[] = {0, 1, 2};
        const long long pos[] = {10, 500, 3};
        check(run(rows, pos, 3, 3, 8, 0, false) == nullptr, "one row per slot");
        check(run(rows, pos, 3, 3, 8, 0, true) == nullptr, "one row per slot through the hand-off");
    }
    // --batch-mtp: a slot's two rows at consecutive positions; also on a layer split (S > slots)
    {
        const int rows[] = {1, 1, 0, 0};
        const long long pos[] = {40, 41, 7, 8};
        check(run(rows, pos, 4, 2, 8, 0, false) == nullptr, "two rows per slot, one stage");
        check(run(rows, pos, 4, 2, 8, 0, true) == nullptr, "two rows per slot across a layer split (S > slots)");
    }
    // eight rows = four slots of two: the whole hand-off
    {
        const int rows[] = {0, 0, 1, 1, 2, 2, 3, 3};
        const long long pos[] = {1, 2, 5, 6, 9, 10, 20, 21};
        check(run(rows, pos, 8, 4, 8, 0, true) == nullptr, "eight rows fill the hand-off");
        check(run(rows, pos, 8, 4, 8, 1, true) != nullptr, "a group past the hand-off's end");
        check(run(rows, pos, 8, 4, 8, 1, false) == nullptr, "no hand-off: the base does not bound the rows");
    }
    // refused layouts
    {
        const int rows[] = {0, 1, 0};
        const long long pos[] = {5, 9, 6};
        check(run(rows, pos, 3, 2, 8, 0, false) != nullptr, "a slot's rows must be contiguous");
    }
    {
        const int rows[] = {0, 0};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 1, 8, 0, false) != nullptr, "a slot's rows need consecutive positions");
    }
    {
        const int rows[] = {0, 2};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 2, 8, 0, false) != nullptr, "a slot past the slots held");
        const int neg[] = {-1};
        const long long p1[] = {1};
        check(run(neg, p1, 1, 2, 8, 0, false) != nullptr, "a negative slot");
    }
    {
        const int rows[] = {0, 0, 0};
        const long long pos[] = {1, 2, 3};
        check(run(rows, pos, 3, 1, 2, 0, false) != nullptr, "more rows than the window holds");
        check(run(rows, pos, 0, 1, 8, 0, false) != nullptr, "an empty window");
        check(run(rows, pos, 3, 1, 8, -1, false) != nullptr, "a negative hand-off base");
        check(run(rows, pos, 3, 1, 8, 0, false) == nullptr, "three rows of one slot are a group");
    }
    // The guard covers every verified row, not only the accepted prefix. A successful
    // poll and a rejected second-row proposal cannot turn poisoned logits into output.
    {
        using strata::core::batch_rows_nonfinite;
        check(!batch_rows_nonfinite(0, 0, 8), "finite synchronous MTP window");
        check(batch_rows_nonfinite(1u << 1, 0, 2), "nonfinite rejected draft row invalidates its slot");
        check(!batch_rows_nonfinite(1u << 1, 2, 2), "rejected draft row does not invalidate another slot");
        for (int group = 0; group < 2; ++group) {
            // Each pipeline owner has a local four-row mask, independent of its physical handoff base.
            const int handoff_base = group * 4;
            const int rows[] = {group * 2, group * 2, group * 2 + 1, group * 2 + 1};
            const long long pos[] = {10, 11, 20, 21};
            check(run(rows, pos, 4, 4, 8, handoff_base, true) == nullptr, "pipeline group row layout");
            for (int poisoned = 0; poisoned < 4; ++poisoned) {
                const uint32_t mask = 1u << poisoned;
                for (int first = 0; first < 4; first += 2)
                    check(batch_rows_nonfinite(mask, first, 2) == (poisoned / 2 == first / 2),
                          "each pipeline group's current and rejected draft rows invalidate only their slot");
            }
        }
        check(batch_rows_nonfinite(1u << 7, 6, 2), "last synchronous slot's rejected draft row");
        check(!batch_rows_nonfinite(1u, -1, 2), "negative row is not a nonfinite range");
        check(!batch_rows_nonfinite(1u, 0, 0), "empty row range");
    }
    dflash2_graph_capacity();
    if (failures != 0) return EXIT_FAILURE;
    std::printf("batch_rows_test: ok\n");
    return EXIT_SUCCESS;
}
