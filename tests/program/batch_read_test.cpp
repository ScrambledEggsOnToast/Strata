// batch_read_piece: the pieces of a --batch prompt read (synthetic: no model, no GPU).
#include "strata/program/batch_read.hpp"
#include "strata/program/serve_controls.hpp"

#include <cstdio>

using strata::program::batch_read_piece;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // no --batch, or the pipelined groups: one run
    CHECK(batch_read_piece(false, false, false, false, 2048, 8) == 0);
    CHECK(batch_read_piece(true, true, true, false, 2048, 8) == 0);
    // one GPU: a chunk per piece, whether or not a slot decodes (the yield point is the chunk boundary)
    CHECK(batch_read_piece(true, false, false, false, 2048, 8) == 2048);
    CHECK(batch_read_piece(true, false, false, true, 2048, 8) == 2048);
    // layer split with slots decoding: a chunk per piece as before
    CHECK(batch_read_piece(true, false, true, true, 2048, 8) == 2048);
    // layer split, nobody decoding: pieces of 8 chunks - the BYIELD of a shorter request can be taken (it was one run)
    CHECK(batch_read_piece(true, false, true, false, 2048, 8) == 8 * 2048);
    CHECK(batch_read_piece(true, false, true, false, 1024, 4) == 4096);
    // STRATA_BATCH_YIELD_CHUNKS=0 keeps the old single run
    CHECK(batch_read_piece(true, false, true, false, 2048, 0) == 0);
    // a degenerate chunk never gives a zero piece
    CHECK(batch_read_piece(true, false, false, false, 0, 8) == 1);
    // a piece is always a multiple of the chunk
    for (int64_t c : {256, 1000, 2048, 4096})
        for (int64_t k : {1, 3, 8, 16}) CHECK(batch_read_piece(true, false, true, false, c, k) % c == 0);
    {
        using strata::program::batch_read_quantum;
        for (int64_t n : {2047, 2048, 2049, 4095, 4096, 4097, 6144, 6145}) {
            CHECK(batch_read_quantum(n, 2048, false, 2048) == std::min<int64_t>(n, 2048));
            const auto span = batch_read_quantum(n, 2048, true, 2048);
            CHECK(span > 0 && span <= n && span <= 4096);
            if (span > 2048) CHECK(n == span || n - span > 2048);
        }
        CHECK(batch_read_quantum(2049, 2048, true, 2048) == 2049);
        CHECK(batch_read_quantum(4095, 2048, true, 2048) == 4095);
        CHECK(batch_read_quantum(4096, 2048, true, 2048) == 4096);
        CHECK(batch_read_quantum(4097, 2048, true, 2048) == 2048);
        CHECK(batch_read_quantum(6144, 2048, true, 2048) == 2048);
        CHECK(batch_read_quantum(6145, 2048, true, 2048) == 4096);
        CHECK(batch_read_quantum(INT64_MAX, INT64_MAX, true, INT64_MAX) == INT64_MAX);
        CHECK(batch_read_quantum(INT64_MAX, INT64_MAX / 2, true, INT64_MAX / 2) == INT64_MAX / 2);
        CHECK(batch_read_quantum(INT64_MAX, INT64_MAX / 4, true, INT64_MAX / 4) == 2 * (INT64_MAX / 4));
        CHECK(batch_read_quantum(4096, 0, true, 2048) == 0);
        CHECK(batch_read_quantum(0, 2048, true, 2048) == 0);
    }
    {
        using strata::program::pop_control_prefix;
        std::deque<std::string> pending{"BGEN 0 next-owner", "BSTOP 0"};
        std::string control = "unchanged";
        CHECK(!pop_control_prefix(pending, control, true));
        CHECK(control == "unchanged" && pending.front() == "BGEN 0 next-owner" && pending.size() == 2);
        pending = {"BSTOP 0", "BGEN 0 next-owner", "BSTOP 0"};
        CHECK(pop_control_prefix(pending, control, true) && control == "BSTOP 0");
        CHECK(!pop_control_prefix(pending, control, true) && pending.size() == 2);
        pending = {"BYIELD 1", "BSTOP 1", "BYIELD 2"};
        CHECK(!pop_control_prefix(pending, control, false) && pending.size() == 3);
        CHECK(pop_control_prefix(pending, control, true) && control == "BYIELD 1");
        CHECK(pending.front() == "BSTOP 1"); // caller can park/return before touching this incarnation
        CHECK(pop_control_prefix(pending, control, false) && control == "BSTOP 1");
        CHECK(!pop_control_prefix(pending, control, false) && pending.front() == "BYIELD 2");
        CHECK(pop_control_prefix(pending, control, true) && control == "BYIELD 2");
        CHECK(!pop_control_prefix(pending, control, true));
        pending = {"BPRIORITY 2 background", "GEN 1 another-owner"};
        CHECK(pop_control_prefix(pending, control, false) && control == "BPRIORITY 2 background");
        CHECK(!pop_control_prefix(pending, control, true) && pending.front() == "GEN 1 another-owner");
    }
    if (fails == 0) std::printf("batch_read_test: ok\n");
    return fails == 0 ? 0 : 1;
}
