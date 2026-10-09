#include "strata/plan/source_receipt.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
}
}

int main() {
    using strata::plan::parse_source_receipt;
    strata::plan::SourceResidencyReceipt receipt;
    const std::string counts = "50292326400 42 9 1800000000000000000 4096 8192 16384 0";
    require(parse_source_receipt("1 " + counts + "\n", 1, receipt), "v1 source rejected");
    require(receipt.expert_mtime_ns == 1800000000000000000ull && receipt.host_payload_bytes == 0,
            "identity or zero host growth changed");
    require(!parse_source_receipt("1 " + counts, 2, receipt), "single-ring measurement admitted two owners");
    for (uint64_t lanes : {2, 3}) {
        const auto wire = "2 " + counts + " " + std::to_string(lanes) + "\n";
        require(parse_source_receipt(wire, lanes, receipt), "matching aggregate rejected");
        require(receipt.lane_count == lanes && receipt.ring_bytes == 4096 && receipt.reader_overhead_bytes == 16384,
                "aggregate measurement multiplied or owner count lost");
        require(!parse_source_receipt(wire, 1, receipt), "aggregate for different live owner count admitted");
    }
    for (const auto& wire : {
            std::string{}, "1 " + counts + " 2", "2 " + counts, "3 " + counts + " 2",
            "2 " + counts + " 0", "2 " + counts + " 4", "2 " + counts + " -1",
            "2 " + counts + " 2 9", "2 " + counts + " 2x",
            "2 " + counts + " 18446744073709551616"}) {
        require(!parse_source_receipt(wire, 2, receipt), "malformed record admitted");
    }
    for (const auto* wire : {
            "2 4096 42 9 1800000000000000000 8192 8192 16384 0 2",
            "2 4096 42 9 1800000000000000000 0 8192 16384 0 2",
            "2 4096 42 9 1800000000000000000 4096 0 16384 0 2",
            "2 4096 42 9 1800000000000000000 4096 8192 0 0 2",
            "2 4096 42 9 1800000000000000000 4096 8192 16384 1125899906842625 2"}) {
        require(!parse_source_receipt(wire, 2, receipt), "invalid measured bytes admitted");
    }
    require(!parse_source_receipt("2 " + counts + " 2", 0, receipt), "zero owners admitted");
    require(!parse_source_receipt(std::string(513, ' '), 1, receipt), "oversized wire admitted");
    std::cout << "source_receipt_test: OK\n";
}
