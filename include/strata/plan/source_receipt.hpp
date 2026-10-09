#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace strata::plan {

struct SourceResidencyReceipt {
    uint64_t expert_bytes = 0, expert_inode = 0, expert_device = 0, expert_mtime_ns = 0;
    uint64_t ring_bytes = 0, guest_pages_bytes = 0, reader_overhead_bytes = 0, host_payload_bytes = 0;
    uint64_t lane_count = 1;
};

// Parse the protected controller's aggregate measurement. A v1 record prices
// exactly one source owner; a v2 record prices its explicit number of live rings.
// File protection and artifact identity are checked by the caller, not this wire parser.
inline bool parse_source_receipt(std::string_view text, uint64_t expected_lanes,
                                 SourceResidencyReceipt& out) {
    if (text.empty() || text.size() > 512 || expected_lanes < 1 || expected_lanes > 3) return false;
    const char* cursor = text.data();
    const char* end = cursor + text.size();
    uint64_t fields[10]{};
    size_t count = 0;
    while (cursor < end) {
        while (cursor < end && (*cursor == ' ' || *cursor == '\n')) ++cursor;
        if (cursor == end) break;
        if (count == 10) return false;
        const auto result = std::from_chars(cursor, end, fields[count++]);
        if (result.ec != std::errc{} || result.ptr == cursor) return false;
        cursor = result.ptr;
        if (cursor < end && *cursor != ' ' && *cursor != '\n') return false;
    }
    if (!((fields[0] == 1 && count == 9) || (fields[0] == 2 && count == 10))) return false;
    const uint64_t lanes = fields[0] == 1 ? 1 : fields[9];
    if (lanes != expected_lanes) return false;
    // The nanosecond mtime is not a byte count and may legitimately exceed 2^50.
    for (size_t i = 5; i < 9; ++i) if (fields[i] > (1ull << 50)) return false;
    if (!fields[5] || fields[5] > fields[1] || !fields[6] || !fields[7]) return false;
    out = {fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7], fields[8], lanes};
    return true;
}

}  // namespace strata::plan
