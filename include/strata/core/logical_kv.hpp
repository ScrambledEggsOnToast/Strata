// Logical image byte runs over a private page table; no CUDA or allocation.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::core {
struct KvLogicalPages {
    const int32_t* pages = nullptr;
    uint64_t logical = 0, physical = 0;
    bool valid(uint64_t count) const {
        if (!pages) return true; // historical identity/authoritative-host layout
        if (count > logical || !physical) return false;
        for (uint64_t p = 0; p < count; ++p)
            if (pages[p] < 0 || uint64_t(pages[p]) >= physical) return false;
        return true;
    }
    template<class Fn> bool visit(size_t page_bytes, size_t offset, size_t bytes, Fn fn) const {
        if (!pages || !page_bytes) return !bytes || fn(offset, bytes, offset);
        if (offset > SIZE_MAX - bytes || uint64_t((offset + bytes) / page_bytes) +
            ((offset + bytes) % page_bytes != 0) > logical) return false;
        while (bytes) {
            const size_t page = offset / page_bytes, inside = offset % page_bytes;
            if (pages[page] < 0 || uint64_t(pages[page]) >= physical ||
                uint64_t(pages[page]) > (SIZE_MAX - inside) / page_bytes) return false;
            size_t n = std::min(bytes, page_bytes - inside);
            while (n < bytes && (offset + n) % page_bytes == 0) {
                const size_t next = (offset + n) / page_bytes;
                if (next >= logical || pages[next] < 0 || uint64_t(pages[next]) >= physical ||
                    uint64_t(pages[next]) != uint64_t(pages[page]) + next - page) break;
                n += std::min(bytes - n, page_bytes);
            }
            const size_t at = size_t(pages[page]) * page_bytes + inside;
            if (n > SIZE_MAX - at || !fn(at, n, offset)) return false;
            offset += n; bytes -= n;
        }
        return true;
    }
};
} // namespace strata::core
