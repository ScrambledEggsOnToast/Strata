#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

void test_complement_plan() {
    using namespace strata::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
}

void test_resident_lend_region() {
    using namespace strata::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_resident_exchange() {
    using namespace strata::core;
    using namespace strata::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

void test_cgroup_memory_budget() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    CgroupMemoryStat clean_cache{40 * GiB, 8 * GiB, 24 * GiB, 32 * GiB, 0, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB, "clean active and inactive cache was not credited");

    CgroupMemoryStat protected_cache{40 * GiB, 8 * GiB, 24 * GiB, 32 * GiB,
                                     3 * GiB, 1 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, protected_cache, bytes), "valid protected cache was rejected");
    require(bytes == 38 * GiB, "shmem, unevictable, dirty or writeback pages were credited");

    CgroupMemoryStat oversized_lru{10 * GiB, 20 * GiB, 20 * GiB, 4 * GiB, 0, 0, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_lru, bytes), "oversized LRU was rejected");
    require(bytes == 6 * GiB, "file charge did not bound cache credit");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max, max, 0, 0, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes), "large counters were rejected");
    require(bytes == 0, "cache accounting overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, 0, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes), "over-limit counters were rejected");
    require(bytes == 0, "over-limit usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

// ================================ HET-017: THE WINDOWED SOURCE ================================

/// Every byte of a fixture is a function of its file offset, so a mis-placed or short read shows up anywhere,
/// not just at a chosen marker.
uint8_t windowed_plane_byte(uint64_t at) {
    return (uint8_t) ((at * 131ull + (at >> 7) * 29ull + 17ull) & 0xff);
}

/// `planned_bytes` is pure: it prices without touching anything, and it refuses absurd geometries by name.
void test_windowed_planned_bytes() {
    using namespace strata::core;
    using strata::kernels::cpu::ExpertLayout;
    std::string err;
    ExpertLayout empty;
    require(WindowedExpertSource::planned_bytes(WindowedExpertSource::Config{}, empty, err) == 0 &&
                err.find("empty") != std::string::npos,
            "an empty layout was priced");
    ExpertLayout small = empty;
    small.n_layers = 1;
    small.n_expert = 2;
    small.total = 8;
    WindowedExpertSource::Config many;
    many.slots = std::numeric_limits<int64_t>::max();
    require(WindowedExpertSource::planned_bytes(many, small, err) == 0 && err.find("slot count") != std::string::npos,
            "an absurd slot count was priced");
}

/// Byte/lifetime regressions require direct IO support. Refusal must fail the read-path test,
/// not silently convert an implementation defect into a passing run.
bool open_windowed(strata::core::WindowedExpertSource& src, const fs::path& dir, int64_t layers, int64_t experts,
                   const strata::core::WindowedExpertSource::Config& cfg, std::string& err) {
    const bool opened = src.open(dir.string(), layers, experts, err, cfg);
    require(opened, "direct source fixture open failed: " + err);
    return opened;
}

void test_windowed_direct_canonical() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 4;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    create_pack(dir.path, total);
    // FULL-BYTE FIXTURE: the whole file is the offset pattern, so every blob is verifiable end to end.
    {
        std::fstream pack(dir.path / "experts.bin", std::ios::binary | std::ios::in | std::ios::out);
        require((bool) pack, "could not open synthetic experts.bin for the byte pattern");
        constexpr uint64_t kChunk = 1u << 20;
        std::vector<char> buf((size_t) kChunk);
        for (uint64_t at = 0; at < total; at += kChunk) {
            const uint64_t n = std::min<uint64_t>(kChunk, total - at);
            for (uint64_t i = 0; i < n; ++i) buf[(size_t) i] = (char) windowed_plane_byte(at + i);
            pack.seekp((std::streamoff) at);
            pack.write(buf.data(), (std::streamsize) n);
            require((bool) pack, "could not write the experts.bin byte pattern");
        }
    }

    WindowedExpertSource::Config cfg;
    const uint64_t priced = WindowedExpertSource::planned_bytes(cfg, expert_layout(), err);
    require(priced >= (uint64_t) experts * ((uint64_t) BLOB + 2 * 4096),
            "the ring price lost the ring's own bytes");

    WindowedExpertSource src;
    open_windowed(src, dir.path, layers, experts, cfg, err);
    require(src.ready() && src.blobs() == layers * experts, "the windowed open left the source incomplete");
    require(src.slots() == experts, "the default ring is one full layer's distinct experts");
    require(src.slot_stride_bytes() >= (uint64_t) BLOB, "a slot cannot take the largest blob");
    require(src.residency_bytes() == (uint64_t) src.slots() * src.slot_stride_bytes(),
            "the resident envelope is the ring, exactly");
    require(src.residency_bytes() <= priced, "the ring exceeded its startup price");
    require(src.io_transient_bytes() == src.slot_stride_bytes(), "the IO transient is one slot");
    require(src.source_bytes() == total, "the source metadata does not name the whole source file");
    require(src.reader_threads() > 0, "the direct reader pool went unreported");

    // BYTE-EXACT across an interior offset boundary, the first and the last byte of whole blobs.
    auto want_blob = [&](int64_t layer, int64_t expert, std::vector<char>& want) {
        want.resize((size_t) BLOB);
        const uint64_t at = (uint64_t) layer * layer_bytes + (uint64_t) expert * (uint64_t) BLOB;
        for (uint64_t i = 0; i < (uint64_t) BLOB; ++i) want[(size_t) i] = (char) windowed_plane_byte(at + i);
    };
    std::vector<char> want;
    const uint8_t* b00 = src.blob(0, 0);
    const uint8_t* b01 = src.blob(0, 1);
    const uint8_t* b02 = src.blob(0, 2);
    require(b00 && b01 && b02, "valid canonical windowed fetches failed");
    require(b00 != b01 && b00 != b02 && b01 != b02, "one epoch's distinct blobs shared a slot");
    want_blob(0, 0, want);
    require(std::memcmp(b00, want.data(), want.size()) == 0, "blob 0,0 is not byte-exact with the source");
    want_blob(0, 1, want);
    require(std::memcmp(b01, want.data(), want.size()) == 0, "blob 0,1 is not byte-exact with the source");
    want_blob(0, 2, want);
    require(std::memcmp(b02, want.data(), want.size()) == 0, "blob 0,2 is not byte-exact with the source");
    require(src.reads() == 3 && src.io_bytes() == 3 * (uint64_t) BLOB,
            "the read and byte counters disagree with the fetches");
    require(src.blob(0, 0) == b00, "a same-epoch refetch moved a blob");
    require(src.reads() == 4 && src.io_bytes() == 3 * (uint64_t) BLOB,
            "a same-epoch refetch re-read the source");

    const uint8_t* b10 = src.blob(1, 0);   // the epoch boundary: layer 0's pointers go stale HERE
    require(b10 != nullptr, "layer 1's first fetch failed across the boundary");
    want_blob(1, 0, want);
    require(std::memcmp(b10, want.data(), want.size()) == 0, "blob 1,0 is not byte-exact with the source");
    require(src.blob(1, 0) == b10, "a layer-1 refetch moved its blob");
    require(src.io_bytes() == 4 * (uint64_t) BLOB, "the boundary changed the byte count");
    const uint8_t* fresh = src.blob(0, 0);   // and back: the ring must serve fresh, correct bytes
    require(fresh != nullptr, "the first fetch of the recycled epoch failed");
    want_blob(0, 0, want);
    require(std::memcmp(fresh, want.data(), want.size()) == 0, "a recycled slot served stale bytes");

    // Refusals must latch and name themselves; on a SECOND source so the main one stays usable.
    WindowedExpertSource bad;
    require(open_windowed(bad, dir.path, layers, experts, cfg, err), "a second open failed");
    require(bad.blob(layers, 0) == nullptr && bad.error().find("geometry") != std::string::npos,
            "an out-of-range lookup was not named");
    require(bad.blob(0, 0) == nullptr, "a failed source kept serving");

    WindowedExpertSource::Config one;
    one.slots = 1;
    WindowedExpertSource tiny;
    require(open_windowed(tiny, dir.path, layers, experts, one, err), "a one-slot ring refused to open");
    require(tiny.blob(0, 0) != nullptr, "the first fetch fits a one-slot ring");
    tiny.begin_layer(0, nullptr, 0);
    const uint8_t* same_layer_epoch = tiny.blob(0, 1);
    require(same_layer_epoch != nullptr, "an explicit same-layer boundary did not recycle the ring");
    want_blob(0, 1, want);
    require(std::memcmp(same_layer_epoch, want.data(), want.size()) == 0,
            "an explicit same-layer epoch served stale bytes");
    require(tiny.blob(0, 2) == nullptr && tiny.error().find("slots") != std::string::npos,
            "an overflowing epoch was not named");
    const uint64_t tiny_priced = WindowedExpertSource::planned_bytes(one, expert_layout(), err);
    require(tiny.residency_bytes() <= tiny_priced, "the one-slot ring exceeded its price");
    tiny.close();
    require(!tiny.ready() && tiny.residency_bytes() == 0, "close did not release the ring");

    // A wrong-size pack refuses BY NAME, before any allocation.  Runs last: it corrupts the pack.
    src.close();
    bad.close();
    fs::resize_file(dir.path / "experts.bin", total - 1);
    WindowedExpertSource short_src;
    require(!short_src.open(dir.path.string(), layers, experts, err, cfg) && !short_src.ready() &&
                err.find(std::to_string(total - 1)) != std::string::npos,
            "a truncated pack was not refused with its size");
}

#if defined(STRATA_NATIVE_EXPERTS)
/// A shard image whose planes sit at `base[r]` (deliberately not page-aligned), `per[r]` bytes each for three
/// experts; every byte is a function of its file offset, so any mis-placed read shows up anywhere.
std::vector<char> windowed_shard_image(const uint64_t base[3], const uint64_t per[3]) {
    uint64_t end = 1;
    for (int r = 0; r < 3; ++r) end = std::max(end, base[r] + per[r] * 3);
    std::vector<char> img((size_t) end, '\0');
    for (int r = 0; r < 3; ++r)
        for (uint64_t i = 0; i < per[r] * 3; ++i)
            img[(size_t) (base[r] + i)] = (char) windowed_plane_byte(base[r] + i);
    return img;
}

void test_windowed_direct_native_bin() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    require(native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err),
            "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    require(native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err),
            "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {(uint64_t) first_fmt.bytes + 123, 'm'}, {layer0_bytes, 'c'},
                                  {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}, {total - 1, 'z'}});

    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load synthetic native layout");
    const uint64_t priced = WindowedExpertSource::planned_bytes(WindowedExpertSource::Config{}, expert_layout(), err);
    require(priced > 0, "the native ring priced to zero");

    WindowedExpertSource src;
    open_windowed(src, dir.path, layers, experts, WindowedExpertSource::Config{}, err);
    require(src.residency_bytes() <= priced, "the native ring exceeded its price");
    require(src.blobs() == layers * experts, "the native blob count is wrong");
    const uint8_t* first = src.blob(0, 0);
    const uint8_t* second = src.blob(0, 1);
    require(first && second && first[0] == 'a' && second[0] == 'b' && second[123] == 'm',
            "same-layer native pointers did not retain the correct expert bytes");
    const uint8_t* next_layer = src.blob(1, 0); // Layer-0 pointers are stale from this epoch boundary.
    const uint8_t* last = src.blob(1, 2);
    require(next_layer && last && next_layer[0] == 'c' && last[(uint64_t) second_fmt.bytes - 1] == 'z',
            "native layer-1 blobs came back wrong");
    require(src.io_bytes() == 2 * (uint64_t) first_fmt.bytes + 2 * (uint64_t) second_fmt.bytes,
            "the native byte count disagrees with the fetches");
    src.close();
}

void test_windowed_direct_native_gguf() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    TempDirectory dir;

    NativeFmt fmts[2];
    std::string err;
    require(native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, fmts[0], err),
            "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    require(native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, 2560, 640, fmts[1], err),
            "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(fmts[0].bytes != fmts[1].bytes, "chosen native formats do not exercise variable layer sizes");
    const uint64_t per[2][3] = {{fmts[0].up_off, fmts[0].down_off - fmts[0].up_off, fmts[0].bytes - fmts[0].down_off},
                                {fmts[1].up_off, fmts[1].down_off - fmts[1].up_off, fmts[1].bytes - fmts[1].down_off}};
    const uint64_t layer_off[2] = {0, (uint64_t) fmts[0].bytes * (uint64_t) experts};
    // plane bases, deliberately NOT page-aligned and never 0 (0 is the layout's "unknown" sentinel)
    uint64_t baseA[3] = {4096 + 123, 0, 0};
    baseA[1] = baseA[0] + per[0][0] * (uint64_t) experts + 777;
    baseA[2] = baseA[1] + per[0][1] * (uint64_t) experts + 333;
    uint64_t baseB[3] = {999, 0, 0};
    baseB[1] = baseB[0] + per[1][0] * (uint64_t) experts + 55;
    baseB[2] = baseB[1] + per[1][1] * (uint64_t) experts + 4444;
    const std::vector<char> imgA = windowed_shard_image(baseA, per[0]);
    const std::vector<char> imgB = windowed_shard_image(baseB, per[1]);
    {
        std::ofstream a(dir.path / "shardA.bin", std::ios::binary | std::ios::trunc);
        require((bool) a, "could not create shard A");
        a.write(imgA.data(), (std::streamsize) imgA.size());
        require((bool) a, "could not write shard A");
    }
    {
        std::ofstream b(dir.path / "shardB.bin", std::ios::binary | std::ios::trunc);
        require((bool) b, "could not create shard B");
        b.write(imgB.data(), (std::streamsize) imgB.size());
        require((bool) b, "could not write shard B");
    }
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << fmts[0].bytes << ' '
                 << baseA[0] << ' ' << baseA[1] << ' ' << baseA[2] << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer_off[1] << ' '
                 << fmts[1].bytes << ' ' << baseB[0] << ' ' << baseB[1] << ' ' << baseB[2] << " shardB.bin\n";
        require((bool) metadata, "could not write native_experts.txt");
    }
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load the GGUF layout: " + err);
    require(expert_layout().gguf_off.size() == (size_t) (3 * layers) &&
                expert_layout().gguf_file.size() == (size_t) (3 * layers) &&
                expert_layout().gguf_file[(size_t) (3 * (layers - 1))] == "shardB.bin" &&
                expert_layout().gguf_file[(size_t) (3 * (layers - 1) + 2)] == "shardB.bin",
            "the plane offsets did not load");

    WindowedExpertSource::Config cfg;
    const uint64_t priced = WindowedExpertSource::planned_bytes(cfg, expert_layout(), err);
    require(priced > 0, "the GGUF ring priced to zero");
    WindowedExpertSource src;
    src.set_gguf((dir.path / "shardA.bin").string());
    open_windowed(src, dir.path, layers, experts, cfg, err);
    require(src.residency_bytes() == priced, "the GGUF-mode ring does not match its own price exactly");
    require(src.slots() == experts, "the default GGUF ring is one layer's experts");
    require(src.source_bytes() == imgA.size() + imgB.size(), "the source metadata lost a shard");
    require(src.reader_threads() >= 2, "a distinct shard got no reader pool");

    for (int64_t l = 0; l < layers; ++l) {
        const NativeFmt& fm = fmts[(size_t) l];
        const uint64_t blob_bytes = (uint64_t) fm.bytes;
        const uint64_t at[3] = {0, (uint64_t) fm.up_off, (uint64_t) fm.down_off};
        const std::vector<char>& img = l == 0 ? imgA : imgB;
        for (int64_t e : {0, 2}) {
            const uint8_t* blob = src.blob(l, e);
            require(blob != nullptr, "a GGUF blob fetch failed");
            std::vector<char> want((size_t) blob_bytes, '\0');
            for (int r = 0; r < 3; ++r) {
                const uint64_t file_base = (l == 0 ? baseA[r] : baseB[r]) + (uint64_t) e * per[(size_t) l][r];
                std::memcpy(want.data() + (size_t) at[r], img.data() + (size_t) file_base,
                            (size_t) per[(size_t) l][r]);
            }
            require(std::memcmp(blob, want.data(), want.size()) == 0,
                    "the assembled GGUF blob is not byte-exact with the shard planes");
        }
    }
    require(src.io_bytes() == 2 * (uint64_t) fmts[0].bytes + 2 * (uint64_t) fmts[1].bytes,
            "the GGUF byte count disagrees with the fetches");
    src.close();
    require(!src.ready(), "close did not release the GGUF ring");
}
#endif
// #633: the host RAM probe with fake /proc and cgroup trees: v2 (a limit, "max", a missing or malformed limit), v1
// (a limit, unlimited), and no cgroup line at all.  Elsewhere than Linux it reads the machine's RAM.
void test_host_memory() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    HostMemory m;
#if defined(__linux__)
    TempDirectory t;
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((t.path / rel).parent_path());
        std::ofstream(t.path / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (t.path / "meminfo").string(), cg = (t.path / "cgroup").string(),
                      root = (t.path / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        return host_available_memory(m, mi, cg, root);
    };
    // no cgroup line at all: MemAvailable alone (before #633: "cannot determine")
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache.  The stat carries all seven counters the
    // credit rule reads (active_file/inactive_file/file/shmem/unevictable/file_dirty/file_writeback): the rule
    // is one rule, and it refuses to credit a group whose counters it cannot see, so a three-key memory.stat
    // (the shape this fixture had before the merge with the fork's conservative credit) is a refusal, not a
    // smaller credit.
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "active_file 0\ninactive_file " + std::to_string(4 * GiB) +
                                  "\nfile " + std::to_string(4 * GiB) +
                                  "\nshmem 0\nunevictable 0\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 36 * GiB && m.cgroup_limit == 48 * GiB, "v2 limit");
    // and the same group with the counters the rule needs missing: refused, not credited from the three keys
    // the pre-merge fixture wrote
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(!probe("0::/box\n"), "a memory.stat without the credit rule's counters was accepted");
    put("fs/box/memory.stat", "active_file 0\ninactive_file " + std::to_string(4 * GiB) +
                                  "\nfile " + std::to_string(4 * GiB) +
                                  "\nshmem 0\nunevictable 0\nfile_dirty 0\nfile_writeback 0\n");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n"), "v2 malformed limit accepted");
    fs::remove(t.path / "fs/box/memory.max");
    require(!probe("0::/box\n"), "v2 group without memory.max accepted");
    // v1: the memory controller's group (a hybrid line list), a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v1 unlimited");
    // a v1 group not visible here (another namespace): MemAvailable alone
    require(probe("4:memory:/elsewhere\n") && m.available == 100 * GiB, "v1 group not mounted");
#else
    require(host_available_memory(m) && m.available > 0 && m.cgroup_limit == ~0ull, "this PC's RAM");
    (void) GiB;
#endif
}

}  // namespace

int main() {
    try {
        test_complement_plan();
        test_resident_lend_region();
        test_resident_exchange();
        test_cgroup_memory_budget();
        test_host_memory();
        test_canonical_layout();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
#endif
        test_windowed_planned_bytes();
        test_windowed_direct_canonical();
#if defined(STRATA_NATIVE_EXPERTS)
        test_windowed_direct_native_bin();
        test_windowed_direct_native_gguf();
#endif
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
