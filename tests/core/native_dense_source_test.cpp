// Host upload sources span all stages; device pricing retains the stage's layer range.
#include "gguf_fixture.hpp"
#include "strata/core/native_dense.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;

int main() {
    const auto dir = fs::temp_directory_path() /
        ("strata-native-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    struct Cleanup {
        fs::path path;
        ~Cleanup() { NativeDense::set_layer_range(-1, -1); std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{dir};
    const std::string shard = (dir / "dense.gguf").string();
    // Each Q8_0 matrix is 32 blocks * 34 bytes = 1088 source bytes. PLE is shared by both stages.
    fixture::write(shard, {}, {{"blk.0.attn_qkv.weight", {256, 4}, 8, 1},
                               {"blk.2.attn_qkv.weight", {256, 4}, 8, 2},
                               {"blk.1.ple_key.weight", {256, 4}, 8, 3},
                               {"blk.2.ffn_gate_inp.weight", {256, 4}, 8, 4}});
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };
    std::string err;
    uint64_t full = 0, source = 0;
    NativeDense::set_layer_range(-1, -1);
    check(NativeDense::planned_bytes({shard}, true, full, err, &source) && source == 3264,
          "whole-model source excludes unsupported names and scratch");
    for (int start : {0, 2}) {
        NativeDense::set_layer_range(start, start + 1);
        uint64_t stage = 0, without_source = 0;
        check(NativeDense::planned_bytes({shard}, true, stage, err, &source) &&
              source == 3264 && full - stage == 1088,
              "each trimmed stage keeps its device subset but prices the complete source union once");
        check(NativeDense::planned_bytes({shard}, true, without_source, err) && without_source == stage,
              "requesting host source accounting does not change device allocation pricing");
    }
    NativeDense::set_layer_range(-1, -1);
    uint64_t stage = 0;
    check(NativeDense::planned_bytes({shard}, true, stage, err, &source, 9, 10) &&
          stage == 0 && source == 3264,
          "an empty explicit device range does not erase other stages' host source demand");
    NativeDense::set_layer_range(0, 1);
    check(NativeDense::planned_bytes({shard}, false, stage, err, &source) && source == 2176,
          "excluding the native PLE key also excludes it from host source demand");
    check(!NativeDense::planned_bytes({shard, shard}, true, stage, err, &source),
          "duplicate source tensors refuse rather than double-counting the shared upload");
    return failures ? 1 : 0;
}
