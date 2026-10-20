#include "strata/core/shared_rope.hpp"
#include "strata/kernels/rope.hpp"
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {
int failures = 0;
void check(bool value, const char* name) {
    if (!value) { std::fprintf(stderr, "FAIL %s\n", name); ++failures; }
}
void table_prefix(const strata::kernels::RopeScaling& scaling) {
    using namespace strata;
    constexpr int capacity = 262144, rows = 4096, width = 64;
    constexpr size_t plane = (size_t)capacity * (width / 2), prefix = (size_t)rows * (width / 2);
    std::vector<float> long_table(plane * 2), short_table(prefix * 2);
    kernels::build_rope_table(width, scaling, capacity, long_table.data(), long_table.data() + plane);
    kernels::build_rope_table(width, scaling, rows, short_table.data(), short_table.data() + prefix);
    const bool cosine_exact = std::memcmp(long_table.data(), short_table.data(), prefix * sizeof(float)) == 0;
    const bool sine_exact = std::memcmp(long_table.data() + plane, short_table.data() + prefix, prefix * sizeof(float)) == 0;
    check(cosine_exact, "first 4096 cosine rows are bit-identical at 262144 capacity");
    check(sine_exact, "first 4096 sine rows are bit-identical at 262144 capacity");
    const core::SharedRopeTable table{long_table.data(), capacity, width, scaling};
    core::SharedRopeView generation;
    check(!table.borrow(0, 0, capacity, width, scaling, generation), "equal-capacity generation borrow");
    {
        core::SharedRopeView short_state;
        check(!table.borrow(0, 0, rows, width, scaling, short_state), "checked short-state borrow");
        check(short_state.cells == rows && short_state.cosine == generation.cosine &&
              short_state.sine == generation.sine && short_state.sine == long_table.data() + plane,
              "short view exposes only required rows but retains the full-capacity sine offset");
        check(std::memcmp(short_state.sine, short_table.data() + prefix, prefix * sizeof(float)) == 0,
              "borrowed sine plane is the actual short table, not a slice of cosine");
    }
    core::SharedRopeView after;
    check(!table.borrow(0, 0, capacity, width, scaling, after) &&
          after.cosine == generation.cosine && after.sine == generation.sine && after.cells == capacity,
          "short-view teardown leaves the immutable generation descriptor intact");
    auto refuses = [&](int device, int64_t count, int64_t n_rot, const kernels::RopeScaling& config,
                       const char* expected) {
        const auto saved = after;
        const char* why = table.borrow(0, device, count, n_rot, config, after);
        check(why && std::strcmp(why, expected) == 0, expected);
        check(after.cosine == saved.cosine && after.sine == saved.sine && after.cells == saved.cells &&
              table.data == long_table.data() && table.cells == capacity,
              "refusal does not mutate a view or replace the published table");
    };
    refuses(1, rows, width, scaling, "shared RoPE: allocation device differs");
    refuses(0, rows, width / 2, scaling, "shared RoPE: rotary width differs");
    refuses(0, capacity + 1, width, scaling, "shared RoPE: requested rows exceed immutable table capacity");
    refuses(0, 0, width, scaling, "shared RoPE: requested rows must be in 1..INT32_MAX");
    refuses(0, -1, width, scaling, "shared RoPE: requested rows must be in 1..INT32_MAX");
    refuses(0, (int64_t)INT32_MAX + 1, width, scaling, "shared RoPE: requested rows must be in 1..INT32_MAX");
    refuses(0, rows, 63, scaling, "shared RoPE: rotary width must be positive, even and at most INT32_MAX");
    auto changed = scaling;
    changed.type = scaling.type == kernels::RopeScalingType::None ? kernels::RopeScalingType::Linear : kernels::RopeScalingType::None;
    refuses(0, rows, width, changed, "shared RoPE: scaling type differs");
    struct Field { double kernels::RopeScaling::* member; const char* reason; };
    const Field fields[] = {
        {&kernels::RopeScaling::freq_base, "shared RoPE: frequency base differs"},
        {&kernels::RopeScaling::factor, "shared RoPE: scaling factor differs"},
        {&kernels::RopeScaling::freq_scale_in, "shared RoPE: frequency scale differs"},
        {&kernels::RopeScaling::orig_ctx, "shared RoPE: original context differs"},
        {&kernels::RopeScaling::ext_factor, "shared RoPE: YaRN extension factor differs"},
        {&kernels::RopeScaling::attn_factor, "shared RoPE: attention factor differs"},
        {&kernels::RopeScaling::beta_fast, "shared RoPE: YaRN fast beta differs"},
        {&kernels::RopeScaling::beta_slow, "shared RoPE: YaRN slow beta differs"}
    };
    for (const auto& field : fields) {
        changed = scaling; changed.*field.member += 1;
        refuses(0, rows, width, changed, field.reason);
    }
    check(plane * 2 * sizeof(float) == 67108864 && prefix * 2 * sizeof(float) == 1048576,
          "long and hypothetical dedicated short tables retain their exact byte prices");
    std::printf("SHARED_ROPE_PREFIX scaling=%s capacity=%d rows=%d bit_exact=%d borrow_device_bytes=0\n",
                scaling.type == kernels::RopeScalingType::None ? "none" : "yarn", capacity, rows, cosine_exact && sine_exact);
}
}
int main() {
    static_assert(std::is_trivially_destructible_v<strata::core::SharedRopeTable>);
    static_assert(std::is_trivially_destructible_v<strata::core::SharedRopeView>);
    strata::kernels::RopeScaling scaling;
    table_prefix(scaling);
    scaling.type = strata::kernels::RopeScalingType::YaRN;
    scaling.factor = 4; scaling.ext_factor = 1; scaling.orig_ctx = 65536;
    table_prefix(scaling);
    strata::core::SharedRopeTable unpublished;
    strata::core::SharedRopeView view;
    const char* why = unpublished.borrow(0, 0, 4096, 64, scaling, view);
    check(why && std::strcmp(why, "shared RoPE: no published process-owned table on this device") == 0,
          "a required borrow cannot manufacture a second table");
    return failures ? 1 : 0;
}
