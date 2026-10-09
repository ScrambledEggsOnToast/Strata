// Host-only independent-oracle boundaries; no CUDA context or library required.
#include "../cuda/swiglu_reference.hpp"

#include <cstdio>
#include <functional>

using namespace swiglu_reference;

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& action, const char* expected) {
    try { action(); }
    catch (const std::runtime_error& e) {
        require(std::string(e.what()).find(expected) != std::string::npos, "wrong rejection kind");
        return;
    }
    throw std::runtime_error("invalid oracle output accepted");
}
void rounding_boundaries() {
    require(rn32(1 + 0x1p-24) == 1, "even tie at one rounded up");
    require(rn32(1 + 3 * 0x1p-24) == 1 + 0x1p-22, "odd tie did not round to even");
    require(rn32(0x1p-150) == 0, "zero tie did not round to even");
    require(std::bit_cast<uint32_t>(rn32(3 * 0x1p-150)) == 2, "subnormal odd tie");
    const double midpoint = kMinNormal - 0x1p-150;
    require(ftz(rn32(below(midpoint))) == 0, "below min-normal midpoint did not flush");
    require(ftz(rn32(midpoint)) == kMinNormal, "min-normal tie must round to normal");
    require(ftz(rn32(above(midpoint))) == kMinNormal, "above min-normal midpoint");
    const double overflow = 0x1p128 - 0x1p103;
    require(rn32(below(overflow)) == std::numeric_limits<float>::max(), "below overflow tie");
    require(std::isinf(rn32(overflow)), "overflow tie must saturate to infinity");
    require(std::bit_cast<uint32_t>(rn32(-0.0)) == 0x80000000u, "negative zero lost");
    const int old = std::fegetround();
    require(std::fesetround(FE_DOWNWARD) == 0, "cannot exercise ambient rounding independence");
    require(rn32(1 + 0x1p-23) == 1 + 0x1p-23, "explicit conversion used ambient rounding");
    rejects([] { (void) exp_interval(1); }, "nearest-even");
    require(std::fesetround(old) == 0, "cannot restore host rounding");
    // exp(-2^-26) rounds to 1. Its allowed lower result uses the spacing
    // at 1, not the half-sized spacing immediately below that binade.
    const Interval crossing = exponential_outputs(0x1p-26f);
    require(crossing.lo <= 1 - 2 * 0x1p-23, "exponential binade crossing under-enclosed");
}
void denominator_boundaries() {
    const float t = float(kDivisionLimit);
    const float lower = std::nextafter(t, 0.0f);
    const float higher = std::nextafter(t, std::numeric_limits<float>::infinity());
    const Outputs below_t = from_denominators(-128, float(kDivisionLimit), {lower, lower});
    const Outputs at_t = from_denominators(-128, float(kDivisionLimit), {t, t});
    const Outputs above_t = from_denominators(-128, float(kDivisionLimit), {higher, higher});
    require(!below_t.zero && below_t.contains(-128), "below threshold lost accurate arm");
    require(!at_t.zero && at_t.contains(-128), "threshold equality incorrectly saturated");
    require(above_t.zero && !above_t.normal && above_t.contains(-0.0f), "strict threshold did not saturate");
    const Outputs spanning = from_denominators(-128, float(kDivisionLimit), {lower, higher});
    require(spanning.contains(-0.0f) && spanning.contains(-128), "straddling union lost an arm");
    require(!spanning.contains(-64), "zero/accurate gap incorrectly convex-hulled");
    const double tie = 0x1p-24;
    const Interval d_below = denominators({0, std::nextafter(float(tie), 0.0f)});
    const Interval d_equal = denominators({0, tie});
    const Interval d_above = denominators({0, std::nextafter(float(tie), 1.0f)});
    require(d_below.lo == 1 && d_below.hi == 1, "below denominator-one midpoint");
    require(d_equal.lo == 1 && d_equal.hi == 1, "denominator-one even tie");
    require(d_above.hi > 1, "above denominator-one midpoint collapsed");
    const Outputs one = from_denominators(100, 1, {1, 1});
    require(one.contains(std::nextafter(100.0f, 101.0f)), "D=1 incorrectly forced exact division");
    require(!one.contains(101), "D=1 division corruption accepted");
}
void ftz_and_signs() {
    const float tiny = std::numeric_limits<float>::min();
    require(checked_swiglu(tiny, 2, 0) == 0, "quotient FTZ not modelled");
    rejects([tiny] { (void) checked_swiglu(tiny, 2, tiny); }, "tail contract");
    require(checked_swiglu(1, tiny, 0) == 0, "final-product FTZ not modelled");
    rejects([tiny] { (void) checked_swiglu(1, tiny, tiny); }, "tail contract");
    for (float gate : {0.0f, -0.0f}) {
        for (float up : {2.0f, -2.0f}) {
            const float zero = std::copysign(0.0f, gate * up);
            require(checked_swiglu(gate, up, zero) == 0, "zero sign rejected");
            rejects([=] { (void) checked_swiglu(gate, up, -zero); }, "tail contract");
        }
    }
    require(checked_swiglu(-100, 2, -0.0f) == 0, "overflow saturation rejected");
    rejects([] { (void) checked_swiglu(-100, 2, -1); }, "tail contract");
    require(checked_swiglu(100, 2, 200) == 0, "underflow tail rejected");
    rejects([] { (void) checked_swiglu(100, 2, 201); }, "tail contract");
    rejects([] { (void) checked_swiglu(100, 2, std::numeric_limits<float>::infinity()); }, "normal-domain");
    rejects([] { (void) checked_swiglu(1, 2, std::numeric_limits<float>::quiet_NaN()); }, "normal-domain");
    rejects([] { (void) checked_swiglu(1, 2, std::numeric_limits<float>::denorm_min()); }, "normal-domain");
    rejects([] { (void) checked_swiglu(300, 2, 600); }, "|gate|<=256");
}
void ordinary_policy_is_not_retried() {
    const float gate = 85.3125f;
    require(checked_swiglu(gate, 1, gate) <= 1, "ordinary input rejected");
    rejects([gate] { (void) checked_swiglu(gate, 1, 86); }, "epilogue bound exceeded");
    rejects([] { (void) checked_swiglu(1, 1, 1); }, "epilogue bound exceeded");
}
}

int main() {
    try {
        rounding_boundaries();
        denominator_boundaries();
        ftz_and_signs();
        ordinary_policy_is_not_retried();
        std::puts("SwiGLU host reference: rounding, saturation union, both FTZ stages, signs and normal rejection passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SwiGLU host reference failed: %s\n", e.what());
        return 1;
    }
}
