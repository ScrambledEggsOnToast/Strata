// HET-014 host oracle for the retained CUDA 12.4 SwiGLU lowering.
// The tail RN/FTZ saturation rule is an operator-approved PROJECT requirement,
// not a claimed full-range NVIDIA __expf guarantee. No GPU result defines a bound.
#pragma once

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef __FAST_MATH__
#error "The independent SwiGLU reference must not use host fast-math"
#endif

namespace swiglu_reference {

constexpr double kUnitRoundoff = 0x1p-24;
constexpr double kMinNormal = 0x1p-126;
constexpr double kDivisionLimit = 0x1p126;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct SwigluReferenceDomain : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void normal_or_zero(double value, const char* stage) {
    if (!std::isfinite(value) || (value != 0 &&
        (std::fabs(value) < std::numeric_limits<float>::min() ||
         std::fabs(value) > std::numeric_limits<float>::max())))
        throw std::runtime_error(std::string(stage) + ": outside the finite normal-domain contract");
}
inline void swiglu_reference_domain(double value, const char* stage) {
    try { normal_or_zero(value, stage); }
    catch (const std::runtime_error& e) { throw SwigluReferenceDomain(e.what()); }
}

// Explicit nearest-even binary64 -> binary32. No host float conversion, FTZ mode,
// infinity cast or ambient rounding mode participates in this conversion.
inline float rn32(double value) {
    static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
    const uint64_t bits = std::bit_cast<uint64_t>(value);
    const uint32_t sign = uint32_t(bits >> 32) & 0x80000000u;
    const unsigned exponent = unsigned((bits >> 52) & 0x7ffu);
    const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
    if (exponent == 0x7ffu)
        return std::bit_cast<float>(sign | (fraction ? 0x7fc00000u : 0x7f800000u));
    if (exponent == 0) return std::bit_cast<float>(sign); // Even binary64's largest subnormal rounds to zero.
    const int e = int(exponent) - 1023;
    if (e > 127) return std::bit_cast<float>(sign | 0x7f800000u);
    if (e < -150) return std::bit_cast<float>(sign);
    const uint64_t significand = (uint64_t{1} << 52) | fraction;
    const unsigned shift = unsigned(e >= -126 ? 29 : -97 - e);
    uint64_t rounded = significand >> shift;
    const uint64_t remainder = significand & ((uint64_t{1} << shift) - 1);
    const uint64_t halfway = uint64_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
    if (e < -126) return std::bit_cast<float>(sign | uint32_t(rounded));
    // Carry into the next binade (including infinity) is intentional.
    return std::bit_cast<float>(sign | ((uint32_t(e + 126) << 23) + uint32_t(rounded)));
}

inline float ftz(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x7f800000u) == 0 ? std::bit_cast<float>(bits & 0x80000000u) : value;
}
inline double below(double value) { return std::nextafter(value, -kInfinity); }
inline double above(double value) { return std::nextafter(value, kInfinity); }

struct Interval { double lo, hi; };

inline void require_host_arithmetic() {
    if (std::fegetround() != FE_TONEAREST)
        throw SwigluReferenceDomain("SwiGLU reference requires host nearest-even arithmetic");
    // All interval intermediates are normal binary64 values. Binary32 subnormal
    // conversions must also be preserved when representing interval endpoints.
    volatile float smallest = std::bit_cast<float>(uint32_t{1});
    volatile double converted = smallest;
    if (converted != 0x1p-149)
        throw SwigluReferenceDomain("SwiGLU reference rejects host denormals-are-zero");
}

// Independent exponential enclosure; no accuracy premise about the host libm's
// exp is needed for the tail oracle. For |x|<=256, r=|x|/256 is exact and <=1.
// Sum through r^20/20!, enclosing every operation outwards. The remainder is
// bounded by t20*r/21 / (1-r/22). Eight squarings recover exp(|x|); reciprocal
// gives exp(x) for x<0. Intermediates are well inside normal binary64 range.
inline Interval exp_interval(float x) {
    require_host_arithmetic();
    if (!std::isfinite(x) || std::fabs(double(x)) > 256)
        throw SwigluReferenceDomain("SwiGLU tail exponential outside |gate|<=256 reference domain");
    if (x == 0) return {1, 1};
    if (std::fabs(double(x)) < 0x1p-20) {
        // For 0<a<2^-20, 1-2a < exp(+-a) < 1+2a. This also keeps
        // Taylor terms away from binary64 underflow for tiny normal FP32 gates.
        const double a = 2 * std::fabs(double(x));
        return {below(1 - a), above(1 + a)};
    }
    const double r = std::ldexp(std::fabs(double(x)), -8);
    Interval term{1, 1}, sum{1, 1};
    for (int n = 1; n <= 20; ++n) {
        term = {below(below(term.lo * r) / n), above(above(term.hi * r) / n)};
        sum = {below(sum.lo + term.lo), above(sum.hi + term.hi)};
    }
    const double next = above(above(term.hi * r) / 21);
    const double remainder = above(next / below(1 - above(r / 22)));
    sum.hi = above(sum.hi + remainder);
    for (int n = 0; n < 8; ++n)
        sum = {below(sum.lo * sum.lo), above(sum.hi * sum.hi)};
    if (x < 0) sum = {below(1 / sum.hi), above(1 / sum.lo)};
    return sum;
}

// Tail policy only: finite rounded exp results get the documented coefficient;
// an overflowing rounded endpoint is positive infinity. FTZ is applied AFTER
// the finite-result allowance, so it cannot be mistaken for a subnormal ULP bound.
inline Interval exponential_outputs(float gate) {
    const Interval exact = exp_interval(-gate);
    const float low = rn32(exact.lo), high = rn32(exact.hi);
    const double units = 2 + std::floor(1.173 * std::fabs(double(gate)));
    // A center interval can cross a binade, where ULP doubles. Its lower
    // allowance is NOT monotone there: use the largest finite center spacing
    // for both endpoints. Overflow itself remains the separate +infinity arm.
    const float largest = std::min(high, std::numeric_limits<float>::max());
    const double spacing = double(largest) < kMinNormal ? 0x1p-149 :
                           std::ldexp(1.0, std::ilogb(largest) - 23);
    const double radius = units * spacing; // Exact: integer times a power of two.
    auto endpoint = [radius](float center, bool upper) -> double {
        if (std::isinf(center)) return kInfinity;
        const double edge = upper ? above(double(center) + radius) :
                                   std::max(0.0, below(double(center) - radius));
        return double(ftz(rn32(edge)));
    };
    return {endpoint(low, false), endpoint(high, true)};
}

inline Interval denominators(Interval exponential) {
    if (!(0 <= exponential.lo && exponential.lo <= exponential.hi))
        throw SwigluReferenceDomain("Invalid independent exponential interval");
    auto add_one = [](double e) {
        // E is an FP32 endpoint. Outside this middle range, +1 cannot
        // change its rounded sum; inside, binary64 addition is exact.
        if (e <= 0x1p-25) return 1.0;
        if (e >= 0x1p25) return e;
        return double(rn32(1 + e));
    };
    return {add_one(exponential.lo), add_one(exponential.hi)};
}

struct Outputs {
    bool zero = false;
    bool normal = false;
    bool negative = false;
    float lo = 0, hi = 0; // Magnitudes; the zero/normal gap is not an accepted interval.

    bool contains(float value) const {
        if (!std::isfinite(value) || std::signbit(value) != negative) return false;
        const double magnitude = std::fabs(double(value));
        if (magnitude == 0) return zero;
        return normal && magnitude >= kMinNormal && magnitude >= lo && magnitude <= hi;
    }
};

inline float ceil32(double value) {
    float result = rn32(value);
    if (double(result) < value) result = std::nextafter(result, std::numeric_limits<float>::infinity());
    return result;
}
inline float floor32(double value) {
    float result = rn32(value);
    if (double(result) > value) result = std::nextafter(result, 0.0f);
    return result;
}

// Reduce a rounded-denominator enclosure. D==2^126 is the accurate arm;
// D>2^126 (including infinity) is the distinct zero arm of div.approx.ftz.
inline Outputs from_denominators(float gate, float up, Interval d) {
    if (!(d.lo >= 1 && d.lo <= d.hi))
        throw SwigluReferenceDomain("Invalid rounded SwiGLU denominator interval");
    Outputs out;
    out.negative = std::signbit(gate) != std::signbit(up);
    out.zero = d.hi > kDivisionLimit || gate == 0 || up == 0;
    if (gate == 0 || up == 0 || d.lo > kDivisionLimit) return out;
    const double a = std::fabs(double(gate));
    const double qmin = below(a / std::min(d.hi, kDivisionLimit));
    const double qmax = above(a / d.lo);
    if (!(qmax > 0 && std::isfinite(qmax)))
        throw SwigluReferenceDomain("Unsupported SwiGLU quotient interval");
    // Next-binade spacing plus 3 ULP covers 2 ULP of approximate division and
    // rounding to its reference. This is only the NEW tail arm, never normal policy.
    const double spacing = std::max(0x1p-149, std::ldexp(1.0, std::ilogb(qmax) + 1 - 23));
    const double radius = 3 * spacing;
    const float qlo = ceil32(std::max(0.0, below(qmin - radius)));
    const float qhi = floor32(above(qmax + radius));
    if (qlo < kMinNormal) out.zero = true;
    if (qhi < kMinNormal) return out;
    if (!std::isfinite(qhi))
        throw SwigluReferenceDomain("SwiGLU quotient overflow outside reference domain");
    const float first_normal = std::max(qlo, std::numeric_limits<float>::min());
    const double b = std::fabs(double(up));
    // Product of two binary32 values is exact in binary64. Flush quotient first.
    const float low_product = ftz(rn32(double(first_normal) * b));
    const float high_product = ftz(rn32(double(qhi) * b));
    if (!std::isfinite(high_product))
        throw SwigluReferenceDomain("SwiGLU product overflow outside reference domain");
    if (low_product == 0) out.zero = true;
    if (high_product != 0) {
        out.normal = true;
        out.lo = std::max(low_product, std::numeric_limits<float>::min());
        out.hi = high_product;
    }
    return out;
}

inline double checked_tail(float gate, float up, float got) {
    const Outputs allowed = from_denominators(gate, up, denominators(exponential_outputs(gate)));
    if (!allowed.contains(got)) throw std::runtime_error("SwiGLU independent tail contract exceeded");
    return 0; // Membership, not a tolerance-normalized ordinary error.
}

// Fixed normal-range acceptance policy, retained without loosening. CUDA 12.4's
// current table uses 1.173, not the historical 1.16 coefficient below: this is
// a stricter policy, not a proved worst-case envelope from that table.
// This checks the epilogue on independently validated observed gate/up, not on
// an ideal hidden value that would wrongly assume continuous requantization.
inline double checked_swiglu(float g, float up, float got) {
    normal_or_zero(g, "SwiGLU gate");
    normal_or_zero(up, "SwiGLU up");
    normal_or_zero(got, "SwiGLU output");
    require_host_arithmetic();
    const double ex = std::exp(-(double) g);
    // Domain selection is input-only. No comparison failure can reach a retry.
    if (g == 0 || up == 0 || !std::isfinite(ex) || ex == 0 ||
        ex < kMinNormal || ex > std::numeric_limits<float>::max())
        return checked_tail(g, up, got);
    swiglu_reference_domain(ex, "exp(-gate)");
    const double exp_error = (3 + std::floor(1.16 * std::fabs((double) g))) *
                             std::ldexp(1.0, std::ilogb(ex) - 23);
    const double denominator = 1 + ex;
    const double denominator_error = exp_error + kUnitRoundoff * (denominator + exp_error);
    if (!(denominator - denominator_error > 0) || denominator + denominator_error >= 0x1p126)
        return checked_tail(g, up, got);
    const double quotient = (double) g / denominator;
    const double quotient_error = std::fabs((double) g) * denominator_error /
        (denominator * (denominator - denominator_error)) +
        4 * kUnitRoundoff * std::fabs((double) g) / (denominator - denominator_error);
    const double reference = quotient * up;
    if ((quotient != 0 && std::fabs(quotient) < kMinNormal) ||
        (reference != 0 && std::fabs(reference) < kMinNormal))
        return checked_tail(g, up, got);
    swiglu_reference_domain(quotient, "SwiGLU quotient");
    swiglu_reference_domain(reference, "SwiGLU reference");
    const double bound = std::fabs((double) up) * quotient_error +
                         kUnitRoundoff * std::fabs((double) up) * (std::fabs(quotient) + quotient_error);
    const double error = std::fabs((double) got - reference);
    if (!(error <= bound)) throw std::runtime_error("SwiGLU independent epilogue bound exceeded");
    return bound > 0 ? error / bound : 0;
}

} // namespace swiglu_reference
