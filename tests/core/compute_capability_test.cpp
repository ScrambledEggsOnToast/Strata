// tests/core/compute_capability_test.cpp - the run-time admission policy, checked on a host with no GPU.
//
// WHY THIS EXISTS: the engine refuses a card older than the floor it was built for, and that refusal lives inside
// device_info(), reachable only through a real cudaGetDeviceProperties.  The case it exists for - an sm_70;sm_86
// binary carried to an older card - cannot be provoked on a machine that has no such card, so the policy is a
// header-only predicate (include/strata/core/device.hpp) that a host can call with any (major, minor), and this
// test calls it with the cards the ticket names.
//
// It is meaningful in BOTH configurations, which is why the expectations are written against
// kMinComputeCapability rather than against a number: a default build must still refuse Volta, and an
// -DSTRATA_EXPERIMENTAL_SM60=ON build must admit it.  Run the same source both ways (the CMake target covers
// whichever configuration it was built in; compiling this file directly with g++ covers the other).
#include "strata/core/device.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using strata::core::compute_capability_problem;
using strata::core::kMinComputeCapability;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string refused(int major, int minor) {
    const std::string why = compute_capability_problem(major, minor);
    require(!why.empty(), "compute capability " + std::to_string(major) + "." + std::to_string(minor) +
                              " should have been refused, but was admitted");
    return why;
}

void admitted(int major, int minor) {
    const std::string why = compute_capability_problem(major, minor);
    require(why.empty(), "compute capability " + std::to_string(major) + "." + std::to_string(minor) +
                             " should have been admitted, but was refused: " + why);
}

// The floor is the whole point of the flag, so the build states which one it carries instead of trusting a log.
void test_floor_matches_the_build() {
#if defined(STRATA_EXPERIMENTAL_SM60)
    require(kMinComputeCapability == 60,
            "an experimental SM60 build must lower the floor to the Pascal generation");
#else
    require(kMinComputeCapability == 75, "a default build must keep the Turing floor");
#endif
    require(kMinComputeCapability == 60 || kMinComputeCapability == 75,
            "the floor is one of the two the build system can emit");
}

// sm_86: the RTX 3090 this programme brings the engine up on.  Admitted by every configuration - it is what the
// release floor was chosen for.
void test_sm86_is_admitted() {
    admitted(8, 6);
    admitted(8, 9);
    admitted(12, 0);
}

// sm_70: the V100.  This is the ticket's target and the reason the flag exists, so the expectation is the flag.
void test_sm70_follows_the_flag() {
    if (kMinComputeCapability <= 70) {
        admitted(7, 0);
    } else {
        const std::string why = refused(7, 0);
        require(why.find("7.0") != std::string::npos, "the refusal must name the capability the card reported");
    }
}

// sm_75: Turing.  At the release floor, so admitted everywhere.
void test_sm75_is_admitted() { admitted(7, 5); }

// The generations the flag does NOT reach.  Maxwell and Kepler sit below the flag's own floor of 6.0, so no build
// may admit them: STRATA_EXPERIMENTAL_SM60 lowers the floor, it does not remove the check.
void test_unsupported_architectures_are_refused() {
    const std::string maxwell = refused(5, 2);
    const std::string kepler = refused(3, 0);
    require(maxwell.find("5.2") != std::string::npos, "the Maxwell refusal must name the reported capability");
    require(kepler.find("3.0") != std::string::npos, "the Kepler refusal must name the reported capability");
}

// Pascal GP10x (6.1) is the generation the flag is named for, so the flag is what decides it.  Pinned here because
// this programme's build targets 70;86: the experiment must not quietly widen or narrow the flag's meaning.
void test_pascal_gp10x_follows_the_flag() {
    if (kMinComputeCapability <= 61) {
        admitted(6, 1);
    } else {
        require(!refused(6, 1).empty(), "a default build must refuse Pascal");
    }
}

// The boundary, both sides of it: the floor itself runs, one minor release below does not.
void test_the_floor_is_the_boundary() {
    admitted(kMinComputeCapability / 10, kMinComputeCapability % 10);
    const int below_major = (kMinComputeCapability - 1) / 10;
    const int below_minor = (kMinComputeCapability - 1) % 10;
    const std::string why = refused(below_major, below_minor);
    require(why.find(std::to_string(kMinComputeCapability / 10) + "." +
                     std::to_string(kMinComputeCapability % 10)) != std::string::npos,
            "the refusal must name the floor this binary needs");
}

// The caller puts the device's name in front, so the message must read as a sentence in that position and must not
// claim a floor the binary does not have.
void test_the_message_is_actionable() {
    const std::string why = refused(5, 2);
    require(why.rfind("reports compute capability", 0) == 0,
            "the sentence must start where device_info concatenates the device name");
    require(why.find("Strata needs compute capability") != std::string::npos,
            "the refusal must state what the binary does need");
#if defined(STRATA_EXPERIMENTAL_SM60)
    require(why.find("experimental SM60") != std::string::npos,
            "an experimental build must say so rather than claim the release floor");
#else
    require(why.find("experimental SM60") == std::string::npos,
            "a default build must not describe itself as an experimental one");
#endif
}

}  // namespace

int main() {
    try {
        test_floor_matches_the_build();
        test_sm86_is_admitted();
        test_sm70_follows_the_flag();
        test_sm75_is_admitted();
        test_pascal_gp10x_follows_the_flag();
        test_unsupported_architectures_are_refused();
        test_the_floor_is_the_boundary();
        test_the_message_is_actionable();
        std::cout << "compute_capability_test: PASS (floor sm_" << kMinComputeCapability << ", sm_70 "
                  << (kMinComputeCapability <= 70 ? "admitted" : "refused") << ", sm_86 admitted)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "compute_capability_test: " << error.what() << '\n';
        return 1;
    }
}
