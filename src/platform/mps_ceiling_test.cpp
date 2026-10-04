// src/platform/mps_ceiling_test.cpp - the declared-MPS-ceiling contract (CPU only, no GPU, no model).
//
// The parse and the identity check are pure, and the runtime check is arithmetic over
// numbers the engine measured, so the whole contract is testable without a device. The
// cases below are the ways this can be wrong: an unreadable declaration, a missing half of
// the pair, a value the driver would reject, a pipe this process cannot reach, and - the
// one the qualification actually measured - a client that carries the variable with nothing
// enforcing it, which reports the whole device and must refuse.
#include "strata/platform/mps_ceiling.hpp"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++failures;
    }
}

void set_value(const char* name, const char* value) {
    // POSIX setenv/unsetenv are not available on MSVC, so the Windows build uses _putenv_s
    // (an empty value removes the variable) rather than not compiling at all.
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) unsetenv(name);
    else setenv(name, value, 1);
#endif
}

void test_parse() {
    uint64_t bytes = 0;
    std::string err;
    check(strata::platform::mps_parse_limit("0=1024M", 0, bytes, err) && bytes == 1024ull * 1024 * 1024,
          "0=1024M parses to 1 GiB");
    check(strata::platform::mps_parse_limit("0=1G", 0, bytes, err) && bytes == 1024ull * 1024 * 1024,
          "0=1G parses to 1 GiB");
    check(strata::platform::mps_parse_limit("0=1073741824", 0, bytes, err) && bytes == 1024ull * 1024 * 1024,
          "a bare byte count parses");
    check(strata::platform::mps_parse_limit("0=2gib", 0, bytes, err) && bytes == 2ull * 1024 * 1024 * 1024,
          "the binary spelling parses case-insensitively");
    check(strata::platform::mps_parse_limit("1=4G,0=1024M", 0, bytes, err) && bytes == 1024ull * 1024 * 1024,
          "the wanted ordinal is selected from a list");
    check(!strata::platform::mps_parse_limit("1=4G", 0, bytes, err), "a missing ordinal refuses");
    check(!strata::platform::mps_parse_limit("0=nonsense", 0, bytes, err), "an unreadable value refuses");
    check(!strata::platform::mps_parse_limit("0=", 0, bytes, err), "an empty value refuses");
    check(!strata::platform::mps_parse_limit("0=0", 0, bytes, err), "a zero ceiling refuses");
    check(!strata::platform::mps_parse_limit("x=1G", 0, bytes, err), "an unreadable ordinal refuses");
}

void test_declaration() {
    strata::platform::MpsCeiling ceiling;
    std::string err;
    set_value("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", nullptr);
    set_value("CUDA_MPS_PIPE_DIRECTORY", nullptr);
    check(strata::platform::mps_ceiling_from_environment(ceiling, err) && !ceiling.declared,
          "no variables means no declaration and no failure");
    check(strata::platform::mps_ceiling_pipe_identity(ceiling, err), "an absent declaration needs no pipe");
    check(strata::platform::mps_device_ceiling(4096, 1024, false) == 4096,
          "with no declaration the device ceiling is the driver total");

    set_value("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "0=1024M");
    set_value("CUDA_MPS_PIPE_DIRECTORY", "/tmp/strata-mps-test-pipe");
    check(strata::platform::mps_ceiling_from_environment(ceiling, err) && ceiling.declared &&
              ceiling.cap_bytes == 1024ull * 1024 * 1024 && ceiling.device == 0,
          "a complete pair declares the ceiling");
    check(!strata::platform::mps_ceiling_pipe_identity(ceiling, err),
          "a pipe directory that does not exist refuses");

    set_value("CUDA_MPS_PIPE_DIRECTORY", nullptr);
    check(!strata::platform::mps_ceiling_from_environment(ceiling, err),
          "the limit without the pipe refuses");

    set_value("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", nullptr);
    set_value("CUDA_MPS_PIPE_DIRECTORY", "/tmp/strata-mps-test-pipe");
    check(!strata::platform::mps_ceiling_from_environment(ceiling, err),
          "the pipe without the limit refuses");

    set_value("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "0=1024M");
    const uint64_t total = 25304236032ull;   // the measured 24 GiB card
    check(strata::platform::mps_ceiling_from_environment(ceiling, err) &&
              strata::platform::mps_device_ceiling(total, ceiling.cap_bytes, ceiling.declared) ==
                  ceiling.cap_bytes,
          "a declared ceiling smaller than the driver total becomes the device ceiling");
    check(strata::platform::mps_device_ceiling(512ull * 1024 * 1024, ceiling.cap_bytes,
                                               ceiling.declared) == 512ull * 1024 * 1024,
          "a declared ceiling larger than the driver total leaves the total in place");
}

#if defined(_WIN32)
void test_pipe_identity() {
    // The identity check is Linux-only along with MPS itself.
    strata::platform::MpsCeiling ceiling;
    ceiling.declared = true;
    ceiling.pipe_directory = "C:\\strata-mps";
    std::string err;
    check(!strata::platform::mps_ceiling_pipe_identity(ceiling, err),
          "a declaration on Windows refuses");
}
#else
void test_pipe_identity() {
    const std::string root = "/tmp/strata-mps-ceiling-test-" + std::to_string(getpid());
    const std::string pipe = root + "/pipe";
    std::string command = "rm -rf " + root + " && mkdir -p " + pipe;
    if (std::system(command.c_str()) != 0) {
        std::printf("  FAIL: could not create the fixture\n");
        ++failures;
        return;
    }
    strata::platform::MpsCeiling ceiling;
    ceiling.declared = true;
    ceiling.cap_bytes = 1024ull * 1024 * 1024;
    ceiling.pipe_directory = pipe;
    std::string err;
    check(strata::platform::mps_ceiling_pipe_identity(ceiling, err),
          "a directory owned by this user satisfies the identity check");

    const std::string link = root + "/link";
    if (std::system(("ln -s " + pipe + " " + link).c_str()) == 0) {
        ceiling.pipe_directory = link;
        check(!strata::platform::mps_ceiling_pipe_identity(ceiling, err), "a symlink refuses");
    }

    const std::string file = root + "/file";
    std::FILE* f = std::fopen(file.c_str(), "wb");
    if (f != nullptr) std::fclose(f);
    ceiling.pipe_directory = file;
    check(!strata::platform::mps_ceiling_pipe_identity(ceiling, err), "a regular file refuses");

    std::system(("rm -rf " + root).c_str());
}
#endif

void test_runtime_holds() {
    std::string err;
    // The qualified values: a 1 GiB declared ceiling and the client's own residual budget.
    check(strata::platform::mps_ceiling_holds(812070608ull, 1024ull * 1024 * 1024, err),
          "a residual budget below the ceiling holds");
    check(strata::platform::mps_ceiling_holds(1024ull * 1024 * 1024, 1024ull * 1024 * 1024, err),
          "a residual equal to the ceiling holds");
    // The measured failure mode: the variable is set, nothing enforces it, and the client
    // sees the whole 24 GiB device instead of its declared 1 GiB.
    check(!strata::platform::mps_ceiling_holds(25304236032ull, 1024ull * 1024 * 1024, err),
          "the whole device above the ceiling refuses");
    check(!strata::platform::mps_ceiling_holds(0, 0, err), "a zero ceiling refuses");
}

}  // namespace

int main() {
    test_parse();
    test_declaration();
    test_pipe_identity();
    test_runtime_holds();
    std::printf("mps_ceiling_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
