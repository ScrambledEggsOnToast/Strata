// #252: which failed capture ends serving, and the one line a dropped capture writes.  CPU only, no model or GPU.
#include "strata/program/checkpoint_preservation.hpp"

#include <cstdio>
#include <string>

namespace preservation = strata::program;
namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)
}

int main() {
    // The class that ends serving: a device sync/launch failed, so the stream is no longer trustworthy.
    CHECK(preservation::preservation_ends_serving(preservation::PreservationFailure::device));
    // The class that must not: a copy, allocation, validation or publication failure only loses optional reuse.
    CHECK(!preservation::preservation_ends_serving(preservation::PreservationFailure::durability));

    // A drop is one bounded line on stderr, never an ERR/FATAL protocol line the frontend would treat as a request
    // terminal (server.py's ERR/FATAL parsing is what turns one into a failed request).
    std::FILE* f = std::tmpfile();
    CHECK(f != nullptr);
    if (f != nullptr) {
        const std::string long_reason(4096, 'x');
        preservation::preservation_report_drop(f, "the outgoing conversation snapshot", long_reason);
        std::rewind(f);
        std::string line(8192, '\0');
        const size_t n = std::fread(line.data(), 1, line.size(), f);
        line.resize(n);
        std::fclose(f);
        CHECK(!line.empty() && line.back() == '\n' && line.find('\n') == line.size() - 1);
        CHECK(line.rfind("strata serve: the outgoing conversation snapshot dropped (", 0) == 0);
        CHECK(line.find("ERR") == std::string::npos && line.find("FATAL") == std::string::npos);
        CHECK(line.size() < 256);   // the reason is bounded, so a driver/allocator message cannot flood the log
    }

    if (failures) std::fprintf(stderr, "checkpoint_preservation_test: %d failure(s)\n", failures);
    else std::printf("checkpoint_preservation_test: ok\n");
    return failures == 0 ? 0 : 1;
}
