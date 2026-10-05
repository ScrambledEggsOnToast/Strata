// Private host-only seams: production uses fixed commands; tests supply readback bytes
// or execute this same finite transport against a CPU-only fixture executable.
#pragma once

#include "strata/platform/mps_ceiling.hpp"

#include <chrono>
#include <cstddef>
#include <string>

namespace strata::platform::mps_detail {

using Deadline = std::chrono::steady_clock::time_point;
constexpr std::size_t kMaxQueryOutput = 64 * 1024;
constexpr std::size_t kMaxServers = 32;
constexpr std::size_t kMaxClients = 4096;
constexpr auto kQueryBudget = std::chrono::seconds(3);

enum class Query { default_limit, physical_free, servers, clients, server_limit };
using QueryRunner = bool (*)(void*, Query, const MpsCeiling&, int, Deadline,
                            std::string&, std::string&);

bool verify_before(const MpsCeiling&, uint64_t, MpsVerification&, std::string&,
                   QueryRunner, void*);
bool verify_after(const MpsCeiling&, const std::string&, uint64_t, MpsVerification&,
                  std::string&, QueryRunner, void*);

#if defined(__linux__)
// The override is private to CPU tests. Public entry points always use nullptr and
// cannot execute user-supplied programs. The query enum fixes argv and stdin content.
bool run_query(Query, const MpsCeiling&, int server_pid, Deadline,
               std::string& output, std::string& err, const char* test_executable = nullptr);
#endif

}  // namespace strata::platform::mps_detail
