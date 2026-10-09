// src/platform/mps_ceiling.cpp - host-only MPS declaration and read-only verification.
#include "strata/platform/mps_ceiling.hpp"
#include "mps_ceiling_internal.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace strata::platform {
namespace {

constexpr const char* kLimitVariable = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT";
constexpr const char* kPipeVariable = "CUDA_MPS_PIPE_DIRECTORY";
constexpr const char* kVisibleVariable = "CUDA_VISIBLE_DEVICES";
constexpr uint64_t kMiB = 1024ull * 1024;

bool whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && whitespace(text.front())) text.remove_prefix(1);
    while (!text.empty() && whitespace(text.back())) text.remove_suffix(1);
    return text;
}

bool decimal(std::string_view text, uint64_t& number) {
    if (text.empty()) return false;
    uint64_t parsed = 0;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
        const unsigned digit = static_cast<unsigned>(c - '0');
        if (parsed > (UINT64_MAX - digit) / 10) return false;
        parsed = parsed * 10 + digit;
    }
    number = parsed;
    return true;
}

char lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c; }

bool parse_number(std::string_view value, uint64_t& bytes) {
    value = trim(value);
    size_t at = 0;
    while (at < value.size() && value[at] >= '0' && value[at] <= '9') ++at;
    uint64_t number = 0;
    if (!decimal(value.substr(0, at), number)) return false;
    auto suffix = value.substr(at);
    uint64_t scale = 1;
    if (!suffix.empty() && lower(suffix.front()) != 'b') {
        switch (lower(suffix.front())) {
            case 'k': scale = 1024ull; break;
            case 'm': scale = kMiB; break;
            case 'g': scale = kMiB * 1024; break;
            case 't': scale = kMiB * 1024 * 1024; break;
            default: return false;
        }
        suffix.remove_prefix(1);
        if (!suffix.empty() && lower(suffix.front()) == 'i') suffix.remove_prefix(1);
    }
    if (!suffix.empty() && lower(suffix.front()) == 'b') suffix.remove_prefix(1);
    if (!suffix.empty() || number > UINT64_MAX / scale) return false;
    bytes = number * scale;
    return true;
}

bool gpu_uuid(std::string_view uuid) {
    if (uuid.size() != 40 || uuid.substr(0, 4) != "GPU-") return false;
    for (size_t i = 4; i < uuid.size(); ++i) {
        if (i == 12 || i == 17 || i == 22 || i == 27) {
            if (uuid[i] != '-') return false;
        } else if (!((uuid[i] >= '0' && uuid[i] <= '9') ||
                     (lower(uuid[i]) >= 'a' && lower(uuid[i]) <= 'f'))) return false;
    }
    return true;
}

bool same_uuid(std::string_view a, std::string_view b) {
    if (!gpu_uuid(a) || !gpu_uuid(b)) return false;
    for (size_t i = 4; i < a.size(); ++i) if (lower(a[i]) != lower(b[i])) return false;
    return true;
}

bool valid_declaration(const MpsCeiling& ceiling, std::string& err) {
#if !defined(__linux__)
    (void)ceiling;
    err = "an MPS ceiling cannot be enforced on this platform (Linux required)";
    return false;
#else
    if (ceiling.devices.empty() || trim(ceiling.pipe_directory).empty() ||
        ceiling.pipe_directory.find('\0') != std::string::npos) {
        err = "MPS requires at least one device ceiling, a pipe directory and complete GPU UUIDs";
        return false;
    }
    for (size_t i = 0; i < ceiling.devices.size(); ++i) {
        const MpsDeviceCeiling& d = ceiling.devices[i];
        if (d.device != (int) i || d.cap_bytes == 0 || !gpu_uuid(d.gpu_uuid)) {
            err = "MPS requires one positive ceiling and a complete GPU UUID per visible device, "
                  "ordered by CUDA_VISIBLE_DEVICES ordinal";
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (same_uuid(d.gpu_uuid, ceiling.devices[j].gpu_uuid)) {
                err = "MPS requires distinct GPU UUIDs: CUDA_VISIBLE_DEVICES names one device twice";
                return false;
            }
        }
    }
    return true;
#endif
}

bool physical_envelope(uint64_t cap_bytes, uint64_t outside, uint64_t free_bytes,
                       std::string& err) {
    if (outside == 0) {
        err = "MPS requires a positive explicit outside-client allowance";
        return false;
    }
    if (cap_bytes > free_bytes || outside > free_bytes - cap_bytes) {
        err = "MPS full client ceiling plus outside-client allowance exceeds pre-context physical free memory";
        return false;
    }
    return true;
}

bool physical_row(std::string_view output, const std::string& expected_uuid,
                  uint64_t& bytes, std::string& err) {
    output = trim(output);
    const size_t comma = output.find(',');
    uint64_t mib = 0;
    if (comma == std::string_view::npos || output.find('\n') != std::string_view::npos ||
        output.find('\r') != std::string_view::npos ||
        !same_uuid(trim(output.substr(0, comma)), expected_uuid) ||
        !decimal(trim(output.substr(comma + 1)), mib) || mib > UINT64_MAX / kMiB) {
        err = "nvidia-smi did not return exactly one matching GPU UUID and whole-MiB physical-free row";
        return false;
    }
    bytes = mib * kMiB;
    return true;
}

template <size_t N>
bool pid_list(std::string_view output, std::array<int, N>& pids, size_t& count,
              std::string& err) {
    count = 0;
    output = trim(output);
    while (!output.empty()) {
        const size_t end = output.find('\n');
        const auto line = trim(output.substr(0, end));
        uint64_t parsed = 0;
        if (!decimal(line, parsed) || parsed == 0 || parsed > INT_MAX || count == N) {
            err = "MPS returned a malformed or oversized PID list";
            return false;
        }
        pids[count++] = static_cast<int>(parsed);
        if (end == std::string_view::npos) break;
        output.remove_prefix(end + 1);
    }
    std::sort(pids.begin(), pids.begin() + count);
    for (size_t i = 1; i < count; ++i) {
        if (pids[i - 1] == pids[i]) {
            err = "MPS returned a duplicate PID";
            return false;
        }
    }
    return true;
}

bool query(mps_detail::QueryRunner runner, void* context, mps_detail::Query kind,
           const MpsCeiling& ceiling, int pid, const std::string& uuid,
           mps_detail::Deadline deadline, std::string& output, std::string& err) {
    output.clear();
    if (std::chrono::steady_clock::now() >= deadline) {
        err = "MPS verification exceeded its total query deadline";
        return false;
    }
    if (!runner(context, kind, ceiling, pid, uuid, deadline, output, err)) return false;
    if (std::chrono::steady_clock::now() >= deadline || output.size() > mps_detail::kMaxQueryOutput) {
        err = "MPS verification exceeded its query deadline or output bound";
        return false;
    }
    return true;
}

bool limit_reply(const std::string& output, uint64_t expected, uint64_t& measured,
                 std::string& err) {
    if (!parse_number(output, measured) || measured == 0) {
        err = "MPS daemon returned an unreadable or zero device memory limit";
        return false;
    }
    if (measured != expected) {
        err = "MPS daemon device memory limit differs from the declared ceiling";
        return false;
    }
    return true;
}

bool production_query(void*, mps_detail::Query kind, const MpsCeiling& ceiling,
                      int pid, const std::string& uuid,
                      mps_detail::Deadline deadline, std::string& output,
                      std::string& err) {
#if defined(__linux__)
    return mps_detail::run_query(kind, ceiling, pid, uuid, deadline, output, err);
#else
    (void)kind; (void)ceiling; (void)pid; (void)uuid; (void)deadline; (void)output;
    err = "MPS queries require Linux";
    return false;
#endif
}

}  // namespace

bool mps_parse_limit(const std::string& text, std::vector<uint64_t>& caps, std::string& err) {
    caps.clear();
    err.clear();
    const auto value = trim(text);
    std::vector<uint64_t> parsed_caps;
    size_t at = 0;
    int expected_ordinal = 0;
    for (;;) {
        const size_t comma = value.find(',', at);
        const auto entry = trim(value.substr(at, comma == std::string_view::npos
                                                 ? std::string_view::npos : comma - at));
        const size_t equals = entry.find('=');
        const auto ordinal_text = equals == std::string_view::npos
                                      ? std::string_view() : trim(entry.substr(0, equals));
        uint64_t parsed = 0;
        // "00" is not the ordinal 0: the daemon's list form names each device once, plainly.
        if (equals == std::string_view::npos || ordinal_text.empty() ||
            (ordinal_text.front() == '0' && ordinal_text.size() != 1) ||
            !decimal(ordinal_text, parsed) || parsed != (uint64_t) expected_ordinal ||
            !parse_number(entry.substr(equals + 1), parsed) || parsed == 0) {
            err = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT requires the full contiguous ordered list "
                  "0=<limit>[,1=<limit>...] of positive byte counts (got \"" + text + "\")";
            return false;
        }
        parsed_caps.push_back(parsed);
        ++expected_ordinal;
        if (comma == std::string_view::npos) break;
        at = comma + 1;
    }
    caps = std::move(parsed_caps);
    return true;
}

bool mps_ceiling_from_environment(MpsCeiling& out, std::string& err) {
    out = MpsCeiling{};
    err.clear();
    const char* limit = std::getenv(kLimitVariable);
    const char* pipe = std::getenv(kPipeVariable);
    if (limit == nullptr && pipe == nullptr) return true;
    if (limit == nullptr || pipe == nullptr || trim(limit).empty() || trim(pipe).empty()) {
        err = "MPS ceiling declared incompletely: limit and pipe must both be nonempty";
        return false;
    }
    // CUDA_VISIBLE_DEVICES is the ordered selection: one complete GPU UUID, or a list with
    // exactly one complete UUID per selected device. The declaration needs one cap each.
    std::vector<std::string> uuids;
    {
        const char* visible = std::getenv(kVisibleVariable);
        std::string_view rest = visible == nullptr ? std::string_view() : visible;
        for (;;) {
            const size_t comma = rest.find(',');
            const auto entry = rest.substr(0, comma == std::string_view::npos
                                                 ? std::string_view::npos : comma);
            if (!gpu_uuid(entry)) {
                err = "MPS ceiling requires CUDA_VISIBLE_DEVICES to be one complete GPU UUID, or an "
                      "ordered list of complete GPU UUIDs with exactly one cap each";
                return false;
            }
            for (const std::string& prior : uuids) {
                if (same_uuid(prior, entry)) {
                    err = "MPS ceiling requires distinct GPU UUIDs: CUDA_VISIBLE_DEVICES names one device twice";
                    return false;
                }
            }
            uuids.emplace_back(entry);
            if (comma == std::string_view::npos) break;
            rest.remove_prefix(comma + 1);
        }
    }
    std::vector<uint64_t> caps;
    if (!mps_parse_limit(limit, caps, err)) return false;
    if (caps.size() != uuids.size()) {
        err = "MPS ceiling requires exactly one cap per selected device: CUDA_VISIBLE_DEVICES names " +
              std::to_string(uuids.size()) + ", the limit list names " + std::to_string(caps.size());
        return false;
    }
    MpsCeiling parsed;
    parsed.declared = true;
    parsed.pipe_directory = pipe;
    parsed.limit_from_environment = true;
    parsed.devices.resize(uuids.size());
    for (size_t i = 0; i < uuids.size(); ++i)
        parsed.devices[i] = MpsDeviceCeiling{static_cast<int>(i), caps[i], uuids[i]};
    if (!valid_declaration(parsed, err)) return false;
    out = std::move(parsed);
    return true;
}

bool mps_ceiling_pipe_identity(const MpsCeiling& ceiling, std::string& err) {
    if (!ceiling.declared) return true;
#if !defined(__linux__)
    err = "an MPS ceiling cannot be enforced on this platform: " + ceiling.pipe_directory;
    return false;
#else
    // lstat("symlink/") follows the link on Linux. Remove trailing slashes for the
    // identity check without changing the pipe path sent to the daemon.
    std::string untrailed;
    const char* path = ceiling.pipe_directory.c_str();
    if (ceiling.pipe_directory.size() > 1 && ceiling.pipe_directory.back() == '/') {
        untrailed = ceiling.pipe_directory;
        while (untrailed.size() > 1 && untrailed.back() == '/') untrailed.pop_back();
        path = untrailed.c_str();
    }
    struct stat info;
    if (ceiling.pipe_directory.find('\0') != std::string::npos ||
        lstat(path, &info) != 0) {
        err = "MPS control pipe directory is not reachable: " + ceiling.pipe_directory;
        return false;
    }
    if (S_ISLNK(info.st_mode) || !S_ISDIR(info.st_mode)) {
        err = "MPS control pipe directory is not a directory: " + ceiling.pipe_directory;
        return false;
    }
    const uid_t uid = geteuid();
    if (info.st_uid != uid && info.st_uid != 0) {
        err = "MPS control pipe directory is not owned by this user or root: " + ceiling.pipe_directory;
        return false;
    }
    return true;
#endif
}

bool mps_ceiling_holds(uint64_t reported_free_bytes, uint64_t declared_cap_bytes,
                       std::string& err) {
    if (declared_cap_bytes == 0) {
        err = "MPS ceiling is zero: nothing could be allocated";
        return false;
    }
    if (reported_free_bytes > declared_cap_bytes) {
        err = "cudaMemGetInfo residual free exceeds the declared MPS ceiling";
        return false;
    }
    return true;
}

uint64_t mps_device_ceiling(uint64_t total_bytes, uint64_t declared_cap_bytes, bool declared) {
    if (!declared) return total_bytes;
    return declared_cap_bytes < total_bytes ? declared_cap_bytes : total_bytes;
}

namespace mps_detail {

bool verify_before(const MpsCeiling& ceiling, uint64_t outside, MpsVerification& receipt,
                   std::string& err, QueryRunner runner, void* context) {
    receipt = MpsVerification{};
    err.clear();
    if (!ceiling.declared) return true;
    if (!valid_declaration(ceiling, err) || outside == 0) {
        if (err.empty()) err = "MPS requires a positive explicit outside-client allowance";
        return false;
    }
    if (!mps_ceiling_pipe_identity(ceiling, err)) return false;
    // One finite deadline is shared by every query of this phase: the whole readback stays
    // bounded no matter how many devices the declaration names.
    const Deadline deadline = std::chrono::steady_clock::now() + kQueryBudget;
    std::string output;
    receipt.devices.resize(ceiling.devices.size());
    for (size_t i = 0; i < ceiling.devices.size(); ++i) {
        const MpsDeviceCeiling& d = ceiling.devices[i];
        MpsDeviceVerification& row = receipt.devices[i];
        row.device = d.device;
        row.gpu_uuid = d.gpu_uuid;
        row.cap_bytes = d.cap_bytes;
        row.outside_client_allowance_bytes = outside;
        // The daemon's device operand IS the full GPU UUID: the control commands accept a
        // UUID string or an ordinal without translating between them, and the MPS server
        // remaps device visibility, so no ordinal of any enumeration is ever assumed.
        if (!query(runner, context, Query::default_limit, ceiling, 0, d.gpu_uuid,
                   deadline, output, err) ||
            !limit_reply(output, d.cap_bytes, row.default_cap_bytes, err)) return false;
        if (!query(runner, context, Query::physical_free, ceiling, 0, d.gpu_uuid,
                   deadline, output, err) ||
            !physical_row(output, d.gpu_uuid, row.physical_free_bytes, err)) return false;
        row.physical_free_measured = true;
        if (i == 0) {
            // The retained single-device fields mirror the front row as it verifies, so a
            // later device's refusal still leaves the measured first-device sample intact.
            receipt.physical_free_bytes = row.physical_free_bytes;
            receipt.physical_free_measured = row.physical_free_measured;
            receipt.default_cap_bytes = row.default_cap_bytes;
            receipt.outside_client_allowance_bytes = outside;
        }
        if (!physical_envelope(d.cap_bytes, outside, row.physical_free_bytes, err)) return false;
    }
    return true;
}

bool verify_after(const MpsCeiling& ceiling, const std::vector<std::string>& actual_uuids,
                  uint64_t primary_residual, MpsVerification& receipt,
                  std::string& err, QueryRunner runner, void* context) {
    err.clear();
    if (!ceiling.declared) return true;
    if (!valid_declaration(ceiling, err)) return false;
    // A refused attachment must not leave a receipt behind: attachment fields clear on entry
    // and fill only after every check holds.
    receipt.server_pid = 0;
    receipt.client_pid = 0;
    receipt.server_cap_bytes = 0;
    for (auto& row : receipt.devices) {
        row.server_cap_bytes = 0;
        row.residual_free_bytes = 0;
    }
    if (actual_uuids.size() != ceiling.devices.size()) {
        err = "post-context verification requires the actual CUDA UUID of every declared device";
        return false;
    }
    for (size_t i = 0; i < ceiling.devices.size(); ++i) {
        if (!gpu_uuid(actual_uuids[i])) {
            err = "the actual CUDA identity of device " + std::to_string(i) +
                  " is not a complete GPU UUID";
            return false;
        }
        if (!same_uuid(ceiling.devices[i].gpu_uuid, actual_uuids[i])) {
            err = "CUDA device " + std::to_string(i) +
                  " UUID differs from the protected MPS GPU identity";
            return false;
        }
    }
    if (receipt.devices.size() != ceiling.devices.size()) {
        err = "MPS post-context verification requires successful pre-context readback";
        return false;
    }
    for (size_t i = 0; i < ceiling.devices.size(); ++i) {
        const MpsDeviceVerification& row = receipt.devices[i];
        if (!row.physical_free_measured || row.default_cap_bytes != ceiling.devices[i].cap_bytes) {
            err = "MPS post-context verification requires successful pre-context readback";
            return false;
        }
        if (!physical_envelope(ceiling.devices[i].cap_bytes, receipt.outside_client_allowance_bytes,
                               row.physical_free_bytes, err)) return false;
    }
    if (!mps_ceiling_holds(primary_residual, ceiling.devices.front().cap_bytes, err)) return false;
#if defined(__linux__)
    const int own_pid = static_cast<int>(getpid());
#else
    const int own_pid = 0;  // valid_declaration already refuses non-Linux.
#endif
    const Deadline deadline = std::chrono::steady_clock::now() + kQueryBudget;
    std::string output;
    std::array<int, kMaxServers> servers;
    size_t server_count = 0;
    if (!query(runner, context, Query::servers, ceiling, 0, std::string(), deadline, output, err) ||
        !pid_list(output, servers, server_count, err)) return false;
    int matched_server = 0;
    std::array<int, kMaxClients> clients;
    for (size_t i = 0; i < server_count; ++i) {
        size_t client_count = 0;
        if (!query(runner, context, Query::clients, ceiling, servers[i], std::string(),
                   deadline, output, err) ||
            !pid_list(output, clients, client_count, err)) return false;
        for (size_t j = 0; j < client_count; ++j) {
            if (clients[j] != own_pid) continue;
            if (matched_server != 0) {
                err = "this process appears in more than one MPS server client list";
                return false;
            }
            matched_server = servers[i];
        }
    }
    if (matched_server == 0) {
        err = "this process is absent from every MPS server client list";
        return false;
    }
    for (size_t i = 0; i < ceiling.devices.size(); ++i) {
        MpsDeviceVerification& row = receipt.devices[i];
        // The matched server's own per-device readback is keyed by the device's full GPU
        // UUID, never by an ordinal assumed from the client's visible order.
        if (!query(runner, context, Query::server_limit, ceiling, matched_server, row.gpu_uuid,
                   deadline, output, err) ||
            !limit_reply(output, ceiling.devices[i].cap_bytes, row.server_cap_bytes, err)) return false;
        row.residual_free_bytes = i == 0 ? primary_residual : 0;
        row.gpu_uuid = actual_uuids[i];
    }
    receipt.server_pid = matched_server;
    receipt.client_pid = own_pid;
    // Retained single-device receipt fields mirror the front row.
    receipt.server_cap_bytes = receipt.devices.front().server_cap_bytes;
    receipt.physical_free_bytes = receipt.devices.front().physical_free_bytes;
    receipt.outside_client_allowance_bytes = receipt.devices.front().outside_client_allowance_bytes;
    receipt.physical_free_measured = receipt.devices.front().physical_free_measured;
    receipt.default_cap_bytes = receipt.devices.front().default_cap_bytes;
    return true;
}

#if defined(__linux__)
namespace {

// Own descriptors and the child on every return path. Only posix_spawn performs child
// setup, so no allocator, environment or C++ callbacks execute in a post-fork CUDA child.
struct QueryProcess {
    int input[2] = {-1, -1};
    int output[2] = {-1, -1};
    pid_t pid = 0;
    bool reaped = false;
    static void close_fd(int& fd) {
        if (fd >= 0) close(fd);
        fd = -1;
    }
    ~QueryProcess() {
        if (pid > 0 && !reaped) {
            kill(-pid, SIGKILL);
            int status = 0;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        }
        for (int& fd : input) close_fd(fd);
        for (int& fd : output) close_fd(fd);
    }
};

bool prepare_fds(int (&fds)[2]) {
    for (int& fd : fds) {
        if (fd > STDERR_FILENO) continue;
        const int duplicate = fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
        if (duplicate < 0) return false;
        close(fd);
        fd = duplicate;
    }
    return true;
}

bool transport_error(std::string& err, const char* what, int code) {
    err = std::string("MPS query ") + what + ": " + std::strerror(code);
    return false;
}

}  // namespace

bool run_query(Query kind, const MpsCeiling& ceiling, int server_pid,
               const std::string& device_uuid, Deadline deadline,
               std::string& output, std::string& err, const char* test_executable) {
    output.clear();
    std::string input;
    int ordinal = -1;
    if (kind == Query::default_limit || kind == Query::server_limit) {
        for (const auto& device : ceiling.devices)
            if (same_uuid(device.gpu_uuid, device_uuid)) ordinal = device.device;
        if (ordinal < 0) {
            err = "MPS control query names no declared device";
            return false;
        }
    }
    switch (kind) {
        case Query::default_limit:
            // The protected daemon has the same explicit UUID order as this client.
            // Driver 550 accepts only ordinal operands, not UUID strings.
            input = "get_default_device_pinned_mem_limit " + std::to_string(ordinal) + "\n";
            break;
        case Query::servers: input = "get_server_list\n"; break;
        case Query::clients: input = "get_client_list " + std::to_string(server_pid) + "\n"; break;
        case Query::server_limit:
            input = "get_device_pinned_mem_limit " + std::to_string(server_pid) + " " +
                    std::to_string(ordinal) + "\n";
            break;
        case Query::physical_free: break;
        default: err = "unsupported MPS read-only query"; return false;
    }
    const bool physical = kind == Query::physical_free;
    const char* executable = test_executable ? test_executable :
        (physical ? "/usr/bin/nvidia-smi" : "/usr/bin/nvidia-cuda-mps-control");
    // The physical sample selects the exact GPU by its full UUID.
    std::string id = physical ? "--id=" + device_uuid : std::string{};
    char* argv[] = {const_cast<char*>(executable), physical ? id.data() : nullptr,
                    const_cast<char*>("--query-gpu=uuid,memory.free"),
                    const_cast<char*>("--format=csv,noheader,nounits"), nullptr};
    // Pin the queried daemon to the supplied declaration, not a mutable ambient value.
    std::string pipe_env = std::string(kPipeVariable) + "=" + ceiling.pipe_directory;
    char locale[] = "LC_ALL=C";
    std::vector<char*> environment;
    for (char** value = environ; *value != nullptr; ++value) {
        if (std::strncmp(*value, "CUDA_MPS_PIPE_DIRECTORY=", sizeof("CUDA_MPS_PIPE_DIRECTORY=") - 1) == 0 ||
            std::strncmp(*value, "LC_ALL=", 7) == 0) continue;
        environment.push_back(*value);
    }
    environment.push_back(pipe_env.data());
    environment.push_back(locale);
    environment.push_back(nullptr);
    if (std::chrono::steady_clock::now() >= deadline) {
        err = "MPS query deadline expired before launch";
        return false;
    }
    QueryProcess process;
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, process.input) != 0 ||
        pipe2(process.output, O_CLOEXEC) != 0 || !prepare_fds(process.input) || !prepare_fds(process.output))
        return transport_error(err, "descriptor setup failed", errno);
    if (fcntl(process.input[0], F_SETFL, O_NONBLOCK) < 0 ||
        fcntl(process.output[0], F_SETFL, O_NONBLOCK) < 0)
        return transport_error(err, "nonblocking setup failed", errno);
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (code != 0) return transport_error(err, "file actions failed", code);
    auto add_dup = [&](int from, int to) {
        if (code == 0) code = posix_spawn_file_actions_adddup2(&actions, from, to);
    };
    add_dup(process.input[1], STDIN_FILENO);
    add_dup(process.output[1], STDOUT_FILENO);
    add_dup(process.output[1], STDERR_FILENO);
    for (int fd : {process.input[0], process.input[1], process.output[0], process.output[1]}) {
        if (code == 0) code = posix_spawn_file_actions_addclose(&actions, fd);
    }
    posix_spawnattr_t attributes;
    const int attr_code = posix_spawnattr_init(&attributes);
    if (attr_code != 0 || code != 0) {
        posix_spawn_file_actions_destroy(&actions);
        if (attr_code == 0) posix_spawnattr_destroy(&attributes);
        return transport_error(err, "spawn setup failed", code != 0 ? code : attr_code);
    }
    sigset_t empty, defaults;
    sigemptyset(&empty);
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    code = posix_spawnattr_setflags(&attributes,
        POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    if (code == 0) code = posix_spawnattr_setpgroup(&attributes, 0);
    if (code == 0) code = posix_spawnattr_setsigmask(&attributes, &empty);
    if (code == 0) code = posix_spawnattr_setsigdefault(&attributes, &defaults);
    pid_t spawned_pid = 0;
    if (code == 0) code = posix_spawn(&spawned_pid, executable, &actions, &attributes, argv, environment.data());
    posix_spawnattr_destroy(&attributes);
    posix_spawn_file_actions_destroy(&actions);
    if (code != 0) return transport_error(err, "launch failed", code);
    process.pid = spawned_pid;
    QueryProcess::close_fd(process.input[1]);
    QueryProcess::close_fd(process.output[1]);
    size_t sent = 0;
    bool eof = false;
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            err = "MPS query exceeded its total deadline";
            return false;
        }
        if (process.input[0] >= 0) {
            if (sent < input.size()) {
                const ssize_t n = send(process.input[0], input.data() + sent, input.size() - sent, MSG_NOSIGNAL);
                if (n > 0) sent += static_cast<size_t>(n);
                else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                    return transport_error(err, "input closed", errno);
            }
            if (sent == input.size()) {
                shutdown(process.input[0], SHUT_WR);
                QueryProcess::close_fd(process.input[0]);
            }
        }
        // A bounded number of bytes per loop keeps a continuously writing child from
        // bypassing the deadline. stderr is merged deliberately: diagnostics fail parsing.
        char buffer[4096];
        if (!eof) {
            const ssize_t n = read(process.output[0], buffer, sizeof(buffer));
            if (n > 0) {
                if (static_cast<size_t>(n) > kMaxQueryOutput - output.size()) {
                    err = "MPS query exceeded its output bound";
                    return false;
                }
                output.append(buffer, static_cast<size_t>(n));
                continue;
            }
            if (n == 0) {
                eof = true;
                QueryProcess::close_fd(process.output[0]);
            } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                return transport_error(err, "output read failed", errno);
        }
        if (eof) {
            int status = 0;
            const pid_t waited = waitpid(process.pid, &status, WNOHANG);
            if (waited == process.pid) {
                process.reaped = true;
                if (std::chrono::steady_clock::now() >= deadline || !WIFEXITED(status) ||
                    WEXITSTATUS(status) != 0 || sent != input.size()) {
                    err = "MPS query exited unsuccessfully or before accepting its input";
                    return false;
                }
                return true;
            }
            if (waited < 0 && errno != EINTR) {
                // An external SIGCHLD reaper must not make cleanup signal a reused PID.
                if (errno == ECHILD) process.reaped = true;
                return transport_error(err, "wait failed", errno);
            }
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            err = "MPS query exceeded its total deadline";
            return false;
        }
        pollfd descriptors[] = {{process.output[0], POLLIN, 0}, {process.input[0], POLLOUT, 0}};
        const int delay = remaining < 10 ? static_cast<int>(remaining) : 10;
        const int ready = poll(descriptors, 2, delay);
        if (ready < 0 && errno != EINTR) return transport_error(err, "poll failed", errno);
        for (const auto& descriptor : descriptors) {
            if (descriptor.revents & POLLNVAL) {
                err = "MPS query encountered an invalid descriptor";
                return false;
            }
        }
    }
}
#endif

}  // namespace mps_detail

bool mps_verify_before_cuda(const MpsCeiling& ceiling, uint64_t outside,
                            MpsVerification& receipt, std::string& err) {
    return mps_detail::verify_before(ceiling, outside, receipt, err, production_query, nullptr);
}

bool mps_verify_after_cuda(const MpsCeiling& ceiling, const std::vector<std::string>& actual_uuids,
                           uint64_t primary_residual_free_bytes, MpsVerification& receipt,
                           std::string& err) {
    return mps_detail::verify_after(ceiling, actual_uuids, primary_residual_free_bytes, receipt,
                                    err, production_query, nullptr);
}

}  // namespace strata::platform
