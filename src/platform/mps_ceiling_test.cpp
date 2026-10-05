// CPU-only declaration, readback and finite-transport behavior. No daemon, CUDA or model.
#include "strata/platform/mps_ceiling.hpp"
#include "mps_ceiling_internal.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using namespace strata::platform;
namespace detail = strata::platform::mps_detail;
constexpr uint64_t kMiB = 1024ull * 1024;
constexpr uint64_t kGiB = 1024ull * kMiB;
constexpr const char* kUuid = "GPU-12345678-1234-abcd-9876-123456789abc";
constexpr const char* kOtherUuid = "GPU-12345678-1234-abcd-9876-123456789abd";
int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what);
        ++failures;
    }
}

void set_value(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) unsetenv(name);
    else setenv(name, value, 1);
#endif
}

struct Environment {
    const char* name;
    bool present;
    std::string value;
    explicit Environment(const char* key) : name(key), present(std::getenv(key) != nullptr),
        value(present ? std::getenv(key) : "") {}
    ~Environment() { set_value(name, present ? value.c_str() : nullptr); }
};

void test_parse() {
    uint64_t bytes = 0;
    std::string err;
    for (const char* value : {"0=1024M", "0=1G", "0=1073741824", " 0 = 1GiB "})
        check(mps_parse_limit(value, 0, bytes, err) && bytes == kGiB, "single ordinal-0 binary limit parses");
    check(mps_parse_limit("0=18446744073709551615", 0, bytes, err) && bytes == UINT64_MAX,
          "the maximum byte count parses without truncation");
    for (const char* value : {"1=4G,0=1024M", "0=1G,1=4G", "0=1G,garbage", "0=1G,",
                              ",0=1G", "0=1G,0=1G", "1=4G", "0=nonsense", "0=", "0=0",
                              "x=1G", "=1G", "1G", "0=1G\nerror 12", "0=1i", "0=1iiB",
                              "0=18446744073709551616", "0=18014398509481984K",
                              "4294967296=1G", "2147483648=1G", "-0=1G", "+0=1G", "00=1G",
                              "0=-1G", "0=+1G", "0=1.5G", "0=1 G", "0=1=G"}) {
        bytes = 123;
        check(!mps_parse_limit(value, 0, bytes, err) && bytes == 0 && !err.empty(),
              "malformed, ambiguous, overflowing or unsupported full declarations refuse");
    }
    check(!mps_parse_limit("0=1G", 1, bytes, err), "unsupported wanted ordinal refuses");
}

void test_declaration() {
    Environment limit("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"), pipe("CUDA_MPS_PIPE_DIRECTORY"),
                uuid("CUDA_VISIBLE_DEVICES");
    MpsCeiling ceiling;
    MpsVerification receipt;
    std::string err;
    set_value(limit.name, nullptr);
    set_value(pipe.name, nullptr);
    set_value(uuid.name, "0,1");
    check(mps_ceiling_from_environment(ceiling, err) && !ceiling.declared,
          "no declaration leaves ordinary CUDA visibility alone");
    check(mps_ceiling_pipe_identity(ceiling, err), "an absent declaration needs no pipe");
    check(mps_verify_before_cuda(ceiling, 0, receipt, err) &&
          mps_verify_after_cuda(ceiling, "", UINT64_MAX, receipt, err), "no declaration avoids all queries");
    check(mps_device_ceiling(4096, 1024, false) == 4096, "no declaration preserves driver total");
    set_value(limit.name, "0=1024M");
    check(!mps_ceiling_from_environment(ceiling, err), "a missing pipe refuses");
    set_value(pipe.name, "/tmp/strata-mps-test-pipe");
    set_value(limit.name, nullptr);
    check(!mps_ceiling_from_environment(ceiling, err), "a missing limit refuses");
    set_value(limit.name, "0=1024M");
    for (const char* value : {"", "0", "GPU-12345678", "GPU-12345678-1234-abcd-9876-123456789abz",
                              " GPU-12345678-1234-abcd-9876-123456789abc", "MIG-12345678",
                              "GPU-12345678-1234-abcd-9876-123456789abc,0"}) {
        set_value(uuid.name, value);
        check(!mps_ceiling_from_environment(ceiling, err) && !ceiling.declared,
              "missing, partial, malformed and multi-device GPU identities refuse");
    }
    set_value(uuid.name, nullptr);
    check(!mps_ceiling_from_environment(ceiling, err), "unset GPU identity refuses");
    set_value(uuid.name, kUuid);
#if defined(__linux__)
    check(mps_ceiling_from_environment(ceiling, err) && ceiling.declared && ceiling.device == 0 &&
          ceiling.cap_bytes == kGiB && ceiling.gpu_uuid == kUuid && ceiling.limit_from_environment,
          "complete single-device declaration preserves its identity");
#else
    check(!mps_ceiling_from_environment(ceiling, err), "declared MPS refuses on non-Linux");
#endif
    check(mps_device_ceiling(24 * kGiB, kGiB, true) == kGiB &&
          mps_device_ceiling(kGiB / 2, kGiB, true) == kGiB / 2, "declared ceiling is bounded by driver total");
    set_value(limit.name, "0=1G,broken");
    check(!mps_ceiling_from_environment(ceiling, err) && !ceiling.declared,
          "a trailing malformed entry is not ignored");
    set_value(limit.name, " ");
    set_value(pipe.name, " ");
    check(!mps_ceiling_from_environment(ceiling, err), "present-but-empty declarations are not absence");
}

void test_runtime_holds() {
    std::string err;
    check(mps_ceiling_holds(812070608ull, kGiB, err), "residual below cap passes the necessary check");
    check(mps_ceiling_holds(kGiB, kGiB, err), "residual equality passes");
    check(!mps_ceiling_holds(24 * kGiB, kGiB, err), "whole-device residual above cap refuses");
    check(!mps_ceiling_holds(0, 0, err), "zero cap refuses");
}

#if defined(__linux__)
struct FixtureDirectory {
    char path[64] = "/tmp/strata-mps-ceiling-XXXXXX";
    bool valid = mkdtemp(path) != nullptr;
    ~FixtureDirectory() { if (valid) rmdir(path); }
};

MpsCeiling declaration(const char* pipe) {
    MpsCeiling ceiling;
    ceiling.declared = true;
    ceiling.device = 0;
    ceiling.cap_bytes = kGiB;
    ceiling.gpu_uuid = kUuid;
    ceiling.pipe_directory = pipe;
    ceiling.limit_from_environment = true;
    return ceiling;
}

void test_pipe_identity() {
    FixtureDirectory root;
    check(root.valid, "pipe fixture directory created");
    if (!root.valid) return;
    auto ceiling = declaration(root.path);
    std::string err;
    check(mps_ceiling_pipe_identity(ceiling, err), "owned directory passes identity check");
    const std::string link = std::string(root.path) + "/link";
    check(symlink(root.path, link.c_str()) == 0, "symlink fixture created");
    ceiling.pipe_directory = link;
    check(!mps_ceiling_pipe_identity(ceiling, err), "symlink refuses");
    ceiling.pipe_directory += "/";
    check(!mps_ceiling_pipe_identity(ceiling, err), "trailing slash cannot hide a symlink");
    unlink(link.c_str());
    const std::string file = std::string(root.path) + "/file";
    const int fd = open(file.c_str(), O_CREAT | O_WRONLY | O_EXCL, 0600);
    check(fd >= 0, "regular-file fixture created");
    if (fd >= 0) close(fd);
    ceiling.pipe_directory = file;
    check(!mps_ceiling_pipe_identity(ceiling, err), "regular file refuses");
    unlink(file.c_str());
    ceiling.pipe_directory = std::string(root.path) + "/missing";
    check(!mps_ceiling_pipe_identity(ceiling, err), "missing pipe refuses");
}

struct Replies {
    std::string default_limit = "1024M\n";
    std::string physical = std::string(kUuid) + ", 1088\n";
    std::string servers = "101\n";
    std::string first_clients = std::to_string(getpid()) + "\n";
    std::string other_clients = "\n";
    std::string server_limit = "1G\n";
    bool unavailable = false;
    bool deadlines_equal = true;
    int calls = 0;
    int queried_server = 0;
    detail::Deadline deadline{};
    static bool run(void* opaque, detail::Query query, const MpsCeiling&, int server,
                    detail::Deadline until, std::string& output, std::string& err) {
        auto& self = *static_cast<Replies*>(opaque);
        if (self.calls++ == 0) self.deadline = until;
        else self.deadlines_equal = self.deadlines_equal && self.deadline == until;
        if (self.unavailable) { err = "fixture: unavailable daemon"; return false; }
        switch (query) {
            case detail::Query::default_limit: output = self.default_limit; break;
            case detail::Query::physical_free: output = self.physical; break;
            case detail::Query::servers: output = self.servers; break;
            case detail::Query::clients: output = server == 101 ? self.first_clients : self.other_clients; break;
            case detail::Query::server_limit: output = self.server_limit; self.queried_server = server; break;
        }
        return true;
    }
};

void test_before_readback() {
    FixtureDirectory root;
    check(root.valid, "pre-context fixture created");
    if (!root.valid) return;
    const auto ceiling = declaration(root.path);
    std::string err;
    MpsVerification receipt;
    Replies replies;
    const auto before = [&](uint64_t outside = 64 * kMiB) {
        replies.calls = 0;
        return detail::verify_before(ceiling, outside, receipt, err, Replies::run, &replies);
    };
    check(before() && receipt.physical_free_measured && receipt.physical_free_bytes == 1088 * kMiB &&
          receipt.outside_client_allowance_bytes == 64 * kMiB && receipt.default_cap_bytes == kGiB &&
          replies.calls == 2 && replies.deadlines_equal, "physical-envelope equality admits with a shared query deadline");
    check(!before(64 * kMiB + 1) && receipt.physical_free_measured,
          "one byte beyond physical envelope refuses but preserves measured sample");
    check(!before(UINT64_MAX), "outside plus cap cannot wrap into an admission");
    check(!before(0) && replies.calls == 0, "zero outside allowance refuses before querying");
    replies.physical = std::string(kUuid) + ", 1023\n";
    check(!before(), "cap exceeding physical free refuses");
    replies = Replies{};
    replies.unavailable = true;
    check(!before() && !receipt.physical_free_measured, "missing daemon refuses without verified physical data");
    replies.unavailable = false;
    for (const char* output : {"", "2G\n", "0\n", "unknown device 0\n", "1G\n1G\n", "-1G\n",
                               "18446744073709551616", "18014398509481984K", "1G trailing", "1.0G"}) {
        replies.default_limit = output;
        check(!before() && !receipt.physical_free_measured && replies.calls == 1,
              "malformed, mismatched, missing and overflowing default readback refuses");
    }
    replies = Replies{};
    const std::vector<std::string> physical_bad = {
        "", std::string(kOtherUuid) + ", 1088\n", "GPU-1234, 1088\n",
        std::string(kUuid) + ", N/A\n", std::string(kUuid) + ", -1\n",
        std::string(kUuid) + ", +1088\n", std::string(kUuid) + ", 1088.5\n",
        std::string(kUuid) + ", 1088 MiB\n", std::string(kUuid) + ", 1088, 1\n",
        std::string(kUuid) + ", 18446744073709551616\n",
        std::string(kUuid) + ", " + std::to_string(UINT64_MAX / kMiB + 1) + "\n",
        std::string(kUuid) + ", 1088\n" + kUuid + ", 1088\n",
        std::string(kUuid) + ", 1088\nwarning 1\n"
    };
    for (const auto& output : physical_bad) {
        replies.physical = output;
        check(!before() && !receipt.physical_free_measured,
              "malformed, multiple, mismatched and overflowing physical rows refuse");
    }
    replies.physical.assign(detail::kMaxQueryOutput + 1, '1');
    check(!before(), "oversized readback refuses at the production query seam");
}

void test_after_readback() {
    FixtureDirectory root;
    check(root.valid, "post-context fixture created");
    if (!root.valid) return;
    const auto ceiling = declaration(root.path);
    MpsVerification before, receipt;
    std::string err;
    Replies replies;
    check(detail::verify_before(ceiling, 64 * kMiB, before, err, Replies::run, &replies),
          "post-context fixture has a valid pre-context receipt");
    const auto after = [&](const std::string& uuid = kUuid, uint64_t residual = kGiB - kMiB) {
        replies.calls = 0;
        receipt = before;
        return detail::verify_after(ceiling, uuid, residual, receipt, err, Replies::run, &replies);
    };
    check(after() && receipt.server_pid == 101 && receipt.client_pid == getpid() &&
          receipt.server_cap_bytes == kGiB && receipt.physical_free_bytes == before.physical_free_bytes &&
          replies.queried_server == 101 && replies.calls == 3 && replies.deadlines_equal,
          "own PID, matched server device cap and preserved physical sample verify");
    check(after("GPU-12345678-1234-ABCD-9876-123456789ABC"), "UUID hex case does not alter device identity");
    check(!after(kOtherUuid) && replies.calls == 0, "actual CUDA UUID mismatch refuses before queries");
    check(!after("GPU-12345678") && replies.calls == 0, "partial actual CUDA UUID refuses");
    check(!after(kUuid, kGiB + 1) && replies.calls == 0, "reported residual above cap refuses");
    MpsVerification absent;
    check(!detail::verify_after(ceiling, kUuid, 0, absent, err, Replies::run, &replies),
          "missing pre-context receipt refuses even when residual is small");
    replies = Replies{};
    replies.unavailable = true;
    check(!after(), "daemon disappearing after context refuses");
    replies = Replies{};
    for (const char* list : {"", "0\n", "-101\n", "+101\n", "101 102\n", "error 101\n",
                             "2147483648\n", "18446744073709551616\n", "101\n101\n", "101\n\n102\n"}) {
        replies.servers = list;
        check(!after() && receipt.client_pid == 0, "empty, malformed, duplicate or overflowing server lists refuse");
    }
    replies.servers.clear();
    for (size_t i = 0; i <= detail::kMaxServers; ++i) replies.servers += std::to_string(100 + i) + "\n";
    check(!after() && replies.calls == 1, "server count is bounded before any per-server queries");
    replies = Replies{};
    const std::string own = std::to_string(getpid());
    for (const auto& list : std::vector<std::string>{"", "0\n", "other process " + own + "\n",
                                                    own + "\n" + own + "\n", "18446744073709551616\n"}) {
        replies.first_clients = list;
        check(!after(), "own PID must occur in a strictly parsed positive unique client list");
    }
    replies.first_clients = std::to_string(getpid() == 7 ? 8 : 7) + "\n";
    check(!after(), "another attached client's PID is not evidence for this process");
    replies.first_clients.clear();
    for (size_t i = 1; i <= detail::kMaxClients + 1; ++i) replies.first_clients += std::to_string(i) + "\n";
    check(!after(), "client lists are bounded as well as server lists");
    replies = Replies{};
    replies.servers = "101\n102\n";
    replies.other_clients = own + "\n";
    check(!after(), "own PID in two servers is ambiguous and refuses");
    replies.other_clients = "diagnostic 102\n";
    check(!after(), "malformed later server is not ignored after an earlier PID match");
    replies.other_clients.clear();
    check(after() && replies.calls == 4 && replies.deadlines_equal,
          "empty other server client list is valid and all queries share one deadline");
    replies = Replies{};
    for (const char* limit : {"", "2G", "0", "device 0: 1G", "1G\n1G\n", "18446744073709551616"}) {
        replies.server_limit = limit;
        check(!after() && receipt.client_pid == 0 && receipt.server_pid == 0,
              "missing, malformed, overflowing and mismatched matched-server caps refuse");
    }
}

// Invoked only by this test executable's private transport override. Real production
// paths remain fixed. Fixture directory names select behavior without shell commands.
int child_fixture(int argc, char** argv) {
    const char* mode = std::getenv("STRATA_MPS_QUERY_FIXTURE");
    if (mode == nullptr) return -1;
    if (std::strcmp(mode, "timeout") == 0) {
        for (;;) pause();
    }
    if (std::strcmp(mode, "eof-timeout") == 0) {
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        for (;;) pause();
    }
    if (std::strcmp(mode, "closed") == 0) {
        close(STDIN_FILENO);
        return 7;
    }
    if (std::strcmp(mode, "flood") == 0 || std::strcmp(mode, "boundary") == 0) {
        const std::array<char, 4096> bytes{};
        const int blocks = std::strcmp(mode, "boundary") == 0 ? 16 : 18;
        for (int i = 0; i != blocks; ++i) {
            if (write(STDOUT_FILENO, bytes.data(), bytes.size()) < 0) return 1;
        }
        return 0;
    }
    if (std::strcmp(mode, "nonzero") == 0) return 3;
    if (std::strcmp(mode, "signal") == 0) { raise(SIGTERM); return 1; }
    if (std::strcmp(mode, "environment") == 0) {
        const char* pipe = std::getenv("CUDA_MPS_PIPE_DIRECTORY");
        const char* locale = std::getenv("LC_ALL");
        if (!pipe || !locale || std::strcmp(locale, "C") != 0) return 1;
        std::printf("%s\n", pipe);
        return 0;
    }
    if (std::strcmp(mode, "physical") == 0) {
        if (argc != 4 || std::string(argv[1]) != std::string("--id=") + kUuid ||
            std::strcmp(argv[2], "--query-gpu=uuid,memory.free") != 0 ||
            std::strcmp(argv[3], "--format=csv,noheader,nounits") != 0) return 1;
        std::printf("%s, 1088\n", kUuid);
        return 0;
    }
    if (std::strcmp(mode, "echo") == 0) {
        char buffer[256];
        for (;;) {
            const ssize_t n = read(STDIN_FILENO, buffer, sizeof(buffer));
            if (n == 0) return 0;
            if (n < 0) { if (errno == EINTR) continue; return 1; }
            if (write(STDOUT_FILENO, buffer, static_cast<size_t>(n)) != n) return 1;
        }
    }
    if (std::strcmp(mode, "diagnostic") == 0) {
        std::fputs("unavailable daemon 123\n", stderr);
        return 0;
    }
    return 2;
}

void test_transport() {
    Environment fixture("STRATA_MPS_QUERY_FIXTURE"), pipe("CUDA_MPS_PIPE_DIRECTORY"), locale("LC_ALL");
    auto ceiling = declaration("/tmp/the-declared-daemon");
    std::string output, err;
    const auto run = [&](detail::Query kind = detail::Query::default_limit,
                         std::chrono::milliseconds budget = std::chrono::milliseconds(1000)) {
        return detail::run_query(kind, ceiling, 101, std::chrono::steady_clock::now() + budget,
                                 output, err, "/proc/self/exe");
    };
    set_value(fixture.name, "echo");
    check(run() && output == "get_default_device_pinned_mem_limit 0\n", "fixed default query reaches child stdin");
    check(run(detail::Query::servers) && output == "get_server_list\n", "fixed server-list command is read-only");
    check(run(detail::Query::clients) && output == "get_client_list 101\n", "fixed client-list command carries selected PID");
    check(run(detail::Query::server_limit) && output == "get_device_pinned_mem_limit 101 0\n",
          "server device-0 readback uses the matched server PID");
    set_value(fixture.name, "physical");
    check(run(detail::Query::physical_free) && output == std::string(kUuid) + ", 1088\n",
          "physical query has only fixed read-only argv and exact UUID selector");
    set_value(fixture.name, "environment");
    set_value(pipe.name, "/tmp/wrong-ambient-daemon");
    set_value(locale.name, "POSIX");
    check(run() && output == ceiling.pipe_directory + "\n", "transport pins pipe and numeric locale in child environment");
    set_value(fixture.name, "diagnostic");
    check(run() && output == "unavailable daemon 123\n", "stderr is captured rather than hidden from strict parsing");
    set_value(fixture.name, "boundary");
    check(run() && output.size() == detail::kMaxQueryOutput, "exact output-size boundary is readable");
    for (const char* mode : {"closed", "nonzero", "signal", "flood"}) {
        set_value(fixture.name, mode);
        check(!run(), "closed input, nonzero exit, signal and output overflow fail closed");
    }
    for (const char* mode : {"timeout", "eof-timeout"}) {
        set_value(fixture.name, mode);
        const auto started = std::chrono::steady_clock::now();
        check(!run(detail::Query::default_limit, std::chrono::milliseconds(50)),
              "non-terminating query times out even after closing all output");
        check(std::chrono::steady_clock::now() - started < std::chrono::seconds(2),
              "timeout remains finite including kill and reap");
    }
    check(!detail::run_query(detail::Query::servers, ceiling, 0, std::chrono::steady_clock::now(), output, err,
                             "/proc/self/exe"), "already expired shared deadline refuses without launching");
    check(!detail::run_query(detail::Query::servers, ceiling, 0,
                             std::chrono::steady_clock::now() + std::chrono::seconds(1), output, err,
                             "/proc/self/no-such-executable"), "missing control executable refuses at the real transport seam");
    int status = 0;
    errno = 0;
    check(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD, "every query child is reaped on success and failure");
    // Exercise child file actions with parent descriptors initially closed; no collision
    // with stdin/stdout/stderr may leave a spurious writer alive or miswire the query.
    set_value(fixture.name, "echo");
    const int saved = dup(STDIN_FILENO);
    if (saved >= 0) {
        close(STDIN_FILENO);
        const bool success = run();
        dup2(saved, STDIN_FILENO);
        close(saved);
        check(success && output == "get_default_device_pinned_mem_limit 0\n", "closed parent stdin does not break spawn descriptor wiring");
    }
}
#else
void test_pipe_identity() {
    MpsCeiling ceiling;
    ceiling.declared = true;
    std::string err;
    check(!mps_ceiling_pipe_identity(ceiling, err), "declared pipe identity refuses on non-Linux");
}
#endif

}  // namespace

int main(int argc, char** argv) {
#if defined(__linux__)
    const int child_result = child_fixture(argc, argv);
    if (child_result >= 0) return child_result;
#else
    (void)argc; (void)argv;
#endif
    test_parse();
    test_declaration();
    test_pipe_identity();
    test_runtime_holds();
#if defined(__linux__)
    test_before_readback();
    test_after_readback();
    test_transport();
#endif
    std::printf("mps_ceiling_test: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
