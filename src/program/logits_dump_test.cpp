#include "logits_dump.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using strata::program::LogitsDump;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
void require(bool ok, const char* what) {
    if (!ok) { std::perror(what); std::exit(2); }
}
constexpr std::array<float, 4> row = {1.0f, -2.0f, 3.0f, 0.0f};
void open_dump(LogitsDump& dump, const std::string& path, int rows = 2) {
    dump.path = path;
    dump.n_vocab = row.size();
    dump.expected_rows = rows;
    dump.f = std::fopen(path.c_str(), "wb");
    require(dump.f != nullptr, "fopen");
    const int32_t header[] = {int32_t(row.size()), rows};
    require(std::fwrite(header, sizeof header, 1, dump.f) == 1, "header");
}
struct CloseFailure { int fd; };
ssize_t cookie_write(void* value, const char* data, size_t size) {
    return ::write(static_cast<CloseFailure*>(value)->fd, data, size);
}
int cookie_close(void* value) {
    require(::close(static_cast<CloseFailure*>(value)->fd) == 0, "close real backing fd");
    errno = EIO;
    return -1;
}
} // namespace

int main() {
    char temporary[] = "/tmp/strata-logits-XXXXXX";
    require(mkdtemp(temporary) != nullptr, "mkdtemp");
    const std::filesystem::path root(temporary);
    const std::string path = (root / "dump.bin").string();
    {
        LogitsDump dump;
        open_dump(dump, path);
        check(dump.write_row(row.data(), 0) && dump.write_row(row.data(), 1), "complete rows write");
        check(dump.finish(), "complete dump succeeds");
        check(std::filesystem::file_size(path) == 8 + 2 * sizeof row, "complete dump exact byte count");
        std::FILE* input = std::fopen(path.c_str(), "rb");
        require(input != nullptr, "read complete dump");
        int32_t header[2];
        std::array<float, 8> payload;
        check(std::fread(header, sizeof header, 1, input) == 1 && header[0] == 4 && header[1] == 2,
              "complete dump truthful header");
        check(std::fread(payload.data(), sizeof payload, 1, input) == 1 &&
              std::memcmp(payload.data(), row.data(), sizeof row) == 0 &&
              std::memcmp(payload.data() + row.size(), row.data(), sizeof row) == 0,
              "complete dump preserves full row bytes");
        std::fclose(input);
    }
    {
        LogitsDump dump;
        open_dump(dump, path);
        check(dump.write_row(row.data(), 0), "early termination wrote one row");
    }
    check(!std::filesystem::exists(path), "unwound incomplete dump removed");
    {
        LogitsDump dump;
        open_dump(dump, path);
        check(dump.write_row(row.data(), 0), "short count wrote one row");
        check(!dump.finish(), "short row count refuses completion");
        check(!std::filesystem::exists(path), "short count removed");
    }
    {
        LogitsDump dump;
        open_dump(dump, path, 1);
        check(dump.write_row(row.data(), 0), "size mismatch wrote row");
        require(std::fputc('x', dump.f) != EOF, "extra byte");
        check(!dump.finish(), "extra bytes refuse completion");
        check(!std::filesystem::exists(path), "extra bytes removed");
    }
    // Real kernel ENOSPC at fwrite and at the buffered fflush boundary. Removal
    // unlinks our temporary symlink, never /dev/full itself.
    for (bool buffered : {false, true}) {
        require(::symlink("/dev/full", path.c_str()) == 0, "symlink /dev/full");
        {
            LogitsDump dump;
            dump.path = path;
            dump.n_vocab = row.size();
            dump.expected_rows = 1;
            dump.f = std::fopen(path.c_str(), "wb");
            require(dump.f != nullptr, "fopen /dev/full");
            char buffer[4096];
            require(std::setvbuf(dump.f, buffered ? buffer : nullptr, buffered ? _IOFBF : _IONBF,
                                buffered ? sizeof buffer : 0) == 0, "setvbuf");
            const bool wrote = dump.write_row(row.data(), 0);
            check(wrote == buffered, buffered ? "buffered row defers ENOSPC" : "unbuffered ENOSPC reported");
            check(!dump.finish(), buffered ? "flush ENOSPC refuses completion" : "write failure refuses completion");
        }
        check(!std::filesystem::is_symlink(path), "failed full-device output removed");
    }
    {
        // libc performs a real fclose on a real file-backed stream, whose close
        // callback fails after successful write/flush. Isolates the close verdict
        // from row-count, fflush and file-size checks (which all succeed).
        CloseFailure backing{::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600)};
        require(backing.fd >= 0, "open close-failure backing");
        LogitsDump dump;
        dump.path = path;
        dump.n_vocab = row.size();
        dump.expected_rows = 1;
        cookie_io_functions_t callbacks{};
        callbacks.write = cookie_write;
        callbacks.close = cookie_close;
        dump.f = fopencookie(&backing, "w", callbacks);
        require(dump.f != nullptr, "fopencookie");
        const int32_t header[] = {int32_t(row.size()), 1};
        require(std::fwrite(header, sizeof header, 1, dump.f) == 1, "cookie header");
        check(dump.write_row(row.data(), 0), "close-failure row writes");
        require(std::fflush(dump.f) == 0, "cookie flush");
        check(std::filesystem::file_size(path) == 8 + sizeof row, "close-failure file already complete");
        check(!dump.finish(), "fclose failure refuses otherwise complete dump");
        check(!std::filesystem::exists(path), "close-failed complete file removed");
    }
    {
        int ready[2];
        require(::pipe(ready) == 0, "pipe");
        const pid_t child = ::fork();
        require(child >= 0, "fork");
        if (child == 0) {
            ::close(ready[0]);
            LogitsDump dump;
            open_dump(dump, path);
            if (!dump.write_row(row.data(), 0) || std::fflush(dump.f) != 0) _exit(2);
            const char byte = 'r';
            if (::write(ready[1], &byte, 1) != 1) _exit(3);
            for (;;) ::pause();
        }
        ::close(ready[1]);
        char byte = 0;
        require(::read(ready[0], &byte, 1) == 1 && byte == 'r', "child flushed partial output");
        ::close(ready[0]);
        require(::kill(child, SIGKILL) == 0, "kill writer");
        int status = 0;
        require(::waitpid(child, &status, 0) == child, "waitpid");
        check(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "abrupt writer has no success exit");
        check(std::filesystem::file_size(path) == 8 + sizeof row, "abrupt file is detectably short of two-row header");
    }
    std::filesystem::remove_all(root);
    if (!failures) std::puts("logits_dump_test: complete, unwind, count, size, write, flush, close and SIGKILL controls passed");
    return failures ? 1 : 0;
}
