#include "strata/core/native_weight_identity.hpp"

#include <atomic>
#include <filesystem>
#include <limits>
#include <stdexcept>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace strata::core {
NativeSourceIdentity native_source_identity(const std::string& path) {
    NativeSourceIdentity result;
    result.path = std::filesystem::canonical(path).string();
#ifndef _WIN32
    struct stat info{};
    if (::stat(result.path.c_str(), &info) != 0 || !S_ISREG(info.st_mode))
        throw std::runtime_error("native identity requires a regular source file");
    result.stamp = {uint64_t(info.st_dev), uint64_t(info.st_ino), uint64_t(info.st_size),
                    uint64_t(info.st_mtim.tv_sec), uint64_t(info.st_mtim.tv_nsec),
                    uint64_t(info.st_ctim.tv_sec), uint64_t(info.st_ctim.tv_nsec)};
#else
    // The DFlash2 exchange is Linux-only. Other backends can still record a
    // load identity; no Linux inode/ctime guarantee is claimed on Windows.
    result.stamp[2] = std::filesystem::file_size(result.path);
    result.stamp[3] = uint64_t(std::filesystem::last_write_time(result.path).time_since_epoch().count());
#endif
    return result;
}
uint64_t next_native_weight_generation() {
    static std::atomic<uint64_t> generation{1};
    uint64_t result = generation.load(std::memory_order_relaxed);
    for (;;) {
        if (result == std::numeric_limits<uint64_t>::max())
            throw std::runtime_error("native weight load generation exhausted");
        if (generation.compare_exchange_weak(result, result + 1, std::memory_order_relaxed)) return result;
    }
}
} // namespace strata::core
