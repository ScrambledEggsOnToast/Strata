// src/platform/mps_ceiling.cpp - see include/strata/platform/mps_ceiling.hpp.
//
// Everything here is host-only and side-effect free apart from `stat`: the parse never
// touches the filesystem, the identity check never opens the control pipe, and the runtime
// check is arithmetic over numbers the caller already measured. The engine refuses on a
// declaration it cannot honour rather than proceeding with an unenforced ceiling.
#include "strata/platform/mps_ceiling.hpp"

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::platform {
namespace {

const char* kLimitVariable = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT";
const char* kPipeVariable = "CUDA_MPS_PIPE_DIRECTORY";

std::string trim(const std::string& text) {
    size_t first = 0, last = text.size();
    while (first < last && std::isspace(static_cast<unsigned char>(text[first]))) ++first;
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) --last;
    return text.substr(first, last - first);
}

/// `<digits><suffix>`; the driver accepts decimal and binary spellings of K/M/G/T.
bool parse_number(const std::string& text, uint64_t& bytes) {
    const std::string value = trim(text);
    if (value.empty()) return false;
    size_t at = 0;
    uint64_t number = 0;
    while (at < value.size() && std::isdigit(static_cast<unsigned char>(value[at]))) {
        const uint64_t digit = static_cast<uint64_t>(value[at] - '0');
        if (number > (UINT64_MAX - digit) / 10) return false;  // overflow refuses
        number = number * 10 + digit;
        ++at;
    }
    if (at == 0) return false;
    std::string suffix;
    while (at < value.size()) suffix.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(value[at++]))));
    if (suffix == "i") suffix.clear();          // "1Mi" without a B is still Mi
    if (suffix.size() > 1 && suffix[1] == 'i') suffix.erase(1, 1);  // "MiB" -> "mb", "mib" -> "mb"
    if (!suffix.empty() && suffix.back() == 'b') suffix.pop_back();
    uint64_t scale = 1;
    if (suffix.empty()) scale = 1;
    else if (suffix == "k") scale = 1024ull;
    else if (suffix == "m") scale = 1024ull * 1024;
    else if (suffix == "g") scale = 1024ull * 1024 * 1024;
    else if (suffix == "t") scale = 1024ull * 1024 * 1024 * 1024;
    else return false;
    if (number != 0 && scale > UINT64_MAX / number) return false;
    bytes = number * scale;
    return true;
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : text) {
        if (c == separator) { parts.push_back(current); current.clear(); }
        else current.push_back(c);
    }
    parts.push_back(current);
    return parts;
}

}  // namespace

bool mps_parse_limit(const std::string& text, int wanted_device, uint64_t& bytes, std::string& err) {
    for (const std::string& entry : split(text, ',')) {
        if (trim(entry).empty()) continue;
        const size_t equals = entry.find('=');
        std::string device_text, value_text;
        if (equals == std::string::npos) {
            // A bare value applies to every device the client can see; the caller's
            // ordinal is the only device this engine uses.
            device_text.clear();
            value_text = entry;
        } else {
            device_text = trim(entry.substr(0, equals));
            value_text = entry.substr(equals + 1);
        }
        int device = wanted_device;
        if (!device_text.empty()) {
            char* end = nullptr;
            const long parsed = std::strtol(device_text.c_str(), &end, 10);
            if (end == nullptr || *end != '\0' || parsed < 0) {
                err = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: unreadable device '" + device_text + "'";
                return false;
            }
            device = static_cast<int>(parsed);
        }
        if (device != wanted_device) continue;
        if (!parse_number(value_text, bytes) || bytes == 0) {
            err = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: unreadable limit '" + trim(value_text) + "'";
            return false;
        }
        return true;
    }
    err = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: no entry for device " + std::to_string(wanted_device);
    return false;
}

bool mps_ceiling_from_environment(MpsCeiling& out, std::string& err) {
    out = MpsCeiling{};
    const char* limit = std::getenv(kLimitVariable);
    const char* pipe = std::getenv(kPipeVariable);
    const bool have_limit = limit != nullptr && !trim(limit).empty();
    const bool have_pipe = pipe != nullptr && !trim(pipe).empty();
    if (!have_limit && !have_pipe) return true;  // nothing declared: not this module's business
    if (have_limit != have_pipe) {
        err = std::string("MPS ceiling declared incompletely: ") +
              (have_limit ? kPipeVariable : kLimitVariable) + " is missing";
        return false;
    }
    out.declared = true;
    out.pipe_directory = pipe;
    // Device ordinals only matter for multi-device clients; the engine plans device 0 and
    // the supervisor sets the limit for that ordinal.
    if (!mps_parse_limit(limit, 0, out.cap_bytes, err)) {
        out = MpsCeiling{};
        return false;
    }
    out.device = 0;
    out.limit_from_environment = true;
    return true;
}

#if defined(_WIN32)
bool mps_ceiling_pipe_identity(const MpsCeiling& ceiling, std::string& err) {
    // MPS is Linux-only, so a declaration reaching a Windows build cannot be honoured.
    if (!ceiling.declared) return true;
    err = "an MPS ceiling cannot be enforced on this platform: " + ceiling.pipe_directory;
    return false;
}
#else
bool mps_ceiling_pipe_identity(const MpsCeiling& ceiling, std::string& err) {
    if (!ceiling.declared) return true;
    struct stat info;
    if (lstat(ceiling.pipe_directory.c_str(), &info) != 0) {
        err = "MPS control pipe directory is not reachable: " + ceiling.pipe_directory;
        return false;
    }
    if (S_ISLNK(info.st_mode) || !S_ISDIR(info.st_mode)) {
        err = "MPS control pipe directory is not a directory: " + ceiling.pipe_directory;
        return false;
    }
    const uid_t uid = geteuid();
    if (info.st_uid != uid && info.st_uid != 0) {
        err = "MPS control pipe directory is not owned by this user or root: " +
              ceiling.pipe_directory;
        return false;
    }
    return true;
}
#endif

bool mps_ceiling_holds(uint64_t reported_free_bytes, uint64_t declared_cap_bytes,
                       std::string& err) {
    if (declared_cap_bytes == 0) {
        err = "MPS ceiling is zero: nothing could be allocated";
        return false;
    }
    if (reported_free_bytes > declared_cap_bytes) {
        err = "declared MPS ceiling is not in force: cudaMemGetInfo reports " +
              std::to_string(reported_free_bytes) + " B free above the declared " +
              std::to_string(declared_cap_bytes) +
              " B ceiling, which is the unattached-client fallback to the whole device";
        return false;
    }
    return true;
}

uint64_t mps_device_ceiling(uint64_t total_bytes, uint64_t declared_cap_bytes, bool declared) {
    if (!declared) return total_bytes;
    return declared_cap_bytes < total_bytes ? declared_cap_bytes : total_bytes;
}

}  // namespace strata::platform
