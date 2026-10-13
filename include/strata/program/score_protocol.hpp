// SCORE v1: final-prompt-row selected FP32 logits, never a generation request.
#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::program::score {
inline constexpr size_t max_wire_bytes = 65536;
inline constexpr size_t max_prompt_tokens = 4096;
inline constexpr size_t max_selected_tokens = 256;
inline constexpr size_t max_pending = 8;

inline bool valid_id(std::string_view id) {
    if (id.empty() || id.size() > 64) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}
inline std::string_view request_id(std::string_view line) {
    if (line.substr(0, 6) != "SCORE ") return {};
    const auto end = line.find(' ', 6);
    return line.substr(6, end == line.npos ? line.size() - 6 : end - 6);
}
struct Request {
    std::string id;
    std::vector<int64_t> prompt, selected;
};
inline bool token_list(std::string_view text, size_t cap, int64_t vocabulary,
                       std::vector<int64_t>& out) {
    out.clear();
    if (text.empty()) return false;
    for (;;) {
        const size_t comma = text.find(',');
        const auto item = text.substr(0, comma);
        int64_t token = -1;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), token);
        if (item.empty() || parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() ||
            token < 0 || token >= vocabulary || out.size() >= cap) return false;
        out.push_back(token);
        if (comma == text.npos) return true;
        text.remove_prefix(comma + 1);
    }
}
inline bool parse(std::string_view line, int64_t context, int64_t vocabulary,
                  Request& request, std::string& error) {
    request.id.clear(); request.prompt.clear(); request.selected.clear();
    const auto id = request_id(line);
    if (!valid_id(id)) { error = "invalid score request ID"; return false; }
    request.id = id;
    if (line.size() > max_wire_bytes) { error = "score wire limit exceeded"; return false; }
    const size_t start = 7 + id.size();
    if (start > line.size() || line.substr(start, 10) != "v1 prompt=") {
        error = "expected SCORE v1 prompt=<ids> selected=<ids>"; return false;
    }
    auto payload = line.substr(start + 10);
    const size_t gap = payload.find(" selected=");
    if (context > 0) request.prompt.reserve(std::min<uint64_t>(max_prompt_tokens, (uint64_t) context));
    request.selected.reserve(max_selected_tokens);
    if (gap == payload.npos || context <= 0 || vocabulary <= 0 ||
        !token_list(payload.substr(0, gap), max_prompt_tokens, vocabulary, request.prompt) ||
        !token_list(payload.substr(gap + 10), max_selected_tokens, vocabulary, request.selected) ||
        request.prompt.size() > (uint64_t) context) {
        request.prompt.clear(); request.selected.clear();
        error = "score token, context or work limit exceeded"; return false;
    }
    return true;
}

// Called under the serving reader's mutex. IDs own cancellation through SDONE publication.
class Pending {
    struct Owner { std::string id; bool cancelled = false; };
    std::array<Owner, max_pending> owners_;
public:
    enum class Admission { accepted, full, duplicate };
    Admission admit(std::string_view id) {
        Owner* free = nullptr;
        for (auto& owner : owners_) {
            if (owner.id == id) return Admission::duplicate;
            if (owner.id.empty()) free = &owner;
        }
        if (!free) return Admission::full;
        free->id = id; free->cancelled = false;
        return Admission::accepted;
    }
    void cancel(std::string_view id) {
        for (auto& owner : owners_) if (owner.id == id) owner.cancelled = true;
    }
    bool cancelled(std::string_view id) const {
        for (const auto& owner : owners_) if (owner.id == id) return owner.cancelled;
        return false;
    }
    void complete(std::string_view id) {
        for (auto& owner : owners_) if (owner.id == id) { owner.id.clear(); owner.cancelled = false; }
    }
};

inline bool select_logits(const float* row, int64_t vocabulary, const std::vector<int64_t>& selected,
                          std::vector<float>& values, std::string& error) {
    values.clear();
    if (!row || vocabulary <= 0 || selected.empty() || selected.size() > max_selected_tokens) {
        error = "invalid selected logit extent"; return false;
    }
    for (int64_t token : selected) if (token < 0 || token >= vocabulary) {
        error = "selected token outside vocabulary"; return false;
    }
    for (int64_t token = 0; token < vocabulary; ++token) if (!std::isfinite(row[token])) {
        error = "non-finite score logits"; return false;
    }
    values.reserve(max_selected_tokens);
    values.resize(selected.size());
    for (size_t i = 0; i < selected.size(); ++i) values[i] = row[selected[i]];
    return true;
}
inline void json_string(std::string& out, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    out += '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += char(c);
    }
    out += '"';
}
inline std::string terminal(std::string_view id, std::string_view status, const Request* request = nullptr,
                            const std::vector<float>* values = nullptr, std::string_view error = {}) {
    // A mismatched or incomplete vector can never become an OK option set.
    if (status == "OK" && (!request || !values || values->size() != request->selected.size() ||
                          values->empty())) {
        status = "ERROR"; error = "incomplete score result"; request = nullptr; values = nullptr;
    }
    if (status == "OK") for (float value : *values) if (!std::isfinite(value)) {
        status = "ERROR"; error = "non-finite score result"; request = nullptr; values = nullptr; break;
    }
    std::string out;
    out.reserve(1024 + (values ? values->size() * 32 : 0));
    out += "SDONE "; out += id; out += ' '; out += status;
    out += " {\"version\":1,\"request_id\":"; json_string(out, id);
    out += ",\"generated_tokens\":0";
    if (status == "OK") {
        out += ",\"prompt_tokens\":" + std::to_string(request->prompt.size());
        out += ",\"selected_token_ids\":[";
        for (size_t i = 0; i < request->selected.size(); ++i) {
            if (i) out += ',';
            out += std::to_string(request->selected[i]);
        }
        out += "],\"logits\":[";
        for (size_t i = 0; i < values->size(); ++i) {
            if (i) out += ',';
            char number[48];
            const auto converted = std::to_chars(number, number + sizeof number, (*values)[i],
                                                 std::chars_format::general, std::numeric_limits<float>::max_digits10);
            out.append(number, converted.ptr);
        }
        out += "],\"receipt\":{\"prompt_tokens\":" + std::to_string(request->prompt.size());
        out += ",\"selected_tokens\":" + std::to_string(request->selected.size());
        out += ",\"logit_passes\":1,\"dtype\":\"float32\",\"scale\":1}";
    } else {
        out += ",\"error\":"; json_string(out, error.substr(0, 512));
    }
    out += "}\n";
    return out;
}
} // namespace strata::program::score
