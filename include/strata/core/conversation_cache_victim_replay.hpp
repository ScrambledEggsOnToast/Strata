// One-shot diagnostic nomination, never a retention policy or authorization source.
#pragma once

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace strata::core {

struct ConversationVictimIdentity {
    uint64_t serial = 0, scope = 0, history = 0, tokens = 0, bytes = 0;
    bool operator==(const ConversationVictimIdentity&) const = default;
};

class ConversationVictimReplay {
public:
    static constexpr size_t max_victims = 8, max_contract_bytes = 4096;
    // Serial is the cache's never-reused allocation generation, not a deque index.
    ConversationVictimIdentity subject;
    std::array<ConversationVictimIdentity, max_victims> victims{};
    uint64_t contract = 0, event = 0, request_sequence = 0, leg = 0;
    size_t count = 0;
    std::string_view site;
    const char* status = "off";
    bool configured = false, alternate = false, consumed = false;

    // Fixed, strict token grammar. Unknown, duplicate or trailing fields fail closed.
    // Both arms use the same pathname; only its frozen contents differ between processes.
    static ConversationVictimReplay parse(std::string_view text) noexcept {
        ConversationVictimReplay out;
        out.status = "malformed";
        out.contract = fingerprint(text);
        if (text.empty() || text.size() > max_contract_bytes) return out;
        auto token = [&]() {
            const auto begin = text.find_first_not_of(" \t\r\n");
            if (begin == text.npos) { text = {}; return std::string_view{}; }
            text.remove_prefix(begin);
            const auto end = text.find_first_of(" \t\r\n");
            const auto value = text.substr(0, end);
            text.remove_prefix(value.size());
            return value;
        };
        auto number = [&](uint64_t& value, int base = 10) {
            const auto word = token();
            if (word.empty()) return false;
            const auto result = std::from_chars(word.data(), word.data() + word.size(), value, base);
            return result.ec == std::errc{} && result.ptr == word.data() + word.size();
        };
        auto identity = [&](ConversationVictimIdentity& value) {
            return number(value.serial) && number(value.scope, 16) && number(value.history, 16) &&
                number(value.tokens) && number(value.bytes);
        };
        if (token() != "strata-cache-victim-replay-v1" || token() != "mode") return out;
        const auto mode = token();
        if (mode != "keep" && mode != "alternate") return out;
        out.alternate = mode == "alternate";
        const auto field = token();
        if (field.empty() && !out.alternate) { out.status = "keep"; return out; }
        if (field != "event" || !number(out.event) || !out.event ||
            token() != "request_sequence" || !number(out.request_sequence) || !out.request_sequence ||
            token() != "leg" || !number(out.leg) || !out.leg || token() != "site") return out;
        const auto site = token();
        // Retain only static strings: the parsing buffer need not outlive this call.
        if (site == "put") out.site = "put";
        else if (site == "reserve_snapshot") out.site = "reserve_snapshot";
        else if (site == "evict_oldest") out.site = "evict_oldest";
        else return out;
        uint64_t count = 0;
        if (token() != "subject" || !identity(out.subject) ||
            token() != "victims" || !number(count) || !count || count > max_victims) return out;
        out.count = size_t(count);
        if (out.site == "put") {
            if (!out.subject.serial || !out.subject.tokens || !out.subject.bytes) return out;
        } else if (out.subject != ConversationVictimIdentity{}) return out;
        if (out.site == "evict_oldest" && out.count != 1) return out;
        for (size_t i = 0; i < out.count; ++i) {
            auto& victim = out.victims[i];
            if (token() != "victim" || !identity(victim) || !victim.serial || !victim.tokens || !victim.bytes)
                return out;
            for (size_t j = 0; j < i; ++j) if (out.victims[j].serial == victim.serial) return out;
        }
        if (!token().empty()) return out;
        out.configured = true;
        out.status = "armed";
        return out;
    }

    static ConversationVictimReplay from_path(const char* path) noexcept {
        ConversationVictimReplay out;
        if (!path || !*path) return out;
        out.status = "missing";
        FILE* file = std::fopen(path, "rb");
        if (!file) return out;
        char bytes[max_contract_bytes + 1];
        const size_t n = std::fread(bytes, 1, sizeof(bytes), file);
        const bool valid = !std::ferror(file) && n <= max_contract_bytes;
        std::fclose(file);
        if (!valid) { out.status = "malformed"; return out; }
        return parse(std::string_view(bytes, n));
    }
    static ConversationVictimReplay& process() noexcept {
        // Includes the absent/malformed case: never reopen, reload, retry or adapt.
        static ConversationVictimReplay value = from_path(std::getenv("STRATA_CACHE_VICTIM_REPLAY"));
        return value;
    }

    bool contains(uint64_t serial) const noexcept {
        for (size_t i = 0; i < count; ++i) if (victims[i].serial == serial) return true;
        return false;
    }

    template<class Image>
    static bool identify(uint64_t serial, const Image& image, ConversationVictimIdentity& out) noexcept {
        // Bounded diagnostic identity work, never an equality/authorization shortcut.
        constexpr size_t max_words = 1u << 20;
        if (image.live.ids.size() > max_words || image.live.imgs.size() >
            (max_words - image.live.ids.size()) / 2) return false;
        out = {serial, 1469598103934665603ull, 1469598103934665603ull,
               uint64_t(image.live.ids.size()), uint64_t(image.bytes())};
        for (auto byte : image.scope.value) out.scope = (out.scope ^ byte) * 1099511628211ull;
        for (auto id : image.live.ids) out.history = (out.history ^ uint32_t(id)) * 1099511628211ull;
        for (const auto& key : image.live.imgs) {
            out.history = (out.history ^ uint64_t(key.start)) * 1099511628211ull;
            out.history = (out.history ^ key.hash) * 1099511628211ull;
        }
        return true;
    }

private:
    static uint64_t fingerprint(std::string_view text) noexcept {
        uint64_t hash = 1469598103934665603ull;
        for (unsigned char byte : text) hash = (hash ^ byte) * 1099511628211ull;
        return hash;
    }
};
} // namespace strata::core
