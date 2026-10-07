// Storage-tier parking (HET-042): conversations that no longer fit the RAM budget, held as session-file
// format images (conversation_file.hpp) in a directory this process owns exclusively.  Pure host code: no CUDA.
//
// - The tier is bounded by a byte budget; admission evicts oldest-first within it.
// - Selection matches the full ConversationCache identity and the same exact token/image prefix rule as
//   ConversationCache::best; the caller consults this tier only when RAM has no longer match.
// - Nothing here is durable: the owned subdirectory is created exclusively per process, and only names this
//   process admitted are ever removed.  A process-salted authorization namespace (server-side) plus the
//   per-process subdirectory mean parked state is never reused across engine or server restarts, by
//   construction.  A crashed process's subdirectory (and any of its unwritten temporaries) is left for its
//   owner to clean up; this process never touches it.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <filesystem>

namespace strata::core {

class ConversationDiskCache {
public:
    struct Entry {
        ConversationIdentity identity;
        bool cvec = true;
        std::vector<int32_t> ids;                    // the parked conversation's tokens, for prefix matching
        std::vector<ConversationImageKey> imgs;
        uint64_t file_bytes = 0;
        std::string name;                            // inside the owned directory; never a path with separators
    };
    struct Hit {
        size_t index = 0;
        int64_t tokens = 0;
    };

    // Creates an exclusive per-process subdirectory of `parent` ("strata-park-<pid>-<random>"): a name
    // collision is retried with a fresh one, so no two engines (or two runs of the same engine) ever share a
    // directory, and a previous process's subdirectory is never opened or cleaned.  false (with `error`) when
    // the parent is missing or unwritable.
    bool open(const std::string& parent, std::string& error) {
        dir_.clear();
        if (parent.empty()) { error = "no directory given"; return false; }
        std::error_code ec;
        if (!std::filesystem::is_directory(std::filesystem::path(parent), ec)) {
            error = parent + " is not a directory";
            return false;
        }
        std::random_device rng;
        for (int attempt = 0; attempt < 64; ++attempt) {
            char tag[16];
            std::snprintf(tag, sizeof tag, "%08x", (unsigned) (rng() & 0xffffffffu));
            const std::filesystem::path p =
                std::filesystem::path(parent) / ("strata-park-" + std::to_string(pid_()) + "-" + tag);
            ec.clear();
            if (std::filesystem::create_directory(p, ec)) {
                // The files carry a conversation's private state: the directory is the owner's only, whatever
                // the umask would have made it (Windows keeps its ACL model; the session-file docs say so).
                std::error_code pec;
                std::filesystem::permissions(p, std::filesystem::perms::owner_all, pec);
                dir_ = p.string();
                return true;
            }
            if (ec) {   // not a collision: the parent is missing or refuses the creation
                error = p.string() + ": " + ec.message();
                return false;
            }
        }
        error = "no unique subdirectory under " + parent;
        return false;
    }
    bool enabled() const { return !dir_.empty(); }
    const std::string& dir() const { return dir_; }
    void set_budget(uint64_t bytes) { budget_ = bytes; }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    uint64_t evictions() const { return evictions_; }

    // Longest exact prefix among this identity's entries, from the indexed tokens alone (no file is read).
    template<class Token>
    Hit best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
             const ConversationIdentity& id) const {
        Hit hit;
        if (!id.known()) return hit;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& e = entries_[i];
            if (e.cvec != cvec || !(e.identity == id)) continue;
            const int64_t n = conversation_prefix(e.ids, e.imgs, prompt, images);
            if (n > hit.tokens) hit = {i, n};
        }
        return hit;
    }

    // Records a file the caller has written and owns; its size joins the budget.
    void admit(Entry&& e) {
        bytes_ += e.file_bytes;
        entries_.push_back(std::move(e));
    }
    // Oldest-first eviction so `incoming` more bytes fit.  false when the file alone exceeds the budget, or
    // when an evicted file cannot be unlinked: the charge then stays, and the caller refuses the park instead
    // of letting the real usage drift past the bound.
    bool make_room(uint64_t incoming) {
        if (incoming > budget_) return false;
        while (!entries_.empty() && bytes_ + incoming > budget_) {
            if (!remove(0)) return false;
            ++evictions_;
        }
        return true;
    }
    const Entry& entry(size_t index) const { return entries_.at(index); }
    // Unlinks the entry's file and drops its record.  Only names this process admitted are ever unlinked.
    // false (record and charge kept) when the file could not be removed: an uncharged footprint would let the
    // real usage drift past the budget.  A file that has already vanished is gone: the record goes too.
    bool remove(size_t index) {
        const Entry& e = entries_.at(index);
        const std::filesystem::path p = std::filesystem::path(dir_) / e.name;
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) {
            std::filesystem::remove(p, ec);
            if (ec || std::filesystem::exists(p, ec)) return false;
        } else if (ec) {
            return false;
        }
        bytes_ -= std::min<uint64_t>(bytes_, e.file_bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        return true;
    }
    // Entries whose model, config or frontend no longer matches can never be selected again (AC-3): unlink
    // and drop them now instead of failing them at restore time.  Other tenants' entries stay: they are
    // valid for their owners.  An entry whose file cannot be removed stays charged (and can never match).
    // Returns how many were discarded.
    size_t invalidate_except(const ConversationIdentity& keep) {
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const ConversationIdentity& e = entries_[i].identity;
            if (e.model != keep.model || e.config != keep.config || e.frontend != keep.frontend) {
                if (remove(i)) {
                    ++dropped;
                    continue;
                }
            }
            ++i;
        }
        return dropped;
    }
    // Best-effort removal of this process's own files and its own (possibly not empty) subdirectory: a file
    // that cannot be unlinked stops the sweep and stays for its owner to clean up - never a busy loop.
    void close() noexcept {
        if (!enabled()) return;
        std::error_code ec;
        while (!entries_.empty())
            if (!remove(0)) break;
        std::filesystem::remove(std::filesystem::path(dir_), ec);   // fails while anything remains: fine
        dir_.clear();
    }

private:
    static int pid_() {
#if defined(_WIN32)
        return _getpid();
#else
        return getpid();
#endif
    }

    std::string dir_;
    uint64_t budget_ = 0, bytes_ = 0, evictions_ = 0;
    std::deque<Entry> entries_;
};

} // namespace strata::core
