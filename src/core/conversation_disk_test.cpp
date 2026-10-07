// HET-042 storage-tier parking: identity-scoped, bounded, exclusively owned.  Pure host code: the tier's
// records and files, with the real session-file writer/reader for the bytes.

#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}

SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
} // namespace

int main() {
    const fs::path parent = fs::temp_directory_path() /
        ("strata-disk-tier-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(parent);
    const fs::path bystander = parent / "user-file.txt";
    { std::ofstream b(bystander); b << "not ours"; }
    const ConversationIdentity t1{0xa, 0xb, 0xc, 1, 0};
    const ConversationIdentity t2{0xa, 0xb, 0xc, 2, 0};
    const ConversationIdentity t1new{0xa, 0xb, 0xd, 1, 0};

    // An exclusive, per-process directory; two tiers never share one.
    ConversationDiskCache first, second;
    std::string error;
    check(first.open(parent.string(), error) && second.open(parent.string(), error), "two tiers open");
    check(first.dir() != second.dir(), "each tier owns its own subdirectory");
    check(fs::is_directory(first.dir()), "the owned subdirectory exists");
    check(first.dir().find("strata-park-") != std::string::npos, "the subdirectory is the tier's own");

    // A park the caller wrote and admitted: the file is bound to the FULL identity, the record to the budget.
    first.set_budget(1 << 20);
    SavedConversation img = image({1, 2, 3, 4});
    size_t bytes = 0;
    const strata::core::SessionFileIdentity bound{t1.model, t1.config, t1.frontend, t1.tenant_lo, t1.tenant_hi};
    check(session_file_write((fs::path(first.dir()) / "park-1.bin").string(), img, bound, bytes,
                             error), "the tier's file writes");
    check(bytes > 96, "the file holds a v2 header and a payload");
    {
        // a real read back through the tier's identity - the header binds frontend and namespace, exactly
        SavedConversation back;
        size_t read_bytes = 0;
        check(session_file_read((fs::path(first.dir()) / "park-1.bin").string(), bound, back, read_bytes, error),
              "the owner's full identity reads the file back");
        check(back.live.ids == img.live.ids && back.live.gdn == img.live.gdn, "the read-back is the same state");
        check(!session_file_read((fs::path(first.dir()) / "park-1.bin").string(),
                                 {t1.model, t1.config, t1.frontend, 99, 0}, back, read_bytes, error) &&
              error.find("namespace") != std::string::npos,
              "another tenant's namespace cannot read the file");
        check(!session_file_read((fs::path(first.dir()) / "park-1.bin").string(),
                                 {t1.model, t1.config}, back, read_bytes, error),
              "no namespace at all cannot read the file");
    }
    check(first.make_room(bytes), "the file fits the budget");
    first.admit({t1, true, img.live.ids, img.live.imgs, (uint64_t) bytes, "park-1.bin"});
    check(first.bytes() == bytes && first.size() == 1, "the record charges its file's bytes");

    // Selection: identity and the exact token prefix, from the index alone.
    const std::vector<int64_t> prompt = {1, 2, 3, 4, 5};
    check(first.best(prompt, {}, true, t1).tokens == 4, "the owner's namespace matches its prefix");
    check(first.best(prompt, {}, true, t2).tokens == 0, "another tenant's namespace matches nothing");
    check(first.best(prompt, {}, true, t1new).tokens == 0, "another frontend matches nothing");
    check(first.best(prompt, {}, true, {}).tokens == 0, "no identity matches nothing (default-deny)");
    check(first.best(std::vector<int64_t>{1, 2, 9}, {}, true, t1).tokens == 0,
          "a diverged suffix is no prefix match");

    // Budget: oldest-first eviction inside the tier's own files only.
    SavedConversation img2 = image({9, 9, 9});
    size_t bytes2 = 0;
    check(session_file_write((fs::path(first.dir()) / "park-2.bin").string(), img2, bound, bytes2,
                             error), "the second file writes");
    check(first.make_room(bytes2), "room for the second file");
    first.admit({t1, true, img2.live.ids, img2.live.imgs, (uint64_t) bytes2, "park-2.bin"});
    const uint64_t after_two = first.bytes();
    check(after_two == bytes + bytes2, "both files are charged");
    const uint64_t small = 1;
    first.set_budget(after_two);   // the incoming byte exceeds the current footprint bound
    check(first.make_room(small) && first.size() == 1 && first.evictions() == 1 && first.bytes() == bytes2,
          "the oldest entry is evicted to fit");
    check(first.entry(0).name == "park-2.bin" && !fs::exists(fs::path(first.dir()) / "park-1.bin"),
          "the evicted file is gone, the kept one remains");
    check(fs::exists(bystander), "the user's own file is never touched");
    check(!first.make_room((uint64_t) 1 << 40), "a file over the whole budget is refused");

    // AC-3: a model/config/frontend change discards the entries that can never be selected again; other
    // tenants of the current identity stay.
    first.admit({t1new, true, img2.live.ids, img2.live.imgs, 8, "park-3.bin"});
    { std::ofstream f(fs::path(first.dir()) / "park-3.bin"); f << "stand-in"; }
    first.admit({t1new, true, img2.live.ids, img2.live.imgs, 8, "park-4.bin"});
    { std::ofstream f(fs::path(first.dir()) / "park-4.bin"); f << "stand-in"; }
    check(first.invalidate_except(t1) == 2, "the other frontend's entries are invalidated");
    check(first.size() == 1 && !fs::exists(fs::path(first.dir()) / "park-3.bin") &&
          !fs::exists(fs::path(first.dir()) / "park-4.bin"),
          "invalidated entries' files are unlinked, the kept ones' are not");
    check(first.invalidate_except(t1) == 0, "no change, nothing to discard");

    // remove() unlinks exactly the entry's own file; close() takes only this process's own files and its own
    // (now empty) directory - the bystander file and the OTHER tier's directory stay.
    check(fs::exists(fs::path(first.dir()) / first.entry(0).name), "the kept entry's file exists");
    first.remove(0);
    check(first.size() == 0 && first.bytes() == 0, "remove drops the record and its bytes");
    check(fs::directory_iterator(first.dir()) == fs::directory_iterator(), "the owned directory is empty");
    second.admit({t1, true, img.live.ids, img.live.imgs, 4, "park-1.bin"});
    { std::ofstream f(fs::path(second.dir()) / "park-1.bin"); f << "other"; }
    first.close();
    second.close();
    check(!fs::exists(first.dir()) && !fs::exists(second.dir()), "each tier's own subdirectory goes with it");
    check(fs::exists(bystander), "the user's file in the parent stays");

    std::filesystem::remove_all(parent);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
}
