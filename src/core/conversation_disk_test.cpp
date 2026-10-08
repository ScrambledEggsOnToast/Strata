// HET-042 storage-tier parking: identity-scoped, bounded, exclusively owned.  Pure host code: the tier's
// records and files, with the real session-file writer/reader for the bytes.

#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_file.hpp"

#include <cerrno>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
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

    // The writer prices its padded temporary peak BEFORE writing, but the tier records only published bytes.
    {
        ConversationDiskCache tier;
        check(tier.open(parent.string(), error), "peak admission tier opens");
        const fs::path kept = fs::path(tier.dir()) / "kept.bin";
        const fs::path incoming = fs::path(tier.dir()) / "incoming.bin";
        size_t kept_bytes = 0;
        check(session_file_write(kept.string(), img, bound, kept_bytes, error), "peak admission seed writes");
        check(kept_bytes < 4096 && kept_bytes % 4096 != 0, "the small image needs final-block padding");
        tier.admit({t1, true, img.live.ids, img.live.imgs, (uint64_t) kept_bytes, "kept.bin"});

        uint64_t asked = 0, budget = 0;
        size_t admissions = 0, transfers = 0, faults = 0;
        SessionWriteOptions opt;
        opt.admit = [&](uint64_t need, std::string& why) {
            asked = need;
            ++admissions;
            for (const auto& e : fs::directory_iterator(tier.dir()))
                check(e.path().extension() != ".tmp", "admission precedes temporary creation");
            if (tier.make_room(need)) return true;
            why = "the storage tier cannot hold the write peak";
            return false;
        };
        opt.progress = [&](uint64_t done, uint64_t total) {
            ++transfers;
            check(done <= total, "transfer progress counts exact serialized bytes");
            uint64_t footprint = 0;
            for (const auto& e : fs::directory_iterator(tier.dir())) footprint += e.file_size();
            check(footprint <= budget, "real temporary and retained files stay within the admitted budget");
        };
        opt.fault = [&](const char*) { ++faults; return 0; };
        for (uint64_t limit : {uint64_t(kept_bytes), uint64_t(4095)}) {
            budget = limit;
            tier.set_budget(budget);
            admissions = transfers = faults = 0;
            size_t incoming_bytes = 0;
            SessionStatus st;
            check(!session_file_write(incoming.string(), img, bound, incoming_bytes, error, opt, &st) &&
                  st.error == SessionError::storage && !st.published && incoming_bytes == 0,
                  "a budget fitting the final file but not the padded peak refuses the real writer");
            check(admissions == 1 && asked == 4096 && transfers == 0 && faults == 0,
                  "peak refusal happens before temporary creation or any write step");
            check(!fs::exists(incoming) && fs::file_size(kept) == kept_bytes &&
                  tier.size() == 1 && tier.bytes() == kept_bytes && tier.evictions() == 0 &&
                  std::distance(fs::directory_iterator(tier.dir()), fs::directory_iterator()) == 1,
                  "an over-budget peak writes nothing and does not evict a retained conversation");
        }

        budget = kept_bytes + 4096;
        tier.set_budget(budget);
        admissions = transfers = 0;
        size_t incoming_bytes = 0;
        SessionStatus st;
        check(session_file_write(incoming.string(), img, bound, incoming_bytes, error, opt, &st) &&
              st.published && admissions == 1 && asked == 4096 && transfers == 2,
              "the retained file plus the exact padded peak fits at equality");
        check(incoming_bytes == kept_bytes && fs::file_size(incoming) == kept_bytes,
              "the published file and returned bytes exclude temporary padding");
        tier.admit({t1, true, img.live.ids, img.live.imgs, (uint64_t) incoming_bytes, "incoming.bin"});
        check(tier.bytes() == 2 * kept_bytes && tier.evictions() == 0,
              "the index charges only exact published bytes, not the admitted peak");

        // Admission may evict first; a later write failure does not roll those evictions back.
        budget = 4096;
        tier.set_budget(budget);
        opt.fault = [](const char* step) { return std::string(step) == "rename" ? EIO : 0; };
        const fs::path failed = fs::path(tier.dir()) / "failed.bin";
        check(!session_file_write(failed.string(), img, bound, incoming_bytes, error, opt, &st) &&
              st.error == SessionError::io && !st.published && tier.evictions() == 2 &&
              tier.size() == 0 && tier.bytes() == 0 &&
              fs::directory_iterator(tier.dir()) == fs::directory_iterator(),
              "post-admission failure cleans its temporary but does not restore evicted entries");

        SavedConversation aligned = img;
        aligned.live.gdn.resize(aligned.live.gdn.size() + 4096 - kept_bytes, 7);
        opt.fault = {};
        admissions = transfers = 0;
        const fs::path boundary = fs::path(tier.dir()) / "boundary.bin";
        check(session_file_write(boundary.string(), aligned, bound, incoming_bytes, error, opt, &st) &&
              st.published && admissions == 1 && asked == 4096 && incoming_bytes == 4096 &&
              fs::file_size(boundary) == 4096 && transfers == 2,
              "an already aligned file is admitted at its exact boundary without another page");
        tier.admit({t1, true, aligned.live.ids, aligned.live.imgs, (uint64_t) incoming_bytes, "boundary.bin"});
        check(tier.bytes() == budget, "the aligned publication charges exactly its budget");
        tier.close();
    }

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
    // An unreadable/corrupt entry must stop matching even when unlink fails.
    const auto blocked = fs::path(first.dir()) / "blocked.bin";
    fs::create_directory(blocked);
    { std::ofstream f(blocked / "child"); f << "prevents directory removal"; }
    first.admit({t1, true, img.live.ids, img.live.imgs, 8, "blocked.bin"});
    check(!first.remove(0) && first.bytes() == 8 && first.size() == 1,
          "failed unlink retains the owned footprint charge");
    check(first.best(prompt, {}, true, t1).tokens == 0,
          "failed unlink cannot offer invalidated state again");
    fs::remove(blocked / "child");
    check(first.remove(0) && first.bytes() == 0, "invalidated footprint can be reclaimed later");
    second.admit({t1, true, img.live.ids, img.live.imgs, 4, "park-1.bin"});
    { std::ofstream f(fs::path(second.dir()) / "park-1.bin"); f << "other"; }
    first.close();
    second.close();
    check(!fs::exists(first.dir()) && !fs::exists(second.dir()), "each tier's own subdirectory goes with it");
    check(fs::exists(bystander), "the user's file in the parent stays");

    std::filesystem::remove_all(parent);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
}
