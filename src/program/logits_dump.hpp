#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace strata::program {
/// **THE DUMP'S ACCOUNTING IS STRICT: THE HEADER DESCRIBES THE FILE, OR THERE IS NO FILE.**  The header
/// promises `logits_selection::row_count(n_prompt - 1 + max_new, stride)` rows of `n_vocab` floats; every
/// writer goes through `write_row`, which counts, and `finish` compares the count at close.  Any path out of
/// `main` before that - a write failure, an early EOS, a CUDA fault after the header - destroys the guard,
/// which removes the file: rows that were not written must not be claimed (defect #57, enforced rather than
/// refused around).  The writers are the per-position token loop, the batched prompt path's head rows and the
/// verify windows' committed rows (#58).
struct LogitsDump {
    std::FILE* f = nullptr;
    std::string path;
    int64_t n_vocab = 0, expected_rows = 0, written_rows = 0;
    ~LogitsDump() {
        if (f == nullptr) return;
        std::fclose(f);
        std::fprintf(stderr, "strata generate: %s held %lld of the %lld logits rows the header claims and is "
                             "removed: the run ended before it produced every row\n",
                     path.c_str(), (long long) written_rows, (long long) expected_rows);
        std::remove(path.c_str());
    }
    /// One complete row, already on the host and checked finite by the caller.  false fails the run.
    bool write_row(const float* row, int64_t pos) {
        if (std::fwrite(row, sizeof(float), (size_t) n_vocab, f) != (size_t) n_vocab) {
            std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) pos);
            return false;
        }
        ++written_rows;
        return true;
    }
    /// The strict close: what the header claims is what is on disk, or the file goes and the run fails.
    bool finish() {
        if (f == nullptr) return true;
        std::FILE* d = f;
        f = nullptr;   // the destructor must not remove what finish() has judged
        if (written_rows != expected_rows) {
            std::fprintf(stderr, "strata generate: the logits dump holds %lld rows, the header claims %lld\n",
                         (long long) written_rows, (long long) expected_rows);
            std::fclose(d);
            std::remove(path.c_str());
            return false;
        }
        const bool flushed = std::fflush(d) == 0;
        std::error_code size_error;
        const auto bytes = std::filesystem::file_size(path, size_error);
        const auto expected_bytes = uint64_t{8} + uint64_t(expected_rows) * uint64_t(n_vocab) * sizeof(float);
        const bool closed = std::fclose(d) == 0;
        if (!flushed || !closed || size_error || bytes != expected_bytes) {
            std::fprintf(stderr, "strata generate: cannot finish logits dump with the declared byte count\n");
            std::remove(path.c_str());
            return false;
        }
        return true;
    }
};

} // namespace strata::program
