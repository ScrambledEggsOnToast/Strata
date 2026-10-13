#include "strata/program/readout_reference.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

int main() {
    namespace ref = strata::program::readout_reference;
    const auto dir = std::filesystem::temp_directory_path() / ("strata-reference-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    const std::string path = (dir / "readout.json").string();
    std::string error;
    std::vector<float> row(1024, -10000.0f);
    row[19] = 10000.0f;
    std::vector<int64_t> selected;
    for (int64_t id = 800; id > 500; --id) selected.push_back(id);
    CHECK(ref::write(path, row.data(), row.size(), 7, selected, error));
    std::ifstream file(path);
    const std::string json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    CHECK(json.find("\"position\":7,\"consumed_prompt_tokens\":7,\"row_position\":6") != std::string::npos);
    CHECK(json.find("\"generated_tokens\":0,\"sampling\":false,\"mtp\":false") != std::string::npos);
    CHECK(json.find("\"selected_token_ids\":[800,799,798") != std::string::npos);
    CHECK(json.find(",501]") != std::string::npos); // 300 selected entries, not a 256-token top-k subset
    CHECK(json.find("\"vocab_logsumexp\":10000") != std::string::npos);
    CHECK(json.find("\"log_probs\":[-20000,-20000") != std::string::npos);
    CHECK(!ref::write(path, row.data(), row.size(), 7, selected, error)); // create-only
    const std::string refused = (dir / "refused.json").string();
    selected.push_back(1024);
    CHECK(!ref::write(refused, row.data(), row.size(), 7, selected, error));
    CHECK(!std::filesystem::exists(refused)); // invalid late target cannot yield a partial set
    selected.pop_back();
    row[0] = std::numeric_limits<float>::quiet_NaN(); // unselected vocabulary entry still poisons the row
    CHECK(!ref::write(refused, row.data(), row.size(), 7, selected, error));
    CHECK(!std::filesystem::exists(refused));
    CHECK(!ref::valid_ids({}, row.size(), error));
    CHECK(!ref::valid_ids({-1}, row.size(), error));
    CHECK(!ref::valid_ids({1}, 0, error));
    CHECK(!ref::write(refused, nullptr, row.size(), 7, selected, error));
    CHECK(!ref::write(refused, row.data(), row.size(), 0, selected, error));
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
