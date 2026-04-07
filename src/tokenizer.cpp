#include <map>
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem; 

using Pair = std::pair<std::string, std::string>;

auto most_frequent(const std::map<Pair, int>& counts) -> Pair {
    Pair best;
    int best_count = 0;
    for (auto& [pair, count] :  counts) {
        if (count > best_count) {
            best_count = count;
            best = pair;
        }
    }
    return best;
}

auto count_pairs(const std::vector<std::string>& tokens) -> std::map<Pair, int> {
    std::map<Pair, int> counts;
    for (std::size_t i = 0; i + 1 < tokens.size(); i++) {
        counts[{tokens[i], tokens[i + 1]}]++;
    }
    return counts;
}

auto merge(const std::vector<std::string>& tokens, const Pair& pair) -> std::vector<std::string> {
    std::vector<std::string> result;
    std::size_t i = 0;
    while (i < tokens.size()) {
        if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
            result.emplace_back(pair.first + pair.second);
            i += 2;
        } else {
            result.emplace_back(tokens[i]);
            i += 1;
        }
    }
    return result;
}

int main() {
    const fs::path filepath = "data.txt";
    if (!fs::exists(filepath)) {
        std::cerr << std::format("error: '{}' not found\n", filepath.string());
    }

    auto filesize = fs::file_size(filepath);
    std::cout << std::format("file: {}, size: {} bytes\n", filepath.string(),filesize);

    FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) {
        std::cerr << "error: fopen failed\n";
        return 1;
    }

    std::string buffer(filesize, '\0');
    std::size_t read_bytes = std::fread(buffer.data(), 1, filesize, fp);
    std::fclose(fp);
    
    std::cout << std::format("buffer: read {} bytes\n", read_bytes);

    auto preview_len = std::min<std::size_t>(read_bytes, 200);
    std::cout << std::format("--- preview (first {} bytes) ---\n", preview_len);
    std::cout << buffer.substr(0, preview_len) << "\n";

    for (auto i = 0; i < 10; i++) {
         std::cout << buffer[i];
    }
    std::cout << std:: endl;

    std::vector<std::string> tokens;
    for (char c : buffer) {
        tokens.emplace_back(std::string(1, c));
    }

    std::cout << std::format("initial tokens: {}\n", tokens.size());
    std::cout << "start tokenizing\n";

    const int num_merges = 200;
    
    std::vector<Pair> merge_rules;

    for (auto step = 0; step < num_merges; step++) {
        auto counts = count_pairs(tokens);
        if (counts.empty()) break;

        auto best = most_frequent(counts);
        if (counts[best] < 2) break;

        tokens = merge(tokens, best);
        merge_rules.push_back(best);
        std::cout << std::format(
            "merge {:2d}: ('{}', '{}') -> '{}' | freq={}, tokens={}\n",
            step + 1,
            best.first, best.second,
            best.first + best.second,
            counts[best],
            tokens.size()
        );
    }

    std::cout << "\n--- final tokens (first 50) ---\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(tokens.size(), 50); i++) {
        std::cout << std::format("[{}] ", tokens[i]);
    }
    std::cout << "\n";

    return 0;
}