#include <tokenizer.hpp>
#include <nlohmann/json.hpp>

#include <vector>
#include <format>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

Tokenizer::Tokenizer(const std::string& vocab_path, const std::string& merges_path) {
    load_vocab(vocab_path);
    load_merges(merges_path);
    build_byte_encoder();
}

void Tokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f) {
        throw std::system_error(
            errno, std::generic_category(),
            std::format("failed to open vocab.json '{}'", vocab_path)
        );
    }

    json j;
    f >> j;

    vocab.reserve(j.size());
    for (const auto& [token, id] : j.items()) {
        vocab[token] = id.get<int>();
    }
}

void Tokenizer::load_merges(const std::string& merges_path) {
    std::ifstream f(merges_path);
    if (!f) {
        throw std::system_error(
            errno, std::generic_category(),
            std::format("failed to open merges.txt '{}'", merges_path)
        );
    }

    bpe_rank.reserve(50000);

    std::string line;
    int rank = 0;
    bool first_line = true;

    while (std::getline(f, line)) {
        if (first_line) {
            first_line = false;
            if (!line.empty() && line[0] == '#') continue;
        }
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::istringstream iss(line);
        std::string right, left;
        if(!(iss >> left >> right)) continue;

        bpe_rank[{left, right}] = rank++;
    }
}

void Tokenizer::build_byte_encoder() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);

    std::vector<int> cs = bs;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool present = false;
        for (int x : bs) { if (x == b) { present = true; break; } }
        if (!present) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    unicode_table.resize(256);
    for (size_t i = 0; i < bs.size(); i++) {
        unicode_table[bs[i]] = utf8_encode(static_cast<uint32_t>(cs[i]));
    }
}

std::string Tokenizer::utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::vector<std::string> utf8_split(const std::string& s) {
    std::vector<std::string> out;
    // out.reserve(s.length());
    for (size_t i = 0; i < s.size(); ) {
        uint8_t first = static_cast<uint8_t>(s[i]);
        size_t len;
        if      ((first & 0x80) == 0)    len = 1;
        else if ((first & 0xE0) == 0xC0) len = 2;
        else if ((first & 0xF0) == 0xE0) len = 3;
        else if ((first & 0xF8) == 0xF0) len = 4;
        else {
            throw std::invalid_argument(
                std::format("invalid UTF-8: unexpected continuation byte 0x{:02X} at offset {}", 
                        first, i)
            );
        }

        if (i + len > s.size()) {
        throw std::invalid_argument(
            std::format("invalid UTF-8: truncated {}-byte sequence at offset {} (size={})",
                    len, i, s.size()));
        }
        out.push_back(s.substr(i, len));
        i = i + len;
    }        
    return out;
}
