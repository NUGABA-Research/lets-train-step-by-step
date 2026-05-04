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

std::vector<std::string> Tokenizer::utf8_split(const std::string& s) {
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

std::string Tokenizer::byte_encode(const std::string& word) const {
    std::string out;
    out.reserve(word.size() * 2);
    for (const auto& w : word) {
        out += unicode_table[static_cast<uint8_t>(w)];
    }
    return out;
}

std::vector<std::string> Tokenizer::bpe(const std::string& encoded) const {
    auto tokens = utf8_split(encoded);
    if (tokens.size() < 2) return tokens;

    std::vector<std::string> next;
    next.reserve(tokens.size());

    while (true) {
        int best_rank = INT_MAX;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto it = bpe_rank.find({tokens[i], tokens[i+1]});
            if (it != bpe_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx == SIZE_MAX) break;

        std::string best_left = tokens[best_idx];
        std::string best_right = tokens[best_idx + 1];

        next.clear();
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() &&
                tokens[i] == best_left && tokens[i+1] == best_right) {
                next.push_back(tokens[i] + tokens[i+1]);
                i += 2;
            } else {
                next.push_back(tokens[i]);
                i += 1;
            }
        }
        std::swap(tokens, next);

        if (tokens.size() == 1) break;
    }

    return tokens;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> parts;

    size_t i = 0;
    const size_t n = text.size();

    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' ||
               c == '\r' || c == '\f' || c == '\v';
    };

    auto is_letter = [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    };

    auto is_digit = [](unsigned char c) {
        return c >= '0' && c <= '9';
    };

    while (i < n) {
        size_t start = i;

        // GPT-2 contraction patterns: 's, 't, 're, 've, 'm, 'll, 'd
        if (text[i] == '\'') {
            if (i + 2 <= n &&
                (text.compare(i, 2, "'s") == 0 ||
                 text.compare(i, 2, "'t") == 0 ||
                 text.compare(i, 2, "'m") == 0 ||
                 text.compare(i, 2, "'d") == 0)) {
                parts.push_back(text.substr(i, 2));
                i += 2;
                continue;
            }

            if (i + 3 <= n &&
                (text.compare(i, 3, "'re") == 0 ||
                 text.compare(i, 3, "'ve") == 0 ||
                 text.compare(i, 3, "'ll") == 0)) {
                parts.push_back(text.substr(i, 3));
                i += 3;
                continue;
            }
        }

        // Optional one leading space before letters/numbers/punctuation
        bool has_leading_space = false;
        if (text[i] == ' ' && i + 1 < n) {
            unsigned char next = static_cast<unsigned char>(text[i + 1]);
            if (!is_space(next)) {
                has_leading_space = true;
                i++;
            }
        }

        if (i >= n) {
            parts.push_back(text.substr(start));
            break;
        }

        unsigned char c = static_cast<unsigned char>(text[i]);

        // ?\p{L}+
        if (is_letter(c)) {
            while (i < n && is_letter(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            parts.push_back(text.substr(start, i - start));
            continue;
        }

        // ?\p{N}+
        if (is_digit(c)) {
            while (i < n && is_digit(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            parts.push_back(text.substr(start, i - start));
            continue;
        }

        // ?[^\s\p{L}\p{N}]+
        if (!is_space(c)) {
            while (i < n) {
                unsigned char x = static_cast<unsigned char>(text[i]);
                if (is_space(x) || is_letter(x) || is_digit(x)) {
                    break;
                }
                i++;
            }
            parts.push_back(text.substr(start, i - start));
            continue;
        }

        // \s+
        while (i < n && is_space(static_cast<unsigned char>(text[i]))) {
            i++;
        }
        parts.push_back(text.substr(start, i - start));
    }

    return parts;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;

    if (text.empty()) {
        return ids;
    }

    std::vector<std::string> words = pretokenize(text);

    for (const std::string& word : words) {
        std::string encoded = byte_encode(word);
        std::vector<std::string> pieces = bpe(encoded);

        for (const std::string& piece : pieces) {
            auto it = vocab.find(piece);

            if (it == vocab.end()) {
                throw std::runtime_error(
                    "token not found in vocab: " + piece
                );
            }

            ids.push_back(it->second);
        }
    }

    return ids;
}
