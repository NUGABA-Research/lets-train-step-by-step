#pragma once
#include <nlohmann/json.hpp>

#include <vector>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

using Pair = std::pair<std::string, std::string>;

struct PairHash {
    std::size_t operator() (const Pair& p) const {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class Tokenizer {
    public:
        Tokenizer(const std::string& vocab_path, 
            const std::string& merges_path);
        ~Tokenizer()=default;
        std::vector<int> encode(const std::string& text) const;
        std::string decode(const std::vector<int>& ids )const;

    private:
        void load_vocab(const std::string& vocab_path);
        void load_merges(const std::string& merges_path);
        void build_byte_encoder();
        static std::string utf8_encode(uint32_t cp);
        static std::vector<std::string> utf8_split(const std::string& s);
        std::string byte_encode(const std::string& word) const;
        std::vector<std::string> bpe(const std::string& encoded) const;
        std::vector<std::string> pretokenize(const std::string& text) const;

    private:
        std::vector<std::string> unicode_table;
        std::unordered_map<std::string, int> vocab;
        std::unordered_map<Pair, int, PairHash> bpe_rank;
        std::vector<std::string> id_to_token;
        std::unordered_map<std::string, unsigned char> byte_decoder;
};