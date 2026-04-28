#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>
#include <set>

using json = nlohmann::json;

using Pair = std::pair<std::string, std::string>;

struct PairHash {
    std::size_t operator()(const Pair& p) const {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

#include <vector>
#include <string>
#include <cstdint>

// 코드포인트 cp를 UTF-8 바이트열로 인코딩해 std::string에 담아 반환
static std::string utf8_encode(uint32_t cp) {
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

// 0~255 → UTF-8로 인코딩된 유니코드 문자열 (멀티바이트일 수 있음)
std::vector<std::string> build_byte_to_unicode() {
    std::vector<int> bs;
    // 1단계: 그대로 매핑되는 범위
    for (int b = 33;  b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;  // 매핑되는 코드포인트 (자기 자신)

    // 2단계: 빠진 바이트들을 256+n으로 채움
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool present = false;
        for (int x : bs) { if (x == b) { present = true; break; } }
        if (!present) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    // bs[i] → cs[i] 의 매핑을 byte → UTF-8 string 배열로 변환
    std::vector<std::string> table(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        table[bs[i]] = utf8_encode(static_cast<uint32_t>(cs[i]));
    }
    return table;
}

int main() {
    // vocab load
    std::ifstream f1("vocab.json");
    if (!f1) {
        std::cerr << "vocab.json open failed\n";
        return 1;
    }

    json j;
    f1 >> j;  // 파일 전체를 파싱

    // JSON object → unordered_map<string, int>
    std::unordered_map<std::string, int> vocab;
    vocab.reserve(j.size());
    for (const auto& [token, id] : j.items()) {
        vocab[token] = id.get<int>();
    }

    // sanity check
    std::cout << "vocab size: " << vocab.size() << '\n';
    std::cout << "Ġthe       -> " << vocab["\xc4\xa0" "the"] << '\n';
    std::cout << "Ġof        -> " << vocab["\xc4\xa0" "of"]  << '\n';
    std::cout << "Ġand       -> " << vocab["\xc4\xa0" "and"] << '\n';
    // std::cout << "Ġthe       -> " << vocab["Ġthe"] << '\n';
    // std::cout << "Ġof        -> " << vocab["Ġof"]  << '\n';
    // std::cout << "Ġand       -> " << vocab["Ġand"] << '\n';
    std::cout << "<|endoftext|> -> " << vocab["<|endoftext|>"] << '\n';

    std::ifstream f("merges.txt");
    if (!f) {
        std::cerr << "merges.txt open failed\n";
        return 1;
    }

    // merges load
    std::unordered_map<Pair, int, PairHash> bpe_ranks;
    bpe_ranks.reserve(50000);

    std::string line;
    int rank = 0;
    bool first_line = true;

    while (std::getline(f, line)) {
        if (first_line) {
            first_line = false;
            if (!line.empty() && line[0] == '#') continue; // 헤더 스킵
        }
        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // 공백 split
        std::istringstream iss(line);
        std::string right, left;
        if (!(iss >> left >> right)) continue;  // 비정상 라인 스킵

        bpe_ranks[{left, right}] = rank++;
    }

    std::cout << "merges loaded: " << bpe_ranks.size() << '\n';
    std::cout << "rank of (Ġ, t):  " << bpe_ranks[{"\xc4\xa0", "t"}] << '\n';
    std::cout << "rank of (Ġ, a):  " << bpe_ranks[{"\xc4\xa0", "a"}] << '\n';
    std::cout << "rank of (h, e):  " << bpe_ranks[{"h", "e"}]    << '\n';

    auto enc = build_byte_to_unicode();

    assert(enc[' '] == "\xc4\xa0");   // space → Ġ
    assert(enc['A'] == "A");          // ASCII printable → 자기 자신
    assert(enc['\n'] == "\xc4\x8a");  // LF → Ċ
    assert(enc[0x00] == "\xc4\x80");  // NUL → Ā

    std::set<std::string> seen;
    for (const auto& s : enc) seen.insert(s);
    assert(seen.size() == 256);

    return 0;
}