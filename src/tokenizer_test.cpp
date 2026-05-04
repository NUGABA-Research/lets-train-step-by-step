// tokenizer_test.cpp
//
// Goal:
//   Verify that your C++ Tokenizer reproduces GPT-2 token ids for known inputs.
//
// Expected public API in tokenizer.hpp:
//
//   class Tokenizer {
//   public:
//       Tokenizer(const std::string& vocab_path, const std::string& merges_path);
//       std::vector<int> encode(const std::string& text) const;
//   };
//
// Build example:
//
//   g++ -std=c++20 -O2 tokenizer.cpp tokenizer_test.cpp -I. -o tokenizer_test
//
// Run example:
//
//   ./tokenizer_test vocab.json merges.txt
//
// Notes:
//   - These expected ids are for the original GPT-2 tokenizer vocabulary/merges.
//   - If any test involving spaces fails, the most likely cause is missing GPT-2
//     pre-tokenization before byte-level BPE.

#include "tokenizer.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

static void print_ids(const std::vector<int>& ids) {
    std::cerr << "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cerr << ", ";
        std::cerr << ids[i];
    }
    std::cerr << "]";
}

static void expect_encode(
    const Tokenizer& tokenizer,
    const std::string& text,
    const std::vector<int>& expected
) {
    const std::vector<int> got = tokenizer.encode(text);

    if (got != expected) {
        std::cerr << "FAILED\n";
        std::cerr << "  text     : " << text << "\n";
        std::cerr << "  expected : ";
        print_ids(expected);
        std::cerr << "\n";
        std::cerr << "  got      : ";
        print_ids(got);
        std::cerr << "\n";
        std::exit(1);
    }

    std::cout << "PASS: \"" << text << "\" -> ";
    print_ids(got);
    std::cout << "\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <vocab.json> <merges.txt>\n";
        return 1;
    }

    const std::string vocab_path = argv[1];
    const std::string merges_path = argv[2];

    Tokenizer tokenizer(vocab_path, merges_path);

    try {
        // Empty input should produce no token ids.
        expect_encode(tokenizer, "", {});

        // Very common GPT-2 reference cases.
        expect_encode(tokenizer, "Hello", {15496});
        expect_encode(tokenizer, " world", {995});
        expect_encode(tokenizer, "Hello world", {15496, 995});
        expect_encode(tokenizer, "Hello, world!", {15496, 11, 995, 0});

        // Checks leading-space word pieces:
        // "This" + " is" + " a" + " test" + "."
        expect_encode(tokenizer, "This is a test.", {1212, 318, 257, 1332, 13});

        std::cout << "\nAll tokenizer tests passed.\n";
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << "\n";
    }

    std::string s = "Hello, this is GPT-2 from openAI";
    auto ids = tokenizer.encode(s);
    auto text = tokenizer.decode(ids);
    std::cout << text << std::endl;
}
