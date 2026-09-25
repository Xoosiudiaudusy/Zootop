// Standalone check of handindex.h against canonical_form():
//   g++ -O2 -std=c++17 -I csrc csrc/tests/test_handindex.cpp -o build/test_handindex && build/test_handindex [full]
// 1. sizes = Waugh's class counts;  2. random hands: unindex(index(h)) has h's canonical form,
// perm_of maps h onto that representative;  3. "full": every flop hand, the index <-> canonical
// form map is a bijection onto [0, size).
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "abstraction.h"
#include "handindex.h"

namespace negp {
const int SUIT_PERMS[24][4] = {
    {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
    {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
    {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
    {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0},
};
}
using namespace negp;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails++ < 10) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

static bool same_form(const CanonicalForm& a, const CanonicalForm& b) {
    if (a.n_board != b.n_board || a.hole[0] != b.hole[0] || a.hole[1] != b.hole[1]) return false;
    for (int i = 0; i < a.n_board; i++) if (a.board[i] != b.board[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    const uint64_t expect[6] = {0, 0, 0, 1286792ULL, 13960050ULL, 123156254ULL};
    std::mt19937_64 rng(12345);
    for (int nb = 3; nb <= 5; nb++) {
        HandIndexer ix(nb);
        CHECK(ix.size() == expect[nb], "n_board %d: size %llu, expected %llu", nb, (unsigned long long)ix.size(), (unsigned long long)expect[nb]);
        for (int t = 0; t < 300000; t++) {
            int deck[52];
            for (int i = 0; i < 52; i++) deck[i] = i;
            for (int i = 0; i < 2 + nb; i++) { int j = i + (int)(rng() % (uint64_t)(52 - i)); std::swap(deck[i], deck[j]); }
            const int* hole = deck; const int* board = deck + 2;
            const uint64_t id = ix.index(hole, board);
            CHECK(id < ix.size(), "index out of range");
            int rh[2], rb[5];
            ix.unindex(id, rh, rb);
            CHECK(ix.index(rh, rb) == id, "index(unindex(i)) != i");
            CanonicalForm a, b;
            canonical_form(hole, board, nb, a);
            canonical_form(rh, rb, nb, b);
            CHECK(same_form(a, b), "representative has another canonical form (n_board %d)", nb);
            int perm[4];
            ix.perm_of(hole, board, perm);
            bool in_rep[52] = {false};
            for (int i = 0; i < 2; i++) in_rep[rh[i]] = true;
            for (int i = 0; i < nb; i++) in_rep[rb[i]] = true;
            bool ok = true;
            for (int i = 0; i < 2; i++) { int c = (hole[i] >> 2) * 4 + perm[hole[i] & 3]; ok = ok && in_rep[c] && (c == rh[0] || c == rh[1]); }
            for (int i = 0; i < nb; i++) { int c = (board[i] >> 2) * 4 + perm[board[i] & 3]; bool on = false; for (int j = 0; j < nb; j++) on = on || rb[j] == c; ok = ok && on; }
            CHECK(ok, "perm_of does not map the hand onto its representative");
        }
        std::printf("n_board %d: size %llu, random checks done\n", nb, (unsigned long long)ix.size());
    }
    if (argc > 1 && std::strcmp(argv[1], "full") == 0) {
        HandIndexer ix(3);
        std::vector<uint64_t> pack(ix.size(), 0);
        uint64_t n = 0;
        for (int h0 = 0; h0 < 52; h0++) for (int h1 = h0 + 1; h1 < 52; h1++)
        for (int b0 = 0; b0 < 52; b0++) for (int b1 = b0 + 1; b1 < 52; b1++) for (int b2 = b1 + 1; b2 < 52; b2++) {
            if (b0 == h0 || b0 == h1 || b1 == h0 || b1 == h1 || b2 == h0 || b2 == h1) continue;
            int hole[2] = {h0, h1}, board[3] = {b0, b1, b2};
            CanonicalForm cf; canonical_form(hole, board, 3, cf);
            const uint64_t p = canonical_pack(cf), id = ix.index(hole, board);
            if (pack[id] == 0) pack[id] = p;
            CHECK(pack[id] == p, "two canonical forms share index %llu", (unsigned long long)id);
            n++;
        }
        uint64_t empty = 0;
        for (uint64_t p : pack) empty += p == 0;
        CHECK(empty == 0, "%llu indices never reached", (unsigned long long)empty);
        std::printf("full flop: %llu hands, bijection checked\n", (unsigned long long)n);
    }
    std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
