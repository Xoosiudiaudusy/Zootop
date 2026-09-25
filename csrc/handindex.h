// Perfect index of a (hole, board) class under suit relabelling: a bijection between the
// canonical forms of abstraction.h (hole as a set, board as a set, suits permuted) and
// [0, size()).  Same equivalence as canonical_form(): two (hole, board) get one index iff
// canonical_form() gives them one form.  Class counts (Waugh 2013, "A Fast and Optimal Hand
// Isomorphism Algorithm", rounds {2, n}): flop 1,286,792, turn 13,960,050, river 123,156,254.
//
// Method (Waugh's, restricted to two rounds): every suit gets its configuration = the ranks it
// holds in the hole (A0) and on the board (A1), ranked within its size pattern (|A0|, |A1|).
// The four suits are sorted by (pattern, rank); the sorted pattern list selects a block of the
// index space, and inside the block every run of suits with one pattern contributes a multiset
// rank (combinations with repetition), mixed radix across runs.  Cost: a few popcounts and
// table lookups, no 24-permutation search.
//
// perm_of(): the suit permutation that maps a hand onto the representative index_to_hand()
// returns, so that per-board tables built on the representative can be reused.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace negp {

namespace hidx {

struct Binom {
    uint64_t c[64][8];
    Binom() {
        for (int n = 0; n < 64; n++)
            for (int k = 0; k < 8; k++) c[n][k] = k == 0 ? 1 : (n == 0 ? 0 : c[n - 1][k - 1] + c[n - 1][k]);
    }
};
inline const Binom& binom() {
    static const Binom b;
    return b;
}
// C(n, k) for k <= 7; n may be large (up to ~2^13 with k <= 4 without overflow)
inline uint64_t C(int64_t n, int k) {
    if (n < 0 || k < 0 || k > 7 || n < k) return 0;
    if (n < 64) return binom().c[n][k];
    uint64_t r = 1;
    for (int i = 0; i < k; i++) r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);  // exact at every step
    return r;
}

inline int ctz32(uint32_t m) {
    int n = 0;
    while (!(m & 1u)) { m >>= 1; n++; }
    return n;
}

inline int popcount13(uint32_t m) {
    int n = 0;
    while (m) { m &= m - 1; n++; }
    return n;
}

// colex rank of a set of ranks: sum over its elements b_0 < b_1 < ... of C(b_i, i + 1)
inline uint64_t colex(uint32_t mask) {
    uint64_t r = 0;
    int i = 0;
    while (mask) {
        int b = ctz32(mask);
        mask &= mask - 1;
        r += C(b, ++i);
    }
    return r;
}
inline uint32_t colex_unrank(uint64_t r, int m) {
    uint32_t mask = 0;
    for (int j = m; j >= 1; j--) {
        int b = j - 1;
        while (C(b + 1, j) <= r) b++;
        r -= C(b, j);
        mask |= 1u << b;
    }
    return mask;
}
// the ranks of `set` renumbered among the ranks outside `used` (both 13-bit masks)
inline uint32_t compress(uint32_t set, uint32_t used) {
    uint32_t out = 0;
    int k = 0;
    for (int r = 0; r < 13; r++) {
        if (used >> r & 1) continue;
        if (set >> r & 1) out |= 1u << k;
        k++;
    }
    return out;
}
inline uint32_t expand(uint32_t rel, uint32_t used) {
    uint32_t out = 0;
    int k = 0;
    for (int r = 0; r < 13; r++) {
        if (used >> r & 1) continue;
        if (rel >> k & 1) out |= 1u << r;
        k++;
    }
    return out;
}

}  // namespace hidx

class HandIndexer {
public:
    // n_board = 3 (flop), 4 (turn) or 5 (river); the hole is always 2 cards
    explicit HandIndexer(int n_board) : nb_(n_board) {
        if (n_board < 3 || n_board > 5) throw std::invalid_argument("HandIndexer: n_board must be 3..5");
        build();
    }
    int n_board() const { return nb_; }
    uint64_t size() const { return size_; }

    // index of (hole[0..1], board[0..nb-1]); cards 0..51, rank = c >> 2, suit = c & 3
    uint64_t index(const int* hole, const int* board) const {
        uint32_t a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        a0[hole[0] & 3] |= 1u << (hole[0] >> 2);
        a0[hole[1] & 3] |= 1u << (hole[1] >> 2);
        for (int i = 0; i < nb_; i++) a1[board[i] & 3] |= 1u << (board[i] >> 2);
        SuitKey k[4];
        for (int s = 0; s < 4; s++) k[s] = suit_key(a0[s], a1[s], s);
        sort4(k);
        return index_sorted(k);
    }

    // a representative (hole, board) of the class `idx` (the same for every member)
    void unindex(uint64_t idx, int* hole, int* board) const {
        if (idx >= size_) throw std::out_of_range("HandIndexer::unindex");
        const Config* cfg = &configs_[0];
        {   // last config with offset <= idx
            size_t lo = 0, hi = configs_.size();
            while (hi - lo > 1) {
                size_t mid = (lo + hi) / 2;
                if (configs_[mid].offset <= idx) lo = mid; else hi = mid;
            }
            cfg = &configs_[lo];
        }
        uint64_t rem = idx - cfg->offset;
        uint64_t vals[4];
        // mixed radix: the first run is the most significant
        uint64_t run_rank[4];
        for (int g = cfg->n_runs - 1; g >= 0; g--) {
            run_rank[g] = rem % cfg->run_size[g];
            rem /= cfg->run_size[g];
        }
        int pos = 0;
        for (int g = 0; g < cfg->n_runs; g++) {
            const int k = cfg->run_len[g];
            const uint64_t M = pattern_count(cfg->pat[pos]);
            uint64_t r = run_rank[g];
            for (int i = 1; i <= k; i++) {  // greedy unrank of a strictly decreasing w_i
                const int t = k - i + 1;
                // largest w in [t - 1, M - 1 + k - i] with C(w, t) <= r
                int64_t lo = t - 1, hi = (int64_t)M - 1 + (k - i);
                while (lo < hi) {
                    const int64_t mid = (lo + hi + 1) / 2;
                    if (hidx::C(mid, t) <= r) lo = mid; else hi = mid - 1;
                }
                const uint64_t w = (uint64_t)lo;
                r -= hidx::C((int64_t)w, t);
                vals[pos + i - 1] = w - (uint64_t)(k - i);
            }
            pos += k;
        }
        int nh = 0, nbd = 0;
        for (int s = 0; s < 4; s++) {
            const int p = cfg->pat[s];
            const int m0 = p >> 3, m1 = p & 7;
            const uint64_t inner = hidx::C(13 - m0, m1);
            const uint32_t A0 = hidx::colex_unrank(vals[s] / inner, m0);
            const uint32_t A1 = hidx::expand(hidx::colex_unrank(vals[s] % inner, m1), A0);
            for (int r = 0; r < 13; r++) {
                if (A0 >> r & 1) hole[nh++] = r * 4 + s;
                if (A1 >> r & 1) board[nbd++] = r * 4 + s;
            }
        }
    }

    // perm[old_suit] = suit in the representative, for a hand whose class representative is
    // unindex(index(hand)): representative card = rank * 4 + perm[card & 3]
    void perm_of(const int* hole, const int* board, int* perm) const {
        uint32_t a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
        a0[hole[0] & 3] |= 1u << (hole[0] >> 2);
        a0[hole[1] & 3] |= 1u << (hole[1] >> 2);
        for (int i = 0; i < nb_; i++) a1[board[i] & 3] |= 1u << (board[i] >> 2);
        SuitKey k[4];
        for (int s = 0; s < 4; s++) k[s] = suit_key(a0[s], a1[s], s);
        sort4(k);
        for (int i = 0; i < 4; i++) perm[k[i].suit] = i;
    }

private:
    struct SuitKey {
        uint64_t order;  // (pattern << 40) | value: sorted descending
        int pat;         // m0 * 8 + m1
        uint64_t val;    // rank of the suit's configuration within its pattern
        int suit;
    };
    struct Config {
        int pat[4];        // sorted descending
        int n_runs = 0;
        int run_len[4];
        uint64_t run_size[4];
        uint64_t offset = 0, size = 1;
    };

    static uint64_t pattern_count(int p) {
        const int m0 = p >> 3, m1 = p & 7;
        return hidx::C(13, m0) * hidx::C(13 - m0, m1);
    }
    SuitKey suit_key(uint32_t a0, uint32_t a1, int s) const {
        const int m0 = hidx::popcount13(a0), m1 = hidx::popcount13(a1);
        SuitKey k;
        k.pat = m0 * 8 + m1;
        k.val = hidx::colex(a0) * hidx::C(13 - m0, m1) + hidx::colex(hidx::compress(a1, a0));
        k.order = ((uint64_t)k.pat << 40) | k.val;
        k.suit = s;
        return k;
    }
    static void sort4(SuitKey* k) {
        // descending by order; ties (identical suits) keep suit order, which only affects perm_of
        auto gt = [](const SuitKey& a, const SuitKey& b) { return a.order > b.order || (a.order == b.order && a.suit < b.suit); };
        for (int i = 1; i < 4; i++) {
            SuitKey x = k[i];
            int j = i - 1;
            while (j >= 0 && gt(x, k[j])) { k[j + 1] = k[j]; j--; }
            k[j + 1] = x;
        }
    }
    static int config_key(const int* pat) { return ((pat[0] * 24 + pat[1]) * 24 + pat[2]) * 24 + pat[3]; }

    uint64_t index_sorted(const SuitKey* k) const {
        int pat[4] = {k[0].pat, k[1].pat, k[2].pat, k[3].pat};
        const Config& cfg = configs_[config_of_[config_key(pat)]];
        uint64_t idx = 0;
        int pos = 0;
        for (int g = 0; g < cfg.n_runs; g++) {
            const int len = cfg.run_len[g];
            uint64_t r = 0;
            for (int i = 1; i <= len; i++) r += hidx::C((int64_t)(k[pos + i - 1].val + (uint64_t)(len - i)), len - i + 1);
            idx = idx * cfg.run_size[g] + r;
            pos += len;
        }
        return cfg.offset + idx;
    }

    void build() {
        // every multiset of four patterns (m0, m1) with sum m0 = 2, sum m1 = nb
        std::vector<int> pats;
        for (int m0 = 2; m0 >= 0; m0--)
            for (int m1 = nb_; m1 >= 0; m1--) pats.push_back(m0 * 8 + m1);
        std::sort(pats.begin(), pats.end(), std::greater<int>());
        config_of_.assign(24 * 24 * 24 * 24, -1);
        const int P = (int)pats.size();
        for (int a = 0; a < P; a++)
            for (int b = a; b < P; b++)
                for (int c = b; c < P; c++)
                    for (int d = c; d < P; d++) {
                        int pat[4] = {pats[a], pats[b], pats[c], pats[d]};
                        int s0 = 0, s1 = 0;
                        for (int i = 0; i < 4; i++) { s0 += pat[i] >> 3; s1 += pat[i] & 7; }
                        if (s0 != 2 || s1 != nb_) continue;
                        Config cfg;
                        for (int i = 0; i < 4; i++) cfg.pat[i] = pat[i];
                        for (int i = 0; i < 4;) {
                            int j = i;
                            while (j < 4 && pat[j] == pat[i]) j++;
                            const int len = j - i;
                            const uint64_t M = pattern_count(pat[i]);
                            cfg.run_len[cfg.n_runs] = len;
                            cfg.run_size[cfg.n_runs] = hidx::C((int64_t)(M + len - 1), len);
                            cfg.size *= cfg.run_size[cfg.n_runs];
                            cfg.n_runs++;
                            i = j;
                        }
                        cfg.offset = size_;
                        size_ += cfg.size;
                        config_of_[config_key(pat)] = (int)configs_.size();
                        configs_.push_back(cfg);
                    }
    }
    int nb_;
    uint64_t size_ = 0;
    std::vector<Config> configs_;
    std::vector<int> config_of_;
};

}  // namespace negp
