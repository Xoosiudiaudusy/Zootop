// Exact potential-aware features (flop and turn): the histogram of the equity after each possible
// next card, with every equity computed exactly instead of by Monte-Carlo runouts.
//
//   turn (hole, board4):  for each of the 46 river cards c (card order): e_c = river_equity_exact(hole,
//                         board4 + c) (all 990 opponent holes); histogram of the 46 values, mean.
//   flop (hole, board3):  for each of the 47 turn cards t (card order): e_t = the mean over the 46
//                         river cards r (card order) of river_equity_exact(hole, board3 + t + r);
//                         histogram of the 47 values, mean.
// Bin of a value e: min(int(e * bins), bins - 1), as in potential_histogram().  Everything is a pure
// function of the cards and invariant under suit relabelling.
//
// Two implementations with the same numbers (the same sums in the same order, tests):
//   * exact_feature():        one hand, by the definition (reference; used for fits and checks);
//   * ExactFeatureBatch:      every hand of one flop or turn board at once.  The river equity of all
//                             1,081 holes of a 5-card board comes from one sorted pass (river_equity_all:
//                             won = holes below + half the ties, minus those sharing a card), so a flop
//                             board costs C(49,2) = 1,176 such passes and a turn board 48.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "evaluator.h"
#include "handindex.h"

namespace negp {

// river_equity_exact for every hole of a 5-card board: out[a * 52 + b] (a != b, both off the board)
// = the same double river_equity_exact(hole={a,b}, board) returns.  `str` and `all` are scratch.
inline void river_equity_all(const int* board, double* out, std::vector<int64_t>& str, std::vector<int64_t>& all) {
    bool on[52] = {false};
    for (int i = 0; i < 5; i++) on[board[i]] = true;
    int rest[52];
    int nr = 0;
    for (int c = 0; c < 52; c++) if (!on[c]) rest[nr++] = c;
    str.assign(52 * 52, 0);
    all.clear();
    int cards[7];
    for (int i = 0; i < 5; i++) cards[2 + i] = board[i];
    for (int i = 0; i < nr; i++)
        for (int j = i + 1; j < nr; j++) {
            cards[0] = rest[i];
            cards[1] = rest[j];
            const int64_t s = evaluate(cards, 7);
            str[(size_t)(rest[i] * 52 + rest[j])] = str[(size_t)(rest[j] * 52 + rest[i])] = s;
            all.push_back(s);
        }
    std::sort(all.begin(), all.end());
    const double n_opp = (double)((nr - 2) * (nr - 3) / 2);  // 990
    for (int i = 0; i < nr; i++)
        for (int j = i + 1; j < nr; j++) {
            const int a = rest[i], b = rest[j];
            const int64_t s = str[(size_t)(a * 52 + b)];
            long long lower = std::lower_bound(all.begin(), all.end(), s) - all.begin();
            long long equal = (std::upper_bound(all.begin(), all.end(), s) - all.begin()) - lower;
            for (int k = 0; k < nr; k++) {
                const int y = rest[k];
                if (y != a) {
                    const int64_t t = str[(size_t)(a * 52 + y)];
                    if (t < s) lower--;
                    else if (t == s) equal--;
                }
                if (y != b) {
                    const int64_t t = str[(size_t)(b * 52 + y)];
                    if (t < s) lower--;
                    else if (t == s) equal--;
                }
            }
            equal++;  // (a, b) itself was removed twice
            // river_equity_exact adds 1.0 / 0.5 per opponent: an exact multiple of 0.5, the same double
            const double e = ((double)lower + 0.5 * (double)equal) / n_opp;
            out[a * 52 + b] = out[b * 52 + a] = e;
        }
}

inline int exact_bin(double e, int bins) {
    int b = (int)(e * (double)bins);
    return b >= bins ? bins - 1 : b;
}

// the feature of one hand by the definition: counts[bins] and the mean (reference implementation)
inline void exact_feature(const int* hole, const int* board, int n_board, int bins, int* counts, double& mean) {
    bool used[52] = {false};
    used[hole[0]] = used[hole[1]] = true;
    for (int i = 0; i < n_board; i++) used[board[i]] = true;
    for (int i = 0; i < bins; i++) counts[i] = 0;
    int b5[5];
    for (int i = 0; i < n_board; i++) b5[i] = board[i];
    double total = 0.0;
    int n = 0;
    if (n_board == 4) {
        for (int c = 0; c < 52; c++) {
            if (used[c]) continue;
            b5[4] = c;
            const double e = river_equity_exact(hole, b5, 5);
            counts[exact_bin(e, bins)]++;
            total += e;
            n++;
        }
    } else {  // flop
        for (int t = 0; t < 52; t++) {
            if (used[t]) continue;
            b5[3] = t;
            double s = 0.0;
            int m = 0;
            for (int r = 0; r < 52; r++) {
                if (used[r] || r == t) continue;
                b5[4] = r;
                s += river_equity_exact(hole, b5, 5);
                m++;
            }
            const double e = s / (double)m;
            counts[exact_bin(e, bins)]++;
            total += e;
            n++;
        }
    }
    mean = total / (double)n;
}

// Features of every hole of one flop (3 cards) or turn (4 cards) board.  After compute(), counts(a, b)
// and mean(a, b) give the feature of hole {a, b} (off the board).
class ExactFeatureBatch {
public:
    explicit ExactFeatureBatch(int bins) : bins_(bins), eq_(52 * 52), acc_(52 * 52 * 52) {}

    void compute(const int* board, int n_board) {
        n_board_ = n_board;
        for (int i = 0; i < n_board; i++) board_[i] = board[i];
        bool on[52] = {false};
        for (int i = 0; i < n_board; i++) on[board[i]] = true;
        int b5[5];
        for (int i = 0; i < n_board; i++) b5[i] = board[i];
        if (n_board == 4) {
            // acc_[c][hole] = equity of the hole on board + c
            for (int c = 0; c < 52; c++) {
                if (on[c]) continue;
                b5[4] = c;
                river_equity_all(b5, eq_.data(), str_, all_);
                std::copy(eq_.begin(), eq_.end(), acc_.begin() + (size_t)c * 52 * 52);
            }
        } else {
            // acc_[t][hole] = sum over rivers r (card order) of the equity on board + t + r; the pair
            // loop visits, for every t, its rivers in increasing order (pairs (r, t) with r < t first,
            // outer index increasing, then (t, r) with r > t), the order exact_feature() sums in
            std::fill(acc_.begin(), acc_.end(), 0.0);
            for (int t = 0; t < 52; t++) {
                if (on[t]) continue;
                for (int r = t + 1; r < 52; r++) {
                    if (on[r]) continue;
                    b5[3] = t;
                    b5[4] = r;
                    river_equity_all(b5, eq_.data(), str_, all_);
                    double* at = &acc_[(size_t)t * 52 * 52];
                    double* ar = &acc_[(size_t)r * 52 * 52];
                    for (int a = 0; a < 52; a++) {
                        if (on[a] || a == t || a == r) continue;
                        for (int b = a + 1; b < 52; b++) {
                            if (on[b] || b == t || b == r) continue;
                            const double e = eq_[(size_t)(a * 52 + b)];
                            at[a * 52 + b] += e;
                            ar[a * 52 + b] += e;
                        }
                    }
                }
            }
        }
        on_ = std::vector<bool>(on, on + 52);
    }

    // the feature of hole {a, b}, a < b, both off the board
    void feature(int a, int b, int* counts, double& mean) const {
        if (a > b) std::swap(a, b);
        for (int i = 0; i < bins_; i++) counts[i] = 0;
        double total = 0.0;
        int n = 0;
        for (int c = 0; c < 52; c++) {
            if (on_[c] || c == a || c == b) continue;
            double e = acc_[(size_t)c * 52 * 52 + (size_t)(a * 52 + b)];
            if (n_board_ == 3) e = e / (double)(52 - 3 - 1 - 2);  // mean over the 46 rivers
            counts[exact_bin(e, bins_)]++;
            total += e;
            n++;
        }
        mean = total / (double)n;
    }

private:
    int bins_;
    int n_board_ = 0;
    int board_[5];
    std::vector<bool> on_;
    std::vector<double> eq_, acc_;
    std::vector<int64_t> str_, all_;
};

// the canonical k-card boards (k = 3 or 4): the lexicographically smallest sorted image under the 24
// suit permutations; every board is a relabelling of exactly one of them
inline std::vector<std::vector<int>> canonical_small_boards(int k) {
    std::vector<std::vector<int>> out;
    std::vector<int> b(k);
    std::vector<int> q(k);
    const auto rec = [&](auto&& self, int i, int start) -> void {
        if (i == k) {
            for (int p = 1; p < 24; p++) {
                for (int j = 0; j < k; j++) q[j] = (b[j] >> 2) * 4 + SUIT_PERMS[p][b[j] & 3];
                std::sort(q.begin(), q.end());
                if (std::lexicographical_compare(q.begin(), q.end(), b.begin(), b.end())) return;
            }
            out.push_back(b);
            return;
        }
        for (int c = start; c < 52; c++) { b[i] = c; self(self, i + 1, c + 1); }
    };
    rec(rec, 0, 0);
    return out;
}

// the exact feature of every flop (n_board 3) or turn (4) class, by class index (handindex.h)
struct ExactFeatureTable {
    int n_board = 0, bins = 0;
    std::vector<uint8_t> counts;  // [class][bins]
    std::vector<double> mean;     // [class]
};

inline void build_exact_features(int n_board, int bins, int threads, ExactFeatureTable& out, std::atomic<uint64_t>* done = nullptr) {
    if (n_board != 3 && n_board != 4) throw std::invalid_argument("exact features: flop (3) or turn (4)");
    if (bins < 1 || bins > 64) throw std::invalid_argument("exact features: bins 1..64");
    const HandIndexer ix(n_board);
    const uint64_t n = ix.size();
    out.n_board = n_board;
    out.bins = bins;
    out.counts.assign(n * (uint64_t)bins, 0);
    out.mean.assign(n, -1.0);
    const std::vector<std::vector<int>> boards = canonical_small_boards(n_board);
    std::atomic<size_t> next{0};
    auto work = [&]() {
        ExactFeatureBatch batch(bins);
        int counts[64];
        for (size_t i = next.fetch_add(1); i < boards.size(); i = next.fetch_add(1)) {
            const int* board = boards[i].data();
            batch.compute(board, n_board);
            bool on[52] = {false};
            for (int k = 0; k < n_board; k++) on[board[k]] = true;
            uint64_t written = 0;
            for (int a = 0; a < 52; a++) {
                if (on[a]) continue;
                for (int b = a + 1; b < 52; b++) {
                    if (on[b]) continue;
                    const int hole[2] = {a, b};
                    const uint64_t id = ix.index(hole, board);
                    double m;
                    batch.feature(a, b, counts, m);
                    // a class met twice gets the same numbers (pure function of the class): benign race
                    for (int j = 0; j < bins; j++) out.counts[id * (uint64_t)bins + (uint64_t)j] = (uint8_t)counts[j];
                    out.mean[id] = m;
                    written++;
                }
            }
            if (done) done->fetch_add(written, std::memory_order_relaxed);
        }
    };
    const int T = threads < 1 ? 1 : threads;
    std::vector<std::thread> pool;
    for (int t = 1; t < T; t++) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    for (uint64_t id = 0; id < n; id++)
        if (out.mean[id] < 0.0) throw std::logic_error("exact features: a class was not covered by the canonical boards");
}

}  // namespace negp
