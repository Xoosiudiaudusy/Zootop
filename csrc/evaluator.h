// Port of negpluribus/evaluator.py: 5..7 card evaluator, higher = better, and the returned
// integer is *identical* to the pure-Python value (category * 15^5 + five packed ranks), so
// the two implementations are interchangeable anywhere a value is stored or compared.
#pragma once
#include <cstdint>
#include <stdexcept>

namespace negp {

enum HandCategory {
    HIGH_CARD = 0, PAIR = 1, TWO_PAIR = 2, TRIPS = 3, STRAIGHT = 4,
    FLUSH = 5, FULL_HOUSE = 6, QUADS = 7, STRAIGHT_FLUSH = 8,
};

namespace detail {
constexpr int64_t EVAL_BASE = 15;

inline int64_t encode(int cat, const int* ranks, int n) {
    int64_t v = cat;
    for (int i = 0; i < 5; i++) {
        int r = i < n ? ranks[i] : 0;
        v = v * EVAL_BASE + r;
    }
    return v;
}

inline int straight_high(int mask) {
    int m = (mask << 1) | ((mask >> 12) & 1);
    for (int t = 13; t > 3; t--) {
        if (((m >> (t - 4)) & 0x1F) == 0x1F) return t - 1;
    }
    return -1;
}
}  // namespace detail

inline int64_t evaluate(const int* cards, int n) {
    using namespace detail;
    if (n < 5 || n > 7) throw std::invalid_argument("evaluate expects 5..7 cards");
    int counts[13] = {0};
    int suit_counts[4] = {0};
    int suit_masks[4] = {0};
    int mask = 0;
    for (int i = 0; i < n; i++) {
        int c = cards[i];
        int r = c >> 2, s = c & 3;
        counts[r]++;
        suit_counts[s]++;
        suit_masks[s] |= 1 << r;
        mask |= 1 << r;
    }
    int64_t flush_value = -1;
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] >= 5) {
            int sh = straight_high(suit_masks[s]);
            if (sh >= 0) { int r[1] = {sh}; return encode(STRAIGHT_FLUSH, r, 1); }
            int fm = suit_masks[s];
            int top[5]; int k = 0;
            for (int r = 12; r >= 0 && k < 5; r--) if ((fm >> r) & 1) top[k++] = r;
            flush_value = encode(FLUSH, top, 5);
            break;
        }
    }
    int quads = -1, trips = -1;
    int pairs[3]; int npairs = 0;
    for (int r = 12; r >= 0; r--) {
        int c = counts[r];
        if (c == 4) quads = r;
        else if (c == 3) { if (trips < 0) trips = r; else pairs[npairs++] = r; }
        else if (c == 2) pairs[npairs++] = r;
    }
    if (quads >= 0) {
        int kicker = -1;
        for (int r = 12; r >= 0; r--) if (counts[r] && r != quads) { kicker = r; break; }
        int rr[2] = {quads, kicker};
        return encode(QUADS, rr, 2);
    }
    if (trips >= 0 && npairs > 0) { int rr[2] = {trips, pairs[0]}; return encode(FULL_HOUSE, rr, 2); }
    if (flush_value >= 0) return flush_value;
    int sh = straight_high(mask);
    if (sh >= 0) { int rr[1] = {sh}; return encode(STRAIGHT, rr, 1); }
    if (trips >= 0) {
        int rr[3] = {trips, 0, 0}; int k = 1;
        for (int r = 12; r >= 0 && k < 3; r--) if (counts[r] && r != trips) rr[k++] = r;
        return encode(TRIPS, rr, k);
    }
    if (npairs >= 2) {
        int p1 = pairs[0], p2 = pairs[1];
        int kicker = -1;
        for (int r = 12; r >= 0; r--) if (counts[r] && r != p1 && r != p2) { kicker = r; break; }
        int rr[3] = {p1, p2, kicker};
        return encode(TWO_PAIR, rr, 3);
    }
    if (npairs == 1) {
        int p = pairs[0];
        int rr[4] = {p, 0, 0, 0}; int k = 1;
        for (int r = 12; r >= 0 && k < 4; r--) if (counts[r] && r != p) rr[k++] = r;
        return encode(PAIR, rr, k);
    }
    int top[5]; int k = 0;
    for (int r = 12; r >= 0 && k < 5; r--) if (counts[r]) top[k++] = r;
    return encode(HIGH_CARD, top, k);
}

}  // namespace negp
