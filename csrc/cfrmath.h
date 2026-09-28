// The arithmetic the CPU trainers and the GPU trainer (gpucfr.cu) share, written once so both sides
// compute the same doubles: CPython's float sum, regret matching, the payoff of a terminal.  Plain C++
// that nvcc also compiles for the device (NEGP_HD); nothing here may depend on the standard library
// beyond <cmath>-free integer and floating-point operations, and the device build must not contract
// a * b + c into an FMA (--fmad=false; the CPU builds use -ffp-contract=off, /fp:precise).
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define NEGP_HD __host__ __device__
#else
#define NEGP_HD
#endif

namespace negp {

constexpr int MAX_ACTIONS = 8;
constexpr int CFR_MAX_PLAYERS = 9;  // engine.h::MAX_PLAYERS (static_assert there)

// what one iteration of the flat trainers needs from its deal (flatcfr.h computes it on the CPU)
struct FlatIter {
    long long t;
    double weight;                        // Linear CFR weight
    int32_t button;
    int32_t bucket[CFR_MAX_PLAYERS][6];   // [absolute seat][n_board], -1 = not dealt
    int64_t strength[CFR_MAX_PLAYERS];    // 7-card values by relative seat
};

NEGP_HD inline double cfr_fabs(double x) { return x < 0.0 ? -x : x; }
NEGP_HD inline bool cfr_isfinite(double x) { return x - x == 0.0; }  // false for inf and nan

// CPython 3.12+ builtins.sum over a non-empty list of floats (start=0): the first item is
// added to the int 0 exactly, the rest with Neumaier compensation.
NEGP_HD inline double py_sum(const double* x, int n) {
    if (n == 0) return 0.0;
    double f = x[0];
    double c = 0.0;
    for (int i = 1; i < n; i++) {
        double xi = x[i];
        double t = f + xi;
        if (cfr_fabs(f) >= cfr_fabs(xi)) c += (f - t) + xi;
        else c += (xi - t) + f;
        f = t;
    }
    if (c != 0.0 && cfr_isfinite(c)) f += c;
    return f;
}

// the current strategy of regrets r[0..n) (regret matching, CPython sum() semantics)
NEGP_HD inline void regret_matching(const double* r, int n, double* out) {
    double pos[MAX_ACTIONS];
    for (int i = 0; i < n; i++) pos[i] = r[i] > 0 ? r[i] : 0.0;
    double s = py_sum(pos, n);
    if (s <= 0) { for (int i = 0; i < n; i++) out[i] = 1.0 / n; return; }
    for (int i = 0; i < n; i++) out[i] = pos[i] / s;
}

// net chips of relative seat `me` at a terminal (HandState::finish, in relative seats): `invested[r]`
// what relative seat r put in, bit r of `folded` set if it folded, `strength[r]` its 7-card value
// (read only when two or more reach showdown).  Side pots by contribution level; a split pot's odd
// chips go first to the seat after the button.
NEGP_HD inline int terminal_net_of(const int32_t* invested, uint16_t folded, int n, int me, const int64_t* strength) {
    int active[CFR_MAX_PLAYERS];
    int n_act = 0;
    for (int r = 0; r < n; r++) if (!(folded >> r & 1)) active[n_act++] = r;
    // the distinct positive contribution levels, increasing
    int levels[CFR_MAX_PLAYERS];
    int n_levels = 0;
    for (int r = 0; r < n; r++) {
        const int v = invested[r];
        if (v <= 0) continue;
        int k = 0;
        while (k < n_levels && levels[k] < v) k++;
        if (k < n_levels && levels[k] == v) continue;
        for (int j = n_levels; j > k; j--) levels[j] = levels[j - 1];
        levels[k] = v;
        n_levels++;
    }
    int won = 0, prev = 0;
    for (int li = 0; li < n_levels; li++) {
        const int lvl = levels[li];
        int portion = 0;
        for (int r = 0; r < n; r++) {
            const int x = (invested[r] < lvl ? invested[r] : lvl) - prev;
            portion += x > 0 ? x : 0;
        }
        int eligible[CFR_MAX_PLAYERS];
        int n_el = 0;
        for (int k = 0; k < n_act; k++) if (invested[active[k]] >= lvl) eligible[n_el++] = active[k];
        if (n_el == 0) for (int k = 0; k < n_act; k++) eligible[n_el++] = active[k];
        if (n_el == 1) {
            if (eligible[0] == me) won += portion;
        } else {
            int64_t best = -1;
            for (int k = 0; k < n_el; k++) if (strength[eligible[k]] > best) best = strength[eligible[k]];
            int ws[CFR_MAX_PLAYERS];
            int n_ws = 0;
            for (int k = 0; k < n_el; k++) if (strength[eligible[k]] == best) ws[n_ws++] = eligible[k];
            const int share = portion / n_ws, odd = portion % n_ws;
            // stable insertion sort by (r - 1) mod n: odd chips first to the seat after the button
            for (int i = 1; i < n_ws; i++) {
                const int x = ws[i];
                const int kx = ((x - 1) % n + n) % n;
                int j = i;
                while (j > 0 && ((ws[j - 1] - 1) % n + n) % n > kx) { ws[j] = ws[j - 1]; j--; }
                ws[j] = x;
            }
            for (int i = 0; i < n_ws; i++) if (ws[i] == me) won += share + (i < odd ? 1 : 0);
        }
        prev = lvl;
    }
    return won - invested[me];
}

}  // namespace negp
