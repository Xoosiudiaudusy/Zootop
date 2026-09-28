// The per-item work of the GPU trainer's kernels (gpucfr.cu), as functions both the device and the host
// compile (NEGP_HD).  gpucfr.cu launches them one thread per item; FlatTrainer's emulation mode
// (flatcfr.h, emulate_gpu) runs the same functions in loops, in the same sequence of steps (level by level,
// exclusive scan, records by key, sort, runs), so everything but the CUDA calls themselves is checked on
// any machine against the CPU trainers.
//
// Items: one (node, job) per slot of the level arrays; job = (iteration - batch start) * n + traverser.
// Strategies: the tables do not change during a batch, so the current strategy of every row (infoset) is
// computed once at the batch start (row_sigma) and read by the items.
// Update records: key = kind (2 bits) | cell or infoset (32-bit keys), value; each item's records go to the
// slots of an exclusive scan of the record counts over the items of the pass.  The records of one cell come
// in increasing job order: a cell belongs to one node, a node to one level, the items of a level keep the
// job order of level 0 (children are emitted in their parents' order), and passes follow each other.  A
// stable sort by key therefore keeps (iteration, traverser) order within every cell: the order of the
// CPU trainers' additions.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cfrmath.h"
#include "philox.h"

namespace negp {

// the update keys: kind in bits [cb, cb + 2), cell or infoset in [0, cb)
struct KeyLayout {
    int cb = 1;
    int bits() const { return cb + 2; }
};
NEGP_HD inline uint32_t make_key(const KeyLayout& kl, uint32_t kind, uint64_t cell) { return (kind << kl.cb) | (uint32_t)cell; }
inline int bits_for(uint64_t n) {  // bits holding 0 .. n - 1 (at least 1)
    int b = 1;
    while (b < 64 && (1ULL << b) < n) b++;
    return b;
}
// the layout for `cells` cells / infosets; cb = 0: they do not fit 32-bit keys
inline KeyLayout key_layout(uint64_t cells) {
    KeyLayout kl;
    kl.cb = bits_for(cells);
    if (kl.bits() > 32) kl.cb = 0;
    return kl;
}
constexpr uint8_t CHOICE_TRAVERSER = 0xFF;  // choice of a traverser node: every child

struct DevGame {
    const uint8_t *rel, *n_board, *na;
    const uint32_t* child_base;
    const int32_t* child;
    const uint64_t *hh_a, *hh_b, *info_base, *cell_base;
    const int32_t* n_cache;
    const int32_t* invested;
    const uint16_t* folded;
    const int32_t* info_dec;  // infoset -> its decision
    int n;
};

// the current strategy of infoset row i (regret matching of the batch-start regrets) into sigma[its cells],
// only when the row's regrets changed since it was last computed (dirty[its first cell], set by apply_run):
// a row whose regrets did not change keeps the same strategy, so only the rows the last batch updated are
// recomputed (3-max wide 64: 150M cells, a batch changes a small share of them)
NEGP_HD inline void row_sigma(const DevGame& g, const double* regret, double* sigma, uint8_t* dirty, size_t i) {
    const int d = g.info_dec[i];
    const int na = g.na[d];
    const uint64_t c = g.cell_base[d] + (uint64_t)(i - g.info_base[d]) * (uint64_t)na;
    if (!dirty[c]) return;
    regret_matching(&regret[c], na, &sigma[c]);
    dirty[c] = 0;
}

struct DevLevel {  // the item arrays of all levels of a pass; level L is [off_L, off_L + m_L)
    int32_t* node;     // >= 0 decision, < 0 ~terminal
    uint32_t* job;
    uint32_t* first;   // after the scan: the first child's index in the next level
    uint32_t* cnt;     // children
    uint8_t* choice;   // an opponent node's sampled action, CHOICE_TRAVERSER for a traverser node
    double* value;
    uint32_t* rcnt;    // update records the item writes in the backward pass
    uint64_t* roff;    // after the scan over all items of the pass: its first record's slot
};

NEGP_HD inline int item_bucket(const DevGame& g, const FlatIter& it, int d, int seat, int* err) {
    const int b = it.bucket[seat][g.n_board[d]];
    if (b < 0 || b >= g.n_cache[d]) { *err = 1; return 0; }
    return b;
}

// level 0: the roots of jobs job0 ..
NEGP_HD inline void item_init(DevLevel L, size_t i, uint32_t job0) {
    L.node[i] = 0;
    L.job[i] = job0 + (uint32_t)i;
}

// forward, part 1: children of item x (cnt), a terminal's value, an opponent's sample (choice);
// returns the number of update records the item will write in the backward pass
NEGP_HD inline uint32_t item_count(const DevGame& g, DevLevel L, size_t x, const FlatIter* iters, const double* sigma_tab, uint64_t seed, int bb,
                                   int* err) {
    const int n = g.n;
    const int32_t node = L.node[x];
    const uint32_t job = L.job[x];
    const FlatIter& it = iters[job / (uint32_t)n];
    const int p = (int)(job % (uint32_t)n);
    if (node < 0) {
        const int tm = ~node;
        const int me = ((p - it.button) % n + n) % n;
        L.value[x] = (double)terminal_net_of(&g.invested[(size_t)tm * n], g.folded[tm], n, me, it.strength) / (double)bb;
        L.cnt[x] = 0;
        L.rcnt[x] = 0;
        return 0;
    }
    const int d = node;
    const int na = g.na[d];
    const int seat = (g.rel[d] + it.button) % n;
    if (seat == p) {
        L.cnt[x] = (uint32_t)na;
        L.choice[x] = CHOICE_TRAVERSER;
        L.rcnt[x] = (uint32_t)na;
        return (uint32_t)na;
    }
    const int b = item_bucket(g, it, d, seat, err);
    const double* sigma = &sigma_tab[g.cell_base[d] + (uint64_t)b * (uint64_t)na];
    const double r = philox_sample_u01(seed, (uint64_t)it.t, p, g.hh_a[d], g.hh_b[d]);
    int a = na - 1;
    double acc = 0.0;
    for (int k = 0; k < na; k++) {
        acc += sigma[k];
        if (r < acc) { a = k; break; }
    }
    L.choice[x] = (uint8_t)a;
    L.cnt[x] = 1;
    L.rcnt[x] = (uint32_t)(na + 1);
    return (uint32_t)(na + 1);
}

// forward, part 2: the children of item x into the next level (first[x] = exclusive scan of cnt)
NEGP_HD inline void item_emit(const DevGame& g, DevLevel L, size_t x, size_t next_off) {
    if (L.cnt[x] == 0) return;
    const int d = L.node[x];
    const uint32_t job = L.job[x];
    const size_t at = next_off + L.first[x];
    const int32_t* ch = &g.child[g.child_base[d]];
    const uint8_t c = L.choice[x];
    if (c != CHOICE_TRAVERSER) {
        L.node[at] = ch[c];
        L.job[at] = job;
        return;
    }
    const int na = g.na[d];
    for (int a = 0; a < na; a++) {
        L.node[at + (size_t)a] = ch[a];
        L.job[at + (size_t)a] = job;
    }
}

// the records item x writes in the backward pass (as item_count returned)
NEGP_HD inline uint32_t item_records(const DevGame& g, DevLevel L, size_t x) {
    const int32_t node = L.node[x];
    if (node < 0) return 0;
    const int na = g.na[node];
    return L.choice[x] == CHOICE_TRAVERSER ? (uint32_t)na : (uint32_t)(na + 1);
}

// backward: item x's value from its children; its records at keys[s ..], vals[s ..]
NEGP_HD inline void item_back(const DevGame& g, DevLevel L, size_t x, size_t next_off, const FlatIter* iters, const double* sigma_tab, const KeyLayout& kl,
                              uint32_t* keys, double* vals, uint64_t s, int* err) {
    const int32_t node = L.node[x];
    if (node < 0) return;
    const int n = g.n;
    const uint32_t job = L.job[x];
    const FlatIter& it = iters[job / (uint32_t)n];
    const int d = node;
    const int na = g.na[d];
    const int seat = (g.rel[d] + it.button) % n;
    const int b = item_bucket(g, it, d, seat, err);
    const uint64_t c = g.cell_base[d] + (uint64_t)b * (uint64_t)na;
    const double* sigma = &sigma_tab[c];
    const size_t fc = next_off + L.first[x];
    if (L.choice[x] != CHOICE_TRAVERSER) {
        L.value[x] = L.value[fc];
        for (int a = 0; a < na; a++) {
            keys[s + (uint64_t)a] = make_key(kl, 1, c + (uint64_t)a);
            vals[s + (uint64_t)a] = it.weight * sigma[a];
        }
        keys[s + (uint64_t)na] = make_key(kl, 2, g.info_base[d] + (uint64_t)b);
        vals[s + (uint64_t)na] = 0.0;
        return;
    }
    double utils[MAX_ACTIONS], prods[MAX_ACTIONS];
    for (int a = 0; a < na; a++) {
        utils[a] = L.value[fc + (size_t)a];
        prods[a] = sigma[a] * utils[a];
    }
    const double u = py_sum(prods, na);
    L.value[x] = u;
    for (int a = 0; a < na; a++) {
        keys[s + (uint64_t)a] = make_key(kl, 0, c + (uint64_t)a);
        vals[s + (uint64_t)a] = it.weight * (utils[a] - u);
    }
}

// after the stable sort (keys[0 .. m) non-decreasing, each key's records in job order): the record i that
// starts a run of equal keys adds the run's values to its cell in order; every other record does nothing
NEGP_HD inline void apply_run(const uint32_t* keys, const double* vals, size_t m, size_t i, const KeyLayout& kl, double* regret, double* ssum,
                              int64_t* visits, uint8_t* touched, uint8_t* dirty) {
    const uint32_t key = keys[i];
    if (i > 0 && keys[i - 1] == key) return;
    const int kind = (int)(key >> kl.cb);
    const uint64_t cell = key & ((1u << kl.cb) - 1u);
    size_t j = i;
    if (kind == 2) {
        while (j < m && keys[j] == key) j++;
        visits[cell] += (int64_t)(j - i);
        return;
    }
    double* t = kind == 0 ? regret : ssum;
    double v = t[cell];
    for (; j < m && keys[j] == key; j++) v += vals[j];
    t[cell] = v;
    touched[cell] = 1;
    if (kind == 0) dirty[cell] = 1;  // its row's strategy is recomputed before the next batch
}

}  // namespace negp
