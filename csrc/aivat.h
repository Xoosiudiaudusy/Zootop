// AIVAT for heads-up no-limit hold'em (2026-09-25): the fast twin of negpluribus/eval/aivat.py and
// negpluribus/eval/aivat_values.py (the Python reference defines the numbers; docs/aivat.md).
//
// Burch, Schmid, Moravcik, Morrill, Bowling, "AIVAT: A New Variance Reduction Technique for Agent
// Evaluation in Imperfect Information Games", AAAI 2018, arXiv 1612.06915.  Per hand, for the known
// player x (the blueprint agent) against an unknown y:
//
//   AIVAT(z) = base value (imaginary observations over x's holes, Eq. 1 first part)
//            + sum of the correction terms k_H(z) of the parts H the hand passed through:
//              the root (y's hole deal and the seat), x's decisions, the flop, turn and river deals.
//
// k_H = sum_a sum_{h in H} pi(h a) u_h(a) / sum_{h in H} pi(h) - sum_{h in H} pi(h a_O) u_h(a_O) /
// sum_{h in H} pi(h a_O), with pi the product of chance's and x's probabilities; the states of a
// part differ in x's hole (1326 combos) and in x's private translation coins (branches).
//
// Everything below mirrors the Python reference operation for operation where the result is a
// per-combo number (branch values, reach products, sampling laws, coin probabilities): those agree
// bit for bit.  The sums over combos (dot products) are sequential here and pairwise in numpy: they
// agree to rounding (tests compare with a relative tolerance).
#pragma once
#include "workers.h"
#if defined(_MSC_VER)
#pragma fp_contract(off)
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "abstraction.h"
#include "bucketcache.h"
#include "buckettable.h"
#include "engine.h"
#include "handindex.h"
#include "mccfr.h"
#include "nodetable.h"
#include "persist.h"

namespace negp {
namespace aiv {

constexpr int NC = 1326;
constexpr uint64_t GOLD = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t EQ_STREAM = 1000000;  // rollout index of the Monte-Carlo equity stream
enum TermKind { T_ROOT = 0, T_SEAT = 1, T_X = 2, T_FLOP = 3, T_TURN = 4, T_RIVER = 5 };

// hole combos (a < b) in the order of Python's [(a, b) for a in range(52) for b in range(a + 1, 52)]
struct Combos {
    int c0[NC], c1[NC];
    int idx[52][52];
    uint64_t mask[NC];
    Combos() {
        for (int a = 0; a < 52; a++)
            for (int b = 0; b < 52; b++) idx[a][b] = -1;
        int k = 0;
        for (int a = 0; a < 52; a++)
            for (int b = a + 1; b < 52; b++) {
                c0[k] = a;
                c1[k] = b;
                idx[a][b] = idx[b][a] = k;
                mask[k] = (1ULL << a) | (1ULL << b);
                k++;
            }
    }
};
inline const Combos& combos() {
    static const Combos t;
    return t;
}

// ------------------------------------------------------------------------------------ RNG
inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
inline uint64_t stream_seed(std::initializer_list<uint64_t> parts) {
    uint64_t h = 0x6A09E667F3BCC908ULL;
    for (uint64_t p : parts) h = mix64((h ^ p) + GOLD);
    return h;
}
struct CounterRng {
    uint64_t s;
    explicit CounterRng(uint64_t seed) : s(seed) {}
    uint64_t next64() {
        s += GOLD;
        return mix64(s);
    }
    double uniform() { return (double)(next64() >> 11) * (1.0 / 9007199254740992.0); }
};

// P(U < p) for U = random.random() = k / 2^53: ceil(p 2^53) / 2^53
inline double coin_prob(double p) {
    if (!(p > 0.0)) return 0.0;
    if (p >= 1.0) return 1.0;
    return std::ceil(p * 9007199254740992.0) / 9007199254740992.0;
}

// P(choice = i) of BlueprintAgent.act (the first i with r < running sum, else the last)
inline void sampling_law(const double* probs, int n, double* out) {
    double acc = 0.0, prev_f = 0.0;
    for (int i = 0; i + 1 < n; i++) {
        acc += probs[i];
        const double f = acc < 1.0 ? std::min(coin_prob(acc), 1.0) : 1.0;
        out[i] = std::max(0.0, f - prev_f);
        prev_f = std::max(prev_f, f);
    }
    out[n - 1] = std::max(0.0, 1.0 - prev_f);
}

// --------------------------------------------------------------- the agent's translation coins
struct TransOut {
    std::string tok[2];
    double p[2] = {1.0, 0.0};
    int n = 0;
    void one(const std::string& t) {
        tok[0] = t;
        p[0] = 1.0;
        n = 1;
    }
    void two(const std::string& lo, const std::string& hi, double p_lo) {
        if (p_lo >= 1.0) return one(lo);
        if (p_lo <= 0.0) return one(hi);
        tok[0] = lo;
        tok[1] = hi;
        p[0] = p_lo;
        p[1] = 1.0 - p_lo;
        n = 2;
    }
};

inline void harmonic_outcomes(double x, std::vector<double> g, TransOut& out) {
    std::sort(g.begin(), g.end());
    if (x <= g.front()) return out.one(raise_name(g.front()));
    if (x >= g.back()) return out.one(raise_name(g.back()));
    for (size_t i = 0; i + 1 < g.size(); i++) {
        const double a = g[i], b = g[i + 1];
        if (a <= x && x <= b) {
            const double p_a = (b - x) * (1 + a) / ((b - a) * (1 + x));
            return out.two(raise_name(a), raise_name(b), coin_prob(p_a));
        }
    }
    out.one(raise_name(g.back()));
}

// the tokens BetGrid.from_concrete(ev, ev.all_in, rng) can give, with their probabilities over the
// agent's coin (negpluribus/eval/aivat.py translation_outcomes)
inline void translation_outcomes(const BetGrid& g, const Event& ev, TransOut& out) {
    out.n = 0;
    if (ev.type != RAISE) {
        std::string t;
        g.from_concrete(ev, t);
        return out.one(t);
    }
    const std::vector<double>& fracs = g.fracs_for(ev.street);
    if (ev.all_in && g.allow_all_in) return out.one("a");
    if (ev.raises_this_street >= g.max_raises_per_street) return out.one(g.allow_all_in ? "a" : "c");  // capped node (QA-1)
    if (fracs.empty()) return out.one("a");
    const double x = BetGrid::observed_frac(ev);
    const int pac = ev.pot_before + ev.to_call;
    if (g.allow_all_in && ev.stack_after >= 0 && pac > 0) {
        const double x_allin = (double)(ev.paid + ev.stack_after - ev.to_call) / (double)pac;
        std::vector<double> below;
        std::vector<double> s(fracs);
        std::sort(s.begin(), s.end());
        for (double f : s) if (f < x_allin) below.push_back(f);
        if (below.empty()) return out.one("a");
        const double top = below.back();
        if (x > top) {
            const double p_top = (x_allin - x) * (1 + top) / ((x_allin - top) * (1 + x));
            return out.two(raise_name(top), "a", coin_prob(p_top));
        }
        return harmonic_outcomes(x, below, out);
    }
    harmonic_outcomes(x, fracs, out);
}

// HistHash::event with a given token (the history text the agent builds for one coin outcome)
inline void hh_push(HistHash& h, int street, const std::string& tok) {
    if (street != h.cur) {
        if (h.cur != -1) h.byte('/');
        h.cur = street;
    } else {
        h.byte(' ');
    }
    h.bytes(tok.data(), tok.size());
    h.n++;
}

// ----------------------------------------------------------------------- suit classes of pairs
// class id of every ordered pair (c, d) of disjoint holes under the 24 suit permutations, -1 on
// overlapping pairs; ids in order of first appearance scanning c then d (aivat_values.pair_orbits)
inline std::vector<int32_t> pair_orbits(int& n_classes) {
    const Combos& cb = combos();
    std::vector<int32_t> orbit((size_t)NC * NC, -1);
    std::unordered_map<uint64_t, int32_t> ids;
    ids.reserve(80000);
    for (int ci = 0; ci < NC; ci++) {
        const int a = cb.c0[ci], b = cb.c1[ci];
        for (int di = 0; di < NC; di++) {
            const int e = cb.c0[di], f = cb.c1[di];
            if (a == e || a == f || b == e || b == f) continue;
            uint64_t best = ~0ULL;
            for (int p = 0; p < 24; p++) {
                const int* perm = SUIT_PERMS[p];
                int m[4];
                const int in[4] = {a, b, e, f};
                for (int k = 0; k < 4; k++) m[k] = (in[k] >> 2) * 4 + perm[in[k] & 3];
                const uint64_t key = ((uint64_t)std::min(m[0], m[1]) << 24) | ((uint64_t)std::max(m[0], m[1]) << 16) |
                                     ((uint64_t)std::min(m[2], m[3]) << 8) | (uint64_t)std::max(m[2], m[3]);
                best = std::min(best, key);
            }
            auto it = ids.find(best);
            if (it == ids.end()) it = ids.emplace(best, (int32_t)ids.size()).first;
            orbit[(size_t)ci * NC + di] = it->second;
        }
    }
    n_classes = (int)ids.size();
    return orbit;
}

// ----------------------------------------------------------------------- fast bucket tables
// The river table from one batch per suit-canonical 5-card board (Bucketer::river_buckets_all: the
// same numbers as bucket(), tested in the core); every class has a representative with a canonical
// board, so every index is set (checked).  Returns false if the bucketer has no river batch.
inline bool build_river_table_fast(const Bucketer& bk, BucketTables& tables, int threads) {
    const HandIndexer& ix = tables.indexer(RIVER);
    const std::vector<std::array<int, 5>> boards = canonical_boards(5);
    std::vector<uint8_t> t(ix.size(), 255);
    std::atomic<size_t> next{0};
    std::atomic<bool> unsupported{false};
    std::atomic<long long> filled{0};
    const Combos& cb = combos();
    auto work = [&]() {
        std::vector<uint8_t> out(NC);
        long long mine = 0;
        for (size_t i = next.fetch_add(1); i < boards.size() && !unsupported.load() && !pool_stopping(); i = next.fetch_add(1)) {
            const int* b = boards[i].data();
            if (!bk.river_buckets_all(b, cb.idx, out.data())) {
                unsupported.store(true);
                return;
            }
            for (int c = 0; c < NC; c++) {
                if (out[(size_t)c] == 255) continue;
                const int hole[2] = {cb.c0[c], cb.c1[c]};
                const uint64_t k = ix.index(hole, b);
                if (t[k] == 255) mine++;
                t[k] = out[(size_t)c];  // equal classes get equal buckets (a pure function of the class)
            }
        }
        filled.fetch_add(mine);
    };
    const int T = std::max(1, threads);
    run_pool(T, "river table from canonical boards", work);
    if (unsupported.load()) return false;
    for (uint8_t v : t)
        if (v == 255) throw std::runtime_error("river table: a class was not reached from the canonical boards");
    tables.set_table(RIVER, std::move(t), bk.identity());
    return true;
}

// the flop or turn table through bucket() (fast when the bucketer's cache is warm: load a saved
// bucket cache first); returns how many values the bucketer had to compute (cache misses)
inline uint64_t build_table_through_bucketer(Bucketer& bk, BucketTables& tables, int street, int threads) {
    const HandIndexer& ix = tables.indexer(street);
    const uint64_t n = ix.size();
    std::vector<uint8_t> t(n, 255);
    const uint64_t before = bk.cache_stats(street).computes;
    std::atomic<uint64_t> next{0};
    auto work = [&]() {
        int hole[2], board[5];
        for (uint64_t lo = next.fetch_add(4096); lo < n && !pool_stopping(); lo = next.fetch_add(4096)) {
            const uint64_t hi = std::min(n, lo + 4096);
            for (uint64_t i = lo; i < hi; i++) {
                ix.unindex(i, hole, board);
                t[i] = (uint8_t)bk.bucket(hole, board, ix.n_board());
            }
        }
    };
    const int T = std::max(1, threads);
    run_pool(T, "bucket table through the bucketer", work);
    const uint64_t computed = bk.cache_stats(street).computes - before;
    tables.set_table(street, std::move(t), bk.identity());
    return computed;
}

// ------------------------------------------------------------------------------ the game
struct Game {
    Spec spec;
    BetGrid grid;
    std::shared_ptr<Bucketer> bucketer;
    std::shared_ptr<const BlueprintTable> blueprint;
    std::vector<int> grid_to_bp;  // grid action id -> index in the blueprint's names (-1 unknown)

    uint64_t serial;              // unique per Game object: owner of the per-thread memo (below)
    // the bucketer's tables when it is a TabulatedBucketer: its bucket() on street s with tab_on[s] is
    // tabs->at(s, tabs->indexer(s).index(hole, board)) (csrc/buckettable.h)
    const BucketTables* tabs = nullptr;
    bool tab_on[4] = {false, false, false, false};

    Game(const Spec& s, std::shared_ptr<Bucketer> bk, std::shared_ptr<const BlueprintTable> bp)
        : spec(s), grid(s.grid()), bucketer(std::move(bk)), blueprint(std::move(bp)), serial(next_serial()) {
        if (!bucketer) throw std::invalid_argument("aivat: a bucketer is needed");
        if (!blueprint) throw std::invalid_argument("aivat: a blueprint is needed");
        if (spec.n_players != 2) throw std::invalid_argument("aivat: heads-up only");
        if (auto* tb = dynamic_cast<const TabulatedBucketer*>(bucketer.get()))
            if (tb->tables())
                for (int st = FLOP; st <= RIVER; st++)
                    if (tb->tables()->has(st)) tabs = tb->tables().get(), tab_on[st] = true;  // (kept alive by the bucketer)
        grid_to_bp.assign(grid.names.size(), -1);
        for (size_t i = 0; i < grid.names.size(); i++) grid_to_bp[i] = blueprint->name_index(grid.names[i].data(), grid.names[i].size());
    }

    // the blueprint's probabilities at `key` for the legal actions `al` (false = unknown key)
    bool policy(const NodeKey& key, const ActionList& al, double* out) const {
        const long long i = blueprint->find(key);
        if (i < 0) return false;
        int legal[8];
        for (int a = 0; a < al.n; a++) legal[a] = grid_to_bp[(size_t)al.a[a].id];
        return blueprint->policy_at(i, legal, al.n, out);
    }

    // The same through a per-thread memo: policy() and bucket() are pure functions of their arguments
    // (a rollout repeats a (key, legal) lookup about 30 times per hand, a (hole, board) bucket about 4
    // times), so a memo hit returns exactly the numbers the call would.  Entries are checked in full
    // (both key words, the legal ids; every card), a collision just overwrites.
    struct Memo {
        static constexpr size_t N_POL = 1 << 14, N_BK = 1 << 16;
        struct Pol {
            uint64_t k1 = 0, k2 = 0, sig = 0;  // sig 0: empty
            bool have = false;
            double p[8];
        };
        struct Bk {
            uint64_t key = 0;
            int b = 0;
        };
        static constexpr size_t N_TOK = 1 << 10;
        struct Tok {  // BetGrid::from_concrete of an event with these fields (k0 0: empty)
            uint64_t k0 = 0, k1 = 0, k2 = 0, k3 = 0;
            uint8_t len = 0;
            char t[23];
        };
        uint64_t owner = 0;
        std::vector<Pol> pol;
        std::vector<Bk> bk;
        std::vector<Tok> tok;
        std::string scratch;
    };
    // the slot of a key (from k1 alone, so it can be prefetched before the legal actions are known)
    Memo::Pol& pol_slot(Memo& m, const NodeKey& key) const { return m.pol[(size_t)((key.k1 ^ key.k1 >> 29) & (Memo::N_POL - 1))]; }
    bool policy_memo(Memo& m, const NodeKey& key, const ActionList& al, double* out) const {
        uint64_t sig = 0;
        for (int a = 0; a < al.n; a++) {
            const int id = al.a[a].id;
            if (id < 0 || id >= 255) return policy(key, al, out);
            sig = (sig << 8) | (uint64_t)(id + 1);
        }
        if (sig == 0) return policy(key, al, out);
        Memo::Pol& e = pol_slot(m, key);
        if (e.sig == sig && e.k1 == key.k1 && e.k2 == key.k2) {
            for (int a = 0; a < al.n; a++) out[a] = e.p[a];
            return e.have;
        }
        const bool have = policy(key, al, out);
        e.k1 = key.k1;
        e.k2 = key.k2;
        e.sig = sig;
        e.have = have;
        if (have) for (int a = 0; a < al.n; a++) e.p[a] = out[a];
        return have;
    }
    static uint64_t bucket_key(const int* hole, const int* board, int n_board) {
        uint64_t k = (uint64_t)n_board;
        k = (k << 6) | (uint64_t)hole[0];
        k = (k << 6) | (uint64_t)hole[1];
        for (int i = 0; i < n_board; i++) k = (k << 6) | (uint64_t)board[i];
        return k | 1ULL << 63;  // (never 0: 0 marks an empty slot; 3 + 42 bits used)
    }
    int bucket_memo(Memo& m, const int* hole, const int* board, int n_board) const {
        const uint64_t k = bucket_key(hole, board, n_board);
        Memo::Bk& e = m.bk[(size_t)(fmix64(k) & (Memo::N_BK - 1))];
        if (e.key == k) return e.b;
        const int b = bucketer->bucket(hole, board, n_board);
        e.key = k;
        e.b = b;
        return b;
    }
    // bucket lookups of a rollout's later streets, started early (the run-out is dealt first, so every board is
    // known): stage 1 prefetches the memo slot; stage 2 (a step later) reads it and on a miss computes the
    // table index and prefetches the table byte; the bucket is read when the street comes.  The same numbers
    // as bucket_memo (the memo holds bucket() values; the table byte is what TabulatedBucketer::bucket reads).
    struct Ahead {
        uint64_t key = 0, idx = 0;
        int b = -1, stage = 0;  // 0 none, 1 memo slot prefetched, 2 table byte prefetched, 3 bucket known
    };
    Memo::Bk& bk_slot(Memo& m, uint64_t key) const { return m.bk[(size_t)(fmix64(key) & (Memo::N_BK - 1))]; }
    void ahead_start(Memo& m, Ahead& a, const int* hole, const int* board, int n_board) const {
        a.key = bucket_key(hole, board, n_board);
        NEGP_PREFETCH(&bk_slot(m, a.key));
        a.stage = 1;
    }
    void ahead_step(Memo& m, Ahead& a, int street, const int* hole, const int* board) const {
        const Memo::Bk& e = bk_slot(m, a.key);
        if (e.key == a.key) {
            a.b = e.b;
            a.stage = 3;
            return;
        }
        a.idx = tabs->indexer(street).index(hole, board);
        NEGP_PREFETCH(tabs->data(street) + a.idx);
        a.stage = 2;
    }
    int ahead_bucket(Memo& m, Ahead& a, int street, const int* hole, const int* board) const {
        if (a.stage == 1) ahead_step(m, a, street, hole, board);
        if (a.stage == 3) return a.b;
        const int b = tabs->at(street, a.idx);
        Memo::Bk& e = bk_slot(m, a.key);
        e.key = a.key;
        e.b = b;
        a.b = b;
        a.stage = 3;
        return b;
    }

    // HistHash::event through the memo: the token is a function of the event's fields (from_concrete reads
    // type, street, all_in, raises_this_street, to_call, pot_before, paid, stack_after; the key holds every field)
    void hash_event(Memo& m, HistHash& h, const Event& ev) const {
        const uint64_t k0 = 1ULL << 63 | (uint64_t)(uint8_t)ev.street | (uint64_t)(uint8_t)ev.seat << 8 | (uint64_t)(uint8_t)ev.type << 16 |
                            (uint64_t)ev.facing_raise << 24 | (uint64_t)ev.all_in << 25 | (uint64_t)(uint16_t)ev.raises_this_street << 32;
        const uint64_t k1 = (uint64_t)(uint32_t)ev.amount | (uint64_t)(uint32_t)ev.to_call << 32;
        const uint64_t k2 = (uint64_t)(uint32_t)ev.pot_before | (uint64_t)(uint32_t)ev.paid << 32;
        const uint64_t k3 = (uint64_t)(uint32_t)ev.stack_after;
        Memo::Tok& e = m.tok[(size_t)(fmix64(k0 ^ fmix64(k1 ^ fmix64(k2 ^ k3))) & (Memo::N_TOK - 1))];
        if (!(e.k0 == k0 && e.k1 == k1 && e.k2 == k2 && e.k3 == k3)) {
            m.scratch.clear();
            grid.from_concrete(ev, m.scratch);
            if (m.scratch.size() > sizeof(e.t)) {  // (a token this long is not memoized)
                h.event(ev, grid, m.scratch);
                return;
            }
            e.k0 = k0;
            e.k1 = k1;
            e.k2 = k2;
            e.k3 = k3;
            e.len = (uint8_t)m.scratch.size();
            std::memcpy(e.t, m.scratch.data(), e.len);
        }
        if (ev.street != h.cur) {
            if (h.cur != -1) h.byte('/');
            h.cur = ev.street;
        } else {
            h.byte(' ');
        }
        h.bytes(e.t, e.len);
        h.n++;
    }
    // this thread's memo (made empty when it last served another Game)
    Memo& memo() const {
        thread_local Memo m;
        if (m.owner != serial) {  // another Game on this thread: start empty
            m.pol.assign(Memo::N_POL, Memo::Pol());
            m.bk.assign(Memo::N_BK, Memo::Bk());
            m.tok.assign(Memo::N_TOK, Memo::Tok());
            m.owner = serial;
        }
        return m;
    }
private:
    static uint64_t next_serial() {
        static std::atomic<uint64_t> n{0};
        return ++n;
    }
};

struct ValueParams {
    int rollouts[4] = {4, 8, 8, 0};  // per street of the state reached (preflop, flop, turn)
    int eq_samples = 2000;           // random boards of a preflop all-in value
    uint64_t seed = 0;
    // v2 "turn exact" (docs/aivat.md, type (b)): a turn state with decisions ahead and its turn card known is
    // valued exactly (the betting trees walked, the river card enumerated) instead of by rollouts
    bool turn_exact = false;
};

// one finished hand (negpluribus/eval/aivat.py AivatHand)
// x's strategy at one decision for every hole, as the agent logged it (docs/aivat.md section 7): the actions in
// column order and uint16 rows (the combos off the board, index order) summing to Q_LOG
constexpr int Q_LOG = 65535;
struct LoggedRowsIn {
    bool present = false;
    std::vector<std::pair<int, int>> actions;
    std::vector<uint16_t> q;
};

struct HandIn {
    int hand_id = 0;
    int stacks[2] = {0, 0};
    int button = 0, sb = 50, bb = 100, known_seat = 0;
    int holes[2][2] = {{0, 0}, {0, 0}};
    std::vector<int> board;
    std::vector<std::pair<int, int>> actions;
    std::vector<LoggedRowsIn> x_rows;  // per decision of x, in order; absent / not present: the blueprint model
};

struct Term {
    int kind = 0, street = 0, k = -1;
    double value = 0.0;
};

struct HandOut {
    int hand_id = 0;
    int net = 0;
    double base = 0.0, value = 0.0;
    std::vector<Term> terms;
    std::vector<std::pair<std::string, std::vector<double>>> trace;  // tests: per-node value vectors
    long long rollouts = 0, rollout_steps = 0, river_trees = 0;
    double seconds = 0.0;
};

// the root table: u_root(seat, c, d) per suit class of (c, d)
struct RootTable {
    std::vector<int32_t> orbit;          // NC x NC
    int n_classes = 0;
    std::vector<double> values[2];       // per seat (0: x is the small blind), per class
    double mean[2] = {0.0, 0.0};         // exact mean over every disjoint pair
    bool empty() const { return values[0].empty(); }
    void finish() {
        std::vector<double> cnt((size_t)n_classes, 0.0);
        for (int32_t o : orbit) if (o >= 0) cnt[(size_t)o] += 1.0;
        for (int s = 0; s < 2; s++) {
            double tot = 0.0, n = 0.0;
            for (int o = 0; o < n_classes; o++) {
                tot += cnt[(size_t)o] * values[s][(size_t)o];
                n += cnt[(size_t)o];
            }
            mean[s] = tot / n;
        }
    }
};

// -------------------------------------------------------------------------- heuristic v1
class Evaluator {
public:
    // rollouts in flight per branch (1..MAX_LANES; set_lanes, the binding's aivat_set_lanes): 1, one after
    // another, is the fastest in the cloud, where the tables sit in its 260 MB L3; more may pay where they do
    // not.  The numbers do not depend on it.
    static constexpr int MAX_LANES = 8;
    static std::atomic<int>& lanes_setting() {
        static std::atomic<int> n{1};
        return n;
    }
    static void set_lanes(int n) { lanes_setting().store(n < 1 ? 1 : (n > MAX_LANES ? MAX_LANES : n)); }
    static int lanes_in_flight() { return lanes_setting().load(std::memory_order_relaxed); }

    Evaluator(std::shared_ptr<const Game> game, const ValueParams& vp, std::shared_ptr<const RootTable> root)
        : g_(std::move(game)), vp_(vp), root_(std::move(root)) {}

    const Game& game() const { return *g_; }
    const ValueParams& params() const { return vp_; }

    struct Ctx {
        const HandIn* hand = nullptr;
        int x = 0, y = 1;
        int d[2] = {0, 0};
        std::unordered_map<uint64_t, std::vector<uint8_t>> river_buckets;
        std::unordered_map<uint64_t, std::vector<int8_t>> river_signs;  // x's showdown sign per combo on a 5-card board
        std::unordered_map<uint64_t, std::vector<int>> buckets;  // x's buckets per board (the agent model)
        std::string tok;
        long long rollouts = 0, rollout_steps = 0, river_trees = 0;
        bool trace = false;
        HandOut* out = nullptr;
    };

    // ---- one hand
    HandOut evaluate(const HandIn& h, bool trace = false) const {
        const auto t0 = std::chrono::steady_clock::now();
        const Combos& cb = combos();
        const BetGrid& grid = g_->grid;
        HandOut out;
        out.hand_id = h.hand_id;
        Ctx ctx;
        ctx.hand = &h;
        ctx.x = h.known_seat;
        ctx.y = 1 - h.known_seat;
        ctx.d[0] = h.holes[ctx.y][0];
        ctx.d[1] = h.holes[ctx.y][1];
        ctx.trace = trace;
        ctx.out = &out;
        const int x = ctx.x;
        int deck[52];
        build_deck(h, deck);
        HandState st(std::vector<int>{h.stacks[0], h.stacks[1]}, h.button, h.sb, h.bb, 0, deck, RIVER);
        const uint64_t dmask = (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);

        struct Branch {
            double prob = 1.0;
            HistHash hh;
            std::vector<double> reach;
        };
        std::vector<Branch> branches(1);
        branches[0].reach.assign(NC, 0.0);
        for (int c = 0; c < NC; c++) branches[0].reach[(size_t)c] = (cb.mask[c] & dmask) ? 0.0 : 1.0;
        int node = 0;

        // root: y's hole deal and the seat
        {
            const int pos = h.known_seat == h.button ? 0 : 1;
            double u_sum = 0.0;
            int n = 0;
            std::vector<double> u0((size_t)NC, 0.0);
            const int di = cb.idx[ctx.d[0]][ctx.d[1]];
            for (int c = 0; c < NC; c++) {
                if (cb.mask[c] & dmask) continue;
                const double u = root_ ? root_->values[pos][(size_t)root_->orbit[(size_t)c * NC + di]] : 0.0;
                u0[(size_t)c] = u;
                u_sum += u;
                n++;
            }
            const double c_pos = root_ ? root_->mean[pos] : 0.0;
            add_term(out, T_ROOT, 0, -1, c_pos - u_sum / n);
            const double c_avg = root_ ? 0.5 * (root_->mean[0] + root_->mean[1]) : 0.0;
            add_term(out, T_SEAT, 0, -1, c_avg - c_pos);
            if (trace) out.trace.emplace_back("root", u0);
            node++;
        }

        std::vector<double> reused;
        bool have_reused = false;
        size_t n_x = 0;  // decisions of x so far
        for (size_t k = 0; k < h.actions.size(); k++) {
            const int atype = h.actions[k].first, amount = h.actions[k].first == RAISE ? h.actions[k].second : 0;
            if (st.terminal) throw std::invalid_argument("aivat: actions continue after the hand ended");
            const int seat = st.to_act;
            const int nb0 = st.n_board;
            have_reused = false;
            if (seat == x) {
                // the agent's sigma for every combo, per branch: its logged rows, or the blueprint model
                const LoggedRowsIn* logged = n_x < h.x_rows.size() && h.x_rows[n_x].present ? &h.x_rows[n_x] : nullptr;
                n_x++;
                std::vector<std::pair<int, int>> acts;
                std::vector<std::vector<double>> sig(branches.size());  // [branch][c * A + j]
                if (logged) {
                    acts = logged->actions;
                    for (auto& a : acts) if (a.first != RAISE) a.second = 0;
                } else {
                    const Obs obs = observe(st, x);
                    ActionList al;
                    grid.abstract_actions(obs, al);
                    int ctype[8], camt[8], col[8];
                    for (int i = 0; i < al.n; i++) {
                        grid.to_concrete(obs, al.a[i], ctype[i], camt[i]);
                        const std::pair<int, int> a(ctype[i], ctype[i] == RAISE ? camt[i] : 0);
                        auto it = std::find(acts.begin(), acts.end(), a);
                        if (it == acts.end()) {
                            acts.push_back(a);
                            col[i] = (int)acts.size() - 1;
                        } else {
                            col[i] = (int)(it - acts.begin());
                        }
                    }
                    const int A0 = (int)acts.size();
                    int call_col = -1;
                    for (int j = 0; j < A0; j++) if (acts[(size_t)j].first == CALL) call_col = j;
                    const std::vector<int>& bk = buckets_of(ctx, st.board, st.n_board);
                    const int rel = ((x - st.button) % 2 + 2) % 2;
                    const int n_active = st.n_active();
                    for (size_t t = 0; t < branches.size(); t++) {
                        std::unordered_map<int, std::array<double, 8>> rows;
                        for (int c = 0; c < NC; c++) {
                            const int b = bk[(size_t)c];
                            if (b < 0 || rows.count(b)) continue;
                            std::array<double, 8> row{};
                            double probs[8];
                            if (!g_->policy_memo(g_->memo(), node_key(st.street, rel, n_active, b, branches[t].hh), al, probs)) {
                                row[(size_t)call_col] = 1.0;
                            } else {
                                double law[8];
                                sampling_law(probs, al.n, law);
                                for (int i = 0; i < al.n; i++) row[(size_t)col[i]] += law[i];
                            }
                            rows.emplace(b, row);
                        }
                        std::vector<double>& s = sig[t];
                        s.assign((size_t)NC * A0, 0.0);
                        for (int c = 0; c < NC; c++) {
                            const int b = bk[(size_t)c];
                            if (b < 0) continue;
                            const std::array<double, 8>& row = rows[b];
                            for (int j = 0; j < A0; j++) s[(size_t)c * A0 + j] = row[(size_t)j];
                        }
                    }
                }
                const int A = (int)acts.size();
                int ia = -1;
                for (int j = 0; j < A; j++) if (acts[(size_t)j].first == atype && acts[(size_t)j].second == amount) ia = j;
                if (ia < 0) throw std::invalid_argument("aivat: x's action is not one the agent can take (hand " + std::to_string(h.hand_id) + ")");
                if (logged) {  // q / Q_LOG for the combos off the board, in index order; the same for every coin outcome
                    uint64_t bm = 0;
                    for (int i = 0; i < st.n_board; i++) bm |= 1ULL << st.board[i];
                    std::vector<double> m((size_t)NC * A, 0.0);
                    size_t r = 0;
                    const size_t n_rows = logged->q.size() / (size_t)A;
                    if (n_rows * (size_t)A != logged->q.size()) throw std::invalid_argument("aivat: logged rows of another width");
                    for (int c = 0; c < NC; c++) {
                        if (cb.mask[c] & bm) continue;
                        if (r >= n_rows) throw std::invalid_argument("aivat: fewer logged rows than combos off the board");
                        long long tot = 0;
                        for (int j = 0; j < A; j++) {
                            const uint16_t v = logged->q[r * (size_t)A + (size_t)j];
                            tot += v;
                            m[(size_t)c * A + j] = (double)v / (double)Q_LOG;
                        }
                        if (tot != Q_LOG) throw std::invalid_argument("aivat: a logged row does not sum to 65535");
                        r++;
                    }
                    if (r != n_rows) throw std::invalid_argument("aivat: more logged rows than combos off the board");
                    for (size_t t = 0; t < branches.size(); t++) sig[t] = m;
                }
                uint64_t bmask = dmask;
                for (int i = 0; i < st.n_board; i++) bmask |= 1ULL << st.board[i];
                std::vector<double> w_c((size_t)NC, 0.0), w_ca((size_t)NC * A, 0.0);
                for (int c = 0; c < NC; c++) {
                    const double ok = (cb.mask[c] & bmask) ? 0.0 : 1.0;
                    double acc = 0.0;
                    for (size_t t = 0; t < branches.size(); t++) acc = acc + branches[t].prob * branches[t].reach[(size_t)c];
                    w_c[(size_t)c] = acc * ok;
                    for (int j = 0; j < A; j++) {
                        double a2 = 0.0;
                        for (size_t t = 0; t < branches.size(); t++)
                            a2 = a2 + branches[t].prob * branches[t].reach[(size_t)c] * sig[t][(size_t)c * A + j];
                        w_ca[(size_t)c * A + j] = a2 * ok;
                    }
                }
                std::vector<std::vector<double>> vals((size_t)A);
                for (int j = 0; j < A; j++) {
                    std::vector<char> need((size_t)NC, 0);
                    bool any = false;
                    for (int c = 0; c < NC; c++) if (w_ca[(size_t)c * A + j] > 0.0) need[(size_t)c] = 1, any = true;
                    vals[(size_t)j].assign((size_t)NC, 0.0);
                    if (any) branch(ctx, st, true, acts[(size_t)j].first, acts[(size_t)j].second, std::vector<int>(), node, need, vals[(size_t)j]);
                    if (trace) out.trace.emplace_back("x" + std::to_string(k) + ":a" + std::to_string(j), vals[(size_t)j]);
                }
                double first = 0.0, wsum = 0.0, second = 0.0, wa = 0.0;
                for (int c = 0; c < NC; c++) wsum += w_c[(size_t)c];
                for (int j = 0; j < A; j++) {
                    double dot = 0.0;
                    for (int c = 0; c < NC; c++) dot += w_ca[(size_t)c * A + j] * vals[(size_t)j][(size_t)c];
                    first += dot;
                    if (j == ia) second = dot;
                }
                for (int c = 0; c < NC; c++) wa += w_ca[(size_t)c * A + ia];
                add_term(out, T_X, st.street, (int)k, first / wsum - second / wa);
                for (size_t t = 0; t < branches.size(); t++)
                    for (int c = 0; c < NC; c++) branches[t].reach[(size_t)c] = branches[t].reach[(size_t)c] * sig[t][(size_t)c * A + ia];
                reused = vals[(size_t)ia];
                have_reused = true;
                node++;
            }
            const HandState before(st);
            const Event ev = st.apply(atype, amount);
            // x's coin for this event (both players' events are translated by the agent)
            TransOut to;
            translation_outcomes(grid, ev, to);
            if (to.n == 1) {
                for (Branch& b : branches) hh_push(b.hh, ev.street, to.tok[0]);
            } else {
                std::vector<Branch> nb;
                nb.reserve(branches.size() * 2);
                for (Branch& b : branches)
                    for (int o = 0; o < 2; o++) {
                        Branch c2;
                        c2.prob = b.prob * to.p[o];
                        c2.hh = b.hh;
                        hh_push(c2.hh, ev.street, to.tok[o]);
                        c2.reach = b.reach;
                        nb.push_back(std::move(c2));
                    }
                branches.swap(nb);
            }
            // chance nodes: the board cards this action dealt, street by street
            if (st.n_board > nb0) {
                std::vector<int> fixed;
                int nb = nb0;
                while (nb < st.n_board) {
                    const int g = nb == 0 ? 3 : 1;
                    std::vector<int> f_obs(st.board + nb, st.board + nb + g);
                    uint64_t bmask = dmask;
                    for (int i = 0; i < nb0; i++) bmask |= 1ULL << st.board[i];
                    for (int c2 : fixed) bmask |= 1ULL << c2;
                    std::vector<double> w_c((size_t)NC, 0.0);
                    std::vector<char> need((size_t)NC, 0), need2((size_t)NC, 0);
                    uint64_t fmask = 0;
                    for (int c2 : f_obs) fmask |= 1ULL << c2;
                    for (int c = 0; c < NC; c++) {
                        double acc = 0.0;
                        for (size_t t = 0; t < branches.size(); t++) acc = acc + branches[t].prob * branches[t].reach[(size_t)c];
                        w_c[(size_t)c] = acc * ((cb.mask[c] & bmask) ? 0.0 : 1.0);
                        need[(size_t)c] = w_c[(size_t)c] > 0.0;
                        need2[(size_t)c] = need[(size_t)c] && !(cb.mask[c] & fmask);
                    }
                    std::vector<double> v_before((size_t)NC, 0.0), v_after((size_t)NC, 0.0);
                    const bool use_reused = have_reused && fixed.empty();
                    if (use_reused) v_before = reused;
                    else branch(ctx, before, true, atype, amount, fixed, node, need, v_before);
                    std::vector<int> fixed2(fixed);
                    fixed2.insert(fixed2.end(), f_obs.begin(), f_obs.end());
                    branch(ctx, before, true, atype, amount, fixed2, node, need2, v_after);
                    if (trace) {
                        out.trace.emplace_back("c" + std::to_string(k) + ":" + std::to_string(nb) + ":before", v_before);
                        out.trace.emplace_back("c" + std::to_string(k) + ":" + std::to_string(nb) + ":after", v_after);
                    }
                    double s1 = 0.0, w1 = 0.0, s2 = 0.0, w2 = 0.0;
                    for (int c = 0; c < NC; c++) {
                        s1 += w_c[(size_t)c] * v_before[(size_t)c];
                        w1 += w_c[(size_t)c];
                        if (!(cb.mask[c] & fmask)) {
                            s2 += w_c[(size_t)c] * v_after[(size_t)c];
                            w2 += w_c[(size_t)c];
                        }
                    }
                    const int street = nb == 0 ? FLOP : (nb == 3 ? TURN : RIVER);
                    add_term(out, street == FLOP ? T_FLOP : (street == TURN ? T_TURN : T_RIVER), street, (int)k, s1 / w1 - s2 / w2);
                    fixed = fixed2;
                    nb += g;
                    node++;
                }
            }
        }
        if (!st.terminal) throw std::invalid_argument("aivat: the actions do not finish the hand (hand " + std::to_string(h.hand_id) + ")");
        out.net = st.net(x);
        // base value
        {
            uint64_t bmask = dmask;
            for (int i = 0; i < st.n_board; i++) bmask |= 1ULL << st.board[i];
            std::vector<double> v((size_t)NC, 0.0);
            std::vector<char> ok((size_t)NC, 0);
            for (int c = 0; c < NC; c++) ok[(size_t)c] = !(cb.mask[c] & bmask);
            terminal_values(ctx, st, ok, v);
            if (trace) out.trace.emplace_back("base", v);
            double s = 0.0, w = 0.0;
            for (int c = 0; c < NC; c++) {
                if (!ok[(size_t)c]) continue;
                double acc = 0.0;
                for (size_t t = 0; t < branches.size(); t++) acc = acc + branches[t].prob * branches[t].reach[(size_t)c];
                s += acc * v[(size_t)c];
                w += acc;
            }
            out.base = s / w;
        }
        out.value = out.base;
        for (const Term& t : out.terms) out.value += t.value;
        out.rollouts = ctx.rollouts;
        out.rollout_steps = ctx.rollout_steps;
        out.river_trees = ctx.river_trees;
        out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return out;
    }

    // ---- many hands on threads (results in input order)
    std::vector<HandOut> evaluate_many(const std::vector<HandIn>& hands, int threads) const {
        std::vector<HandOut> res(hands.size());
        std::vector<std::string> errors(hands.size());
        std::atomic<size_t> next{0};
        auto work = [&]() {
            for (size_t i = next.fetch_add(1); i < hands.size() && !pool_stopping(); i = next.fetch_add(1)) {
                try {
                    res[i] = evaluate(hands[i], false);
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                    res[i].hand_id = hands[i].hand_id;
                    res[i].value = std::nan("");
                }
            }
        };
        const int T = std::max(1, threads);
        run_pool(T, "evaluating hands", work);
        for (size_t i = 0; i < hands.size(); i++)
            if (!errors[i].empty()) {
                res[i].trace.emplace_back("error: " + errors[i], std::vector<double>());
            }
        return res;
    }

    // ---- the root table: K rollouts from the root per seat and class, stream (seed, seat, class, k)
    static std::shared_ptr<RootTable> build_root_table(const Game& g, int rollouts, uint64_t seed, int threads) {
        auto rt = std::make_shared<RootTable>();
        rt->orbit = pair_orbits(rt->n_classes);
        const int nc = rt->n_classes;
        std::vector<int> rep_c((size_t)nc, -1), rep_d((size_t)nc, -1);
        for (int ci = 0; ci < NC; ci++)
            for (int di = 0; di < NC; di++) {
                const int32_t o = rt->orbit[(size_t)ci * NC + di];
                if (o >= 0 && rep_c[(size_t)o] < 0) rep_c[(size_t)o] = ci, rep_d[(size_t)o] = di;
            }
        rt->values[0].assign((size_t)nc, 0.0);
        rt->values[1].assign((size_t)nc, 0.0);
        ValueParams vp;
        vp.seed = seed;
        Evaluator ev(std::make_shared<Game>(g), vp, nullptr);
        std::atomic<int> next{0};
        auto work = [&]() {  // (an error leaves to run_pool: the other workers stop, the error is raised)
            for (int o = next.fetch_add(1); o < nc && !pool_stopping(); o = next.fetch_add(1))
                for (int pos = 0; pos < 2; pos++) rt->values[pos][(size_t)o] = ev.root_value(pos, rep_c[(size_t)o], rep_d[(size_t)o], o, rollouts);
        };
        const int T = std::max(1, threads);
        run_pool(T, "root table", work);
        rt->finish();
        return rt;
    }

    // the mean of `rollouts` self-play rollouts from the start of a hand, x in seat `pos` (0: small
    // blind) holding combo ci, y holding di (aivat_values.root_rollout)
    double root_value(int pos, int ci, int di, int cls, int rollouts) const {
        const Combos& cb = combos();
        const Spec& sp = g_->spec;
        HandIn h;
        h.stacks[0] = h.stacks[1] = sp.stack_bb * sp.bb;
        h.button = 0;
        h.sb = sp.sb;
        h.bb = sp.bb;
        h.known_seat = pos == 0 ? 0 : 1;
        const int x = h.known_seat, y = 1 - x;
        h.holes[x][0] = cb.c0[ci];
        h.holes[x][1] = cb.c1[ci];
        h.holes[y][0] = cb.c0[di];
        h.holes[y][1] = cb.c1[di];
        int deck[52];
        build_deck(h, deck);
        HandState st(std::vector<int>{h.stacks[0], h.stacks[1]}, h.button, h.sb, h.bb, 0, deck, RIVER);
        Ctx ctx;
        ctx.hand = &h;
        ctx.x = x;
        ctx.y = y;
        ctx.d[0] = h.holes[y][0];
        ctx.d[1] = h.holes[y][1];
        const int c[2] = {cb.c0[ci], cb.c1[ci]};
        double tot = 0.0;
        for (int r = 0; r < rollouts; r++) {
            CounterRng rng(stream_seed({vp_.seed, (uint64_t)pos, (uint64_t)cls, (uint64_t)r}));
            tot += rollout(ctx, st, false, 0, 0, std::vector<int>(), c, rng);
        }
        return tot / rollouts;
    }

    // tests / reference comparisons: the value vector of one branch of a hand (the state reached by
    // actions[0..k) then actions[k] with the next cards `fixed`), as aivat_values.SelfPlayValues.branch
    std::vector<double> branch_values(const HandIn& h, int k, const std::vector<std::pair<int, int>>& action_override, const std::vector<int>& fixed,
                                      int node, const std::vector<int>& combos_needed) const {
        Ctx ctx;
        ctx.hand = &h;
        ctx.x = h.known_seat;
        ctx.y = 1 - h.known_seat;
        ctx.d[0] = h.holes[ctx.y][0];
        ctx.d[1] = h.holes[ctx.y][1];
        int deck[52];
        build_deck(h, deck);
        HandState st(std::vector<int>{h.stacks[0], h.stacks[1]}, h.button, h.sb, h.bb, 0, deck, RIVER);
        for (int j = 0; j < k; j++) st.apply(h.actions[(size_t)j].first, h.actions[(size_t)j].second);
        std::pair<int, int> a = action_override.empty() ? h.actions[(size_t)k] : action_override[0];
        std::vector<char> need((size_t)NC, 0);
        for (int c : combos_needed) need[(size_t)c] = 1;
        std::vector<double> out((size_t)NC, 0.0);
        branch(ctx, st, true, a.first, a.first == RAISE ? a.second : 0, fixed, node, need, out);
        return out;
    }

private:
    std::shared_ptr<const Game> g_;
    ValueParams vp_;
    std::shared_ptr<const RootTable> root_;

    static void add_term(HandOut& out, int kind, int street, int k, double v) {
        Term t;
        t.kind = kind;
        t.street = street;
        t.k = k;
        t.value = v;
        out.terms.push_back(t);
    }

    // seat 0's hole, seat 1's hole, the board, then every other card in increasing order
    static void build_deck(const HandIn& h, int* deck) {
        uint64_t used = 0;
        int pos = 0;
        for (int s = 0; s < 2; s++)
            for (int i = 0; i < 2; i++) deck[pos++] = h.holes[s][i], used |= 1ULL << h.holes[s][i];
        for (int c : h.board) deck[pos++] = c, used |= 1ULL << c;
        for (int c = 0; c < 52; c++) if (!(used >> c & 1)) deck[pos++] = c;
        if (pos != 52) throw std::invalid_argument("aivat: holes and board share a card");
    }

    const std::vector<int>& buckets_of(Ctx& ctx, const int* board, int n_board) const {
        uint64_t key = (uint64_t)n_board;
        for (int i = 0; i < n_board; i++) key = key * 53 + (uint64_t)board[i] + 1;
        auto it = ctx.buckets.find(key);
        if (it != ctx.buckets.end()) return it->second;
        const Combos& cb = combos();
        uint64_t bm = 0;
        for (int i = 0; i < n_board; i++) bm |= 1ULL << board[i];
        std::vector<int> b((size_t)NC, -1);
        for (int c = 0; c < NC; c++) {
            if (cb.mask[c] & bm) continue;
            const int hole[2] = {cb.c0[c], cb.c1[c]};
            b[(size_t)c] = g_->bucketer->bucket(hole, board, n_board);
        }
        return ctx.buckets.emplace(key, std::move(b)).first->second;
    }

    const std::vector<uint8_t>& river_buckets(Ctx& ctx, const int* board5) const {
        int s[5];
        std::copy(board5, board5 + 5, s);
        std::sort(s, s + 5);
        uint64_t key = 0;
        for (int i = 0; i < 5; i++) key = key * 53 + (uint64_t)s[i] + 1;
        auto it = ctx.river_buckets.find(key);
        if (it != ctx.river_buckets.end()) return it->second;
        const Combos& cb = combos();
        std::vector<uint8_t> b((size_t)NC, 255);
        if (!g_->bucketer->river_buckets_all(board5, cb.idx, b.data())) {
            uint64_t bm = 0;
            for (int i = 0; i < 5; i++) bm |= 1ULL << board5[i];
            for (int c = 0; c < NC; c++) {
                if (cb.mask[c] & bm) continue;
                const int hole[2] = {cb.c0[c], cb.c1[c]};
                b[(size_t)c] = (uint8_t)g_->bucketer->bucket(hole, board5, 5);
            }
        }
        return ctx.river_buckets.emplace(key, std::move(b)).first->second;
    }

    // parent + action (if any) with x holding c, y holding d, the next board cards `runout` then the
    // other cards in increasing order (aivat_values._apply_probe); `deck` must outlive the state
    HandState probe(const Ctx& ctx, const HandState& parent, bool has_action, int atype, int amount, const int* c, const int* runout, int n_runout,
                    int* deck, int& dealt) const {
        int holes[2][2];
        holes[ctx.x][0] = c[0];
        holes[ctx.x][1] = c[1];
        holes[ctx.y][0] = ctx.d[0];
        holes[ctx.y][1] = ctx.d[1];
        uint64_t used = 0;
        int pos = 0;
        for (int s = 0; s < 2; s++)
            for (int i = 0; i < 2; i++) deck[pos++] = holes[s][i], used |= 1ULL << holes[s][i];
        for (int i = 0; i < parent.n_board; i++) deck[pos++] = parent.board[i], used |= 1ULL << parent.board[i];
        for (int i = 0; i < n_runout; i++) deck[pos++] = runout[i], used |= 1ULL << runout[i];
        for (int k = 0; k < 52 && pos < 52; k++) if (!(used >> k & 1)) deck[pos++] = k;
        HandState st(parent);
        st.deck = deck;
        st.deck_pos = 4 + parent.n_board;
        st.players[ctx.x].hole[0] = c[0];
        st.players[ctx.x].hole[1] = c[1];
        st.players[ctx.y].hole[0] = ctx.d[0];
        st.players[ctx.y].hole[1] = ctx.d[1];
        const int n0 = st.n_board;
        if (has_action) st.apply(atype, amount);
        dealt = st.n_board - n0;
        return st;
    }

    // two free cards (for probes whose x hole does not matter)
    static void any_hole(const Ctx& ctx, const int* board, int n_board, const std::vector<int>& fixed, int* c) {
        uint64_t used = (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);
        for (int i = 0; i < n_board; i++) used |= 1ULL << board[i];
        for (int f : fixed) used |= 1ULL << f;
        int n = 0;
        for (int k = 0; k < 52 && n < 2; k++) if (!(used >> k & 1)) c[n++] = k;
    }

    static bool folded(const HandState& st) { return st.n_active() == 1; }

    void terminal_values(Ctx& ctx, const HandState& st, const std::vector<char>& ok, std::vector<double>& out) const {
        const Combos& cb = combos();
        if (folded(st)) {
            const double net = (double)st.net(ctx.x);
            for (int c = 0; c < NC; c++) out[(size_t)c] = ok[(size_t)c] ? net : 0.0;
            return;
        }
        const double m = (double)std::min(st.players[ctx.x].invested, st.players[ctx.y].invested);
        int cards[7];
        for (int i = 0; i < 5; i++) cards[2 + i] = st.board[i];
        cards[0] = ctx.d[0];
        cards[1] = ctx.d[1];
        const int64_t sd = negp::evaluate(cards, 7);
        for (int c = 0; c < NC; c++) {
            if (!ok[(size_t)c]) continue;
            cards[0] = cb.c0[c];
            cards[1] = cb.c1[c];
            const int64_t sc = negp::evaluate(cards, 7);
            out[(size_t)c] = sc > sd ? m : (sc < sd ? -m : 0.0);
        }
    }

    // ---- heuristic v1: the value of the state reached by `action` from `parent`, next cards `fixed`
    void branch(Ctx& ctx, const HandState& parent, bool has_action, int atype, int amount, const std::vector<int>& fixed, int node,
                const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        std::fill(out.begin(), out.end(), 0.0);
        bool any = false;
        for (int c = 0; c < NC; c++) any = any || need[(size_t)c];
        if (!any) return;
        int deck[52], ph[2], dealt = 0;
        any_hole(ctx, parent.board, parent.n_board, fixed, ph);
        const HandState pr = probe(ctx, parent, has_action, atype, amount, ph, fixed.data(), (int)fixed.size(), deck, dealt);
        const int n_random = std::max(0, dealt - (int)fixed.size());
        int board[5];
        int nbrd = 0;
        for (int i = 0; i < parent.n_board; i++) board[nbrd++] = parent.board[i];
        for (int f : fixed) board[nbrd++] = f;
        if (pr.terminal) {
            if (folded(pr)) {
                const double net = (double)pr.net(ctx.x);
                for (int c = 0; c < NC; c++) if (need[(size_t)c]) out[(size_t)c] = net;
                return;
            }
            const double m = (double)std::min(pr.players[ctx.x].invested, pr.players[ctx.y].invested);
            const int missing = 5 - nbrd;
            if (missing == 0) return showdown(ctx, board, m, need, out);
            if (missing <= 2) return equity_exact(ctx, board, nbrd, m, need, out);
            return equity_mc(ctx, board, nbrd, m, node, need, out);
        }
        if (n_random == 0 && pr.street == RIVER) return river_tree(ctx, pr, need, out);
        if (vp_.turn_exact && n_random == 0 && pr.street == TURN) return turn_exact(ctx, pr, need, out);
        if (n_random == 1 && pr.street == RIVER) return river_card(ctx, parent, has_action, atype, amount, fixed, need, out);
        const int k = vp_.rollouts[std::min(pr.street, 3)];
        if (k <= 0) throw std::runtime_error("aivat: no rollouts set for street " + std::to_string(pr.street));
        const uint64_t hid = (uint64_t)ctx.hand->hand_id;
        // every rollout's state has the probe's events (the holes and cards do not change them)
        HistHash hh0;
        hh0.catch_up(pr, g_->grid, ctx.tok);
        Start start;
        start.tmpl = &pr;
        start.dealt = dealt;
        start.hh = &hh0;
        // the rollouts (c, r) in order, LANES in flight at a time; each value is summed in r order as before
        std::vector<int> list;
        for (int c = 0; c < NC; c++) if (need[(size_t)c]) list.push_back(c);
        const int n_jobs = (int)list.size() * k;
        std::vector<double> vals((size_t)n_jobs, 0.0);
        const int LANES = lanes_in_flight();
        Lane lanes[MAX_LANES];
        Game::Memo& mm = g_->memo();
        int next = 0, active = 0;
        auto load = [&](Lane& L) {
            while (next < n_jobs) {
                const int job = next++;
                const int c = list[(size_t)(job / k)], r = job % k;
                const int hole[2] = {cb.c0[c], cb.c1[c]};
                L.rng = CounterRng(stream_seed({vp_.seed, hid, (uint64_t)node, (uint64_t)c, (uint64_t)r}));
                lane_init(ctx, L, parent, has_action, atype, amount, fixed, hole, &start);
                L.job = job;
                if (!L.st.terminal) return true;
                vals[(size_t)job] = (double)L.st.net(ctx.x);
            }
            L.job = -1;
            return false;
        };
        for (int w = 0; w < LANES; w++) active += load(lanes[w]) ? 1 : 0;
        while (active > 0)
            for (int w = 0; w < LANES; w++) {
                Lane& L = lanes[w];
                if (L.job < 0) continue;
                lane_step(ctx, L, mm);
                if (!L.st.terminal) continue;
                vals[(size_t)L.job] = (double)L.st.net(ctx.x);
                if (!load(L)) active--;
            }
        for (size_t i = 0; i < list.size(); i++) {
            double tot = 0.0;
            for (int r = 0; r < k; r++) tot += vals[i * (size_t)k + (size_t)r];
            out[(size_t)list[i]] = tot / k;
        }
    }

    void showdown(Ctx& ctx, const int* board, double m, const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        int cards[7];
        for (int i = 0; i < 5; i++) cards[2 + i] = board[i];
        cards[0] = ctx.d[0];
        cards[1] = ctx.d[1];
        const int64_t sd = negp::evaluate(cards, 7);
        for (int c = 0; c < NC; c++) {
            if (!need[(size_t)c]) continue;
            cards[0] = cb.c0[c];
            cards[1] = cb.c1[c];
            const int64_t sc = negp::evaluate(cards, 7);
            out[(size_t)c] = sc > sd ? m : (sc < sd ? -m : 0.0);
        }
    }

    // m * (P(win) - P(lose)) over every run-out of the 1 or 2 missing cards: per run-out y's hand is
    // evaluated once, then every combo that does not hold a run-out card (the counts are integers,
    // so the order of the loops does not change the numbers of aivat_values._equity_exact)
    void equity_exact(Ctx& ctx, const int* board, int nbrd, double m, const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        uint64_t known = (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);
        for (int i = 0; i < nbrd; i++) known |= 1ULL << board[i];
        const int missing = 5 - nbrd;
        int cx[7], cy[7];
        for (int i = 0; i < nbrd; i++) cx[2 + i] = cy[2 + i] = board[i];
        cy[0] = ctx.d[0];
        cy[1] = ctx.d[1];
        int rest[52], nr = 0;
        for (int k = 0; k < 52; k++) if (!(known >> k & 1)) rest[nr++] = k;
        std::vector<int> list;
        for (int c = 0; c < NC; c++) if (need[(size_t)c]) list.push_back(c);
        std::vector<long long> tot(list.size(), 0), cnt(list.size(), 0);
        auto one = [&](uint64_t rmask) {
            const int64_t sd = negp::evaluate(cy, 7);
            for (size_t t = 0; t < list.size(); t++) {
                const int c = list[t];
                if (cb.mask[c] & rmask) continue;
                cx[0] = cb.c0[c];
                cx[1] = cb.c1[c];
                const int64_t sc = negp::evaluate(cx, 7);
                tot[t] += sc > sd ? 1 : (sc < sd ? -1 : 0);
                cnt[t]++;
            }
        };
        if (missing == 1) {
            for (int i = 0; i < nr; i++) {
                cx[6] = cy[6] = rest[i];
                one(1ULL << rest[i]);
            }
        } else {
            for (int i = 0; i < nr; i++)
                for (int j = i + 1; j < nr; j++) {
                    cx[5] = cy[5] = rest[i];
                    cx[6] = cy[6] = rest[j];
                    one((1ULL << rest[i]) | (1ULL << rest[j]));
                }
        }
        for (size_t t = 0; t < list.size(); t++) out[(size_t)list[t]] = m * (double)tot[t] / (double)cnt[t];
    }

    void equity_mc(Ctx& ctx, const int* board, int nbrd, double m, int node, const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        uint64_t known = (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);
        for (int i = 0; i < nbrd; i++) known |= 1ULL << board[i];
        const int missing = 5 - nbrd;
        const uint64_t hid = (uint64_t)ctx.hand->hand_id;
        int cx[7], cy[7];
        for (int i = 0; i < nbrd; i++) cx[2 + i] = cy[2 + i] = board[i];
        cy[0] = ctx.d[0];
        cy[1] = ctx.d[1];
        for (int c = 0; c < NC; c++) {
            if (!need[(size_t)c]) continue;
            CounterRng rng(stream_seed({vp_.seed, hid, (uint64_t)node, (uint64_t)c, EQ_STREAM}));
            const uint64_t used = known | cb.mask[c];
            int base_rest[52], nb = 0;
            for (int k = 0; k < 52; k++) if (!(used >> k & 1)) base_rest[nb++] = k;
            cx[0] = cb.c0[c];
            cx[1] = cb.c1[c];
            long long tot = 0;
            for (int s = 0; s < vp_.eq_samples; s++) {
                int rest[52];
                std::memcpy(rest, base_rest, sizeof(int) * (size_t)nb);
                int nr = nb;
                for (int j = 0; j < missing; j++) {
                    const int i = (int)(rng.uniform() * (double)nr);
                    cx[2 + nbrd + j] = cy[2 + nbrd + j] = rest[i];
                    std::memmove(rest + i, rest + i + 1, sizeof(int) * (size_t)(nr - i - 1));
                    nr--;
                }
                const int64_t sc = negp::evaluate(cx, 7), sd = negp::evaluate(cy, 7);
                tot += sc > sd ? 1 : (sc < sd ? -1 : 0);
            }
            out[(size_t)c] = m * (double)tot / (double)vp_.eq_samples;
        }
    }

    // (A, S) of the river subtree per needed x bucket: V(c) = A[b(c)] + S[b(c)] s(c)
    void river_walk(Ctx& ctx, const HandState& s, HistHash hh, int by, const std::vector<int>& needed, double* A, double* S) const {
        const size_t nb = needed.size();
        if (s.terminal) {
            if (folded(s)) {
                const double net = (double)s.net(ctx.x);
                for (size_t i = 0; i < nb; i++) A[i] = net, S[i] = 0.0;
            } else {
                const double m = (double)std::min(s.players[ctx.x].invested, s.players[ctx.y].invested);
                for (size_t i = 0; i < nb; i++) A[i] = 0.0, S[i] = m;
            }
            return;
        }
        const BetGrid& grid = g_->grid;
        const int seat = s.to_act;
        const Obs obs = observe(s, seat);
        ActionList al;
        grid.abstract_actions(obs, al);
        hh.catch_up(s, grid, ctx.tok);
        int ctype[8], camt[8];
        for (int i = 0; i < al.n; i++) grid.to_concrete(obs, al.a[i], ctype[i], camt[i]);
        const int rel = ((seat - s.button) % 2 + 2) % 2;
        const int n_active = s.n_active();
        Game::Memo& mm = g_->memo();
        for (size_t i = 0; i < nb; i++) A[i] = 0.0, S[i] = 0.0;
        std::vector<double> A2(nb), S2(nb);
        if (seat == ctx.y) {
            double p[8];
            if (!g_->policy_memo(mm, node_key(s.street, rel, n_active, by, hh), al, p))
                for (int j = 0; j < al.n; j++) p[j] = al.a[j].id == 1 ? 1.0 : 0.0;
            for (int j = 0; j < al.n; j++) {
                if (p[j] == 0.0) continue;
                HandState ch(s);
                ch.apply(ctype[j], ctype[j] == RAISE ? camt[j] : 0);
                river_walk(ctx, ch, hh, by, needed, A2.data(), S2.data());
                for (size_t i = 0; i < nb; i++) {
                    A[i] = A[i] + p[j] * A2[i];
                    S[i] = S[i] + p[j] * S2[i];
                }
            }
            return;
        }
        std::vector<std::array<double, 8>> P(nb);
        for (size_t i = 0; i < nb; i++) {
            double p[8];
            if (!g_->policy_memo(mm, node_key(s.street, rel, n_active, needed[i], hh), al, p))
                for (int j = 0; j < al.n; j++) p[j] = al.a[j].id == 1 ? 1.0 : 0.0;
            for (int j = 0; j < al.n; j++) P[i][(size_t)j] = p[j];
        }
        for (int j = 0; j < al.n; j++) {
            bool anyp = false;
            for (size_t i = 0; i < nb; i++) anyp = anyp || P[i][(size_t)j] != 0.0;
            if (!anyp) continue;
            HandState ch(s);
            ch.apply(ctype[j], ctype[j] == RAISE ? camt[j] : 0);
            river_walk(ctx, ch, hh, by, needed, A2.data(), S2.data());
            for (size_t i = 0; i < nb; i++) {
                A[i] = A[i] + P[i][(size_t)j] * A2[i];
                S[i] = S[i] + P[i][(size_t)j] * S2[i];
            }
        }
    }

    void river_tree(Ctx& ctx, const HandState& st, const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        const std::vector<uint8_t>& bk = river_buckets(ctx, st.board);
        std::vector<int> needed;
        for (int c = 0; c < NC; c++)
            if (need[(size_t)c] && std::find(needed.begin(), needed.end(), (int)bk[(size_t)c]) == needed.end()) needed.push_back(bk[(size_t)c]);
        std::sort(needed.begin(), needed.end());
        const int by = g_->bucketer->bucket(ctx.d, st.board, 5);
        std::vector<double> A(needed.size()), S(needed.size());
        ctx.river_trees++;
        river_walk(ctx, st, HistHash(), by, needed, A.data(), S.data());
        int cards[7];
        for (int i = 0; i < 5; i++) cards[2 + i] = st.board[i];
        cards[0] = ctx.d[0];
        cards[1] = ctx.d[1];
        const int64_t sd = negp::evaluate(cards, 7);
        for (int c = 0; c < NC; c++) {
            if (!need[(size_t)c]) continue;
            cards[0] = cb.c0[c];
            cards[1] = cb.c1[c];
            const int64_t sc = negp::evaluate(cards, 7);
            const double s = sc > sd ? 1.0 : (sc < sd ? -1.0 : 0.0);
            const size_t i = (size_t)(std::lower_bound(needed.begin(), needed.end(), (int)bk[(size_t)c]) - needed.begin());
            out[(size_t)c] = A[i] + S[i] * s;
        }
    }

    // ---- v2 "turn exact": the value of turn state `st` (turn card known, decisions ahead) for every needed combo:
    // E over the river card (uniform over the cards off the board, y's hole and the combo) and over both players'
    // blueprint policies (per bucket, unknown keys check / call, as the rollouts play) of x's net.  The rollouts of
    // v1 estimate the same expectation (their law: the same policies sampled with one uniform per decision).
    // Walks: the turn's betting tree with x's probabilities per turn bucket; at each path into the river the river's
    // tree once, with x's probabilities per river bucket and y's per its river bucket (the tree does not depend on
    // the river card, only y's bucket does); the combos' values are then summed over the river cards.
    struct TurnEx {
        std::vector<int> tb;             // x's turn bucket index (into xb_of) per combo, -1 not needed
        std::vector<int> xb_of;          // x's turn buckets needed
        int nT = 0, by_t = 0;
        std::vector<int> rcards;         // the river cards (off the board and y's hole)
        std::vector<int> ky;             // per river card: index of y's river bucket in ybs
        std::vector<int> ybs;            // distinct y river buckets
        std::vector<int> xbs;            // x's river buckets needed (sorted)
        std::vector<int> ir;             // [card * NC + c]: index into xbs, -1 when the card is in c or c not needed
        std::vector<int8_t> sg;          // [card * NC + c]: x's showdown sign
        std::vector<double> inv_cnt;     // per combo: 1 / number of river cards off it
        std::vector<double> fold_acc, eq_acc;  // per turn bucket: sum of path weight x net at folds; x m at all-in showdowns
        std::vector<double> val;         // per combo: the river paths' part
    };
    void river_leafs(Ctx& ctx, Game::Memo& mm, const HandState& s, HistHash hh, const TurnEx& T, const std::vector<double>& px,
                     const std::vector<double>& py, std::vector<double>& A, std::vector<double>& S) const {
        const size_t nX = T.xbs.size(), nY = T.ybs.size();
        if (s.terminal) {
            const bool fold = folded(s);
            const double v = fold ? (double)s.net(ctx.x) : (double)std::min(s.players[ctx.x].invested, s.players[ctx.y].invested);
            std::vector<double>& D = fold ? A : S;
            for (size_t i = 0; i < nX; i++) {
                if (px[i] == 0.0) continue;
                const double a = px[i] * v;
                for (size_t k = 0; k < nY; k++) D[i * nY + k] += a * py[k];
            }
            return;
        }
        const BetGrid& grid = g_->grid;
        const int seat = s.to_act;
        const Obs obs = observe(s, seat);
        ActionList al;
        grid.abstract_actions(obs, al);
        hh.catch_up(s, grid, ctx.tok);
        int ctype[8], camt[8];
        for (int i = 0; i < al.n; i++) grid.to_concrete(obs, al.a[i], ctype[i], camt[i]);
        const int rel = ((seat - s.button) % 2 + 2) % 2;
        const int n_active = s.n_active();
        const bool is_y = seat == ctx.y;
        const std::vector<int>& bs = is_y ? T.ybs : T.xbs;
        const std::vector<double>& pin = is_y ? py : px;
        std::vector<std::array<double, 8>> P(bs.size());
        for (size_t i = 0; i < bs.size(); i++) {
            double p[8];
            if (pin[i] == 0.0 || !g_->policy_memo(mm, node_key(s.street, rel, n_active, bs[i], hh), al, p))
                for (int j = 0; j < al.n; j++) p[j] = al.a[j].id == 1 ? 1.0 : 0.0;
            for (int j = 0; j < al.n; j++) P[i][(size_t)j] = p[j];
        }
        std::vector<double> pn(bs.size());
        for (int j = 0; j < al.n; j++) {
            bool any = false;
            for (size_t i = 0; i < bs.size(); i++) {
                pn[i] = pin[i] * P[i][(size_t)j];
                any = any || pn[i] != 0.0;
            }
            if (!any) continue;
            HandState ch(s);
            ch.apply(ctype[j], ctype[j] == RAISE ? camt[j] : 0);
            if (is_y) river_leafs(ctx, mm, ch, hh, T, px, pn, A, S);
            else river_leafs(ctx, mm, ch, hh, T, pn, py, A, S);
        }
    }
    void turn_walk(Ctx& ctx, Game::Memo& mm, const HandState& s, HistHash hh, TurnEx& T, const std::vector<double>& px, double py) const {
        const Combos& cb = combos();
        if (s.terminal) {
            const bool fold = folded(s);
            const double v = fold ? (double)s.net(ctx.x) : (double)std::min(s.players[ctx.x].invested, s.players[ctx.y].invested);
            std::vector<double>& D = fold ? T.fold_acc : T.eq_acc;  // (not folded: an all-in, the river run out)
            for (int i = 0; i < T.nT; i++) D[(size_t)i] += px[(size_t)i] * py * v;
            return;
        }
        const BetGrid& grid = g_->grid;
        if (s.street == RIVER) {  // a path into the river: its tree, then the combos over the river cards
            const size_t nX = T.xbs.size(), nY = T.ybs.size();
            std::vector<double> A(nX * nY, 0.0), S(nX * nY, 0.0);
            ctx.river_trees++;
            river_leafs(ctx, mm, s, hh, T, std::vector<double>(nX, 1.0), std::vector<double>(nY, 1.0), A, S);
            const size_t NR = T.rcards.size();
            for (int c = 0; c < NC; c++) {
                const int t = T.tb[(size_t)c];
                if (t < 0 || px[(size_t)t] == 0.0) continue;
                double tot = 0.0;
                for (size_t r = 0; r < NR; r++) {
                    const int i = T.ir[r * NC + (size_t)c];
                    if (i < 0) continue;
                    const size_t q = (size_t)i * nY + (size_t)T.ky[r];
                    tot += A[q] + S[q] * (double)T.sg[r * NC + (size_t)c];
                }
                T.val[(size_t)c] += px[(size_t)t] * py * tot * T.inv_cnt[(size_t)c];
            }
            (void)cb;
            return;
        }
        const int seat = s.to_act;
        const Obs obs = observe(s, seat);
        ActionList al;
        grid.abstract_actions(obs, al);
        hh.catch_up(s, grid, ctx.tok);
        int ctype[8], camt[8];
        for (int i = 0; i < al.n; i++) grid.to_concrete(obs, al.a[i], ctype[i], camt[i]);
        const int rel = ((seat - s.button) % 2 + 2) % 2;
        const int n_active = s.n_active();
        if (seat == ctx.y) {
            double p[8];
            if (!g_->policy_memo(mm, node_key(s.street, rel, n_active, T.by_t, hh), al, p))
                for (int j = 0; j < al.n; j++) p[j] = al.a[j].id == 1 ? 1.0 : 0.0;
            for (int j = 0; j < al.n; j++) {
                if (p[j] == 0.0) continue;
                HandState ch(s);
                ch.apply(ctype[j], ctype[j] == RAISE ? camt[j] : 0);
                turn_walk(ctx, mm, ch, hh, T, px, py * p[j]);
            }
            return;
        }
        std::vector<std::array<double, 8>> P((size_t)T.nT);
        for (int i = 0; i < T.nT; i++) {
            double p[8];
            if (px[(size_t)i] == 0.0 || !g_->policy_memo(mm, node_key(s.street, rel, n_active, T.xb_of[(size_t)i], hh), al, p))
                for (int j = 0; j < al.n; j++) p[j] = al.a[j].id == 1 ? 1.0 : 0.0;
            for (int j = 0; j < al.n; j++) P[(size_t)i][(size_t)j] = p[j];
        }
        std::vector<double> pn((size_t)T.nT);
        for (int j = 0; j < al.n; j++) {
            bool any = false;
            for (int i = 0; i < T.nT; i++) {
                pn[(size_t)i] = px[(size_t)i] * P[(size_t)i][(size_t)j];
                any = any || pn[(size_t)i] != 0.0;
            }
            if (!any) continue;
            HandState ch(s);
            ch.apply(ctype[j], ctype[j] == RAISE ? camt[j] : 0);
            turn_walk(ctx, mm, ch, hh, T, pn, py);
        }
    }
    const std::vector<int8_t>& river_signs(Ctx& ctx, const int* board5) const {
        int s5[5];
        std::copy(board5, board5 + 5, s5);
        std::sort(s5, s5 + 5);
        uint64_t key = 0;
        for (int i = 0; i < 5; i++) key = key * 53 + (uint64_t)s5[i] + 1;
        auto it = ctx.river_signs.find(key);
        if (it != ctx.river_signs.end()) return it->second;
        const Combos& cb = combos();
        std::vector<int8_t> sg((size_t)NC, 0);
        int cards[7];
        for (int i = 0; i < 5; i++) cards[2 + i] = board5[i];
        cards[0] = ctx.d[0];
        cards[1] = ctx.d[1];
        const int64_t sd = negp::evaluate(cards, 7);
        uint64_t bm = 0;
        for (int i = 0; i < 5; i++) bm |= 1ULL << board5[i];
        for (int c = 0; c < NC; c++) {
            if (cb.mask[c] & bm) continue;
            cards[0] = cb.c0[c];
            cards[1] = cb.c1[c];
            const int64_t sc = negp::evaluate(cards, 7);
            sg[(size_t)c] = sc > sd ? 1 : (sc < sd ? -1 : 0);
        }
        return ctx.river_signs.emplace(key, std::move(sg)).first->second;
    }
    void turn_exact(Ctx& ctx, const HandState& st, const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        TurnEx T;
        const std::vector<int>& bk = buckets_of(ctx, st.board, 4);  // x's turn buckets (the agent model's)
        T.tb.assign((size_t)NC, -1);
        std::vector<int> xb_of;
        for (int c = 0; c < NC; c++) {
            if (!need[(size_t)c]) continue;
            const int b = bk[(size_t)c];
            auto it = std::find(xb_of.begin(), xb_of.end(), b);
            if (it == xb_of.end()) {
                xb_of.push_back(b);
                it = xb_of.end() - 1;
            }
            T.tb[(size_t)c] = (int)(it - xb_of.begin());
        }
        T.nT = (int)xb_of.size();
        T.xb_of = xb_of;
        T.by_t = g_->bucketer->bucket(ctx.d, st.board, 4);
        uint64_t known = (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);
        for (int i = 0; i < 4; i++) known |= 1ULL << st.board[i];
        for (int r = 0; r < 52; r++) if (!(known >> r & 1)) T.rcards.push_back(r);
        const size_t NR = T.rcards.size();
        T.ir.assign(NR * NC, -1);
        T.sg.assign(NR * NC, 0);
        T.ky.assign(NR, 0);
        T.inv_cnt.assign((size_t)NC, 0.0);
        std::vector<int> cnt((size_t)NC, 0);
        std::vector<int> xpos(256, -1);
        for (size_t r = 0; r < NR; r++) {
            int b5[5] = {st.board[0], st.board[1], st.board[2], st.board[3], T.rcards[r]};
            const std::vector<uint8_t>& rb = river_buckets(ctx, b5);
            const std::vector<int8_t>& sg = river_signs(ctx, b5);
            const int by = g_->bucketer->bucket(ctx.d, b5, 5);
            auto it = std::find(T.ybs.begin(), T.ybs.end(), by);
            if (it == T.ybs.end()) {
                T.ybs.push_back(by);
                it = T.ybs.end() - 1;
            }
            T.ky[r] = (int)(it - T.ybs.begin());
            for (int c = 0; c < NC; c++) {
                if (T.tb[(size_t)c] < 0 || (cb.mask[c] >> T.rcards[r] & 1)) continue;
                const int b = rb[(size_t)c];
                if (xpos[(size_t)b] < 0) {
                    xpos[(size_t)b] = (int)T.xbs.size();
                    T.xbs.push_back(b);
                }
                T.ir[r * NC + (size_t)c] = xpos[(size_t)b];
                T.sg[r * NC + (size_t)c] = sg[(size_t)c];
                cnt[(size_t)c]++;
            }
        }
        for (int c = 0; c < NC; c++) if (cnt[(size_t)c] > 0) T.inv_cnt[(size_t)c] = 1.0 / (double)cnt[(size_t)c];
        T.fold_acc.assign((size_t)T.nT, 0.0);
        T.eq_acc.assign((size_t)T.nT, 0.0);
        T.val.assign((size_t)NC, 0.0);
        HistHash hh;
        hh.catch_up(st, g_->grid, ctx.tok);
        Game::Memo& mm = g_->memo();
        turn_walk(ctx, mm, st, hh, T, std::vector<double>((size_t)T.nT, 1.0), 1.0);
        for (int c = 0; c < NC; c++) {
            const int t = T.tb[(size_t)c];
            if (t < 0) continue;
            double eq = 0.0;  // x's mean showdown sign over the river cards (an all-in on the turn)
            if (T.eq_acc[(size_t)t] != 0.0) {
                for (size_t r = 0; r < NR; r++)
                    if (T.ir[r * NC + (size_t)c] >= 0) eq += (double)T.sg[r * NC + (size_t)c];
                eq *= T.inv_cnt[(size_t)c];
            }
            out[(size_t)c] = T.fold_acc[(size_t)t] + T.eq_acc[(size_t)t] * eq + T.val[(size_t)c];
        }
    }

    void river_card(Ctx& ctx, const HandState& parent, bool has_action, int atype, int amount, const std::vector<int>& fixed,
                    const std::vector<char>& need, std::vector<double>& out) const {
        const Combos& cb = combos();
        uint64_t known = (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);
        for (int i = 0; i < parent.n_board; i++) known |= 1ULL << parent.board[i];
        for (int f : fixed) known |= 1ULL << f;
        std::vector<double> tot((size_t)NC, 0.0), cnt((size_t)NC, 0.0), vals((size_t)NC, 0.0);
        std::vector<char> sub((size_t)NC, 0);
        for (int r = 0; r < 52; r++) {
            if (known >> r & 1) continue;
            bool anyc = false;
            for (int c = 0; c < NC; c++) {
                sub[(size_t)c] = need[(size_t)c] && !(cb.mask[c] >> r & 1);
                anyc = anyc || sub[(size_t)c];
            }
            if (!anyc) continue;
            std::vector<int> f2(fixed);
            f2.push_back(r);
            int deck[52], ph[2], dealt = 0;
            any_hole(ctx, parent.board, parent.n_board, f2, ph);
            const HandState pr = probe(ctx, parent, has_action, atype, amount, ph, f2.data(), (int)f2.size(), deck, dealt);
            std::fill(vals.begin(), vals.end(), 0.0);
            river_tree(ctx, pr, sub, vals);
            for (int c = 0; c < NC; c++)
                if (sub[(size_t)c]) {
                    tot[(size_t)c] = tot[(size_t)c] + vals[(size_t)c];
                    cnt[(size_t)c] += 1.0;
                }
        }
        for (int c = 0; c < NC; c++) if (need[(size_t)c]) out[(size_t)c] = tot[(size_t)c] / cnt[(size_t)c];
    }

    // one blueprint self-play rollout (aivat_values.SelfPlayValues._rollout): the unknown board
    // cards first (index floor(u * n) into the sorted remaining cards), then one uniform per decision
    // A branch's rollouts share its probe state `tmpl` (probe() with any two holes): the action's events and
    // whether the hand ended do not depend on the cards, so a rollout's state is `tmpl` with x's hole, the
    // board cards the action dealt taken from the run-out, and the rollout's deck; `hh_start`: its events hashed.
    struct Start {
        const HandState* tmpl = nullptr;
        int dealt = 0;
        const HistHash* hh = nullptr;
    };
    // One rollout in flight: rollout() runs one to the end; branch() keeps several in flight and steps them in
    // turn, so the cache misses of one overlap the work of the others (each has its own random stream and
    // state, so the order of the steps does not change any number).
    struct Lane {
        HandState st;
        int deck[52];
        int full[5];
        int memo[2][6];
        Game::Ahead ah[2][4];
        bool ahead = false;
        HistHash hh;
        CounterRng rng{0};
        int job = -1;
    };
    // the unknown board cards first (index floor(u * n) into the sorted remaining cards), the state, the
    // lookups ahead; L.rng must be set
    void lane_init(Ctx& ctx, Lane& L, const HandState& parent, bool has_action, int atype, int amount, const std::vector<int>& fixed, const int* c,
                   const Start* start) const {
        uint64_t known = (1ULL << c[0]) | (1ULL << c[1]) | (1ULL << ctx.d[0]) | (1ULL << ctx.d[1]);
        for (int i = 0; i < parent.n_board; i++) known |= 1ULL << parent.board[i];
        for (int f : fixed) known |= 1ULL << f;
        int rest[52], nr = 0;
        for (int k = 0; k < 52; k++) if (!(known >> k & 1)) rest[nr++] = k;
        int runout[5], nrun = 0;
        for (int f : fixed) runout[nrun++] = f;
        const int need = 5 - parent.n_board - (int)fixed.size();
        for (int j = 0; j < need; j++) {
            const int i = (int)(L.rng.uniform() * (double)nr);
            runout[nrun++] = rest[i];
            std::memmove(rest + i, rest + i + 1, sizeof(int) * (size_t)(nr - i - 1));
            nr--;
        }
        int dealt = 0;
        HandState& st = L.st;
        int* deck = L.deck;
        if (start && start->tmpl) {  // = probe(ctx, parent, has_action, atype, amount, c, runout, nrun, deck, dealt)
            int pos = 0;
            for (int s = 0; s < 2; s++) {
                const int* h = s == ctx.x ? c : ctx.d;
                deck[pos++] = h[0];
                deck[pos++] = h[1];
            }
            for (int i = 0; i < parent.n_board; i++) deck[pos++] = parent.board[i];
            for (int i = 0; i < nrun; i++) deck[pos++] = runout[i];
            for (int i = 0; i < nr; i++) deck[pos++] = rest[i];  // the other cards, increasing
            st = *start->tmpl;
            dealt = start->dealt;
            st.deck = deck;
            st.players[ctx.x].hole[0] = c[0];
            st.players[ctx.x].hole[1] = c[1];
            for (int i = 0; i < dealt; i++) st.board[parent.n_board + i] = runout[i];
        } else {
            st = probe(ctx, parent, has_action, atype, amount, c, runout, nrun, deck, dealt);
        }
        ctx.rollouts++;
        for (int s = 0; s < 2; s++)
            for (int j = 0; j < 6; j++) L.memo[s][j] = -1;
        Game::Memo& mm = g_->memo();
        L.hh = HistHash();
        if (start && start->hh && start->hh->n <= st.n_events) L.hh = *start->hh;  // the events of the branch's state, hashed once
        // the buckets of both players on the later streets of this run-out, looked up ahead (Ahead above)
        for (int i = 0; i < parent.n_board; i++) L.full[i] = parent.board[i];
        for (int i = 0; i < nrun; i++) L.full[parent.n_board + i] = runout[i];
        for (int s = 0; s < 2; s++)
            for (int street = 0; street < 4; street++) L.ah[s][street] = Game::Ahead();
        L.ahead = false;
        if (g_->tabs && parent.n_board + nrun == 5 && !st.terminal)
            for (int street = FLOP; street <= RIVER; street++) {
                const int len = street == FLOP ? 3 : (street == TURN ? 4 : 5);
                if (len <= st.n_board || !g_->tab_on[street]) continue;
                for (int s = 0; s < 2; s++) g_->ahead_start(mm, L.ah[s][street], st.players[s].hole, L.full, len);
                L.ahead = true;
            }
    }
    // one decision of a lane whose state is not terminal: one uniform per decision (the agent's rule)
    void lane_step(Ctx& ctx, Lane& L, Game::Memo& mm) const {
        const BetGrid& grid = g_->grid;
        HandState& st = L.st;
        HistHash& hh = L.hh;
        const int seat = st.to_act;
        while (hh.n < st.n_events) g_->hash_event(mm, hh, st.events[hh.n]);  // = hh.catch_up(st, grid, ctx.tok)
        if (L.ahead && st.street + 1 <= RIVER)  // the next street's lookups: a street of betting to hide the miss
            for (int s = 0; s < 2; s++)
                if (L.ah[s][st.street + 1].stage == 1) g_->ahead_step(mm, L.ah[s][st.street + 1], st.street + 1, st.players[s].hole, L.full);
        int& b = L.memo[seat][st.n_board];
        if (b < 0) {
            Game::Ahead& a = L.ah[seat][st.street];
            if (st.n_board > 0 && a.stage > 0)
                b = g_->ahead_bucket(mm, a, st.street, st.players[seat].hole, L.full);
            else
                b = g_->bucket_memo(mm, st.players[seat].hole, st.board, st.n_board);
        }
        const int rel = ((seat - st.button) % 2 + 2) % 2;
        const NodeKey key = node_key(st.street, rel, st.n_active(), b, hh);
        NEGP_PREFETCH(&g_->pol_slot(mm, key));  // (its line is read after the legal actions)
        g_->blueprint->prefetch(key);                  // (read on a memo miss)
        const Obs obs = observe(st, seat);
        ActionList al;
        grid.abstract_actions(obs, al);
        double p[8];
        const bool have = g_->policy_memo(mm, key, al, p);
        const double u = L.rng.uniform();
        int j = al.n - 1;
        if (!have) {
            for (int i = 0; i < al.n; i++) if (al.a[i].id == 1) j = i;
        } else {
            double acc = 0.0;
            for (int i = 0; i < al.n; i++) {
                acc += p[i];
                if (u < acc) {
                    j = i;
                    break;
                }
            }
        }
        int type, amt;
        grid.to_concrete(obs, al.a[j], type, amt);
        st.apply(type, type == RAISE ? amt : 0);
        ctx.rollout_steps++;
    }
    double rollout(Ctx& ctx, const HandState& parent, bool has_action, int atype, int amount, const std::vector<int>& fixed, const int* c,
                   CounterRng& rng, const Start* start = nullptr) const {
        Lane L;
        L.rng = rng;
        lane_init(ctx, L, parent, has_action, atype, amount, fixed, c, start);
        Game::Memo& mm = g_->memo();
        while (!L.st.terminal) lane_step(ctx, L, mm);
        rng = L.rng;
        return (double)L.st.net(ctx.x);
    }
};

}  // namespace aiv
}  // namespace negp
