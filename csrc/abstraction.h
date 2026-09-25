// Port of negpluribus/abstraction/*: bet grid with deterministic pseudo-harmonic translation,
// suit-canonical form, E[HS] bucketer (Monte-Carlo on the canonical representative, seeded from
// the CPython tuple hash of the canonical key, cached per canonical form), potential-aware
// bucketer (next-street equity histogram on the canonical representative, nearest EMD centroid;
// negpluribus/abstraction/potential.py) and the infoset key string, byte for byte as
// negpluribus/abstraction/infoset.py builds it: "F|BB|2|b6|r1 c/c".
//
// Bucket caches are BOUNDED (FormCache below): the number of distinct canonical forms a long
// run visits is large on the turn and river (13,960,050 / 123,156,254 forms of our sorted-board
// key exist; docs/backends.md), so the per-form caches
// are fixed-size, lock-free, set-associative tables with a per-street capacity set at
// construction; memory stays flat.  Because every cached value is a pure function of the
// canonical key, eviction only costs a recomputation: results are bit-identical whatever the cap.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine.h"
#include "equity.h"
#include "pyrandom.h"

namespace negp {

// ------------------------------------------------------------------ observation
struct Obs {
    int seat, street, pot, to_call, stack, street_bets_max;
    int min_raise_to, max_raise_to;
    bool can_raise, can_fold;
    int raises_this_street, n_active;
};

inline Obs observe(const HandState& st, int seat) {
    Obs o;
    o.seat = seat;
    o.street = st.street;
    o.pot = st.pot();
    o.to_call = st.to_call_for(seat);
    o.stack = st.players[seat].stack;
    o.street_bets_max = st.street_bets_max();
    st.raise_bounds(seat, o.can_raise, o.min_raise_to, o.max_raise_to);
    o.can_fold = o.to_call > 0;
    o.raises_this_street = st.raises_this_street;
    o.n_active = st.n_active();
    return o;
}

// Python's round(float) -> int: ties to even
inline long long py_round(double x) { return (long long)std::nearbyint(x); }

// f"r{frac:g}"
inline std::string raise_name(double frac) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "r%g", frac);
    return std::string(buf);
}

// ------------------------------------------------------------------ bet grid
// Abstract actions carry a small integer id (0 "f", 1 "c", 2 "a", 3.. the raise fractions of
// the grid in preflop-then-postflop order); ``BetGrid::names`` maps id -> name.  Nodes store ids.
struct AbstractAction {
    int id;
    int kind;     // FOLD, CALL, RAISE, or ALL_IN_KIND
    double frac;  // RAISE only
};
constexpr int ALL_IN_KIND = 3;

struct ActionList {
    AbstractAction a[8];
    int n = 0;
    void clear() { n = 0; }
    void push(const AbstractAction& x) { a[n++] = x; }
};

struct BetGrid {
    std::vector<double> preflop_fracs;
    std::vector<double> postflop_fracs;
    bool allow_all_in = true;
    int max_raises_per_street = 4;
    bool forbid_open_limp = false;
    std::vector<std::string> names;  // id -> name
    std::vector<int> preflop_ids, postflop_ids;

    void finalize() {
        names = {"f", "c", "a"};
        preflop_ids.clear(); postflop_ids.clear();
        for (double f : preflop_fracs) { preflop_ids.push_back((int)names.size()); names.push_back(raise_name(f)); }
        for (double f : postflop_fracs) { postflop_ids.push_back((int)names.size()); names.push_back(raise_name(f)); }
    }

    int id_of(const std::string& s) {
        for (size_t i = 0; i < names.size(); i++) if (names[i] == s) return (int)i;
        names.push_back(s);  // foreign name (checkpoint from another grid): keep it addressable
        return (int)names.size() - 1;
    }

    const std::vector<double>& fracs_for(int street) const {
        return street == PREFLOP ? preflop_fracs : postflop_fracs;
    }
    const std::vector<int>& ids_for(int street) const {
        return street == PREFLOP ? preflop_ids : postflop_ids;
    }

    static long long raise_to_for_frac(const Obs& obs, double frac) {
        double pot_after_call = (double)(obs.pot + obs.to_call);
        double increment = frac * pot_after_call;
        return py_round((double)obs.street_bets_max + increment);
    }

    static int clamp_raise(const Obs& obs, long long amount) {
        long long a = std::max<long long>(obs.min_raise_to, std::min<long long>(obs.max_raise_to, amount));
        return (int)a;
    }

    void abstract_actions(const Obs& obs, ActionList& out) const {
        out.clear();
        if (obs.can_fold) out.push({0, FOLD, 0.0});
        bool limp = obs.street == PREFLOP && obs.raises_this_street == 0 && obs.to_call > 0;
        if (!(forbid_open_limp && limp && obs.can_raise)) out.push({1, CALL, 0.0});
        if (!obs.can_raise) return;
        bool capped = obs.raises_this_street >= max_raises_per_street;
        int seen[16]; int n_seen = 0;
        if (!capped) {
            const std::vector<double>& fracs = fracs_for(obs.street);
            const std::vector<int>& ids = ids_for(obs.street);
            for (size_t k = 0; k < fracs.size(); k++) {
                double frac = fracs[k];
                int amt = clamp_raise(obs, raise_to_for_frac(obs, frac));
                bool dup = false;
                for (int i = 0; i < n_seen; i++) if (seen[i] == amt) { dup = true; break; }
                if (dup) continue;
                if (amt == obs.max_raise_to && allow_all_in) continue;
                if (n_seen < 16) seen[n_seen++] = amt;
                out.push({ids[k], RAISE, frac});
            }
        }
        if (allow_all_in) {
            bool dup = false;
            for (int i = 0; i < n_seen; i++) if (seen[i] == obs.max_raise_to) { dup = true; break; }
            if (!dup) out.push({2, ALL_IN_KIND, 0.0});
        }
    }

    // -> (type, amount)
    void to_concrete(const Obs& obs, const AbstractAction& a, int& type, int& amount) const {
        amount = 0;
        if (a.kind == FOLD) { type = obs.can_fold ? FOLD : CALL; return; }
        if (a.kind == CALL) { type = CALL; return; }
        if (!obs.can_raise) { type = CALL; return; }
        type = RAISE;
        if (a.kind == ALL_IN_KIND) { amount = clamp_raise(obs, obs.max_raise_to); return; }
        amount = clamp_raise(obs, raise_to_for_frac(obs, a.frac));
    }

    // the abstract action with id `id` on `street`, exactly as abstract_actions() builds it
    AbstractAction action_from_id(int street, int id) const {
        if (id == 0) return {0, FOLD, 0.0};
        if (id == 1) return {1, CALL, 0.0};
        if (id == 2) return {2, ALL_IN_KIND, 0.0};
        const std::vector<int>& ids = ids_for(street);
        const std::vector<double>& fracs = fracs_for(street);
        for (size_t k = 0; k < ids.size(); k++) if (ids[k] == id) return {id, RAISE, fracs[k]};
        throw std::logic_error("unknown action id");
    }

    AbstractAction action_from_name(const std::string& s) {
        if (s == "f") return {0, FOLD, 0.0};
        if (s == "c") return {1, CALL, 0.0};
        if (s == "a") return {2, ALL_IN_KIND, 0.0};
        return {id_of(s), RAISE, std::strtod(s.c_str() + 1, nullptr)};
    }

    static double observed_frac(const Event& ev) {
        int increment = ev.paid - ev.to_call;
        int pot_after_call = ev.pot_before + ev.to_call;
        if (pot_after_call > 0) return std::max(0.0, (double)increment / (double)pot_after_call);
        return 0.0;
    }

    // pseudo-harmonic translation of x onto a SORTED, non-empty grid g[0..n)
    static double pseudo_harmonic_sorted(double x, const double* g, size_t n) {
        if (x <= g[0]) return g[0];
        if (x >= g[n - 1]) return g[n - 1];
        for (size_t i = 0; i + 1 < n; i++) {
            double a = g[i], b = g[i + 1];
            if (a <= x && x <= b) {
                double p_a = (b - x) * (1 + a) / ((b - a) * (1 + x));
                return p_a >= 0.5 ? a : b;
            }
        }
        return g[n - 1];
    }

    static double pseudo_harmonic(double x, std::vector<double> grid) {
        std::sort(grid.begin(), grid.end());
        return pseudo_harmonic_sorted(x, grid.data(), grid.size());
    }

    void append_frac_name(int street, double g, std::string& out) const {
        const std::vector<double>& fracs = fracs_for(street);
        const std::vector<int>& ids = ids_for(street);
        for (size_t k = 0; k < fracs.size(); k++) if (fracs[k] == g) { out += names[ids[k]]; return; }
        out += raise_name(g);
    }

    // deterministic translation (event_rng=None in the reference).  Ganzfried & Sandholm (IJCAI
    // 2013): translate between the two neighbouring ABSTRACT actions, the all-in being one of
    // them; grid sizes at or above the actor's all-in are the all-in in that spot.  Mirrors
    // BetGrid.from_concrete in negpluribus/abstraction/actions.py operation for operation.
    // (Since 2026-09-24 the sorted grid lives on the stack instead of two vector copies per raise:
    // the same sorted values, the same comparisons, so the same token.)
    void from_concrete(const Event& ev, std::string& out) const {
        if (ev.type == FOLD) { out += 'f'; return; }
        if (ev.type == CALL) { out += 'c'; return; }
        const std::vector<double>& fracs = fracs_for(ev.street);
        if (ev.all_in && allow_all_in) { out += 'a'; return; }
        if (fracs.empty()) { out += 'a'; return; }
        double x = observed_frac(ev);
        int pot_after_call = ev.pot_before + ev.to_call;
        double stack_buf[16];
        std::vector<double> heap_buf;
        const size_t n = fracs.size();
        double* g = stack_buf;
        if (n > 16) { heap_buf.assign(fracs.begin(), fracs.end()); g = heap_buf.data(); }
        else std::copy(fracs.begin(), fracs.end(), g);
        std::sort(g, g + n);
        if (allow_all_in && ev.stack_after >= 0 && pot_after_call > 0) {
            double x_allin = (double)(ev.paid + ev.stack_after - ev.to_call) / (double)pot_after_call;
            size_t nb = 0;  // the grid sizes below the actor's all-in, still sorted
            for (size_t i = 0; i < n; i++) if (g[i] < x_allin) g[nb++] = g[i];
            if (nb == 0) { out += 'a'; return; }
            double top = g[nb - 1];
            if (x > top) {
                double p_top = (x_allin - x) * (1 + top) / ((x_allin - top) * (1 + x));
                if (p_top >= 0.5) append_frac_name(ev.street, top, out); else out += 'a';
                return;
            }
            append_frac_name(ev.street, pseudo_harmonic_sorted(x, g, nb), out);
            return;
        }
        append_frac_name(ev.street, pseudo_harmonic_sorted(x, g, n), out);
    }

    void history_string(const Event* events, int n_events, std::string& out) const {
        out.clear();
        int cur = -1;
        for (int i = 0; i < n_events; i++) {
            const Event& ev = events[i];
            if (ev.street != cur) {
                if (cur != -1) out += '/';
                cur = ev.street;
            } else if (i > 0) {
                out += ' ';
            }
            from_concrete(ev, out);
        }
    }
};

// ------------------------------------------------------------- canonical form
struct CanonicalForm {
    int hole[2];
    int board[5];
    int n_board;
};

extern const int SUIT_PERMS[24][4];

inline void canonical_form(const int* hole, const int* board, int n_board, CanonicalForm& best) {
    bool first = true;
    for (int p = 0; p < 24; p++) {
        const int* perm = SUIT_PERMS[p];
        int h[2] = {(hole[0] >> 2) * 4 + perm[hole[0] & 3], (hole[1] >> 2) * 4 + perm[hole[1] & 3]};
        if (h[0] > h[1]) std::swap(h[0], h[1]);
        int b[5];
        for (int i = 0; i < n_board; i++) b[i] = (board[i] >> 2) * 4 + perm[board[i] & 3];
        std::sort(b, b + n_board);
        bool better = first;
        if (!first) {
            int c = 0;
            if (h[0] != best.hole[0]) c = h[0] < best.hole[0] ? -1 : 1;
            else if (h[1] != best.hole[1]) c = h[1] < best.hole[1] ? -1 : 1;
            else {
                for (int i = 0; i < n_board; i++) {
                    if (b[i] != best.board[i]) { c = b[i] < best.board[i] ? -1 : 1; break; }
                }
            }
            better = c < 0;
        }
        if (better) {
            best.hole[0] = h[0]; best.hole[1] = h[1];
            for (int i = 0; i < n_board; i++) best.board[i] = b[i];
            best.n_board = n_board;
            first = false;
        }
    }
}

// hash(canonical_key) as CPython computes it for the tuple hole + (-1,) + board
inline int64_t canonical_key_hash(const CanonicalForm& cf) {
    int64_t items[8];
    int k = 0;
    items[k++] = cf.hole[0];
    items[k++] = cf.hole[1];
    items[k++] = -1;
    for (int i = 0; i < cf.n_board; i++) items[k++] = cf.board[i];
    return py_tuple_hash(items, k);
}

// 44-bit cache key of a postflop canonical form: n_board - 2 (1..3, never 0) in the top two
// bits, then hole[0], hole[1], board[0..4] in 6 bits each (63 = unused board slot).
inline uint64_t canonical_pack(const CanonicalForm& cf) {
    uint64_t v = (uint64_t)(cf.n_board - 2);
    v = (v << 6) | (uint64_t)cf.hole[0];
    v = (v << 6) | (uint64_t)cf.hole[1];
    for (int i = 0; i < 5; i++) v = (v << 6) | (uint64_t)(i < cf.n_board ? cf.board[i] : 63);
    return v;
}

inline int street_of_board(int n_board) { return n_board == 3 ? FLOP : (n_board == 4 ? TURN : RIVER); }

// ------------------------------------------------------------------ 169 classes
struct HoleClasses {
    int index[13][13][2];  // [r1][r2][suited] with r1 >= r2 -> index in ALL_HOLE_CLASSES
    HoleClasses() {
        int k = 0;
        for (int i = 12; i >= 0; i--) {
            for (int j = 12; j >= 0; j--) {
                if (i == j) { index[i][j][0] = index[i][j][1] = k++; }
                else if (i > j) { index[i][j][1] = k++; index[i][j][0] = k++; }
            }
        }
    }
    int of(int c1, int c2) const {
        int r1 = c1 >> 2, r2 = c2 >> 2;
        if (r1 < r2) std::swap(r1, r2);
        int suited = (c1 & 3) == (c2 & 3) ? 1 : 0;
        return index[r1][r2][suited];
    }
};

// ------------------------------------------------------------- bounded form cache
// Fixed-capacity, lock-free, 8-way set-associative cache: 44-bit packed canonical form ->
// 20-bit non-negative integer, one 64-bit atomic word per slot (0 = empty; a valid word is never
// 0 because the key's top bits are nonzero), one set = one 64-byte cache line.  A lookup is up
// to 8 relaxed loads, an insert one relaxed store: no locks, no rehashing, no allocation after
// the first insert on a street, so the memory of a training run is flat.  Replacement: the first
// empty way, else a per-thread round-robin victim (random-like, no per-entry metadata, no
// global LRU list).  A lost concurrent insert or an eviction only costs a recomputation because
// every value is a pure function of the key.
struct FormCacheStats {
    size_t capacity = 0;   // slots
    size_t size = 0;       // occupied slots (a scan)
    uint64_t computes = 0; // values computed (= cache misses)
    uint64_t evictions = 0;
};

class FormCache {
public:
    static constexpr int WAYS = 8;
    static constexpr int VALUE_BITS = 20;
    static constexpr uint64_t VALUE_MASK = (1ULL << VALUE_BITS) - 1;
    static constexpr uint64_t MAX_VALUE = VALUE_MASK;

    FormCache() = default;
    FormCache(const FormCache&) = delete;
    FormCache& operator=(const FormCache&) = delete;

    // capacity in entries, rounded up to a power-of-two number of 8-way sets; 0 disables the cache.
    // Not thread-safe against concurrent get/put: call before training.
    void set_capacity(size_t capacity) {
        std::lock_guard<std::mutex> lk(alloc_mu_);
        table_.store(nullptr, std::memory_order_release);
        storage_.reset();
        n_sets_ = 0;
        log2_sets_ = 0;
        if (capacity > 0) {
            n_sets_ = 1;
            while (n_sets_ * WAYS < capacity) { n_sets_ <<= 1; log2_sets_++; }
        }
        computes_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
    }

    size_t capacity() const { return n_sets_ * WAYS; }

    bool get(uint64_t key, uint32_t& value) const {
        const std::atomic<uint64_t>* t = table_.load(std::memory_order_acquire);
        if (!t) return false;
        const std::atomic<uint64_t>* set = t + set_index(key) * WAYS;
        const uint64_t want = key << VALUE_BITS;
        for (int w = 0; w < WAYS; w++) {
            uint64_t word = set[w].load(std::memory_order_relaxed);
            if ((word & ~VALUE_MASK) == want) { value = (uint32_t)(word & VALUE_MASK); return true; }
        }
        return false;
    }

    void put(uint64_t key, uint32_t value) {
        computes_.fetch_add(1, std::memory_order_relaxed);  // every put is a value just computed
        if (n_sets_ == 0) return;
        std::atomic<uint64_t>* t = table_.load(std::memory_order_acquire);
        if (!t) t = allocate();
        std::atomic<uint64_t>* set = t + set_index(key) * WAYS;
        const uint64_t want = key << VALUE_BITS;
        const uint64_t word = want | ((uint64_t)value & VALUE_MASK);
        for (int w = 0; w < WAYS; w++) {
            uint64_t cur = set[w].load(std::memory_order_relaxed);
            if (cur == 0 && set[w].compare_exchange_strong(cur, word, std::memory_order_relaxed)) return;
            if ((cur & ~VALUE_MASK) == want) return;  // already there (same value by construction)
        }
        static thread_local uint32_t rr = 0;
        set[rr++ % WAYS].store(word, std::memory_order_relaxed);
        evictions_.fetch_add(1, std::memory_order_relaxed);
    }

    // every occupied slot's word (key << VALUE_BITS | value): the cache's contents, for saving
    std::vector<uint64_t> words() const {
        std::vector<uint64_t> out;
        const std::atomic<uint64_t>* t = table_.load(std::memory_order_acquire);
        if (!t) return out;
        const size_t total = n_sets_ * WAYS;
        for (size_t i = 0; i < total; i++) {
            const uint64_t w = t[i].load(std::memory_order_relaxed);
            if (w != 0) out.push_back(w);
        }
        return out;
    }
    // insert saved words (not counted as computes); returns how many did not fit without an eviction
    size_t load(const uint64_t* w, size_t n) {
        if (n_sets_ == 0) return n;
        std::atomic<uint64_t>* t = table_.load(std::memory_order_acquire);
        if (!t) t = allocate();
        size_t evicted = 0;
        for (size_t i = 0; i < n; i++) {
            const uint64_t word = w[i];
            const uint64_t key = word >> VALUE_BITS;
            std::atomic<uint64_t>* set = t + set_index(key) * WAYS;
            bool placed = false;
            for (int k = 0; k < WAYS && !placed; k++) {
                uint64_t cur = set[k].load(std::memory_order_relaxed);
                if (cur == 0 && set[k].compare_exchange_strong(cur, word, std::memory_order_relaxed)) placed = true;
                else if ((cur & ~VALUE_MASK) == (word & ~VALUE_MASK)) placed = true;
            }
            if (!placed) {
                set[i % WAYS].store(word, std::memory_order_relaxed);
                evicted++;
                evictions_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return evicted;
    }

    // occupied slots: a scan of the whole table (for reporting only)
    size_t size() const {
        const std::atomic<uint64_t>* t = table_.load(std::memory_order_acquire);
        if (!t) return 0;
        size_t n = 0;
        size_t total = n_sets_ * WAYS;
        for (size_t i = 0; i < total; i++) if (t[i].load(std::memory_order_relaxed) != 0) n++;
        return n;
    }

    FormCacheStats stats() const {
        FormCacheStats s;
        s.capacity = capacity();
        s.size = size();
        s.computes = computes_.load(std::memory_order_relaxed);
        s.evictions = evictions_.load(std::memory_order_relaxed);
        return s;
    }

private:
    std::atomic<std::atomic<uint64_t>*> table_{nullptr};
    std::unique_ptr<std::atomic<uint64_t>[]> storage_;
    mutable std::mutex alloc_mu_;
    size_t n_sets_ = 0;
    int log2_sets_ = 0;
    std::atomic<uint64_t> computes_{0};
    std::atomic<uint64_t> evictions_{0};

    size_t set_index(uint64_t key) const {
        if (log2_sets_ == 0) return 0;
        return (size_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - log2_sets_));
    }

    std::atomic<uint64_t>* allocate() {
        std::lock_guard<std::mutex> lk(alloc_mu_);
        std::atomic<uint64_t>* t = table_.load(std::memory_order_acquire);
        if (t) return t;
        size_t total = n_sets_ * WAYS;
        storage_.reset(new std::atomic<uint64_t>[total]);
        for (size_t i = 0; i < total; i++) storage_[i].store(0, std::memory_order_relaxed);
        t = storage_.get();
        table_.store(t, std::memory_order_release);
        return t;
    }
};

// ------------------------------------------------------------------ bucketers
// What a checkpoint records about the card abstraction it was trained with, so that a resume with
// other buckets is refused: the kind, its parameters and a fingerprint of every fitted number
// (cut points, centroids) taken bit for bit.
struct BucketerIdentity {
    std::string kind;  // "ehs" or "potential" (abstraction.bucketer_kind)
    int n_buckets = 0, samples = 0, bins = 0;
    uint64_t fingerprint = 0;
};

class IdentityHash {
public:
    void u64(uint64_t v) {
        h_ ^= v + 0x9E3779B97F4A7C15ULL + (h_ << 6) + (h_ >> 2);
        h_ ^= h_ >> 33;
        h_ *= 0xFF51AFD7ED558CCDULL;
        h_ ^= h_ >> 33;
    }
    void f64(double v) {
        uint64_t b;
        std::memcpy(&b, &v, 8);
        u64(b);
    }
    void doubles(const std::vector<double>& v) {
        u64(v.size());
        for (double x : v) f64(x);
    }
    uint64_t value() const { return h_; }

private:
    uint64_t h_ = 0x6A09E667F3BCC908ULL;
};

// Common interface of the card abstractions (the trainers hold a shared_ptr<Bucketer>):
// preflop = the 169 classes, postflop = whatever the concrete class computes, cached per
// canonical form in one bounded FormCache per street.
class Bucketer {
public:
    // cap of a street without an explicit one (a core object built without cache_caps); the Python
    // trainers always pass fast.DEFAULT_CACHE_CAPS (4M / 32M / 4M entries: flop / turn / river)
    static constexpr size_t DEFAULT_CACHE_CAP = (size_t)1 << 22;  // 4,194,304 entries = 32 MB per street

    Bucketer() { set_cache_caps({}); }
    virtual ~Bucketer() = default;
    virtual bool fitted() const = 0;
    virtual int bucket(const int* hole, const int* board, int n_board) = 0;
    virtual BucketerIdentity identity() const = 0;
    // the postflop bucket of a canonical form, computed without the cache (thread-safe, const):
    // bucket() == bucket_of_form(canonical_form(...)) for every postflop hand (bucket tables)
    virtual int bucket_of_form(const CanonicalForm& cf) const = 0;
    const HoleClasses& classes() const { return classes_; }

    // {flop, turn, river} capacities in entries (missing -> default); call before training
    void set_cache_caps(const std::vector<size_t>& caps) {
        for (int s = FLOP; s <= RIVER; s++) {
            size_t i = (size_t)(s - FLOP);
            caches_[s].set_capacity(i < caps.size() ? caps[i] : DEFAULT_CACHE_CAP);
        }
    }
    std::vector<size_t> cache_caps() const {
        return {caches_[FLOP].capacity(), caches_[TURN].capacity(), caches_[RIVER].capacity()};
    }
    size_t cache_size() const {
        return caches_[FLOP].size() + caches_[TURN].size() + caches_[RIVER].size();
    }
    FormCacheStats cache_stats(int street) const { return caches_[street].stats(); }
    // the cached values of a street (saved to disk and loaded back by persist.h)
    std::vector<uint64_t> cache_words(int street) const { return caches_[street].words(); }
    size_t load_cache_words(int street, const uint64_t* w, size_t n) { return caches_[street].load(w, n); }

    // River buckets of every hole on a 5-card board at once, when the abstraction allows a batch
    // with the same numbers as bucket() (false: use bucket()); out[combo index] (255 on the board).
    virtual bool river_buckets_all(const int* /*board5*/, const int (*/*combo_index*/)[52], uint8_t* /*out*/) const { return false; }

protected:
    HoleClasses classes_;
    FormCache caches_[4];  // by street; PREFLOP unused
};

class EquityBucketer : public Bucketer {
public:
    int n_buckets = 10;
    int samples = 200;
    std::vector<double> boundaries[4];  // by street (1..3)

    EquityBucketer() = default;
    EquityBucketer(int n_buckets_, int samples_) : n_buckets(n_buckets_), samples(samples_) {
        if (samples < 1 || (uint64_t)samples * 2 > FormCache::MAX_VALUE) throw std::invalid_argument("samples must be 1..524287");
    }

    bool fitted() const override { return !boundaries[FLOP].empty(); }

    BucketerIdentity identity() const override {
        BucketerIdentity id;
        id.kind = "ehs";
        id.n_buckets = n_buckets;
        id.samples = samples;
        IdentityHash h;
        h.u64(1);  // kind tag
        for (int s = FLOP; s <= RIVER; s++) h.doubles(boundaries[s]);
        id.fingerprint = h.value();
        return id;
    }

    // E[HS] vs one random hand of the CANONICAL representative (so the value is a pure function
    // of the canonical key), cached as k = 2 * won, an exact integer: with one opponent every
    // sample adds 1 or 1/2 to ``won``, so (k / 2.0) / samples is the very same double as the
    // reference's won / samples.
    double ehs(const int* hole, const int* board, int n_board) {
        CanonicalForm cf;
        canonical_form(hole, board, n_board, cf);
        return ehs_canonical(cf);
    }

    double ehs_canonical(const CanonicalForm& cf) {
        int street = street_of_board(cf.n_board);
        uint64_t key = canonical_pack(cf);
        uint32_t k;
        if (caches_[street].get(key, k)) return decode(k);
        uint64_t seed = (uint64_t)canonical_key_hash(cf) & 0xFFFFFFFFULL;
        PyRandom rng(seed);
        double won = equity_won_vs_random(cf.hole, 2, cf.board, cf.n_board, 1, samples, rng);
        k = (uint32_t)(won * 2.0);
        caches_[street].put(key, k);
        return decode(k);
    }

    int bucket(const int* hole, const int* board, int n_board) override {
        if (n_board == 0) return classes_.of(hole[0], hole[1]);
        int street = street_of_board(n_board);
        const std::vector<double>& cuts = boundaries[street];
        if (cuts.empty()) throw std::runtime_error("bucketer not fitted");
        double e = ehs(hole, board, n_board);
        return (int)(std::upper_bound(cuts.begin(), cuts.end(), e) - cuts.begin());
    }

    int bucket_of_form(const CanonicalForm& cf) const override {
        const std::vector<double>& cuts = boundaries[street_of_board(cf.n_board)];
        if (cuts.empty()) throw std::runtime_error("bucketer not fitted");
        uint64_t seed = (uint64_t)canonical_key_hash(cf) & 0xFFFFFFFFULL;
        PyRandom rng(seed);
        double won = equity_won_vs_random(cf.hole, 2, cf.board, cf.n_board, 1, samples, rng);
        double e = decode((uint32_t)(won * 2.0));
        return (int)(std::upper_bound(cuts.begin(), cuts.end(), e) - cuts.begin());
    }

private:
    double decode(uint32_t k) const { return ((double)k / 2.0) / (double)samples; }
};

// ---------------------------------------------------- potential-aware features (potential.py)
constexpr int MAX_BINS = 64;

// river_equity(): exact equity vs one random hand on a 5-card board, all C(45,2) opponent
// combos in (i < j) card order; won accumulates 1.0 / 0.5 exactly as the reference does.
inline double river_equity_exact(const int* hole, const int* board, int n_board) {
    bool used[52] = {false};
    used[hole[0]] = used[hole[1]] = true;
    for (int i = 0; i < n_board; i++) used[board[i]] = true;
    int rest[52];
    int n_rest = 0;
    for (int c = 0; c < 52; c++) if (!used[c]) rest[n_rest++] = c;
    int cards[7];
    cards[0] = hole[0]; cards[1] = hole[1];
    for (int i = 0; i < n_board; i++) cards[2 + i] = board[i];
    int64_t mine = evaluate(cards, 2 + n_board);
    double won = 0.0;
    int n = 0;
    for (int i = 0; i < n_rest; i++) {
        cards[0] = rest[i];
        for (int j = i + 1; j < n_rest; j++) {
            cards[1] = rest[j];
            int64_t v = evaluate(cards, 2 + n_board);
            if (mine > v) won += 1.0;
            else if (mine == v) won += 0.5;
            n++;
        }
    }
    return won / (double)n;
}

// next_street_histogram(): E[HS] after each possible next card (card order), Monte-Carlo with
// `samples` runouts from ONE generator seeded with hash(canonical_key(hole, board)) & 0xFFFFFFFF
// and consumed sequentially; counts over `bins` equal bins of [0, 1] and the mean equity.
inline void potential_histogram(const int* hole, const int* board, int n_board, int samples, int bins,
                                int* counts, double& mean) {
    CanonicalForm cf;
    canonical_form(hole, board, n_board, cf);
    uint64_t seed = (uint64_t)canonical_key_hash(cf) & 0xFFFFFFFFULL;
    PyRandom rng(seed);
    bool used[52] = {false};
    used[hole[0]] = used[hole[1]] = true;
    for (int i = 0; i < n_board; i++) used[board[i]] = true;
    int next_board[5];
    for (int i = 0; i < n_board; i++) next_board[i] = board[i];
    for (int i = 0; i < bins; i++) counts[i] = 0;
    double total = 0.0;
    int n = 0;
    for (int c = 0; c < 52; c++) {
        if (used[c]) continue;
        next_board[n_board] = c;
        double e = equity_vs_random(hole, 2, next_board, n_board + 1, 1, samples, rng);
        int b = (int)(e * (double)bins);
        if (b >= bins) b = bins - 1;
        counts[b]++;
        total += e;
        n++;
    }
    mean = total / (double)n;
}

// PotentialAwareBucketer.bucket(): the feature is computed on the canonical representative
// (so the result is the same in every process), then the nearest centroid under EMD = L1 on
// CDFs, summed in bin order, first minimum wins; river: exact equity vs the cut points.
class PotentialBucketer : public Bucketer {
public:
    int n_buckets = 8;
    int samples = 40;
    int bins = 10;
    std::vector<std::vector<double>> centroids[4];  // by street (FLOP, TURN): n_buckets CDFs of `bins` values
    std::vector<double> boundaries[4];              // RIVER: n_buckets - 1 cut points

    PotentialBucketer() = default;
    PotentialBucketer(int n_buckets_, int samples_, int bins_) : n_buckets(n_buckets_), samples(samples_), bins(bins_) {
        if (bins < 1 || bins > MAX_BINS) throw std::invalid_argument("bins must be 1..64");
    }

    bool fitted() const override { return !centroids[FLOP].empty(); }

    BucketerIdentity identity() const override {
        BucketerIdentity id;
        id.kind = "potential";
        id.n_buckets = n_buckets;
        id.samples = samples;
        id.bins = bins;
        IdentityHash h;
        h.u64(2);  // kind tag
        for (int s = FLOP; s <= TURN; s++) {
            h.u64(centroids[s].size());
            for (const auto& c : centroids[s]) h.doubles(c);
        }
        h.doubles(boundaries[RIVER]);
        id.fingerprint = h.value();
        return id;
    }

    // (CDF, mean equity) on the canonical representative: PotentialAwareBucketer.feature()
    void feature(const int* hole, const int* board, int n_board, std::vector<double>& cdf, double& mean) const {
        CanonicalForm cf;
        canonical_form(hole, board, n_board, cf);
        int counts[MAX_BINS];
        potential_histogram(cf.hole, cf.board, cf.n_board, samples, bins, counts, mean);
        cdf.resize(bins);
        int n = 0;
        for (int i = 0; i < bins; i++) n += counts[i];
        int cum = 0;
        for (int i = 0; i < bins; i++) { cum += counts[i]; cdf[i] = (double)cum / (double)n; }
    }

    double river_ehs(const int* hole, const int* board, int n_board) const {
        CanonicalForm cf;
        canonical_form(hole, board, n_board, cf);
        return river_equity_exact(cf.hole, cf.board, cf.n_board);
    }

    // bucket() of every hole on one river board: the 1081 hands of the 47 other cards are
    // evaluated once and sorted; a hole's opponents are those 1081 minus the 91 sharing one of its
    // cards, so its wins and ties are counts over the sorted list minus those 91.  won = wins +
    // ties / 2 is the same number river_equity_exact accumulates (sums of 1.0 and 0.5, exact), won /
    // 990 the same double, so the same bucket (checked in tests against bucket()).
    bool river_buckets_all(const int* board, const int (*idx)[52], uint8_t* out) const override {
        const std::vector<double>& cuts = boundaries[RIVER];
        if (cuts.empty()) return false;
        bool on_board[52] = {false};
        for (int i = 0; i < 5; i++) on_board[board[i]] = true;
        int rest[52];
        int nr = 0;
        for (int c = 0; c < 52; c++) if (!on_board[c]) rest[nr++] = c;
        static thread_local std::vector<int64_t> str, all;
        str.assign(52 * 52, 0);
        all.clear();
        int cards[7];
        for (int i = 0; i < 5; i++) cards[2 + i] = board[i];
        for (int i = 0; i < nr; i++) {
            for (int j = i + 1; j < nr; j++) {
                cards[0] = rest[i];
                cards[1] = rest[j];
                const int64_t s = evaluate(cards, 7);
                str[(size_t)(rest[i] * 52 + rest[j])] = str[(size_t)(rest[j] * 52 + rest[i])] = s;
                all.push_back(s);
            }
        }
        std::sort(all.begin(), all.end());
        for (int c = 0; c < 52; c++)
            for (int d = c + 1; d < 52; d++)
                if (on_board[c] || on_board[d]) out[idx[c][d]] = 255;
        for (int i = 0; i < nr; i++) {
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
                const double won = (double)lower + 0.5 * (double)equal;
                const double e = won / (double)((nr - 2) * (nr - 3) / 2);
                out[idx[a][b]] = (uint8_t)(std::upper_bound(cuts.begin(), cuts.end(), e) - cuts.begin());
            }
        }
        return true;
    }

    int bucket(const int* hole, const int* board, int n_board) override {
        if (n_board == 0) return classes_.of(hole[0], hole[1]);
        CanonicalForm cf;
        canonical_form(hole, board, n_board, cf);
        int street = street_of_board(cf.n_board);
        uint64_t key = canonical_pack(cf);
        uint32_t v;
        if (caches_[street].get(key, v)) return (int)v;
        int b = bucket_canonical(cf);
        caches_[street].put(key, (uint32_t)b);
        return b;
    }

    int bucket_of_form(const CanonicalForm& cf) const override { return bucket_canonical(cf); }

private:
    int bucket_canonical(const CanonicalForm& cf) const {
        int street = street_of_board(cf.n_board);
        if (street == RIVER) {
            const std::vector<double>& cuts = boundaries[RIVER];
            if (cuts.empty()) throw std::runtime_error("bucketer not fitted");
            double e = river_equity_exact(cf.hole, cf.board, cf.n_board);
            return (int)(std::upper_bound(cuts.begin(), cuts.end(), e) - cuts.begin());
        }
        const std::vector<std::vector<double>>& cens = centroids[street];
        if (cens.empty()) throw std::runtime_error("bucketer not fitted");
        int counts[MAX_BINS];
        double mean;
        potential_histogram(cf.hole, cf.board, cf.n_board, samples, bins, counts, mean);
        double cdf[MAX_BINS];
        int n = 0;
        for (int i = 0; i < bins; i++) n += counts[i];
        int cum = 0;
        for (int i = 0; i < bins; i++) { cum += counts[i]; cdf[i] = (double)cum / (double)n; }
        int best = 0;
        double best_d = 0.0;
        for (size_t j = 0; j < cens.size(); j++) {
            const std::vector<double>& cen = cens[j];
            double d = 0.0;
            for (int i = 0; i < bins; i++) d += std::fabs(cdf[i] - cen[i]);
            if (j == 0 || d < best_d) { best = (int)j; best_d = d; }
        }
        return best;
    }
};

// ------------------------------------------------------------------ infoset key
// The key for a known bucket ("infoset_key_for_bucket" in infoset.py).  The trainers pass the
// bucket from their per-iteration memo (mccfr.h::memo_bucket); infoset_key() asks the bucketer.
inline void infoset_key_for_bucket(const HandState& st, const Obs& obs, int b, const BetGrid& grid,
                                   std::string& key, std::string& hist_buf) {
    grid.history_string(st.events, st.n_events, hist_buf);
    key.clear();
    key += street_letter(obs.street);
    key += '|';
    key += position_name(obs.seat, st.button, st.n);
    key += '|';
    key += std::to_string(obs.n_active);
    key += "|b";
    key += std::to_string(b);
    key += '|';
    key += hist_buf;
}

inline void infoset_key(const HandState& st, const Obs& obs, Bucketer& bk, const BetGrid& grid,
                        std::string& key, std::string& hist_buf) {
    const Player& p = st.players[obs.seat];
    int b = bk.bucket(p.hole, st.board, st.n_board);
    infoset_key_for_bucket(st, obs, b, grid, key, hist_buf);
}

}  // namespace negp
