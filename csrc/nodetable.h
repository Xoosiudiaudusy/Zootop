// Node storage of the MCCFR / RNR traversals: numeric infoset keys and a flat, lock-free node table
// (2026-09-24; replaces the string-keyed, sharded unordered_map).
//
// Key.  An infoset key string "R|BTN/SB|2|b3|c r0.5 c/c c/c r1 c/c" (abstraction/infoset.py) is
// identified by NodeKey: the fields street, position index (seat relative to the button), n_active
// and bucket, plus a 128-bit hash (two independent 64-bit lanes) of the history part, the text
// after the 4th '|'.  The traversals never build that text: HistHash consumes, event by event,
// exactly the bytes BetGrid::history_string would write ('/' when the street changes except before
// the first event, ' ' between two events of one street, then the token from_concrete appends),
// so a node's key costs a few multiplications per new event.  KeyCodec::of_string parses a key
// string (checkpoint import, lookups from Python) into the same NodeKey; a string that is not in
// the canonical form our code builds (unknown position, leading zeros, missing fields...) becomes
// a "foreign" key, a hash of the whole string with the FOREIGN bit set, which no traversal key can
// equal.  So two keys are equal iff their strings are equal, up to 128-bit hash collisions.  The
// string itself is kept per node (built once, at creation, with today's functions) for export and
// import; in test mode (Trainer::verify_keys) every lookup compares it with the string
// infoset_key would build, which checks the bijection on every key a test visits.
//
// Table.  Power-of-two open addressing with linear probing; 32-byte slots {k1, k2, node, key
// string}; grown at load 1/2.  find: lock-free (acquire load of the node pointer, then the keys).
// insert: the node and its key string are prepared first (the calling thread's arena), then the
// empty slot is claimed with one CAS (nullptr -> CLAIMED), the keys and the key pointer are
// written and the node pointer is published with a release store; a thread that meets a CLAIMED
// slot waits for that publication (a few stores).  Slots never become empty again, so every
// inserter of a key walks the same probe sequence and meets the first claim for it: a node is
// never duplicated or dropped.  Growth needs exclusive access: when a table passes load 1/2, its
// TableGroup (one per trainer, shared by the trainer's tables) raises a flag, every worker parks
// at its next lookup, the last one to park doubles the table(s) and wakes the others ("stop the
// world", a few times per run).  An inserter reserves room before it claims a slot and never
// takes the table past 3/4, parking instead, so workers that have not seen the flag yet cannot
// fill it (found by the stress test: 16 threads on a 16-slot table).  Outside train() an insert
// grows the table at once.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define NEGP_HAVE_MM_PAUSE 1
#endif

#include "abstraction.h"
#include "engine.h"

namespace negp {

inline void cpu_relax() {
#if defined(NEGP_HAVE_MM_PAUSE)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

constexpr int MAX_ACTIONS = 8;

struct SpinLock {
    std::atomic<bool> flag{false};
    void lock() { while (flag.exchange(true, std::memory_order_acquire)) { while (flag.load(std::memory_order_relaxed)) {} } }
    void unlock() { flag.store(false, std::memory_order_release); }
};

struct Node {
    uint8_t n = 0;
    uint8_t acts[MAX_ACTIONS];
    double regret[MAX_ACTIONS];
    double strategy_sum[MAX_ACTIONS];
    long long visits = 0;
    SpinLock lock;

    void init(const uint8_t* ids, int k) {
        n = (uint8_t)k;
        for (int i = 0; i < k; i++) { acts[i] = ids[i]; regret[i] = 0.0; strategy_sum[i] = 0.0; }
        visits = 0;
    }
    // Node.current_strategy() with CPython sum() semantics; caller holds the lock
    void current_strategy(double* out) const {
        double pos[MAX_ACTIONS];
        for (int i = 0; i < n; i++) pos[i] = regret[i] > 0 ? regret[i] : 0.0;
        double s = py_sum(pos, n);
        if (s <= 0) { for (int i = 0; i < n; i++) out[i] = 1.0 / n; return; }
        for (int i = 0; i < n; i++) out[i] = pos[i] / s;
    }
    void average_strategy(double* out) const {
        double s = py_sum(strategy_sum, n);
        if (s <= 0) { for (int i = 0; i < n; i++) out[i] = 1.0 / n; return; }
        for (int i = 0; i < n; i++) out[i] = strategy_sum[i] / s;
    }
};

// ------------------------------------------------------------------ numeric keys
inline uint64_t fmix64(uint64_t k) {  // MurmurHash3 finaliser: a bijection with full avalanche
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDULL;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ULL;
    k ^= k >> 33;
    return k;
}

// Hash of the history text, two independent 64-bit lanes; for a fixed byte each lane update is a
// bijection of the lane state.  `n` = events hashed so far, `cur` = history_string's `cur`.
struct HistHash {
    uint64_t a = 0x243F6A8885A308D3ULL;
    uint64_t b = 0x13198A2E03707344ULL;
    int cur = -1;
    int n = 0;

    void byte(unsigned char c) {
        a ^= c;
        a *= 0x9E3779B97F4A7C15ULL;
        a = (a << 23) | (a >> 41);
        b += (uint64_t)c + 0x632BE59BD9B4E019ULL;
        b *= 0xBF58476D1CE4E5B9ULL;
        b ^= b >> 29;
    }
    void bytes(const char* s, size_t len) {
        for (size_t i = 0; i < len; i++) byte((unsigned char)s[i]);
    }
    // one more event: the separator history_string writes before it, then its token
    void event(const Event& ev, const BetGrid& grid, std::string& scratch) {
        if (ev.street != cur) {
            if (cur != -1) byte('/');
            cur = ev.street;
        } else {
            byte(' ');
        }
        scratch.clear();
        grid.from_concrete(ev, scratch);
        bytes(scratch.data(), scratch.size());
        n++;
    }
    // hash the events of `st` not hashed yet (a child state has one more than its parent)
    void catch_up(const HandState& st, const BetGrid& grid, std::string& scratch) {
        while (n < st.n_events) event(st.events[n], grid, scratch);
    }
};

struct NodeKey {
    uint64_t k1 = 0, k2 = 0;
    bool operator==(const NodeKey& o) const { return k1 == o.k1 && k2 == o.k2; }
    bool operator!=(const NodeKey& o) const { return !(*this == o); }
};
struct NodeKeyHash {
    size_t operator()(const NodeKey& k) const { return (size_t)k.k1; }
};

constexpr uint64_t FOREIGN_KEY_BIT = 1ULL << 63;  // in k2: the string was not in canonical form

// the key of (street, position index, n_active, bucket, history hash); each field < 256 except the
// bucket (< 2^31), so the packed prefix is injective
inline NodeKey node_key(int street, int rel, int n_active, int bucket, const HistHash& h) {
    const uint64_t p = (uint64_t)(street & 0xFF) | ((uint64_t)(rel & 0xFF) << 8) | ((uint64_t)(n_active & 0xFF) << 16) |
                       ((uint64_t)(uint32_t)bucket << 24);
    NodeKey k;
    k.k1 = fmix64(h.a ^ fmix64(p ^ 0xA0761D6478BD642FULL));
    k.k2 = fmix64(h.b ^ fmix64(p + 0xE7037ED1A0B428DBULL)) & ~FOREIGN_KEY_BIT;
    return k;
}

// Numeric keys of one game (the position names depend on the number of players).
class KeyCodec {
public:
    explicit KeyCodec(int n_players = 2) { set_players(n_players); }

    void set_players(int n) {
        n_ = n < 2 ? 2 : n;
        // position_name() is defined for the relative seats 0..5 (6-max labels; the reference
        // raises for more), so only those can be parsed
        n_names_ = n_ < 6 ? n_ : 6;
        for (int r = 0; r < n_names_; r++) pos_names_[r] = position_name(r, 0, n_);
    }
    int n_players() const { return n_; }
    int rel(int seat, int button) const { return ((seat - button) % n_ + n_) % n_; }

    // the key of a decision node: what infoset_key_for_bucket(st, obs, bucket, ...) would spell
    NodeKey key(const Obs& obs, int button, int bucket, const HistHash& h) const {
        return node_key(obs.street, rel(obs.seat, button), obs.n_active, bucket, h);
    }

    // the key of a key string: canonical strings map to node_key(fields, hash(history)), anything
    // else to a foreign key
    NodeKey of_string(const char* s, size_t len) const {
        size_t bar[4];
        int nb = 0;
        for (size_t i = 0; i < len && nb < 4; i++) if (s[i] == '|') bar[nb++] = i;
        if (nb == 4) {
            int street = -1;
            if (bar[0] == 1) {
                for (int st = PREFLOP; st <= SHOWDOWN; st++) if (s[0] == street_letter(st)[0]) { street = st; break; }
            }
            int rel = -1;
            const size_t pl = bar[1] - bar[0] - 1;
            for (int r = 0; r < n_names_; r++) {
                if (pos_names_[r].size() == pl && std::memcmp(pos_names_[r].data(), s + bar[0] + 1, pl) == 0) { rel = r; break; }
            }
            uint64_t n_active = 0, bucket = 0;
            const bool ok = street >= 0 && rel >= 0 && parse_uint(s + bar[1] + 1, bar[2] - bar[1] - 1, 255, n_active) &&
                            bar[3] - bar[2] >= 2 && s[bar[2] + 1] == 'b' &&
                            parse_uint(s + bar[2] + 2, bar[3] - bar[2] - 2, 0x7FFFFFFFULL, bucket);
            if (ok) {
                HistHash h;
                h.bytes(s + bar[3] + 1, len - bar[3] - 1);
                return node_key(street, rel, (int)n_active, (int)bucket, h);
            }
        }
        HistHash h;
        h.bytes(s, len);
        NodeKey k;
        k.k1 = fmix64(h.a ^ 0x94D049BB133111EBULL);
        k.k2 = fmix64(h.b ^ 0x2545F4914F6CDD1DULL) | FOREIGN_KEY_BIT;
        return k;
    }
    NodeKey of_string(const std::string& s) const { return of_string(s.data(), s.size()); }

private:
    // canonical decimal (what std::to_string writes for a non-negative int): digits only, no
    // leading zero unless the number is 0
    static bool parse_uint(const char* s, size_t len, uint64_t max, uint64_t& out) {
        if (len == 0 || len > 10) return false;
        if (len > 1 && s[0] == '0') return false;
        uint64_t v = 0;
        for (size_t i = 0; i < len; i++) {
            if (s[i] < '0' || s[i] > '9') return false;
            v = v * 10 + (uint64_t)(s[i] - '0');
        }
        if (v > max) return false;
        out = v;
        return true;
    }

    int n_ = 2;
    int n_names_ = 2;
    std::string pos_names_[6];
};

// ------------------------------------------------------------------ node arena
// Per-thread bump allocator for nodes and key strings; the memory lives as long as the table (a
// node prepared for an insert that another thread won is simply never published).
class NodeArena {
public:
    static constexpr size_t NODES_PER_CHUNK = 1024;
    static constexpr size_t CHARS_PER_CHUNK = 64 * 1024;

    Node* new_node() {
        if (node_left_ == 0) {
            node_chunks_.emplace_back(new Node[NODES_PER_CHUNK]);
            node_next_ = node_chunks_.back().get();
            node_left_ = NODES_PER_CHUNK;
        }
        node_left_--;
        return node_next_++;
    }
    const char* copy_key(const char* s, size_t len) {
        const size_t need = len + 1;
        if (need > char_left_) {
            const size_t sz = need > CHARS_PER_CHUNK ? need : CHARS_PER_CHUNK;
            char_chunks_.emplace_back(new char[sz]);
            char_bytes_ += sz;
            char_next_ = char_chunks_.back().get();
            char_left_ = sz;
        }
        char* p = char_next_;
        std::memcpy(p, s, len);
        p[len] = '\0';
        char_next_ += need;
        char_left_ -= need;
        return p;
    }
    size_t node_bytes() const { return node_chunks_.size() * NODES_PER_CHUNK * sizeof(Node); }
    size_t char_bytes() const { return char_bytes_; }
    void clear() {
        node_chunks_.clear();
        char_chunks_.clear();
        node_next_ = nullptr;
        char_next_ = nullptr;
        node_left_ = char_left_ = char_bytes_ = 0;
    }

private:
    std::vector<std::unique_ptr<Node[]>> node_chunks_;
    std::vector<std::unique_ptr<char[]>> char_chunks_;
    Node* node_next_ = nullptr;
    char* char_next_ = nullptr;
    size_t node_left_ = 0, char_left_ = 0, char_bytes_ = 0;
};

class FlatNodeTable;

// ------------------------------------------------------------------ resize coordinator
// The worker threads of one trainer and the tables they share.  begin(T) before the workers
// start, leave() when a worker is done, end() after the join.  A table that passes its load limit
// calls request(); while workers run, that raises pending() and every worker parks in arrive() at
// its next lookup; the last one to park grows every table that asked, then all continue.
class TableGroup {
public:
    void add(FlatNodeTable* t) { tables_.push_back(t); }
    bool pending() const { return pending_.load(std::memory_order_relaxed); }

    void begin(int workers) {
        std::lock_guard<std::mutex> lk(mu_);
        if (pending_.load(std::memory_order_relaxed)) grow_and_release();  // left over from an aborted run
        running_ = true;
        active_ = workers;
        waiting_ = 0;
    }
    void leave() {
        std::lock_guard<std::mutex> lk(mu_);
        --active_;
        if (pending_.load(std::memory_order_relaxed) && waiting_ >= active_) grow_and_release();
    }
    void end() {
        std::lock_guard<std::mutex> lk(mu_);
        running_ = false;
        if (pending_.load(std::memory_order_relaxed)) grow_and_release();
    }
    inline void request();
    void arrive() {
        std::unique_lock<std::mutex> lk(mu_);
        if (!pending_.load(std::memory_order_relaxed)) return;
        if (!running_) { grow_and_release(); return; }
        const uint64_t g = gen_;
        if (++waiting_ >= active_) { grow_and_release(); return; }
        cv_.wait(lk, [&] { return gen_ != g; });
    }
    long long resizes() const { return resizes_; }

private:
    inline void grow_and_release();  // mu_ held, no worker inside a table

    std::vector<FlatNodeTable*> tables_;
    std::atomic<bool> pending_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    bool running_ = false;
    int active_ = 0, waiting_ = 0;
    uint64_t gen_ = 0;
    long long resizes_ = 0;
};

// ------------------------------------------------------------------ flat node table
class FlatNodeTable {
public:
    struct Found {
        Node* node = nullptr;
        const char* key = nullptr;  // the node's key string
        bool created = false;
    };
    static constexpr size_t DEFAULT_CAPACITY = 4096;  // slots; doubled whenever the load reaches 1/2

    explicit FlatNodeTable(int n_arenas = 1, size_t initial_capacity = DEFAULT_CAPACITY) : initial_(initial_capacity) {
        set_arenas(n_arenas);
        allocate(initial_);
    }
    FlatNodeTable(const FlatNodeTable&) = delete;
    FlatNodeTable& operator=(const FlatNodeTable&) = delete;

    // one arena per worker thread plus one for inserts outside train(); call before inserting
    void set_arenas(int n) {
        while ((int)arenas_.size() < n) arenas_.emplace_back(new NodeArena());
    }
    int arenas() const { return (int)arenas_.size(); }
    void attach(TableGroup* g) { group_ = g; }

    // Find the node of `k` or insert one.  `init(Node&, NodeArena&) -> const char*` fills a new
    // node and returns its key string (copied into the arena); it runs before the node is
    // published, at most once per call, and only when the key is absent.
    //
    // Room: an inserter reserves a slot in size_ before its claim (size_ = published slots + open
    // reservations) and never goes past hard_limit_ (3/4 of the slots), so an empty slot always
    // exists and every probe ends.  Past the limit it releases the reservation, parks until the
    // table has grown (workers still running may have filled it faster than they parked) and
    // starts over; nothing is claimed while it waits.
    template <class Init>
    Found get_or_create(const NodeKey& k, int arena, Init&& init) {
        Node* prepared = nullptr;
        const char* prepared_key = nullptr;
        for (;;) {
            if (group_ && group_->pending()) group_->arrive();
            Slot* slots = slots_.get();
            const size_t mask = mask_;
            size_t i = (size_t)k.k1 & mask;
            bool reserved = false, full = false;
            for (size_t probes = 0; probes <= mask; probes++, i = (i + 1) & mask) {
                Slot& s = slots[i];
                Node* p = s.node.load(std::memory_order_acquire);
                if (p == nullptr) {
                    if (!prepared) {
                        NodeArena& a = *arenas_[arena];
                        prepared = a.new_node();
                        prepared_key = init(*prepared, a);
                    }
                    if (!reserved) {
                        if (size_.fetch_add(1, std::memory_order_acq_rel) + 1 > hard_limit_) {
                            size_.fetch_sub(1, std::memory_order_acq_rel);
                            full = true;
                            break;
                        }
                        reserved = true;
                    }
                    Node* expected = nullptr;
                    if (s.node.compare_exchange_strong(expected, claimed(), std::memory_order_acq_rel, std::memory_order_acquire)) {
                        s.k1.store(k.k1, std::memory_order_relaxed);
                        s.k2.store(k.k2, std::memory_order_relaxed);
                        s.key.store(prepared_key, std::memory_order_relaxed);
                        s.node.store(prepared, std::memory_order_release);
                        if (size_.load(std::memory_order_relaxed) >= grow_at_) request_growth();
                        return {prepared, prepared_key, true};
                    }
                    p = expected;  // claimed by another thread first: maybe for this very key
                }
                while (p == claimed()) {
                    cpu_relax();
                    p = s.node.load(std::memory_order_acquire);
                }
                if (s.k1.load(std::memory_order_relaxed) == k.k1 && s.k2.load(std::memory_order_relaxed) == k.k2) {
                    if (reserved) size_.fetch_sub(1, std::memory_order_acq_rel);
                    return {p, s.key.load(std::memory_order_relaxed), false};
                }
            }
            if (!full) throw std::runtime_error("node table full");  // unreachable: at most 3/4 of the slots are used
            request_growth();                   // workers running: raise the flag; none: grow now
            if (group_) group_->arrive();       // park until the table has grown (at once if it has)
        }
    }

    // lock-free lookup without insert
    Found find(const NodeKey& k) const {
        const Slot* slots = slots_.get();
        const size_t mask = mask_;
        size_t i = (size_t)k.k1 & mask;
        for (size_t probes = 0; probes <= mask; probes++, i = (i + 1) & mask) {
            const Slot& s = slots[i];
            Node* p = s.node.load(std::memory_order_acquire);
            if (p == nullptr) return {};
            while (p == claimed()) {
                cpu_relax();
                p = s.node.load(std::memory_order_acquire);
            }
            if (s.k1.load(std::memory_order_relaxed) == k.k1 && s.k2.load(std::memory_order_relaxed) == k.k2)
                return {p, s.key.load(std::memory_order_relaxed), false};
        }
        return {};
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }
    size_t capacity() const { return mask_ + 1; }
    // changes whenever Node pointers handed out before become invalid (clear())
    uint64_t epoch() const { return epoch_; }
    size_t slot_bytes() const { return capacity() * sizeof(Slot); }
    size_t node_bytes() const { size_t s = 0; for (auto& a : arenas_) s += a->node_bytes(); return s; }
    size_t key_bytes() const { size_t s = 0; for (auto& a : arenas_) s += a->char_bytes(); return s; }

    // f(const char* key, Node& node) for every node, in slot order; not concurrent with inserts
    template <class F>
    void for_each(F f) {
        const size_t cap = capacity();
        Slot* slots = slots_.get();
        for (size_t i = 0; i < cap; i++) {
            Node* p = slots[i].node.load(std::memory_order_acquire);
            if (p != nullptr && p != claimed()) f(slots[i].key.load(std::memory_order_relaxed), *p);
        }
    }
    // the same with the numeric key: f(const NodeKey&, const char* key, Node& node)
    template <class F>
    void for_each_keyed(F f) {
        const size_t cap = capacity();
        Slot* slots = slots_.get();
        for (size_t i = 0; i < cap; i++) {
            Node* p = slots[i].node.load(std::memory_order_acquire);
            if (p == nullptr || p == claimed()) continue;
            NodeKey k;
            k.k1 = slots[i].k1.load(std::memory_order_relaxed);
            k.k2 = slots[i].k2.load(std::memory_order_relaxed);
            f(k, slots[i].key.load(std::memory_order_relaxed), *p);
        }
    }
    // the node in slot i, if any (i < capacity(); not concurrent with inserts)
    bool at(size_t i, NodeKey& k, const char*& key, Node*& node) const {
        const Slot& s = slots_[i];
        Node* p = s.node.load(std::memory_order_acquire);
        if (p == nullptr || p == claimed()) return false;
        k.k1 = s.k1.load(std::memory_order_relaxed);
        k.k2 = s.k2.load(std::memory_order_relaxed);
        key = s.key.load(std::memory_order_relaxed);
        node = p;
        return true;
    }
    uint64_t k2_at(size_t i) const { return slots_[i].k2.load(std::memory_order_relaxed); }

    // room for `more` nodes on top of size() without growing (loads of known size; not concurrent
    // with anything)
    void reserve(size_t more) {
        const size_t want = size() + more;
        size_t cap = capacity();
        while (want >= cap / 2) cap <<= 1;
        if (cap != capacity()) rehash(cap);
    }

    // not concurrent with anything; invalidates every Node pointer (epoch() changes)
    void clear() {
        epoch_++;
        for (auto& a : arenas_) a->clear();
        allocate(initial_);
        grow_requested_.store(false, std::memory_order_relaxed);
    }

    // double until the load is below 1/2; exclusive access (TableGroup)
    void grow() {
        size_t cap = capacity();
        const size_t n = size();
        while (n >= cap / 2) cap <<= 1;
        rehash(cap);
    }
    bool grow_requested() const { return grow_requested_.load(std::memory_order_relaxed); }

    // tests only (not concurrent): give the node of `k` another stored key string
    bool debug_set_key(const NodeKey& k, int arena, const std::string& key) {
        const size_t mask = mask_;
        size_t i = (size_t)k.k1 & mask;
        for (size_t probes = 0; probes <= mask; probes++, i = (i + 1) & mask) {
            Slot& s = slots_[i];
            if (s.node.load(std::memory_order_relaxed) == nullptr) return false;
            if (s.k1.load(std::memory_order_relaxed) == k.k1 && s.k2.load(std::memory_order_relaxed) == k.k2) {
                s.key.store(arenas_[arena]->copy_key(key.data(), key.size()), std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }

private:
    struct alignas(32) Slot {
        std::atomic<uint64_t> k1{0};
        std::atomic<uint64_t> k2{0};
        std::atomic<Node*> node{nullptr};        // nullptr: empty; claimed(): being written; else published
        std::atomic<const char*> key{nullptr};   // NUL-terminated key string, written before `node`
    };
    static Node* claimed() { return reinterpret_cast<Node*>(static_cast<uintptr_t>(1)); }

    // move every node into a fresh array of `cap` slots (a power of two above twice the size)
    void rehash(size_t cap) {
        std::unique_ptr<Slot[]> fresh(new Slot[cap]);
        const size_t m = cap - 1;
        const size_t old_cap = capacity();
        for (size_t i = 0; i < old_cap; i++) {
            Slot& s = slots_[i];
            Node* p = s.node.load(std::memory_order_relaxed);
            if (p == nullptr) continue;
            const uint64_t k1 = s.k1.load(std::memory_order_relaxed);
            size_t j = (size_t)k1 & m;
            while (fresh[j].node.load(std::memory_order_relaxed) != nullptr) j = (j + 1) & m;
            fresh[j].k1.store(k1, std::memory_order_relaxed);
            fresh[j].k2.store(s.k2.load(std::memory_order_relaxed), std::memory_order_relaxed);
            fresh[j].key.store(s.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
            fresh[j].node.store(p, std::memory_order_relaxed);
        }
        slots_ = std::move(fresh);
        mask_ = m;
        grow_at_ = cap / 2;
        hard_limit_ = cap - cap / 4;
        grow_requested_.store(false, std::memory_order_relaxed);
    }

    void allocate(size_t cap) {
        size_t c = 16;
        while (c < cap) c <<= 1;
        slots_.reset(new Slot[c]);
        mask_ = c - 1;
        grow_at_ = c / 2;
        hard_limit_ = c - c / 4;
        size_.store(0, std::memory_order_relaxed);
    }
    void request_growth() {
        if (grow_requested_.exchange(true, std::memory_order_acq_rel)) return;
        if (group_) group_->request();
        else grow();
    }

    std::unique_ptr<Slot[]> slots_;
    size_t mask_ = 0;
    size_t grow_at_ = 0;     // load at which growth is requested (1/2)
    size_t hard_limit_ = 0;  // most slots ever claimed (3/4)
    std::atomic<size_t> size_{0};  // published slots + open reservations (exact whenever no insert runs)
    std::atomic<bool> grow_requested_{false};
    std::vector<std::unique_ptr<NodeArena>> arenas_;
    TableGroup* group_ = nullptr;
    size_t initial_;
    uint64_t epoch_ = 0;
};

// Lookups by key string (Python, checkpoints).  The stored string must match too, so the answer
// is the one a string-keyed table gives even under a (never observed) numeric collision.
inline FlatNodeTable::Found find_by_string(const FlatNodeTable& t, const KeyCodec& c, const std::string& key) {
    FlatNodeTable::Found f = t.find(c.of_string(key));
    if (f.node != nullptr && std::strcmp(f.key, key.c_str()) != 0) return {};
    return f;
}

// Find or insert by key string (imports, merges); a new node gets the action ids and the string as
// given.  Two different strings with one numeric key raise instead of sharing a node.
inline FlatNodeTable::Found get_or_create_by_string(FlatNodeTable& t, const KeyCodec& c, const std::string& key,
                                                    const uint8_t* ids, int k, int arena) {
    FlatNodeTable::Found f = t.get_or_create(c.of_string(key), arena, [&](Node& n, NodeArena& a) {
        n.init(ids, k);
        return a.copy_key(key.data(), key.size());
    });
    if (!f.created && std::strcmp(f.key, key.c_str()) != 0)
        throw std::runtime_error("numeric infoset key collision: '" + key + "' and '" + std::string(f.key) + "'");
    return f;
}

inline void TableGroup::request() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!running_) {  // no workers: the caller is the only user of the tables
        for (FlatNodeTable* t : tables_) if (t->grow_requested()) { t->grow(); resizes_++; }
        return;
    }
    pending_.store(true, std::memory_order_release);
}

inline void TableGroup::grow_and_release() {
    for (FlatNodeTable* t : tables_) if (t->grow_requested()) { t->grow(); resizes_++; }
    waiting_ = 0;
    pending_.store(false, std::memory_order_release);
    ++gen_;
    cv_.notify_all();
}

}  // namespace negp
