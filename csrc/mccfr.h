// External-sampling MCCFR (+ Linear CFR) on the abstracted game: port of
// negpluribus/cfr/mccfr.py::MCCFRTrainer, multithreaded.
//
// Determinism / race semantics
// ----------------------------
// * threads == 1 reproduces the Python trainer bit for bit for the same seed (same deals,
//   same sampled actions, same floating-point sums).
// * threads > 1: iterations are handed out in chunks of ITER_CHUNK to whichever thread is free
//   (hybrid CPUs: P and E cores finish together), each thread has its own RNG stream, so the
//   deals are statistically the same as with one thread but *which* deals are played depends on
//   the scheduling, as does the interleaving of table updates.  Every node carries a spinlock: the current strategy is read,
//   and regrets / strategy sums are added, under that lock, so no update is lost.  What
//   remains racy is *staleness*: a traverser computes its regret update with the strategy it
//   read before descending, while other threads may have updated the same node in between.
//   That is the semantics of lock-free MCCFR implementations (Pluribus and friends) and is
//   benign for convergence.
//
// Nodes live in a FlatNodeTable under numeric keys (nodetable.h): the traversal carries an
// incremental hash of the history instead of rebuilding the key string at every node; the string
// is built once, when a node is created, and kept for export.  With verify_keys (test mode) every
// lookup also builds today's key string and checks it against the node's.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "engine.h"
#include "histtree.h"
#include "nodetable.h"
#include "pyrandom.h"

namespace negp {

struct Spec {
    int n_players = 2;
    int stack_bb = 20;
    int sb = 50, bb = 100, ante = 0;
    int max_street = FLOP;
    std::vector<double> preflop_fracs{1.0};
    std::vector<double> postflop_fracs{0.5, 1.0};
    int max_raises_per_street = 3;
    int n_buckets = 8;
    bool forbid_open_limp = false;
    bool allow_all_in = true;

    BetGrid grid() const {
        BetGrid g;
        g.preflop_fracs = preflop_fracs;
        g.postflop_fracs = postflop_fracs;
        g.allow_all_in = allow_all_in;
        g.max_raises_per_street = max_raises_per_street;
        g.forbid_open_limp = forbid_open_limp;
        g.finalize();
        return g;
    }
    std::vector<int> stacks() const { return std::vector<int>(n_players, stack_bb * bb); }
};

// Per-thread scratch of the traversals.
//
// bucket_memo: within one iteration every traversal replays the same deal (ctx.order fixes the
// seats' hole cards and the board of every street, and the button), so the bucket of
// (seat, board size) is a constant of the iteration.  It is computed on first use and forgotten
// at the start of the next iteration.  A bucket is a pure function of the cards, so the memo
// only removes repeated canonical_form() + cache lookups (2-player 100bb river game: about 52 per
// iteration down to at most 2 seats x 3 streets), never changes a value.
struct ThreadCtx {
    PyRandom rng;
    long long nodes_touched = 0;
    int tid = 0;               // worker index = this thread's arena in the node tables
    std::string key, hist;     // key strings of new nodes (and every node in test mode)
    std::string tok;           // one history token
    std::vector<int> order;
    int bucket_memo[MAX_PLAYERS][6];  // [seat][n_board], -1 = not computed in this iteration
    // history-tree traversal: the deal of the iteration and the showdown strengths by relative seat
    int button = 0;
    bool strength_done = false;
    int64_t strength[MAX_PLAYERS];

    ThreadCtx() { reset_bucket_memo(); }
    void reset_bucket_memo() {
        for (int s = 0; s < MAX_PLAYERS; s++)
            for (int b = 0; b < 6; b++) bucket_memo[s][b] = -1;
        strength_done = false;
    }
};

// the bucket of `seat` in `st` (what infoset_key() would compute), memoised for the iteration
inline int memo_bucket(Bucketer& bk, const HandState& st, int seat, ThreadCtx& ctx) {
    int& slot = ctx.bucket_memo[seat][st.n_board];
    if (slot < 0) slot = bk.bucket(st.players[seat].hole, st.board, st.n_board);
    return slot;
}

// ---- history-tree traversal helpers (histtree.h), shared by Trainer and RNRTrainer

// the value of terminal `h` for absolute seat `traverser` in bb (HandState::net / bb)
inline double tree_terminal_value(const HistNode* h, int n, int traverser, int bb, ThreadCtx& ctx) {
    if (h->n_act > 1 && !ctx.strength_done) {  // showdown strengths: once per seat and iteration
        for (int r = 0; r < n; r++) {
            const int seat = (r + ctx.button) % n;
            int cards[7] = {ctx.order[2 * seat], ctx.order[2 * seat + 1]};
            for (int i = 0; i < 5; i++) cards[2 + i] = ctx.order[2 * n + i];
            ctx.strength[r] = evaluate(cards, 7);
        }
        ctx.strength_done = true;
    }
    const int me = ((traverser - ctx.button) % n + n) % n;
    return (double)terminal_net(h, n, me, ctx.strength) / (double)bb;
}

// the bucket of absolute `seat` with `n_board` board cards in the iteration's deal (memoised)
inline int tree_bucket(Bucketer& bk, int n_players, int seat, int n_board, ThreadCtx& ctx) {
    int& slot = ctx.bucket_memo[seat][n_board];
    if (slot < 0) slot = bk.bucket(&ctx.order[2 * seat], &ctx.order[2 * n_players], n_board);
    return slot;
}

// the key string infoset_key_for_bucket() spells for decision node `h` and bucket `b`
inline void tree_key(const HistNode* h, int b, int n, std::string& key) {
    key.clear();
    key += street_letter(h->street);
    key += '|';
    key += position_name(h->rel, 0, n);
    key += '|';
    key += std::to_string(h->n_active);
    key += "|b";
    key += std::to_string(b);
    key += '|';
    key += h->hist;
}

// Test mode (verify_keys): the stored key string of every node a traversal reaches must equal the
// string infoset_key builds from the state.  A mismatch would mean two key strings share a numeric
// key; the first one is kept and reported when train() returns.
class KeyChecker {
public:
    void check(const char* stored, const HandState& st, const Obs& obs, int bucket, const BetGrid& grid, ThreadCtx& ctx) {
        infoset_key_for_bucket(st, obs, bucket, grid, ctx.key, ctx.hist);
        if (std::strcmp(stored, ctx.key.c_str()) != 0) report("node found for '" + ctx.key + "' is stored as '" + stored + "'");
    }
    void report(const std::string& msg) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!failed_.load(std::memory_order_relaxed)) {
            msg_ = msg;
            failed_.store(true, std::memory_order_relaxed);
        }
    }
    // after the workers joined: throw the first mismatch, if any
    void rethrow() {
        if (!failed_.load(std::memory_order_relaxed)) return;
        failed_.store(false, std::memory_order_relaxed);
        throw std::runtime_error("numeric infoset key mismatch: " + msg_);
    }

private:
    std::mutex mu_;
    std::atomic<bool> failed_{false};
    std::string msg_;
};

class Trainer {
public:
    Spec spec;
    BetGrid grid;
    std::shared_ptr<Bucketer> bucketer;  // EquityBucketer or PotentialBucketer
    FlatNodeTable nodes;
    KeyCodec codec;
    bool linear = true;
    int threads = 1;
    uint64_t seed = 0;
    bool verify_keys = false;  // test mode, see KeyChecker
    std::mutex api_mu;         // held by train() and by the Python-facing table accessors (bindings)

    Trainer(const Spec& spec_, std::shared_ptr<Bucketer> bk, uint64_t seed_, bool linear_, int threads_, bool verify_keys_ = false)
        : spec(spec_), grid(spec_.grid()), bucketer(std::move(bk)), codec(spec_.n_players), linear(linear_),
          threads(std::max(1, threads_)), seed(seed_), verify_keys(verify_keys_),
          tree_(spec.stacks(), spec.sb, spec.bb, spec.ante, spec.max_street, grid, std::max(1, bucketer->identity().n_buckets)) {
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        if (grid.preflop_fracs.size() > 5 || grid.postflop_fracs.size() > 5)
            throw std::invalid_argument("too many grid fractions for the C++ core (max 5 per street)");
        nodes.set_arenas(threads + 1);  // one per worker, one for imports (main_arena())
        nodes.attach(&group_);
        group_.add(&nodes);
        ctxs_.resize(threads);
        for (int t = 0; t < threads; t++) {
            ctxs_[t].tid = t;
            if (t == 0) ctxs_[t].rng.seed(seed);
            else ctxs_[t].rng.seed_words({(uint32_t)(seed & 0xffffffffU), (uint32_t)(seed >> 32), (uint32_t)t});
        }
    }

    long long iteration() const { return iteration_; }
    void set_iteration(long long t) { iteration_ = t; }
    long long nodes_touched() const { return nodes_touched_; }
    void set_nodes_touched(long long v) { nodes_touched_ = v; }
    int main_arena() const { return threads; }
    long long table_resizes() const { return group_.resizes(); }

    // run `iterations` iterations on `threads` threads
    static constexpr long long ITER_CHUNK = 16;

    void train(long long iterations) {
        long long base = iteration_;
        long long target = base + iterations;
        int T = threads;
        root_ = tree_.root(nodes.epoch());
        group_.begin(T);
        // T == 1: iterations in order (bit-identical to the Python trainer).  T > 1: chunks of
        // iterations handed out dynamically, so faster cores (P vs E cores, a busy sibling
        // hyperthread) do more of them instead of waiting for the slowest thread at the end;
        // each thread still draws its deals from its own RNG stream.
        std::atomic<long long> next{base + 1};
        auto work = [&](int tid) {
            ThreadCtx& ctx = ctxs_[tid];
            if (T == 1) {
                for (long long t = base + 1; t <= target; t++) run_iteration(t, ctx);
            } else {
                for (;;) {
                    const long long lo = next.fetch_add(ITER_CHUNK, std::memory_order_relaxed);
                    if (lo > target) break;
                    const long long hi = lo + ITER_CHUNK - 1 < target ? lo + ITER_CHUNK - 1 : target;
                    for (long long t = lo; t <= hi; t++) run_iteration(t, ctx);
                }
            }
            group_.leave();
        };
        if (T == 1) {
            work(0);
        } else {
            std::vector<std::thread> pool;
            for (int t = 1; t < T; t++) pool.emplace_back(work, t);
            work(0);
            for (auto& th : pool) th.join();
        }
        group_.end();
        iteration_ = target;
        long long touched = 0;
        for (auto& c : ctxs_) { touched += c.nodes_touched; c.nodes_touched = 0; }
        nodes_touched_ += touched;
        checker_.rethrow();
    }

    // RNG state of thread 0 (interop with random.Random.getstate()/setstate())
    PyRandom& rng0() { return ctxs_[0].rng; }
    // every thread's generator (checkpoints restore them so a resumed run continues its streams)
    PyRandom& rng(int tid) { return ctxs_[tid].rng; }

    size_t tree_nodes() const { return tree_.nodes(); }

private:
    TableGroup group_;
    KeyChecker checker_;
    std::vector<ThreadCtx> ctxs_;
    long long iteration_ = 0;
    long long nodes_touched_ = 0;
    HistTree tree_;
    HistNode* root_ = nullptr;

    void run_iteration(long long t, ThreadCtx& ctx) {
        double weight = linear ? (double)t : 1.0;
        ctx.order.resize(52);
        for (int i = 0; i < 52; i++) ctx.order[i] = i;
        ctx.rng.shuffle(ctx.order);
        ctx.reset_bucket_memo();  // a new deal
        int button = (int)(t % spec.n_players);
        ctx.button = button;
        for (int traverser = 0; traverser < spec.n_players; traverser++) traverse_tree(root_, traverser, weight, ctx);
    }

    // ---- the traversal on the history tree (histtree.h); same numbers, same random draws as
    // traverse() on the engine, which remains for the rare node whose stored actions differ
    Node* tree_node(HistNode* h, int b, ThreadCtx& ctx, bool& cached) {
        cached = false;
        if (b < h->n_cache && !verify_keys) {  // test mode: every visit goes through the table and the key check
            Node* p = h->cache[b].load(std::memory_order_acquire);
            if (p) { cached = true; return p; }
        }
        const int n = spec.n_players;
        FlatNodeTable::Found f = nodes.get_or_create(node_key(h->street, h->rel, h->n_active, b, h->hh), ctx.tid,
                                                     [&](Node& nd, NodeArena& arena) {
            nd.init(h->ids, h->na);
            tree_key(h, b, n, ctx.key);
            return arena.copy_key(ctx.key.data(), ctx.key.size());
        });
        if (verify_keys) {
            tree_key(h, b, n, ctx.key);
            if (std::strcmp(f.key, ctx.key.c_str()) != 0)
                checker_.report("node found for '" + ctx.key + "' is stored as '" + f.key + "'");
        }
        return f.node;
    }

    double traverse_tree(HistNode* h, int traverser, double weight, ThreadCtx& ctx) {
        const int n = spec.n_players;
        if (h->terminal) return tree_terminal_value(h, n, traverser, spec.bb, ctx);
        const int seat = (h->rel + ctx.button) % n;
        const int b = tree_bucket(*bucketer, n, seat, h->n_board, ctx);
        bool cached;
        Node* node = tree_node(h, b, ctx, cached);
        if (!cached) {
            bool same = node->n == h->na;
            if (same) for (int i = 0; i < h->na; i++) if (node->acts[i] != h->ids[i]) { same = false; break; }
            if (!same) {  // a node imported with another action list: the engine traversal copes
                HandState st = tree_.replay(h, ctx.order.data(), ctx.button);
                return traverse(st, HistHash(), traverser, weight, ctx);
            }
            if (b < h->n_cache) h->cache[b].store(node, std::memory_order_release);
        }
        ctx.nodes_touched++;
        const int na = h->na;
        double sigma[MAX_ACTIONS];
        node->lock.lock();
        node->current_strategy(sigma);
        node->lock.unlock();

        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            for (int i = 0; i < na; i++) utils[i] = traverse_tree(tree_.child(h, i), traverser, weight, ctx);
            double prods[MAX_ACTIONS];
            for (int i = 0; i < na; i++) prods[i] = sigma[i] * utils[i];
            double u = py_sum(prods, na);
            node->lock.lock();
            for (int i = 0; i < na; i++) node->regret[i] += weight * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        node->lock.lock();
        for (int i = 0; i < na; i++) node->strategy_sum[i] += weight * sigma[i];
        node->visits += 1;
        node->lock.unlock();
        int a = sample(sigma, na, ctx.rng);
        return traverse_tree(tree_.child(h, a), traverser, weight, ctx);
    }

    static int sample(const double* probs, int n, PyRandom& rng) {
        double r = rng.random();
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += probs[i];
            if (r < acc) return i;
        }
        return n - 1;
    }

    // `hh` hashes the history of the parent state; the events added since are hashed here
    double traverse(HandState& st, HistHash hh, int traverser, double weight, ThreadCtx& ctx) {
        if (st.terminal) return (double)st.net(traverser) / (double)spec.bb;
        hh.catch_up(st, grid, ctx.tok);
        int seat = st.to_act;
        Obs obs = observe(st, seat);
        ActionList actions;
        grid.abstract_actions(obs, actions);
        uint8_t ids[MAX_ACTIONS];
        int na = actions.n;
        for (int i = 0; i < na; i++) ids[i] = (uint8_t)actions.a[i].id;
        const int b = memo_bucket(*bucketer, st, seat, ctx);
        FlatNodeTable::Found f = nodes.get_or_create(codec.key(obs, st.button, b, hh), ctx.tid, [&](Node& n, NodeArena& arena) {
            n.init(ids, na);
            infoset_key_for_bucket(st, obs, b, grid, ctx.key, ctx.hist);
            return arena.copy_key(ctx.key.data(), ctx.key.size());
        });
        if (verify_keys) checker_.check(f.key, st, obs, b, grid, ctx);
        Node* node = f.node;
        ctx.nodes_touched++;
        bool same = node->n == na;
        if (same) for (int i = 0; i < na; i++) if (node->acts[i] != ids[i]) { same = false; break; }
        if (!same) {
            // should not happen in self-play; keep going safely with the node's own list
            actions.clear();
            for (int i = 0; i < node->n; i++) actions.push(grid.action_from_name(grid.names[node->acts[i]]));
            na = actions.n;
        }
        double sigma[MAX_ACTIONS];
        node->lock.lock();
        node->current_strategy(sigma);
        node->lock.unlock();

        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            for (int i = 0; i < na; i++) {
                HandState child(st);
                int type, amount;
                grid.to_concrete(obs, actions.a[i], type, amount);
                child.apply(type, amount);
                utils[i] = traverse(child, hh, traverser, weight, ctx);
            }
            double prods[MAX_ACTIONS];
            for (int i = 0; i < na; i++) prods[i] = sigma[i] * utils[i];
            double u = py_sum(prods, na);
            node->lock.lock();
            for (int i = 0; i < na; i++) node->regret[i] += weight * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        node->lock.lock();
        for (int i = 0; i < na; i++) node->strategy_sum[i] += weight * sigma[i];
        node->visits += 1;
        node->lock.unlock();
        int a = sample(sigma, na, ctx.rng);
        int type, amount;
        grid.to_concrete(obs, actions.a[a], type, amount);
        st.apply(type, amount);
        return traverse(st, hh, traverser, weight, ctx);
    }
};

}  // namespace negp
