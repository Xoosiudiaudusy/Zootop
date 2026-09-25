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
#include "philox.h"
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
struct alignas(64) ThreadCtx {  // one cache line boundary per thread: no false sharing between neighbours in ctxs_
    PyRandom rng;
    long long nodes_touched = 0;
    int tid = 0;               // worker index = this thread's arena in the node tables
    std::string key, hist;     // key strings of new nodes (and every node in test mode)
    std::string tok;           // one history token
    std::vector<int> order;
    int bucket_memo[MAX_PLAYERS][6];  // [seat][n_board], -1 = not computed in this iteration
    // history-tree traversal: the deal of the iteration and the showdown strengths by relative seat
    int button = 0;
    bool prune = false;        // this iteration prunes (Trainer::prune_below)
    double prune_limit = 0.0;  // regrets below this are pruned
    double floor_base = 0.0;   // < 0: regrets are clamped at this (relative pruning: x the node's weight)
    long long pruned = 0;      // actions skipped
    bool strength_done = false;
    int64_t strength[MAX_PLAYERS];

    // batched mode (Trainer::batch_size): the updates of this thread's iterations of the current batch
    struct Upd {
        uint64_t k1, k2;   // node key: the sort order that makes the application deterministic
        long long t;       // iteration
        uint32_t seq;      // order within the iteration's traversals
        uint16_t a;        // action (visits: 0)
        uint8_t kind;      // 0 regret, 1 strategy sum, 2 visits
        double v;
        Node* node;
    };
    std::vector<Upd> upd;
    uint32_t upd_seq = 0;

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
inline double tree_terminal_value(const HistTerminal* h, int n, int traverser, int bb, ThreadCtx& ctx) {
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
    // Regret-based pruning (Pluribus, Brown & Sandholm 2019, supplement): in a share `prune_prob` of the
    // iterations after `prune_after`, the traverser does not explore an action whose accumulated
    // regret is below -prune_below (the stored units: bb x iteration weight; Pluribus: -300,000,000
    // in its own units); never on the last betting street (unless prune_last_street, for
    // measurements on one-street games), never an action that ends the hand, and never every action
    // of a node.  prune_below <= 0: off (the default; then nothing changes, not even the draws).
    long long linear_until = 0;  // > 0: Linear CFR weights stop growing after this iteration (off: 0)
    // relative pruning: prune_below is in bb of regret per unit of the node's own traverser weight
    // (the weight-averaged regret per visit), which does not grow with t or with the bucket count;
    // nodes created while it is on carry that weight (Node::tw, 8 bytes), older nodes are never pruned
    bool prune_relative = false;
    // Pluribus-like scale: the threshold grows with the iteration, -prune_below x t (Pluribus kept its
    // threshold fixed on regrets that, once the discounting stopped, grow like t; ours grow like t^2)
    bool prune_scale_t = false;
    // regret floor (Pluribus: -310M against a -300M threshold, "for every action", so a pruned action
    // that improves comes back): regrets are clamped at regret_floor x the pruning threshold after
    // every update (e.g. 1.033); 0: no floor.  Applied in every iteration once t > prune_after.
    double regret_floor = 0.0;
    double prune_below = 0.0;
    double prune_prob = 0.95;
    long long prune_after = 0;
    bool prune_last_street = false;
    std::mutex api_mu;         // held by train() and by the Python-facing table accessors (bindings)

    Trainer(const Spec& spec_, std::shared_ptr<Bucketer> bk, uint64_t seed_, bool linear_, int threads_, bool verify_keys_ = false)
        : spec(spec_), grid(spec_.grid()), bucketer(std::move(bk)), codec(spec_.n_players), linear(linear_),
          threads(std::max(1, threads_)), seed(seed_), verify_keys(verify_keys_),
          tree_(spec.stacks(), spec.sb, spec.bb, spec.ante, spec.max_street, grid, std::max(1, bucketer->identity().n_buckets)) {
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        if (grid.preflop_fracs.size() > 5 || grid.postflop_fracs.size() > 5)
            throw std::invalid_argument("too many grid fractions for the C++ core (max 5 per street)");
        nodes.set_arenas(threads + 1);  // one per worker, one for imports (main_arena())
        nodes.set_dense(true);          // reached through the history tree's pointer caches
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

    // Batched synchronous mode (batch_size > 0; the CPU reference of the GPU trainer): the iterations of
    // a batch all read the strategy of the tables as they were at the start of the batch; their updates
    // are recorded and applied after the batch, summed per cell in the order (node key, kind, action,
    // iteration, sequence), so the result does not depend on the number of threads or their schedule.
    // Deals and samples come from Philox4x32-10 addressed by (seed, iteration, ...) (philox.h).  Another
    // algorithm than the sequential trainer when batch_size > 1 (every iteration of a batch sees the
    // same strategy); judge it by the result.  Pruning is not supported in this mode.
    long long batch_size = 0;

    void train(long long iterations) {
        if (batch_size > 0) { train_batched(iterations); return; }
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
        for (auto& c : ctxs_) { touched += c.nodes_touched; c.nodes_touched = 0; pruned_ += c.pruned; c.pruned = 0; }
        nodes_touched_ += touched;
        checker_.rethrow();
    }
    long long pruned_actions() const { return pruned_; }

    void train_batched(long long iterations) {
        if (prune_below > 0.0) throw std::invalid_argument("pruning is not supported in batched mode");
        const long long target = iteration_ + iterations;
        const int T = threads;
        root_ = tree_.root(nodes.epoch());
        std::vector<ThreadCtx::Upd> all;
        while (iteration_ < target) {
            const long long lo = iteration_ + 1;
            // batches are aligned on absolute iterations (1..B, B+1..2B, ...): a run split into several
            // train() calls or resumed from a checkpoint forms the same batches
            const long long hi = std::min(target, (iteration_ / batch_size + 1) * batch_size);
            // phase 1: traverse, reading the tables only (nodes may be created)
            group_.begin(T);
            std::atomic<long long> next{lo};
            auto work = [&](int tid) {
                ThreadCtx& ctx = ctxs_[tid];
                for (long long t = next.fetch_add(1); t <= hi; t = next.fetch_add(1)) run_iteration_batched(t, ctx);
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
            // phase 2: apply the updates in a fixed order
            all.clear();
            for (auto& c : ctxs_) { all.insert(all.end(), c.upd.begin(), c.upd.end()); c.upd.clear(); }
            std::sort(all.begin(), all.end(), [](const ThreadCtx::Upd& x, const ThreadCtx::Upd& y) {
                if (x.k1 != y.k1) return x.k1 < y.k1;
                if (x.k2 != y.k2) return x.k2 < y.k2;
                if (x.kind != y.kind) return x.kind < y.kind;
                if (x.a != y.a) return x.a < y.a;
                if (x.t != y.t) return x.t < y.t;
                return x.seq < y.seq;
            });
            for (const ThreadCtx::Upd& u : all) {
                if (u.kind == 0) u.node->regret()[u.a] += u.v;
                else if (u.kind == 1) u.node->strategy_sum()[u.a] += u.v;
                else u.node->visits += 1;
            }
            iteration_ = hi;
        }
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
    size_t tree_bytes() const { return tree_.bytes(); }

private:
    TableGroup group_;
    KeyChecker checker_;
    std::vector<ThreadCtx> ctxs_;
    long long iteration_ = 0;
    long long nodes_touched_ = 0;
    HistTree tree_;
    HistNode* root_ = nullptr;
    long long pruned_ = 0;

    void run_iteration(long long t, ThreadCtx& ctx) {
        // Linear CFR: weight t; with linear_until > 0 the weight stops growing at that iteration (Pluribus
        // stopped its linear discounting after 400 minutes), i.e. plain CFR from there on
        double weight = linear ? (double)(linear_until > 0 && t > linear_until ? linear_until : t) : 1.0;
        ctx.order.resize(52);
        for (int i = 0; i < 52; i++) ctx.order[i] = i;
        ctx.rng.shuffle(ctx.order);
        ctx.reset_bucket_memo();  // a new deal
        int button = (int)(t % spec.n_players);
        ctx.button = button;
        ctx.prune_limit = 0.0;
        ctx.prune = false;
        ctx.floor_base = 0.0;
        if (prune_below > 0.0 && t > prune_after) {
            const double base = prune_scale_t ? -prune_below * (double)t : -prune_below;
            if (regret_floor > 0.0) ctx.floor_base = regret_floor * base;
            if (ctx.rng.random() < prune_prob) {
                ctx.prune = true;
                ctx.prune_limit = base;
            }
        }
        for (int traverser = 0; traverser < spec.n_players; traverser++) traverse_tree(root_, traverser, weight, ctx);
    }

    void run_iteration_batched(long long t, ThreadCtx& ctx) {
        const double weight = linear ? (double)(linear_until > 0 && t > linear_until ? linear_until : t) : 1.0;
        ctx.order.resize(52);
        philox_deal(seed, (uint64_t)t, ctx.order.data());
        ctx.reset_bucket_memo();
        ctx.button = (int)(t % spec.n_players);
        ctx.upd_seq = 0;
        for (int traverser = 0; traverser < spec.n_players; traverser++) traverse_batched(root_, traverser, weight, t, ctx);
    }

    // the tree traversal of batched mode: reads the tables, records the updates (see batch_size)
    double traverse_batched(HistNode* node_h, int traverser, double weight, long long t, ThreadCtx& ctx) {
        const int n = spec.n_players;
        if (node_h->terminal) return tree_terminal_value(static_cast<const HistTerminal*>(node_h), n, traverser, spec.bb, ctx);
        HistDecision* h = static_cast<HistDecision*>(node_h);
        const int seat = (h->rel + ctx.button) % n;
        const int b = tree_bucket(*bucketer, n, seat, h->n_board, ctx);
        bool cached;
        Node* node = tree_node(h, b, ctx, cached);
        if (!cached) {
            bool same = node->n == h->na;
            if (same) for (int i = 0; i < h->na; i++) if (node->acts[i] != h->ids[i]) { same = false; break; }
            if (!same) throw std::runtime_error("batched mode: a node with another action list (imported table?)");
            if (b < h->n_cache) h->cache()[b].store(node, std::memory_order_release);
        }
        const NodeKey nk = node_key(h->street, h->rel, h->n_active, b, h->hh);
        ctx.nodes_touched++;
        const int na = h->na;
        double sigma[MAX_ACTIONS];
        node->current_strategy(sigma);  // no writer during the traversal phase
        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            for (int i = 0; i < na; i++) utils[i] = traverse_batched(tree_.child(h, i), traverser, weight, t, ctx);
            double prods[MAX_ACTIONS];
            for (int i = 0; i < na; i++) prods[i] = sigma[i] * utils[i];
            const double u = py_sum(prods, na);
            for (int i = 0; i < na; i++) ctx.upd.push_back({nk.k1, nk.k2, t, ctx.upd_seq++, (uint16_t)i, 0, weight * (utils[i] - u), node});
            return u;
        }
        for (int i = 0; i < na; i++) ctx.upd.push_back({nk.k1, nk.k2, t, ctx.upd_seq++, (uint16_t)i, 1, weight * sigma[i], node});
        ctx.upd.push_back({nk.k1, nk.k2, t, ctx.upd_seq++, 0, 2, 0.0, node});
        const double r = philox_sample_u01(seed, (uint64_t)t, traverser, h->hh.a, h->hh.b);
        int a = na - 1;
        double acc = 0.0;
        for (int i = 0; i < na; i++) {
            acc += sigma[i];
            if (r < acc) { a = i; break; }
        }
        return traverse_batched(tree_.child(h, a), traverser, weight, t, ctx);
    }

    // ---- the traversal on the history tree (histtree.h); same numbers, same random draws as
    // traverse() on the engine, which remains for the rare node whose stored actions differ
    Node* tree_node(HistDecision* h, int b, ThreadCtx& ctx, bool& cached) {
        cached = false;
        if (b < h->n_cache && !verify_keys) {  // test mode: every visit goes through the table and the key check
            Node* p = h->cache()[b].load(std::memory_order_acquire);
            if (p) { cached = true; return p; }
        }
        const int n = spec.n_players;
        const int extra = prune_relative ? 1 : 0;
        FlatNodeTable::Found f = nodes.get_or_create(node_key(h->street, h->rel, h->n_active, b, h->hh), ctx.tid, h->na, extra,
                                                     [&](Node& nd, NodeArena&) {
            nd.init(h->ids, h->na);
            if (extra) { nd.has_tw = 1; *nd.tw() = 0.0; }
            return nd.refer_to_tree(h, b);  // key string spelled on demand from the tree (no copy)
        });
        if (verify_keys) {
            tree_key(h, b, n, ctx.key);
            if (std::strcmp(f.key, ctx.key.c_str()) != 0)
                checker_.report("node found for '" + ctx.key + "' is stored as '" + f.key + "'");
        }
        return f.node;
    }

    double traverse_tree(HistNode* node_h, int traverser, double weight, ThreadCtx& ctx) {
        const int n = spec.n_players;
        if (node_h->terminal) return tree_terminal_value(static_cast<const HistTerminal*>(node_h), n, traverser, spec.bb, ctx);
        HistDecision* h = static_cast<HistDecision*>(node_h);
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
            if (b < h->n_cache) h->cache()[b].store(node, std::memory_order_release);
        }
        ctx.nodes_touched++;
        const int na = h->na;
        double sigma[MAX_ACTIONS];
        double reg[MAX_ACTIONS];
        const bool prune_here = ctx.prune && seat == traverser && (prune_last_street || h->street < spec.max_street);
        node->lock.lock();
        node->current_strategy(sigma);
        double limit = ctx.prune_limit;
        if (prune_here) {
            for (int i = 0; i < na; i++) reg[i] = node->regret()[i];
            if (prune_relative) limit = node->has_tw ? ctx.prune_limit * *node->tw() : -1e300;
        }
        node->lock.unlock();

        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            bool explore[MAX_ACTIONS];
            int n_explore = 0;
            for (int i = 0; i < na; i++) {
                explore[i] = true;
                if (prune_here && reg[i] < limit && !tree_.child(h, i)->terminal) explore[i] = false;
                n_explore += explore[i];
            }
            if (n_explore == 0) for (int i = 0; i < na; i++) explore[i] = true;
            for (int i = 0; i < na; i++) {
                if (explore[i]) utils[i] = traverse_tree(tree_.child(h, i), traverser, weight, ctx);
                else { utils[i] = 0.0; ctx.pruned++; }
            }
            double prods[MAX_ACTIONS];
            for (int i = 0; i < na; i++) prods[i] = explore[i] ? sigma[i] * utils[i] : 0.0;
            double u = py_sum(prods, na);
            node->lock.lock();
            for (int i = 0; i < na; i++) if (explore[i]) node->regret()[i] += weight * (utils[i] - u);
            if (ctx.floor_base < 0.0) {
                const double fl = prune_relative ? (node->has_tw ? ctx.floor_base * *node->tw() : -1e300) : ctx.floor_base;
                for (int i = 0; i < na; i++) if (node->regret()[i] < fl) node->regret()[i] = fl;
            }
            if (node->has_tw) *node->tw() += weight;
            node->lock.unlock();
            return u;
        }
        node->lock.lock();
        for (int i = 0; i < na; i++) node->strategy_sum()[i] += weight * sigma[i];
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
        FlatNodeTable::Found f = nodes.get_or_create(codec.key(obs, st.button, b, hh), ctx.tid, na, [&](Node& n, NodeArena& arena) {
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
            for (int i = 0; i < na; i++) node->regret()[i] += weight * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        node->lock.lock();
        for (int i = 0; i < na; i++) node->strategy_sum()[i] += weight * sigma[i];
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
