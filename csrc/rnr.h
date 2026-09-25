// Restricted Nash Response trainer: port of negpluribus/exploit/rnr.py::RNRTrainer.
//
// Two node tables (hero seat / rational part of the opponents), hero seat and button rotate
// with the iteration, the opponent model is a *table* {key: (names, probs)} materialised in
// Python from the model object (OpponentModel / BlueprintStrategy) with the blueprint's own
// action lists, so that ``model.policy(key, actions)`` is reproduced exactly in self-play.
// Warm start from a blueprint table with ``warm_visits`` pseudo-visits per key, as in the
// reference.  Single thread == bit-identical to the Python trainer.
//
// Node keys are numeric as in mccfr.h (nodetable.h); the model table is indexed by the same
// numeric keys, the warm start (read once, when a node is created) by the key string.
#pragma once
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "mccfr.h"

namespace negp {

struct StratEntry {
    std::vector<std::string> names;
    std::vector<double> probs;
};
using StratTable = std::unordered_map<std::string, StratEntry>;

// BlueprintStrategy.policy(key, legal): probabilities aligned with ``legal`` or None (false)
inline bool blueprint_policy(const StratEntry& e, const ActionList& legal, const BetGrid& grid, double* out) {
    double vals[MAX_ACTIONS];
    for (int i = 0; i < legal.n; i++) {
        const std::string& name = grid.names[legal.a[i].id];
        double v = 0.0;
        for (size_t j = 0; j < e.names.size(); j++) if (e.names[j] == name) v = e.probs[j];  // dict(zip(...)): last wins
        vals[i] = v;
    }
    double s = py_sum(vals, legal.n);
    if (s <= 0) return false;
    for (int i = 0; i < legal.n; i++) out[i] = vals[i] / s;
    return true;
}

class RNRTrainer {
public:
    Spec spec;
    BetGrid grid;
    std::shared_ptr<Bucketer> bucketer;  // EquityBucketer or PotentialBucketer
    FlatNodeTable hero_nodes, opp_nodes;
    KeyCodec codec;
    bool linear = true;
    int threads = 1;
    uint64_t seed = 0;
    double p_model = 0.0;
    StratTable model;       // materialised opponent model (set with set_tables)
    StratTable warm_start;  // blueprint for warm start (may be empty)
    double warm_visits = 30.0;
    double regret_scale_bb = 1.0;
    long long planned_iters = 0;
    bool verify_keys = false;  // test mode, see KeyChecker (mccfr.h)
    std::mutex api_mu;         // held by train() and by the Python-facing table accessors (bindings)

    RNRTrainer(const Spec& spec_, std::shared_ptr<Bucketer> bk, uint64_t seed_, bool linear_, int threads_,
               double p_model_, double warm_visits_, double regret_scale_bb_, bool verify_keys_ = false)
        : spec(spec_), grid(spec_.grid()), bucketer(std::move(bk)), codec(spec_.n_players), linear(linear_),
          threads(std::max(1, threads_)), seed(seed_), p_model(p_model_), warm_visits(warm_visits_),
          regret_scale_bb(regret_scale_bb_), verify_keys(verify_keys_),
          tree_(spec.stacks(), spec.sb, spec.bb, spec.ante, spec.max_street, grid, std::max(1, bucketer->identity().n_buckets), 2) {
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        if (grid.preflop_fracs.size() > 5 || grid.postflop_fracs.size() > 5)
            throw std::invalid_argument("too many grid fractions for the C++ core (max 5 per street)");
        for (FlatNodeTable* t : {&hero_nodes, &opp_nodes}) {
            t->set_arenas(threads + 1);  // one per worker, one for imports (main_arena())
            t->attach(&group_);
            group_.add(t);
        }
        ctxs_.resize(threads);
        for (int t = 0; t < threads; t++) {
            ctxs_[t].tid = t;
            if (t == 0) ctxs_[t].rng.seed(seed);
            else ctxs_[t].rng.seed_words({(uint32_t)(seed & 0xffffffffU), (uint32_t)(seed >> 32), (uint32_t)t});
        }
    }

    // the opponent model and the warm start; the model is indexed by numeric key
    void set_tables(StratTable model_, StratTable warm_start_) {
        model = std::move(model_);
        warm_start = std::move(warm_start_);
        model_idx_.clear();
        model_idx_.reserve(model.size());
        for (const auto& kv : model) {
            if (!model_idx_.emplace(codec.of_string(kv.first), &kv).second)
                throw std::runtime_error("numeric infoset key collision in the opponent model: '" + kv.first + "'");
        }
    }

    long long iteration() const { return iteration_; }
    void set_iteration(long long t) { iteration_ = t; }
    long long nodes_touched() const { return nodes_touched_; }
    void set_nodes_touched(long long v) { nodes_touched_ = v; }
    PyRandom& rng0() { return ctxs_[0].rng; }
    PyRandom& rng(int tid) { return ctxs_[tid].rng; }
    int main_arena() const { return threads; }
    long long table_resizes() const { return group_.resizes(); }

    // planned_iters (warm-start weight) is set by the caller once per outer train() call, as in
    // RNRTrainer.train(); chunked calls must not shrink it
    static constexpr long long ITER_CHUNK = 16;

    void train(long long iterations) {
        long long base = iteration_;
        long long target = base + iterations;
        int T = threads;
        root_ = tree_.root(hero_nodes.epoch() + opp_nodes.epoch());  // both only grow: the sum changes with either
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

private:
    TableGroup group_;
    KeyChecker checker_;
    std::unordered_map<NodeKey, const StratTable::value_type*, NodeKeyHash> model_idx_;
    std::vector<ThreadCtx> ctxs_;
    long long iteration_ = 0;
    long long nodes_touched_ = 0;
    HistTree tree_;
    HistNode* root_ = nullptr;

    void run_iteration(long long t, ThreadCtx& ctx) {
        int n = spec.n_players;
        double weight = linear ? (double)t : 1.0;
        ctx.order.resize(52);
        for (int i = 0; i < 52; i++) ctx.order[i] = i;
        ctx.rng.shuffle(ctx.order);
        ctx.reset_bucket_memo();  // a new deal (every traverser below replays it: mccfr.h ThreadCtx)
        int button = (int)(t % n);
        int hero_seat = (int)((t / n) % n);
        ctx.button = button;
        for (int traverser = 0; traverser < n; traverser++) traverse_tree(root_, traverser, weight, hero_seat, ctx);
    }

    // ---- traversal on the history tree (histtree.h, mccfr.h): same draws, same sums as traverse()
    static void tree_actions(const HistNode* h, ActionList& out) {
        out.clear();
        for (int i = 0; i < h->na; i++) out.push(h->acts[i]);
    }

    Node* tree_node(HistNode* h, int b, bool is_hero, const NodeKey& nk, ThreadCtx& ctx, bool& cached) {
        cached = false;
        const int slot = (is_hero ? 0 : 1) * h->n_cache + b;
        if (b < h->n_cache && !verify_keys) {
            Node* p = h->cache[slot].load(std::memory_order_acquire);
            if (p) { cached = true; return p; }
        }
        const int n = spec.n_players;
        FlatNodeTable& table = is_hero ? hero_nodes : opp_nodes;
        FlatNodeTable::Found f = table.get_or_create(nk, ctx.tid, h->na, [&](Node& node, NodeArena& arena) {
            node.init(h->ids, h->na);
            tree_key(h, b, n, ctx.key);
            if (!warm_start.empty() && warm_visits > 0) {
                auto it = warm_start.find(ctx.key);
                if (it != warm_start.end()) {
                    ActionList actions;
                    tree_actions(h, actions);
                    double base[MAX_ACTIONS];
                    if (blueprint_policy(it->second, actions, grid, base)) {
                        double mean_w = linear ? std::max(1.0, (double)planned_iters / 2.0) : 1.0;
                        double w = warm_visits * mean_w;
                        for (int i = 0; i < actions.n; i++) {
                            node.regret()[i] = w * regret_scale_bb * base[i];
                            node.strategy_sum()[i] = w * base[i];
                        }
                    }
                }
            }
            return arena.copy_key(ctx.key.data(), ctx.key.size());
        });
        if (verify_keys) {
            tree_key(h, b, n, ctx.key);  // leaves today's key string in ctx.key (model_policy check)
            if (std::strcmp(f.key, ctx.key.c_str()) != 0)
                checker_.report("node found for '" + ctx.key + "' is stored as '" + f.key + "'");
        }
        return f.node;
    }

    double traverse_tree(HistNode* h, int traverser, double weight, int hero_seat, ThreadCtx& ctx) {
        const int n = spec.n_players;
        if (h->terminal) return tree_terminal_value(h, n, traverser, spec.bb, ctx);
        const int seat = (h->rel + ctx.button) % n;
        const int b = tree_bucket(*bucketer, n, seat, h->n_board, ctx);
        const bool is_hero = seat == hero_seat;
        const NodeKey nk = node_key(h->street, h->rel, h->n_active, b, h->hh);
        bool cached;
        Node* node = tree_node(h, b, is_hero, nk, ctx, cached);
        if (!cached) {
            bool same = node->n == h->na;
            if (same) for (int i = 0; i < h->na; i++) if (node->acts[i] != h->ids[i]) { same = false; break; }
            if (!same) {  // imported with another action list: the engine traversal copes
                HandState st = tree_.replay(h, ctx.order.data(), ctx.button);
                return traverse(st, HistHash(), traverser, weight, hero_seat, ctx);
            }
            if (b < h->n_cache) h->cache[(is_hero ? 0 : 1) * h->n_cache + b].store(node, std::memory_order_release);
        }
        ctx.nodes_touched++;
        const int na = h->na;
        double sigma[MAX_ACTIONS];
        node->lock.lock();
        node->current_strategy(sigma);
        node->lock.unlock();

        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            for (int i = 0; i < na; i++) utils[i] = traverse_tree(tree_.child(h, i), traverser, weight, hero_seat, ctx);
            double prods[MAX_ACTIONS];
            for (int i = 0; i < na; i++) prods[i] = sigma[i] * utils[i];
            double u = py_sum(prods, na);
            node->lock.lock();
            for (int i = 0; i < na; i++) node->regret()[i] += weight * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        double probs[MAX_ACTIONS];
        const double* use = sigma;
        if (!is_hero && ctx.rng.random() < p_model) {
            ActionList actions;
            tree_actions(h, actions);
            if (model_policy(nk, actions, probs, verify_keys ? &ctx.key : nullptr)) use = probs;
        } else {
            node->lock.lock();
            for (int i = 0; i < na; i++) node->strategy_sum()[i] += weight * sigma[i];
            node->lock.unlock();
        }
        int a = sample(use, na, ctx.rng);
        return traverse_tree(tree_.child(h, a), traverser, weight, hero_seat, ctx);
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

    // RNRTrainer._node_in: get/create with the warm start prior (applied before the node is published)
    FlatNodeTable::Found node_in(FlatNodeTable& table, const NodeKey& nk, const ActionList& actions, const HandState& st,
                                 const Obs& obs, int bucket, ThreadCtx& ctx) {
        uint8_t ids[MAX_ACTIONS];
        for (int i = 0; i < actions.n; i++) ids[i] = (uint8_t)actions.a[i].id;
        return table.get_or_create(nk, ctx.tid, actions.n, [&](Node& node, NodeArena& arena) {
            node.init(ids, actions.n);
            infoset_key_for_bucket(st, obs, bucket, grid, ctx.key, ctx.hist);
            if (!warm_start.empty() && warm_visits > 0) {
                auto it = warm_start.find(ctx.key);
                if (it != warm_start.end()) {
                    double base[MAX_ACTIONS];
                    if (blueprint_policy(it->second, actions, grid, base)) {
                        double mean_w = linear ? std::max(1.0, (double)planned_iters / 2.0) : 1.0;
                        double w = warm_visits * mean_w;
                        for (int i = 0; i < actions.n; i++) {
                            node.regret()[i] = w * regret_scale_bb * base[i];
                            node.strategy_sum()[i] = w * base[i];
                        }
                    }
                }
            }
            return arena.copy_key(ctx.key.data(), ctx.key.size());
        });
    }

    // model.policy(key, actions) from the materialised table; false == None.  `key_string` (test
    // mode) is today's key string of this node: the numeric lookup must find the same entry.
    bool model_policy(const NodeKey& nk, const ActionList& actions, double* out, const std::string* key_string) {
        auto it = model_idx_.find(nk);
        const StratEntry* found = it == model_idx_.end() ? nullptr : &it->second->second;
        if (key_string) {
            auto sit = model.find(*key_string);
            if ((sit == model.end() ? nullptr : &sit->second) != found)
                checker_.report("the opponent model entry of '" + *key_string + "' differs from its numeric lookup");
        }
        if (!found) return false;
        const StratEntry& e = *found;
        bool same = (int)e.names.size() == actions.n;
        if (same) for (int i = 0; i < actions.n; i++) if (e.names[i] != grid.names[actions.a[i].id]) { same = false; break; }
        if (same) { for (int i = 0; i < actions.n; i++) out[i] = e.probs[i]; return true; }
        return blueprint_policy(e, actions, grid, out);
    }

    // `hh` hashes the history of the parent state; the events added since are hashed here
    double traverse(HandState& st, HistHash hh, int traverser, double weight, int hero_seat, ThreadCtx& ctx) {
        if (st.terminal) return (double)st.net(traverser) / (double)spec.bb;
        hh.catch_up(st, grid, ctx.tok);
        int seat = st.to_act;
        Obs obs = observe(st, seat);
        ActionList actions;
        grid.abstract_actions(obs, actions);
        const int b = memo_bucket(*bucketer, st, seat, ctx);
        const NodeKey nk = codec.key(obs, st.button, b, hh);
        bool is_hero = seat == hero_seat;
        FlatNodeTable::Found f = node_in(is_hero ? hero_nodes : opp_nodes, nk, actions, st, obs, b, ctx);
        if (verify_keys) checker_.check(f.key, st, obs, b, grid, ctx);  // leaves today's key string in ctx.key
        Node* node = f.node;
        ctx.nodes_touched++;
        int na = actions.n;
        bool same = node->n == na;
        if (same) for (int i = 0; i < na; i++) if (node->acts[i] != actions.a[i].id) { same = false; break; }
        if (!same) {
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
                utils[i] = traverse(child, hh, traverser, weight, hero_seat, ctx);
            }
            double prods[MAX_ACTIONS];
            for (int i = 0; i < na; i++) prods[i] = sigma[i] * utils[i];
            double u = py_sum(prods, na);
            node->lock.lock();
            for (int i = 0; i < na; i++) node->regret()[i] += weight * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        // sampled node
        double probs[MAX_ACTIONS];
        const double* use = sigma;
        if (!is_hero && ctx.rng.random() < p_model) {
            if (model_policy(nk, actions, probs, verify_keys ? &ctx.key : nullptr)) use = probs;
        } else {
            node->lock.lock();
            for (int i = 0; i < na; i++) node->strategy_sum()[i] += weight * sigma[i];
            node->lock.unlock();
        }
        int a = sample(use, na, ctx.rng);
        int type, amount;
        grid.to_concrete(obs, actions.a[a], type, amount);
        st.apply(type, amount);
        return traverse(st, hh, traverser, weight, hero_seat, ctx);
    }
};

}  // namespace negp
