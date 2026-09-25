// The GPU algorithm on the CPU (GPU_DESIGN.md, step 2): batched synchronous external-sampling MCCFR on
// the flat game (flatgame.h), traversed level by level instead of depth first.
//
// The same algorithm as Trainer's batched mode (mccfr.h, batch_size), and the same numbers bit for
// bit (tests/test_flatcfr.py):
//   * batches aligned on absolute iterations; within a batch every iteration reads the tables as they
//     were at the start of the batch, and the updates are applied after it;
//   * iteration t: the Philox deal of (seed, t), button t % n, Linear CFR weight t (linear_until);
//     traversers 0 .. n-1; an opponent node samples its action with the Philox draw addressed by
//     (seed, t, traverser, history hash) -- the same draw whatever the traversal order;
//   * the updates of a cell are summed in the order (iteration, traverser).  A (history, bucket) cell
//     is reached at most once per (iteration, traverser) (a history is one path of the tree), so this
//     is the reference's order (iteration, sequence) and every sum is the same double.
//
// A pass handles `pass_iterations` iterations x n traversers ("jobs") at once:
//   forward, level L -> L+1:  every item (node, job) of level L computes its current strategy from the
//                             snapshot; a traverser node emits all its children, an opponent node
//                             records its strategy-sum and visit updates and emits the sampled child;
//                             a terminal gets its value.  Children are appended in item order (on the
//                             GPU: a prefix sum over the child counts).
//   backward, level L:        a traverser node's value is the py_sum of sigma x the child values (in
//                             action order) and records its regret updates; an opponent node takes its
//                             child's value.
// The pass size changes the memory, never the result.  Threads run passes in parallel; the updates
// are sorted by (kind, cell, iteration, traverser) and applied sequentially after the batch.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "evaluator.h"
#include "flatgame.h"
#include "histtree.h"
#include "mccfr.h"
#include "philox.h"

namespace negp {

class FlatTrainer {
public:
    Spec spec;
    BetGrid grid;
    std::shared_ptr<Bucketer> bucketer;
    uint64_t seed = 0;
    bool linear = true;
    long long linear_until = 0;
    long long batch_size = 1;
    int threads = 1;
    int pass_iterations = 64;
    FlatGame game;
    std::vector<double> regret, strategy_sum;  // [cell]
    std::vector<int64_t> visits;               // [infoset]
    std::vector<uint8_t> touched;              // [infoset]: an update was applied (the sparse table has the node)

    FlatTrainer(const Spec& spec_, std::shared_ptr<Bucketer> bk, uint64_t seed_, bool linear_, int threads_)
        : spec(spec_), grid(spec_.grid()), bucketer(std::move(bk)), seed(seed_), linear(linear_), threads(std::max(1, threads_)),
          tree_(spec.stacks(), spec.sb, spec.bb, spec.ante, spec.max_street, grid, std::max(1, bucketer->identity().n_buckets)) {
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        game.build(tree_, tree_.root(0), spec.n_players);
        regret.assign(game.n_cells, 0.0);
        strategy_sum.assign(game.n_cells, 0.0);
        visits.assign(game.n_infosets, 0);
        touched.assign(game.n_infosets, 0);
    }

    long long iteration() const { return iteration_; }

    void train(long long iterations) {
        if (batch_size < 1) throw std::invalid_argument("flat trainer: batch_size >= 1");
        const long long target = iteration_ + iterations;
        const int T = threads;
        std::vector<std::vector<Rec>> recs(T);
        std::vector<Rec> all;
        while (iteration_ < target) {
            const long long lo = iteration_ + 1;
            const long long hi = std::min(target, (iteration_ / batch_size + 1) * batch_size);
            const long long P = std::max(1, pass_iterations);
            std::atomic<long long> next{lo};
            auto work = [&](int tid) {
                Scratch sc;
                for (long long a = next.fetch_add(P); a <= hi; a = next.fetch_add(P)) run_pass(a, std::min(hi, a + P - 1), sc, recs[tid]);
            };
            if (T == 1) {
                work(0);
            } else {
                std::vector<std::thread> pool;
                for (int t = 1; t < T; t++) pool.emplace_back(work, t);
                work(0);
                for (auto& th : pool) th.join();
            }
            all.clear();
            for (auto& r : recs) { all.insert(all.end(), r.begin(), r.end()); r.clear(); }
            std::sort(all.begin(), all.end(), [](const Rec& x, const Rec& y) {
                if (x.kind != y.kind) return x.kind < y.kind;
                if (x.cell != y.cell) return x.cell < y.cell;
                if (x.t != y.t) return x.t < y.t;
                return x.p < y.p;
            });
            for (const Rec& u : all) {
                if (u.kind == 0) regret[u.cell] += u.v;
                else if (u.kind == 1) strategy_sum[u.cell] += u.v;
                else { visits[u.cell] += 1; touched[u.cell] = 1; }
                if (u.kind == 0) touched[info_of_cell(u)] = 1;
            }
            iteration_ = hi;
        }
    }

    // every touched infoset: (key, regret, strategy sum, visits)
    template <class F>
    void for_each_touched(F&& f) const {
        std::string key;
        for (size_t d = 0; d < game.n_decisions(); d++) {
            const int na = game.na[d];
            for (int b = 0; b < game.n_cache[d]; b++) {
                const uint64_t i = game.info_base[d] + (uint64_t)b;
                if (!touched[i]) continue;
                game.key(d, b, key);
                const uint64_t c = game.cell_base[d] + (uint64_t)b * (uint64_t)na;
                f(key, &regret[c], &strategy_sum[c], na, visits[i]);
            }
        }
    }

private:
    struct Rec {
        uint64_t cell;  // regret / strategy sum: cell; visits: infoset
        long long t;
        uint32_t p;     // traverser
        uint32_t kind;  // 0 regret, 1 strategy sum, 2 visits
        uint32_t info;  // regret: infoset - info_base of its decision (the bucket), to mark it touched
        uint32_t d;
        double v;
    };
    struct Iter {
        long long t;
        double weight;
        int button;
        int bucket[MAX_PLAYERS][6];  // [absolute seat][n_board]
        int64_t strength[MAX_PLAYERS];  // by relative seat
    };
    struct Item {
        int32_t node;        // >= 0 decision, < 0 ~terminal
        uint32_t job;        // iteration index in the pass * n + traverser
        uint32_t first = 0;  // children in the next level
        double value = 0.0;
    };
    struct Scratch {
        std::vector<Iter> iters;
        std::vector<std::vector<Item>> levels;
        int order[52];
    };

    HistTree tree_;
    long long iteration_ = 0;

    uint64_t info_of_cell(const Rec& u) const { return game.info_base[u.d] + u.info; }

    void run_pass(long long lo, long long hi, Scratch& sc, std::vector<Rec>& out) {
        const int n = spec.n_players;
        const int n_iter = (int)(hi - lo + 1);
        // per iteration: deal, buckets, showdown strengths (on the GPU: computed beside the traversal)
        sc.iters.resize((size_t)n_iter);
        for (int k = 0; k < n_iter; k++) {
            Iter& it = sc.iters[(size_t)k];
            it.t = lo + k;
            it.weight = linear ? (double)(linear_until > 0 && it.t > linear_until ? linear_until : it.t) : 1.0;
            it.button = (int)(it.t % n);
            philox_deal(seed, (uint64_t)it.t, sc.order);
            for (int s = 0; s < n; s++)
                for (int nb = 0; nb < 6; nb++) it.bucket[s][nb] = -1;
            for (int s = 0; s < n; s++)
                for (int nb : {0, 3, 4, 5})
                    if (nb == 0 || nb <= board_cards_by_street(spec.max_street)) it.bucket[s][nb] = bucketer->bucket(&sc.order[2 * s], &sc.order[2 * n], nb);
            for (int r = 0; r < n; r++) {
                const int seat = (r + it.button) % n;
                int cards[7] = {sc.order[2 * seat], sc.order[2 * seat + 1]};
                for (int i = 0; i < 5; i++) cards[2 + i] = sc.order[2 * n + i];
                it.strength[r] = evaluate(cards, 7);
            }
        }
        // forward
        if (sc.levels.size() < (size_t)game.depth + 1) sc.levels.resize((size_t)game.depth + 1);
        for (auto& l : sc.levels) l.clear();
        for (uint32_t j = 0; j < (uint32_t)(n_iter * n); j++) sc.levels[0].push_back({0, j, 0, 0.0});
        int L = 0;
        double sigma[MAX_ACTIONS];
        for (; L < (int)sc.levels.size() && !sc.levels[(size_t)L].empty(); L++) {
            std::vector<Item>& cur = sc.levels[(size_t)L];
            std::vector<Item>* nxt = L + 1 < (int)sc.levels.size() ? &sc.levels[(size_t)L + 1] : nullptr;
            for (Item& item : cur) {
                const Iter& it = sc.iters[item.job / (uint32_t)n];
                const int p = (int)(item.job % (uint32_t)n);
                if (item.node < 0) {
                    const int tm = ~item.node;
                    const int me = ((p - it.button) % n + n) % n;
                    item.value = (double)terminal_net_of(&game.invested[(size_t)tm * n], game.folded[(size_t)tm], n, me, it.strength) / (double)spec.bb;
                    continue;
                }
                const size_t d = (size_t)item.node;
                const int na = game.na[d];
                const int seat = (game.rel[d] + it.button) % n;
                const int b = row_of(d, it.bucket[seat][game.n_board[d]]);
                const uint64_t c = game.cell_base[d] + (uint64_t)b * (uint64_t)na;
                regret_matching(&regret[c], na, sigma);
                const int32_t* ch = &game.child[game.child_base[d]];
                item.first = (uint32_t)nxt->size();
                if (seat == p) {
                    for (int a = 0; a < na; a++) nxt->push_back({ch[a], item.job, 0, 0.0});
                    continue;
                }
                for (int a = 0; a < na; a++) out.push_back({c + (uint64_t)a, it.t, (uint32_t)p, 1, 0, 0, it.weight * sigma[a]});
                out.push_back({game.info_base[d] + (uint64_t)b, it.t, (uint32_t)p, 2, 0, 0, 0.0});
                const double r = philox_sample_u01(seed, (uint64_t)it.t, p, game.hh_a[d], game.hh_b[d]);
                int a = na - 1;
                double acc = 0.0;
                for (int i = 0; i < na; i++) {
                    acc += sigma[i];
                    if (r < acc) { a = i; break; }
                }
                nxt->push_back({ch[a], item.job, 0, 0.0});
            }
        }
        // backward
        for (int l = L - 1; l >= 0; l--) {
            std::vector<Item>& cur = sc.levels[(size_t)l];
            for (Item& item : cur) {
                if (item.node < 0) continue;
                const std::vector<Item>& nxt = sc.levels[(size_t)l + 1];
                const Iter& it = sc.iters[item.job / (uint32_t)n];
                const int p = (int)(item.job % (uint32_t)n);
                const size_t d = (size_t)item.node;
                const int na = game.na[d];
                const int seat = (game.rel[d] + it.button) % n;
                if (seat != p) { item.value = nxt[item.first].value; continue; }
                const int b = row_of(d, it.bucket[seat][game.n_board[d]]);
                const uint64_t c = game.cell_base[d] + (uint64_t)b * (uint64_t)na;
                regret_matching(&regret[c], na, sigma);
                double utils[MAX_ACTIONS], prods[MAX_ACTIONS];
                for (int a = 0; a < na; a++) {
                    utils[a] = nxt[item.first + (uint32_t)a].value;
                    prods[a] = sigma[a] * utils[a];
                }
                const double u = py_sum(prods, na);
                for (int a = 0; a < na; a++) out.push_back({c + (uint64_t)a, it.t, (uint32_t)p, 0, (uint32_t)b, (uint32_t)d, it.weight * (utils[a] - u)});
                item.value = u;
            }
        }
    }

    int row_of(size_t d, int b) const {
        if (b < 0 || b >= game.n_cache[d]) throw std::runtime_error("flat trainer: bucket outside the node's rows");
        return b;
    }
};

}  // namespace negp
