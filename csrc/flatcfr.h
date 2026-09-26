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
#include <chrono>
#include <exception>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "evaluator.h"
#include "flatgame.h"
#include "gpucfr.h"
#include "gpukernels.h"
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

    int gpu_pass = 0;  // GPU mode: iterations traversed at once (0: the whole batch); memory only
    // emulation of the GPU trainer on the host: the kernels' functions (gpukernels.h) in loops, the same
    // steps as GpuFlatTrainer::run_batch (a check of everything but the CUDA calls, on any machine)
    bool emulate_gpu = false;
    // GPU / emulation mode, wall time since construction: the host's deals, buckets and strengths, and run_batch
    // (the preparation of batch i + 1 overlaps batch i; ms_wait_prepare: the device side waited for it)
    double ms_prepare_total = 0.0, ms_device_total = 0.0, ms_wait_prepare = 0.0;

    // move the tables to CUDA device `device` and train there from now on (a CUDA build is needed)
    void use_gpu(int device) {
        std::string why;
        if (!GpuFlatTrainer::available(why)) throw std::runtime_error("no usable CUDA device: " + why);
        FlatGameView v;
        v.n_players = game.n_players;
        v.n_decisions = game.n_decisions();
        v.n_terminals = game.n_terminals();
        v.n_child = game.child.size();
        v.rel = game.rel.data();
        v.n_board = game.n_board.data();
        v.na = game.na.data();
        v.child_base = game.child_base.data();
        v.child = game.child.data();
        v.hh_a = game.hh_a.data();
        v.hh_b = game.hh_b.data();
        v.info_base = game.info_base.data();
        v.cell_base = game.cell_base.data();
        v.n_cache = game.n_cache.data();
        v.invested = game.invested.data();
        v.folded = game.folded.data();
        v.n_infosets = game.n_infosets;
        v.n_cells = game.n_cells;
        sync();
        gpu_.reset(new GpuFlatTrainer(v, spec.bb, seed, device));
        std::vector<uint8_t> tc(game.n_cells, 0);  // touched per cell: every cell of a touched infoset
        for (size_t d = 0; d < game.n_decisions(); d++)
            for (int b = 0; b < game.n_cache[d]; b++)
                if (touched[game.info_base[d] + (uint64_t)b])
                    for (int a = 0; a < game.na[d]; a++) tc[game.cell_base[d] + (uint64_t)b * game.na[d] + (uint64_t)a] = 1;
        gpu_->upload(regret.data(), strategy_sum.data(), visits.data(), tc.data());
    }
    bool on_gpu() const { return (bool)gpu_; }
    std::string gpu_device() const { return gpu_ ? gpu_->device_name() : std::string(); }
    GpuStats gpu_stats() const { return gpu_ ? gpu_->stats() : GpuStats(); }

    // GPU mode: bring the host copies of the tables up to date
    void sync() {
        if (!host_stale_) return;
        std::vector<uint8_t> dl;
        if (gpu_) {
            dl.resize(game.n_cells);
            gpu_->download(regret.data(), strategy_sum.data(), visits.data(), dl.data());
        }
        const std::vector<uint8_t>& tc = gpu_ ? dl : emu_touched_;
        for (size_t d = 0; d < game.n_decisions(); d++)
            for (int b = 0; b < game.n_cache[d]; b++) {
                const uint64_t i = game.info_base[d] + (uint64_t)b;
                bool t = visits[i] > 0;
                for (int a = 0; a < game.na[d] && !t; a++) t = tc[game.cell_base[d] + (uint64_t)b * game.na[d] + (uint64_t)a] != 0;
                touched[i] = t;
            }
        host_stale_ = false;
    }

    void train(long long iterations) {
        if (batch_size < 1) throw std::invalid_argument("flat trainer: batch_size >= 1");
        const long long target = iteration_ + iterations;
        if (gpu_ || emulate_gpu) { train_gpu(target); return; }
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

    // every touched infoset: (key, action ids, regret, strategy sum, number of actions, visits)
    template <class F>
    void for_each_touched(F&& f) {
        sync();
        std::string key;
        for (size_t d = 0; d < game.n_decisions(); d++) {
            const int na = game.na[d];
            for (int b = 0; b < game.n_cache[d]; b++) {
                const uint64_t i = game.info_base[d] + (uint64_t)b;
                if (!touched[i]) continue;
                game.key(d, b, key);
                const uint64_t c = game.cell_base[d] + (uint64_t)b * (uint64_t)na;
                f(key, &game.ids[d * MAX_ACTIONS], &regret[c], &strategy_sum[c], na, visits[i]);
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
    using Iter = FlatIter;
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

    // what iteration t needs from its deal: weight, button, buckets, showdown strengths
    void prepare_iter(long long t, Iter& it, int* order) {
        const int n = spec.n_players;
        it.t = t;
        it.weight = linear ? (double)(linear_until > 0 && t > linear_until ? linear_until : t) : 1.0;
        it.button = (int)(t % n);
        philox_deal(seed, (uint64_t)t, order);
        for (int s = 0; s < CFR_MAX_PLAYERS; s++)
            for (int nb = 0; nb < 6; nb++) it.bucket[s][nb] = -1;
        for (int s = 0; s < n; s++)
            for (int nb : {0, 3, 4, 5})
                if (nb == 0 || nb <= board_cards_by_street(spec.max_street)) it.bucket[s][nb] = bucketer->bucket(&order[2 * s], &order[2 * n], nb);
        for (int r = 0; r < CFR_MAX_PLAYERS; r++) it.strength[r] = 0;
        for (int r = 0; r < n; r++) {
            const int seat = (r + it.button) % n;
            int cards[7] = {order[2 * seat], order[2 * seat + 1]};
            for (int i = 0; i < 5; i++) cards[2 + i] = order[2 * n + i];
            it.strength[r] = evaluate(cards, 7);
        }
    }

    // ---- GPU mode (gpucfr.h): the tables live on the device; the host copies are refreshed by sync()
    std::unique_ptr<GpuFlatTrainer> gpu_;
    bool host_stale_ = false;
    std::vector<Iter> gpu_iters_;

    std::vector<uint8_t> emu_touched_;  // emulation: touched per cell (the device's layout)
    std::vector<int32_t> emu_info_dec_;  // emulation: infoset -> decision
    std::vector<double> emu_sigma_;      // emulation: the strategy table, kept between batches
    std::vector<uint8_t> emu_dirty_;     // emulation: per cell, its row's strategy must be recomputed

    void emu_run_batch(const Iter* its, int k, int pass) {
        const int n = spec.n_players;
        const KeyLayout kl = key_layout(std::max(game.n_cells, game.n_infosets));
        if (kl.cb == 0) throw std::invalid_argument("table too large for 32-bit update keys");
        if (emu_info_dec_.size() != game.n_infosets) {
            emu_info_dec_.resize(game.n_infosets);
            for (size_t d = 0; d < game.n_decisions(); d++)
                for (int b = 0; b < game.n_cache[d]; b++) emu_info_dec_[game.info_base[d] + (uint64_t)b] = (int32_t)d;
        }
        if (emu_touched_.size() != game.n_cells) {  // first batch: every cell of a touched infoset
            emu_touched_.assign(game.n_cells, 0);
            for (size_t d = 0; d < game.n_decisions(); d++)
                for (int b = 0; b < game.n_cache[d]; b++)
                    if (touched[game.info_base[d] + (uint64_t)b])
                        for (int a = 0; a < game.na[d]; a++) emu_touched_[game.cell_base[d] + (uint64_t)b * game.na[d] + (uint64_t)a] = 1;
        }
        if (pass < 1) pass = k;
        DevGame g{game.rel.data(), game.n_board.data(), game.na.data(), game.child_base.data(), game.child.data(), game.hh_a.data(),
                  game.hh_b.data(), game.info_base.data(), game.cell_base.data(), game.n_cache.data(), game.invested.data(),
                  game.folded.data(), emu_info_dec_.data(), n};
        if (emu_sigma_.size() != game.n_cells) {  // first batch: every row computed
            emu_sigma_.assign(game.n_cells, 0.0);
            emu_dirty_.assign(game.n_cells, 1);
        }
        std::vector<double>& sigma_tab = emu_sigma_;  // the strategies of the batch (rows changed since: recomputed)
        for (size_t i = 0; i < game.n_infosets; i++) row_sigma(g, regret.data(), sigma_tab.data(), emu_dirty_.data(), i);
        std::vector<int32_t> node;
        std::vector<uint32_t> job, first, cnt;
        std::vector<uint8_t> choice;
        std::vector<double> value;
        std::vector<uint32_t> rcnt;
        std::vector<uint64_t> roff;
        std::vector<uint32_t> keys;
        std::vector<double> vals;
        std::vector<uint32_t> rec_job;  // emulation only: the job of every record, to check the order claim
        int err = 0;
        auto grow = [&](size_t m) {
            if (node.size() >= m) return;
            node.resize(m); job.resize(m); first.resize(m); cnt.resize(m); choice.resize(m); value.resize(m); rcnt.resize(m); roff.resize(m);
        };
        auto lv = [&]() { return DevLevel{node.data(), job.data(), first.data(), cnt.data(), choice.data(), value.data(), rcnt.data(), roff.data()}; };
        for (int lo = 0; lo < k; lo += pass) {
            const uint32_t jobs = (uint32_t)std::min(pass, k - lo) * (uint32_t)n;
            std::vector<size_t> off{0}, size{jobs};
            grow(jobs);
            for (uint32_t i = 0; i < jobs; i++) item_init(lv(), i, (uint32_t)lo * (uint32_t)n);
            uint64_t n_rec = 0;
            for (size_t L = 0; size[L] > 0; L++) {
                const size_t o = off[L], m = size[L];
                for (size_t i = 0; i < m; i++) n_rec += item_count(g, lv(), o + i, its, sigma_tab.data(), seed, spec.bb, &err);
                uint32_t acc = 0;  // exclusive scan
                for (size_t i = 0; i < m; i++) { first[o + i] = acc; acc += cnt[o + i]; }
                const size_t next_off = o + m;
                grow(next_off + acc);
                for (size_t i = 0; i < m; i++) item_emit(g, lv(), o + i, next_off);
                off.push_back(next_off);
                size.push_back(acc);
            }
            // record slots: exclusive scan of the record counts of all items of the pass (as on the device);
            // the backward pass then runs its items in reverse order: the slots, not the order, place the records
            const size_t items = off.back();
            uint64_t acc = 0;
            for (size_t i = 0; i < items; i++) { roff[i] = acc; acc += rcnt[i]; }
            if (acc != n_rec) throw std::logic_error("emulation: record count mismatch");
            const uint64_t base = keys.size();
            keys.resize(base + n_rec);
            vals.resize(base + n_rec);
            rec_job.resize(base + n_rec);
            uint64_t at = base;
            for (size_t L = size.size() - 1; L-- > 0;)
                for (size_t i = size[L]; i-- > 0;) {
                    const size_t x = off[L] + i;
                    if (!rcnt[x]) continue;
                    item_back(g, lv(), x, off[L + 1], its, sigma_tab.data(), kl, keys.data(), vals.data(), base + roff[x], &err);
                    for (uint32_t r = 0; r < rcnt[x]; r++) rec_job[base + roff[x] + r] = job[x];
                    at += rcnt[x];
                }
            if (at != base + n_rec) throw std::logic_error("emulation: record count mismatch");
        }
        if (err) throw std::runtime_error("GPU emulation: a bucket outside its node's rows");
        // the stable radix sort by key, then the runs; checked here: within a key, the jobs increase
        std::vector<size_t> perm(keys.size());
        for (size_t i = 0; i < perm.size(); i++) perm[i] = i;
        const uint32_t used = kl.bits() >= 32 ? ~0u : (1u << kl.bits()) - 1u;
        for (uint32_t kk : keys)
            if (kk & ~used) throw std::logic_error("emulation: a key outside its layout");
        std::stable_sort(perm.begin(), perm.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });
        std::vector<uint32_t> sk(keys.size());
        std::vector<double> sv(keys.size());
        for (size_t i = 0; i < perm.size(); i++) { sk[i] = keys[perm[i]]; sv[i] = vals[perm[i]]; }
        for (size_t i = 1; i < sk.size(); i++)
            if (sk[i] == sk[i - 1] && rec_job[perm[i]] <= rec_job[perm[i - 1]]) throw std::logic_error("emulation: records of a cell out of job order");
        for (size_t i = 0; i < sk.size(); i++) apply_run(sk.data(), sv.data(), sk.size(), i, kl, regret.data(), strategy_sum.data(), visits.data(), emu_touched_.data(), emu_dirty_.data());
    }

    // the FlatIter of iterations lo .. lo + k - 1, on `threads` threads
    void prepare_batch(long long lo, int k, std::vector<Iter>& out) {
        out.resize((size_t)k);
        std::atomic<int> next{0};
        auto work = [&]() {
            int order[52];
            for (int i = next.fetch_add(64); i < k; i = next.fetch_add(64))
                for (int j = i; j < std::min(k, i + 64); j++) prepare_iter(lo + j, out[(size_t)j], order);
        };
        if (threads == 1) {
            work();
        } else {
            std::vector<std::thread> pool;
            for (int t = 1; t < threads; t++) pool.emplace_back(work);
            work();
            for (auto& th : pool) th.join();
        }
    }

    // the host prepares batch i + 1 while the device runs batch i (the preparation is a pure function of
    // the iteration numbers, so the overlap changes the time, never the result)
    void train_gpu(long long target) {
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        auto batch_end = [&](long long done) { return std::min(target, (done / batch_size + 1) * batch_size); };
        if (iteration_ >= target) return;
        std::vector<Iter> buf[2];
        long long lo = iteration_ + 1, hi = batch_end(iteration_);
        const auto p0 = clk::now();
        prepare_batch(lo, (int)(hi - lo + 1), buf[0]);
        ms_prepare_total += ms(p0, clk::now());
        for (int w = 0;; w ^= 1) {
            const bool more = hi < target;
            const long long nlo = hi + 1, nhi = more ? batch_end(hi) : hi;
            std::exception_ptr prep_error;
            double prep_ms = 0.0;
            std::thread prep;
            if (more) prep = std::thread([&, w]() {
                try {
                    const auto a = clk::now();
                    prepare_batch(nlo, (int)(nhi - nlo + 1), buf[w ^ 1]);
                    prep_ms = ms(a, clk::now());
                } catch (...) { prep_error = std::current_exception(); }
            });
            const auto d0 = clk::now();
            try {
                if (gpu_) gpu_->run_batch(buf[w].data(), (int)(hi - lo + 1), gpu_pass);
                else emu_run_batch(buf[w].data(), (int)(hi - lo + 1), gpu_pass);
            } catch (...) {
                if (prep.joinable()) prep.join();
                throw;
            }
            ms_device_total += ms(d0, clk::now());
            const auto j0 = clk::now();
            if (prep.joinable()) prep.join();
            ms_wait_prepare += ms(j0, clk::now());
            if (prep_error) std::rethrow_exception(prep_error);
            ms_prepare_total += prep_ms;
            host_stale_ = true;
            iteration_ = hi;
            if (!more) break;
            lo = nlo;
            hi = nhi;
        }
    }

    void run_pass(long long lo, long long hi, Scratch& sc, std::vector<Rec>& out) {
        const int n = spec.n_players;
        const int n_iter = (int)(hi - lo + 1);
        // per iteration: deal, buckets, showdown strengths (on the GPU: computed beside the traversal)
        sc.iters.resize((size_t)n_iter);
        for (int k = 0; k < n_iter; k++) prepare_iter(lo + k, sc.iters[(size_t)k], sc.order);
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
