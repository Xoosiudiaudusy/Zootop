// Real-time subgame search, part 1 (2026-09-25): the layout of Pluribus as verified against its
// supplement (docs/search_design.md, docs/search_research.md sections 1 and 7).
//
//   root     the public state at the START of the current betting round, rebuilt from the hand's
//            real actions: exact pot and stacks whatever sizes were used before
//   ranges   every live player's weights over all 1326 hole combos: Bayes over that player's
//            actions before the root with sigma = the blueprint (C++ lookup), board cards removed;
//            for any (street, seat) the caller may pass per-combo likelihoods instead (a previous
//            search's average strategy: SubgameSearch::likelihood)
//   actions  at every node the blueprint grid (+ all-in); each off-grid action actually taken in
//            this round is INSERTED as one more valid action at the node where it was taken
//   fixed    our own actions already taken in this round are forced for our actual hole only;
//            our other holes and every opponent stay free
//   cards    lossless on the current round (infoset = canonical form of hole + board), the
//            blueprint's buckets on later rounds
//   solver   Linear external-sampling MCCFR on N threads, to the end of the hand (depth limits and
//            continuation leaves are part 2); returns the final-iteration strategy of our actual
//            hole at our current decision, its average strategy, and timing; the node table stays
//            for range updates of later rounds
//
// Sampling.  A deal draws every live player's hole independently from its range and rejects the
// draw on any shared card: exactly the joint distribution of the subgame, prod r_i(h_i) over
// disjoint holes.  In a share `focus` of OUR traversals our hole is our actual hole (the opponents
// drawn conditionally, by the same rejection) and the opponents' actions on the real path of this
// round are taken as they happened, the traversal weighted by the product of their current
// probabilities of those actions: an unbiased estimate of the regrets of our actual hole at our
// current decision, which is then updated in every such traversal instead of only when the
// opponents' sampled actions happen to follow the real path.  Average strategies are accumulated
// only in the other, natural traversals.
#pragma once
#include <algorithm>
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
#include "engine.h"
#include "mccfr.h"
#include "nodetable.h"
#include "persist.h"

namespace negp {

constexpr int N_COMBOS = 1326;
constexpr uint8_t INSERTED_ID = 255;  // Node::acts id of an inserted (off-grid) action

// hole combos (a < b), in the order of Python's [(a, b) for a in range(52) for b in range(a + 1, 52)]
struct ComboTable {
    int c0[N_COMBOS], c1[N_COMBOS];
    int idx[52][52];
    ComboTable() {
        int k = 0;
        for (int a = 0; a < 52; a++)
            for (int b = 0; b < 52; b++) idx[a][b] = -1;
        for (int a = 0; a < 52; a++)
            for (int b = a + 1; b < 52; b++) {
                c0[k] = a;
                c1[k] = b;
                idx[a][b] = idx[b][a] = k;
                k++;
            }
    }
};
inline const ComboTable& combo_table() {
    static const ComboTable t;
    return t;
}
inline int combo_index(int a, int b) { return (a >= 0 && a < 52 && b >= 0 && b < 52) ? combo_table().idx[a][b] : -1; }

// xoshiro256** (the search needs speed, not Python's streams)
struct FastRng {
    uint64_t s[4];
    explicit FastRng(uint64_t seed = 0) { reseed(seed); }
    void reseed(uint64_t seed) {
        uint64_t z = seed;
        for (int i = 0; i < 4; i++) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            s[i] = x ^ (x >> 31);
        }
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    double uniform() { return (double)(next() >> 11) * 0x1.0p-53; }
    int below(int n) { return (int)(((next() >> 32) * (uint64_t)n) >> 32); }
};

// hash of the public action sequence since the root (action indices); with the root fixed it
// determines the public state, so it identifies the node
struct PathHash {
    uint64_t a = 0x6C62272E07BB0142ULL;
    uint64_t b = 0x62B821756295C58DULL;
    void step(int idx) {
        a = fmix64(a ^ ((uint64_t)idx + 0x9E3779B97F4A7C15ULL));
        b = fmix64(b + (uint64_t)idx * 0xC2B2AE3D27D4EB4FULL + 0x165667B19E3779F9ULL);
    }
};

inline NodeKey search_key(int street, int seat, int cards, const PathHash& h) {
    const uint64_t p = (uint64_t)(street & 0xFF) | ((uint64_t)(seat & 0xFF) << 8) | ((uint64_t)(uint32_t)cards << 16);
    NodeKey k;
    k.k1 = fmix64(h.a ^ fmix64(p ^ 0xA0761D6478BD642FULL));
    k.k2 = fmix64(h.b ^ fmix64(p + 0xE7037ED1A0B428DBULL));
    return k;
}

// the actions of a subgame node: the grid's, then (on the real path) the inserted one
struct NodeActions {
    int n = 0;
    int type[MAX_ACTIONS];
    int amount[MAX_ACTIONS];
    uint8_t id[MAX_ACTIONS];
};

// ------------------------------------------------------------------ inputs and outputs
struct SearchGame {
    Spec spec;
    BetGrid grid;
    std::shared_ptr<Bucketer> bucketer;
    std::shared_ptr<const BlueprintTable> blueprint;  // null: no prior (uniform ranges)

    SearchGame(const Spec& s, std::shared_ptr<Bucketer> bk, std::shared_ptr<const BlueprintTable> bp)
        : spec(s), grid(s.grid()), bucketer(std::move(bk)), blueprint(std::move(bp)) {
        if (!bucketer) throw std::invalid_argument("a bucketer is needed");
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        if (blueprint && blueprint->codec.n_players() != spec.n_players)
            throw std::invalid_argument("the blueprint's numeric keys are for " + std::to_string(blueprint->codec.n_players()) +
                                        " players, the game has " + std::to_string(spec.n_players));
    }
};

struct LikelihoodOverride {  // replaces sigma = blueprint for one seat's actions on one street
    int street = 0, seat = 0;
    std::vector<double> w;  // 1326 per-combo likelihoods of those actions
};

struct HandInput {
    std::vector<int> stacks;                  // every seat's stack before blinds and antes
    int button = 0;
    std::vector<std::pair<int, int>> actions;  // (type, raise-to amount) of every action so far
    std::vector<int> board;                   // the real board so far
    int our_seat = 0;
    int our_hole[2] = {-1, -1};
    std::vector<LikelihoodOverride> overrides;
};

struct SearchParams {
    long long iterations = 0;  // stop after this many iterations (0: no limit)
    double time_budget = 2.0;  // seconds (0: no limit); at least one limit is needed
    int threads = 15;
    uint64_t seed = 0;
    double focus = 0.5;        // share of our traversals on our actual hole along the real path
    double min_prob = 1e-3;    // floor of sigma(action) in the range update
    bool linear = true;        // Linear CFR weights (iteration t counts t times)
};

struct SearchResult {
    std::vector<int> types, amounts, ids;       // the actions at our current decision
    std::vector<double> final_strategy;         // our actual hole, after the last iteration
    std::vector<double> average_strategy;       // our actual hole, average over the iterations
    bool visited = false;                       // our current infoset was reached
    long long iterations = 0, traversals = 0, focused = 0, nodes_touched = 0, forced = 0, redeals = 0;
    size_t table_size = 0;
    double seconds = 0.0;
    int threads = 0;
};

struct PathStep {  // one real action of the current round
    int actor = 0;
    int index = 0;        // its index in the node's action list
    bool inserted = false;
    int type = 0, amount = 0;
};

// ------------------------------------------------------------------ the search
class SubgameSearch {
public:
    struct PreRootStep {  // one real action before the root, as the blueprint saw it
        int seat, street, rel, n_active, n_board, a_idx;
        HistHash hist;           // the blueprint history before the action
        std::vector<int> legal;  // the legal actions as indices into the blueprint's names (-1: unknown name)
    };

    SubgameSearch(std::shared_ptr<const SearchGame> game, const HandInput& hand, const SearchParams& params)
        : game_(std::move(game)), hand_(hand), params_(params) {
        const Spec& sp = game_->spec;
        const int n = (int)hand_.stacks.size();
        if (n != sp.n_players) throw std::invalid_argument("stacks: one per seat of the game (" + std::to_string(sp.n_players) + ")");
        if (hand_.our_seat < 0 || hand_.our_seat >= n) throw std::invalid_argument("our_seat out of range");
        our_combo_ = combo_index(hand_.our_hole[0], hand_.our_hole[1]);
        if (our_combo_ < 0 || hand_.our_hole[0] == hand_.our_hole[1]) throw std::invalid_argument("our hole: two different cards 0..51");
        if (params_.threads < 1) params_.threads = 1;
        uint64_t used = 0;
        for (int c : hand_.board) {
            if (c < 0 || c > 51 || (used >> c & 1)) throw std::invalid_argument("board: distinct cards 0..51");
            used |= 1ULL << c;
        }
        board_mask_ = used;
        if ((used >> hand_.our_hole[0] & 1) || (used >> hand_.our_hole[1] & 1)) throw std::invalid_argument("our hole shares a card with the board");
        build_deck();
        replay();
        compute_classes();
        compute_ranges();
    }
    // root_ / current_ point into deck_: the object must not move
    SubgameSearch(const SubgameSearch&) = delete;
    SubgameSearch& operator=(const SubgameSearch&) = delete;

    // ---------------------------------------------------------- what the search is built on
    const SearchGame& game() const { return *game_; }
    const HandState& root() const { return root_; }
    const HandState& current() const { return current_; }
    int root_street() const { return root_street_; }
    const std::vector<PathStep>& path() const { return path_; }
    const std::vector<PreRootStep>& pre_root() const { return pre_; }
    // [seat] 1326 weights (empty for a seat that folded before the root)
    const std::vector<std::vector<double>>& reach() const { return reach_; }
    int class_of(int combo) const { return class_of_[combo]; }
    int n_classes() const { return n_classes_; }
    double range_seconds() const { return range_seconds_; }
    NodeActions actions_at(const HandState& st, int k) const {
        NodeActions na;
        build_actions(st, observe(st, st.to_act), k < (int)path_.size() ? &path_[k] : nullptr, na);
        return na;
    }

    // ---------------------------------------------------------- solve
    SearchResult solve() {
        if (current_.terminal || current_.to_act != hand_.our_seat) throw std::runtime_error("it is not our turn to act");
        if (params_.iterations <= 0 && params_.time_budget <= 0) throw std::invalid_argument("set iterations or time_budget");
        const int T = params_.threads;
        table_.reset(new FlatNodeTable(T + 1, (size_t)1 << 16));
        group_.reset(new TableGroup());
        table_->attach(group_.get());
        group_->add(table_.get());
        std::atomic<long long> next_t{1};
        std::atomic<bool> stop{false};
        std::vector<Ctx> ctxs((size_t)T);
        const auto t0 = std::chrono::steady_clock::now();
        const auto deadline = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double>(params_.time_budget > 0 ? params_.time_budget : 0.0));
        std::vector<int> traversers;
        for (int s = 0; s < root_.n; s++) if (root_.players[s].can_act()) traversers.push_back(s);
        group_->begin(T);
        std::mutex err_mu;
        std::string err;
        auto work = [&](int tid) {
            Ctx& ctx = ctxs[(size_t)tid];
            ctx.tid = tid;
            ctx.rng.reseed(params_.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)tid * 0xD1B54A32D192ED03ULL + 1);
            std::memcpy(ctx.deck, deck_, sizeof deck_);
            try {
                while (!stop.load(std::memory_order_relaxed)) {
                    const long long t = next_t.fetch_add(1);
                    if (params_.iterations > 0 && t > params_.iterations) break;
                    const double weight = params_.linear ? (double)t : 1.0;
                    for (int trav : traversers) {
                        const bool focused = trav == hand_.our_seat && ctx.rng.uniform() < params_.focus;
                        deal(ctx, focused);
                        HandState st(root_);
                        st.deck = ctx.deck;
                        for (int s = 0; s < st.n; s++) {
                            st.players[s].hole[0] = ctx.holes[s][0];
                            st.players[s].hole[1] = ctx.holes[s][1];
                        }
                        traverse(st, PathHash(), 0, true, trav, weight, 1.0, focused, ctx);
                        ctx.traversals++;
                        if (focused) ctx.focused++;
                    }
                    ctx.iterations++;
                    if (params_.time_budget > 0 && std::chrono::steady_clock::now() >= deadline) stop.store(true, std::memory_order_relaxed);
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = e.what();
                stop.store(true);
            }
            group_->leave();
        };
        if (T == 1) {
            work(0);
        } else {
            std::vector<std::thread> pool;
            for (int t = 1; t < T; t++) pool.emplace_back(work, t);
            work(0);
            for (auto& th : pool) th.join();
        }
        group_->end();
        if (!err.empty()) throw std::runtime_error("search failed: " + err);
        SearchResult r;
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        r.threads = T;
        for (const Ctx& c : ctxs) {
            r.iterations += c.iterations;
            r.traversals += c.traversals;
            r.focused += c.focused;
            r.nodes_touched += c.nodes;
            r.forced += c.forced;
            r.redeals += c.redeals;
        }
        r.table_size = table_->size();
        NodeActions na;
        build_actions(current_, observe(current_, hand_.our_seat), nullptr, na);
        for (int i = 0; i < na.n; i++) {
            r.types.push_back(na.type[i]);
            r.amounts.push_back(na.amount[i]);
            r.ids.push_back(na.id[i]);
        }
        PathHash ph;
        for (const PathStep& s : path_) ph.step(s.index);
        const FlatNodeTable::Found f = table_->find(search_key(root_street_, hand_.our_seat, class_of_[our_combo_], ph));
        r.final_strategy.assign((size_t)na.n, 1.0 / na.n);
        r.average_strategy.assign((size_t)na.n, 1.0 / na.n);
        if (f.node && f.node->n == na.n) {
            r.visited = true;
            double buf[MAX_ACTIONS];
            f.node->current_strategy(buf);
            r.final_strategy.assign(buf, buf + na.n);
            f.node->average_strategy(buf);
            r.average_strategy.assign(buf, buf + na.n);
        }
        return r;
    }

    // Per combo of `seat`: the probability, under this search's average strategy, of `seat`'s
    // actions among `actions` (the real actions of the round from its start: the solved path,
    // possibly continued).  Combos never reached, or actions the subgame did not contain, count 1
    // (no information); `missing` counts those (step, combo) lookups.
    std::vector<double> likelihood(int seat, const std::vector<std::pair<int, int>>& actions, long long& missing) const {
        if (!table_) throw std::runtime_error("solve first");
        std::vector<double> w((size_t)N_COMBOS, 0.0);
        for (int c = 0; c < N_COMBOS; c++) if (class_of_[c] >= 0) w[(size_t)c] = 1.0;
        missing = 0;
        HandState st(root_);
        st.deck = deck_;
        PathHash ph;
        for (size_t k = 0; k < actions.size(); k++) {
            if (st.terminal || st.street != root_street_) break;
            const int actor = st.to_act;
            NodeActions na;
            build_actions(st, observe(st, actor), k < path_.size() ? &path_[k] : nullptr, na);
            int a = -1;
            const int type = actions[k].first, amount = actions[k].first == RAISE ? actions[k].second : 0;
            for (int i = 0; i < na.n; i++) if (na.type[i] == type && na.amount[i] == amount) { a = i; break; }
            if (actor == seat) {
                for (int c = 0; c < N_COMBOS; c++) {
                    if (class_of_[c] < 0) continue;
                    const FlatNodeTable::Found f = a < 0 ? FlatNodeTable::Found() : table_->find(search_key(root_street_, seat, class_of_[c], ph));
                    if (!f.node || f.node->n != na.n) { missing++; continue; }
                    double avg[MAX_ACTIONS];
                    f.node->average_strategy(avg);
                    w[(size_t)c] *= avg[a];
                }
            }
            if (a < 0) break;  // not in this subgame: nothing further is known about the path
            st.apply(na.type[a], na.amount[a]);
            ph.step(a);
        }
        return w;
    }

    // tests: the strategy the solver uses at real-path node k for `seat` holding `hole` (after solve)
    std::vector<double> probe_path(int k, int h0, int h1) const {
        if (!table_) throw std::runtime_error("solve first");
        if (k < 0 || k > (int)path_.size()) throw std::invalid_argument("k: 0..len(path)");
        HandState st(root_);
        st.deck = deck_;
        PathHash ph;
        for (int j = 0; j < k; j++) {
            NodeActions na;
            build_actions(st, observe(st, st.to_act), &path_[(size_t)j], na);
            st.apply(na.type[path_[(size_t)j].index], na.amount[path_[(size_t)j].index]);
            ph.step(path_[(size_t)j].index);
        }
        const int seat = st.to_act;
        NodeActions na;
        build_actions(st, observe(st, seat), k < (int)path_.size() ? &path_[(size_t)k] : nullptr, na);
        std::vector<double> out((size_t)na.n, 0.0);
        const int c = combo_index(h0, h1);
        if (k < (int)path_.size() && seat == hand_.our_seat && c == our_combo_) {
            out[(size_t)path_[(size_t)k].index] = 1.0;  // fixed
            return out;
        }
        const FlatNodeTable::Found f = c < 0 || class_of_[c] < 0 ? FlatNodeTable::Found()
                                                                 : table_->find(search_key(root_street_, seat, class_of_[c], ph));
        if (!f.node || f.node->n != na.n) return std::vector<double>();
        double buf[MAX_ACTIONS];
        f.node->current_strategy(buf);
        out.assign(buf, buf + na.n);
        return out;
    }

    // Exact exploitability of a strategy profile inside this subgame, for 2 live players and a root
    // on the river (no chance left): best responses over all 1326 x 1326 hole pairs (card removal,
    // exact showdowns), the root ranges as the players' distributions.  kind 0: the search's
    // average strategy, 1: its final iteration (both need solve(); unvisited infosets play
    // uniformly; our actual hole plays its taken actions on the real path), 2: the blueprint as the
    // agent plays it in these spots (bucket keys with the translated history, "c" when unknown).
    // Returns {exploitability, BR gain of seat a, BR gain of seat b} in big blinds per deal of the
    // subgame, a < b the two live seats.
    std::vector<double> river_exploitability(int kind) const {
        if (root_street_ != RIVER || root_.n_board != 5) throw std::runtime_error("river_exploitability: the root must be on the river");
        std::vector<int> live;
        for (int s = 0; s < root_.n; s++) if (!root_.players[s].folded) live.push_back(s);
        if (live.size() != 2) throw std::runtime_error("river_exploitability: exactly two live players");
        if (kind < 2 && !table_) throw std::runtime_error("solve first");
        if (kind == 2 && !game_->blueprint) throw std::runtime_error("no blueprint");
        const ComboTable& ct = combo_table();
        Exploit ex;
        ex.kind = kind;
        ex.strength.assign((size_t)N_COMBOS, 0);
        for (int c = 0; c < N_COMBOS; c++) {
            if (class_of_[(size_t)c] < 0) continue;
            int cards[7] = {ct.c0[c], ct.c1[c], root_.board[0], root_.board[1], root_.board[2], root_.board[3], root_.board[4]};
            ex.strength[(size_t)c] = evaluate(cards, 7);
        }
        if (kind == 2) {
            ex.bucket.assign((size_t)N_COMBOS, -1);
            for (int c = 0; c < N_COMBOS; c++) {
                if (class_of_[(size_t)c] < 0) continue;
                const int hole[2] = {ct.c0[c], ct.c1[c]};
                ex.bucket[(size_t)c] = game_->bucketer->bucket(hole, root_.board, 5);
            }
        }
        double z = 0.0;  // total weight of the disjoint pairs
        const std::vector<double>& ra = reach_[(size_t)live[0]];
        const std::vector<double>& rb = reach_[(size_t)live[1]];
        for (int c = 0; c < N_COMBOS; c++)
            for (int d = 0; d < N_COMBOS; d++)
                if (!overlap(c, d)) z += ra[(size_t)c] * rb[(size_t)d];
        std::vector<double> out(3, 0.0);
        for (int i = 0; i < 2; i++) {
            const int p = live[(size_t)i], o = live[(size_t)(1 - i)];
            const std::vector<double>& rp = reach_[(size_t)p];
            double br = 0.0, val = 0.0;
            for (int mode = 0; mode < 2; mode++) {
                HandState st(root_);
                st.deck = deck_;
                const std::vector<double> v = exploit_walk(st, PathHash(), 0, true, p, o, reach_[(size_t)o], mode == 1, ex);
                double tot = 0.0;
                for (int c = 0; c < N_COMBOS; c++) tot += rp[(size_t)c] * v[(size_t)c];
                (mode == 1 ? br : val) = tot / z;
            }
            out[(size_t)(1 + i)] = br - val;
        }
        out[0] = 0.5 * (out[1] + out[2]);
        return out;
    }

    // tests: the raw node of real-path node k's actor holding `hole` (false if never created)
    bool node_at_path(int k, int h0, int h1, std::vector<double>& regret, std::vector<double>& ssum, long long& visits) const {
        if (!table_) throw std::runtime_error("solve first");
        if (k < 0 || k > (int)path_.size()) throw std::invalid_argument("k: 0..len(path)");
        HandState st(root_);
        st.deck = deck_;
        PathHash ph;
        for (int j = 0; j < k; j++) {
            NodeActions na;
            build_actions(st, observe(st, st.to_act), &path_[(size_t)j], na);
            st.apply(na.type[path_[(size_t)j].index], na.amount[path_[(size_t)j].index]);
            ph.step(path_[(size_t)j].index);
        }
        const int c = combo_index(h0, h1);
        if (c < 0 || class_of_[(size_t)c] < 0) return false;
        const FlatNodeTable::Found f = table_->find(search_key(root_street_, st.to_act, class_of_[(size_t)c], ph));
        if (!f.node) return false;
        regret.assign(f.node->regret, f.node->regret + f.node->n);
        ssum.assign(f.node->strategy_sum, f.node->strategy_sum + f.node->n);
        visits = f.node->visits;
        return true;
    }

    // tests: `n` deals as the solver draws them (holes per seat, then the rest of the board)
    std::vector<std::vector<int>> sample_deals(int n, bool focused, uint64_t seed) const {
        Ctx ctx;
        ctx.rng.reseed(seed);
        std::memcpy(ctx.deck, deck_, sizeof deck_);
        std::vector<std::vector<int>> out;
        for (int i = 0; i < n; i++) {
            deal(ctx, focused);
            std::vector<int> row;
            for (int s = 0; s < root_.n; s++) {
                row.push_back(ctx.holes[s][0]);
                row.push_back(ctx.holes[s][1]);
            }
            for (int j = root_.deck_pos; j < root_.deck_pos + 5 - root_.n_board; j++) row.push_back(ctx.deck[j]);
            out.push_back(std::move(row));
        }
        return out;
    }

private:
    struct Exploit {
        int kind = 0;
        std::vector<int64_t> strength;  // hand value on the river board, per combo
        std::vector<int> bucket;        // blueprint bucket per combo (kind 2)
    };

    static bool overlap(int c, int d) {
        const ComboTable& ct = combo_table();
        return ct.c0[c] == ct.c0[d] || ct.c0[c] == ct.c1[d] || ct.c1[c] == ct.c0[d] || ct.c1[c] == ct.c1[d];
    }

    // per combo of `seat` at this node: its probabilities under the profile (rows of `out`, stride MAX_ACTIONS)
    void profile_rows(const HandState& st, const PathHash& ph, int k, bool path_node, const NodeActions& na, int seat, const Exploit& ex,
                      std::vector<double>& out) const {
        out.assign((size_t)N_COMBOS * MAX_ACTIONS, 0.0);
        HistHash hh;
        std::vector<int> legal_bp;
        int call_idx = 0;
        if (ex.kind == 2) {
            std::string tok;
            hh.catch_up(st, game_->grid, tok);
            for (int i = 0; i < na.n; i++) {
                if (na.id[i] == 1) call_idx = i;
                if (na.id[i] == INSERTED_ID) { legal_bp.push_back(-1); continue; }
                const std::string& nm = game_->grid.names[na.id[i]];
                legal_bp.push_back(game_->blueprint->name_index(nm.data(), nm.size()));
            }
        }
        const int rel = ((seat - st.button) % st.n + st.n) % st.n;
        for (int c = 0; c < N_COMBOS; c++) {
            if (class_of_[(size_t)c] < 0) continue;
            double* row = &out[(size_t)c * MAX_ACTIONS];
            if (path_node && seat == hand_.our_seat && c == our_combo_) {  // our actual hole: the action taken
                row[path_[(size_t)k].index] = 1.0;
                continue;
            }
            if (ex.kind < 2) {
                const FlatNodeTable::Found f = table_->find(search_key(root_street_, seat, class_of_[(size_t)c], ph));
                if (!f.node || f.node->n != na.n) {
                    for (int i = 0; i < na.n; i++) row[i] = 1.0 / na.n;
                } else if (ex.kind == 0) {
                    f.node->average_strategy(row);
                } else {
                    f.node->current_strategy(row);
                }
                continue;
            }
            const long long i = game_->blueprint->find(node_key(st.street, rel, st.n_active(), ex.bucket[(size_t)c], hh));
            if (i < 0 || !game_->blueprint->policy_at(i, legal_bp.data(), na.n, row)) {
                for (int a = 0; a < na.n; a++) row[a] = a == call_idx ? 1.0 : 0.0;  // the agent's fallback: check / call
            }
        }
    }

    // counterfactual values of p's combos (weighted by the opponent's reach `ro`); `br`: p best-responds
    std::vector<double> exploit_walk(HandState& st, const PathHash& ph, int k, bool on_path, int p, int o, const std::vector<double>& ro,
                                     bool br, const Exploit& ex) const {
        const ComboTable& ct = combo_table();
        std::vector<double> v((size_t)N_COMBOS, 0.0);
        if (st.terminal) {
            if (st.n_active() == 1) {  // a fold: the same result for every pair of holes
                const double net = (double)st.net(p) / (double)game_->spec.bb;
                double card_sum[52] = {0.0};
                double total = 0.0;
                for (int d = 0; d < N_COMBOS; d++) {
                    total += ro[(size_t)d];
                    card_sum[ct.c0[d]] += ro[(size_t)d];
                    card_sum[ct.c1[d]] += ro[(size_t)d];
                }
                for (int c = 0; c < N_COMBOS; c++) {
                    if (class_of_[(size_t)c] < 0) continue;
                    v[(size_t)c] = net * (total - card_sum[ct.c0[c]] - card_sum[ct.c1[c]] + ro[(size_t)c]);
                }
                return v;
            }
            // showdown between the two: the winner nets the smaller investment, a tie nets 0
            const double m = (double)std::min(st.players[p].invested, st.players[o].invested) / (double)game_->spec.bb;
            for (int c = 0; c < N_COMBOS; c++) {
                if (class_of_[(size_t)c] < 0) continue;
                const int64_t sc = ex.strength[(size_t)c];
                double acc = 0.0;
                for (int d = 0; d < N_COMBOS; d++) {
                    const double w = ro[(size_t)d];
                    if (w == 0.0 || overlap(c, d)) continue;
                    const int64_t sd = ex.strength[(size_t)d];
                    if (sc > sd) acc += w;
                    else if (sc < sd) acc -= w;
                }
                v[(size_t)c] = m * acc;
            }
            return v;
        }
        const int seat = st.to_act;
        const bool path_node = on_path && k < (int)path_.size();
        NodeActions na;
        build_actions(st, observe(st, seat), path_node ? &path_[(size_t)k] : nullptr, na);
        std::vector<double> rows;
        if (seat == o || !br) profile_rows(st, ph, k, path_node, na, seat, ex, rows);
        if (seat == o) {
            std::vector<double> r2((size_t)N_COMBOS);
            for (int a = 0; a < na.n; a++) {
                for (int d = 0; d < N_COMBOS; d++) r2[(size_t)d] = ro[(size_t)d] * rows[(size_t)d * MAX_ACTIONS + a];
                HandState child(st);
                child.apply(na.type[a], na.amount[a]);
                PathHash ch = ph;
                ch.step(a);
                const std::vector<double> va = exploit_walk(child, ch, k + 1, path_node && a == path_[(size_t)k].index, p, o, r2, br, ex);
                for (int c = 0; c < N_COMBOS; c++) v[(size_t)c] += va[(size_t)c];
            }
            return v;
        }
        std::vector<std::vector<double>> vs((size_t)na.n);
        for (int a = 0; a < na.n; a++) {
            HandState child(st);
            child.apply(na.type[a], na.amount[a]);
            PathHash ch = ph;
            ch.step(a);
            vs[(size_t)a] = exploit_walk(child, ch, k + 1, path_node && a == path_[(size_t)k].index, p, o, ro, br, ex);
        }
        if (!br) {
            for (int c = 0; c < N_COMBOS; c++)
                for (int a = 0; a < na.n; a++) v[(size_t)c] += rows[(size_t)c * MAX_ACTIONS + a] * vs[(size_t)a][(size_t)c];
            return v;
        }
        // best response per infoset (class): the action with the largest value summed over its holes
        const std::vector<double>& rp = reach_[(size_t)p];
        std::vector<double> score((size_t)n_classes_ * MAX_ACTIONS, 0.0);
        for (int c = 0; c < N_COMBOS; c++) {
            const int cl = class_of_[(size_t)c];
            if (cl < 0) continue;
            for (int a = 0; a < na.n; a++) score[(size_t)cl * MAX_ACTIONS + a] += rp[(size_t)c] * vs[(size_t)a][(size_t)c];
        }
        for (int c = 0; c < N_COMBOS; c++) {
            const int cl = class_of_[(size_t)c];
            if (cl < 0) continue;
            int best = 0;
            for (int a = 1; a < na.n; a++)
                if (score[(size_t)cl * MAX_ACTIONS + a] > score[(size_t)cl * MAX_ACTIONS + best]) best = a;
            v[(size_t)c] = vs[(size_t)best][(size_t)c];
        }
        return v;
    }

    struct Ctx {
        FastRng rng;
        int tid = 0;
        int deck[52];
        int holes[MAX_PLAYERS][2];
        int combo[MAX_PLAYERS];
        int bucket_memo[MAX_PLAYERS][6];
        bool actual = false;
        long long iterations = 0, traversals = 0, focused = 0, nodes = 0, forced = 0, redeals = 0;
    };

    std::shared_ptr<const SearchGame> game_;
    HandInput hand_;
    SearchParams params_;
    int deck_[52];
    uint64_t board_mask_ = 0;
    HandState root_, current_;
    int root_street_ = PREFLOP;
    int our_combo_ = -1;
    std::vector<PreRootStep> pre_;
    std::vector<PathStep> path_;
    std::vector<int> class_of_ = std::vector<int>((size_t)N_COMBOS, -1);
    int n_classes_ = 0;
    std::vector<std::vector<double>> reach_;
    std::vector<std::vector<double>> cdf_;
    double range_seconds_ = 0.0;
    std::unique_ptr<FlatNodeTable> table_;
    std::unique_ptr<TableGroup> group_;

    // our hole at our seat, placeholders for the others, then the real board, then the rest: the
    // engine deals holes from the front and board cards from deck_pos on
    void build_deck() {
        const int n = (int)hand_.stacks.size();
        uint64_t used = board_mask_ | (1ULL << hand_.our_hole[0]) | (1ULL << hand_.our_hole[1]);
        int next_free = 0;
        auto take = [&]() {
            while (used >> next_free & 1) next_free++;
            used |= 1ULL << next_free;
            return next_free;
        };
        int pos = 0;
        for (int s = 0; s < n; s++) {
            if (s == hand_.our_seat) {
                deck_[pos++] = hand_.our_hole[0];
                deck_[pos++] = hand_.our_hole[1];
            } else {
                deck_[pos++] = take();
                deck_[pos++] = take();
            }
        }
        for (int c : hand_.board) deck_[pos++] = c;
        while (pos < 52) deck_[pos++] = take();
    }

    void replay() {
        const Spec& sp = game_->spec;
        const BetGrid& grid = game_->grid;
        // the street of the decision now
        {
            HandState st(hand_.stacks, hand_.button, sp.sb, sp.bb, sp.ante, deck_, sp.max_street);
            for (const auto& a : hand_.actions) {
                if (st.terminal) throw std::invalid_argument("actions continue after the hand ended");
                st.apply(a.first, a.second);
            }
            if (st.terminal) throw std::invalid_argument("the hand is over");
            root_street_ = st.street;
        }
        if ((int)hand_.board.size() != board_cards_by_street(root_street_))
            throw std::invalid_argument("board: " + std::to_string(board_cards_by_street(root_street_)) + " cards expected on " +
                                        street_letter(root_street_) + ", got " + std::to_string(hand_.board.size()));
        HandState st(hand_.stacks, hand_.button, sp.sb, sp.bb, sp.ante, deck_, sp.max_street);
        HistHash hh;
        std::string tok;
        size_t j = 0;
        for (; j < hand_.actions.size() && st.street < root_street_; j++) {
            const int seat = st.to_act;
            const Obs obs = observe(st, seat);
            hh.catch_up(st, grid, tok);
            PreRootStep step;
            step.seat = seat;
            step.street = st.street;
            step.rel = ((seat - st.button) % st.n + st.n) % st.n;
            step.n_active = obs.n_active;
            step.n_board = st.n_board;
            step.hist = hh;
            ActionList al;
            grid.abstract_actions(obs, al);
            std::vector<std::string> legal_names;
            for (int i = 0; i < al.n; i++) legal_names.push_back(grid.names[(size_t)al.a[i].id]);
            for (const std::string& nm : legal_names)
                step.legal.push_back(game_->blueprint ? game_->blueprint->name_index(nm.data(), nm.size()) : -1);
            const Event ev = st.apply(hand_.actions[j].first, hand_.actions[j].second);
            tok.clear();
            grid.from_concrete(ev, tok);
            step.a_idx = -1;
            for (size_t i = 0; i < legal_names.size(); i++) if (legal_names[i] == tok) { step.a_idx = (int)i; break; }
            pre_.push_back(std::move(step));
        }
        root_ = st;
        for (; j < hand_.actions.size(); j++) {
            const int seat = st.to_act;
            NodeActions na;
            build_actions(st, observe(st, seat), nullptr, na);
            PathStep ps;
            ps.actor = seat;
            ps.type = hand_.actions[j].first;
            ps.amount = ps.type == RAISE ? hand_.actions[j].second : 0;
            ps.index = -1;
            for (int i = 0; i < na.n; i++) if (na.type[i] == ps.type && na.amount[i] == ps.amount) { ps.index = i; break; }
            if (ps.index < 0) {
                if (na.n >= MAX_ACTIONS) throw std::runtime_error("no room to insert an off-grid action at a node with 8 grid actions");
                ps.inserted = true;
                ps.index = na.n;
            }
            path_.push_back(ps);
            st.apply(ps.type, ps.amount);
        }
        current_ = st;
    }

    void build_actions(const HandState&, const Obs& obs, const PathStep* real, NodeActions& na) const {
        ActionList al;
        game_->grid.abstract_actions(obs, al);
        na.n = 0;
        for (int i = 0; i < al.n; i++) {
            int type, amount;
            game_->grid.to_concrete(obs, al.a[i], type, amount);
            na.type[na.n] = type;
            na.amount[na.n] = type == RAISE ? amount : 0;
            na.id[na.n] = (uint8_t)al.a[i].id;
            na.n++;
        }
        if (real && real->inserted) {
            na.type[na.n] = real->type;
            na.amount[na.n] = real->amount;
            na.id[na.n] = INSERTED_ID;
            na.n++;
        }
    }

    // current round: one infoset per canonical form of hole + board (lossless)
    void compute_classes() {
        const ComboTable& ct = combo_table();
        std::unordered_map<uint64_t, int> ids;
        for (int c = 0; c < N_COMBOS; c++) {
            if ((board_mask_ >> ct.c0[c] & 1) || (board_mask_ >> ct.c1[c] & 1)) continue;
            uint64_t key;
            if (root_.n_board == 0) {
                key = (uint64_t)game_->bucketer->classes().of(ct.c0[c], ct.c1[c]);
            } else {
                const int hole[2] = {ct.c0[c], ct.c1[c]};
                CanonicalForm cf;
                canonical_form(hole, root_.board, root_.n_board, cf);
                key = canonical_pack(cf);
            }
            auto it = ids.find(key);
            if (it == ids.end()) it = ids.emplace(key, (int)ids.size()).first;
            class_of_[(size_t)c] = it->second;
        }
        n_classes_ = (int)ids.size();
    }

    // sigma(real action) at a pre-root step for a hole in `bucket`
    double blueprint_sigma(const PreRootStep& s, int bucket) const {
        const int n_legal = (int)s.legal.size();
        if (!game_->blueprint || n_legal == 0) return n_legal ? 1.0 / n_legal : 1.0;
        const long long i = game_->blueprint->find(node_key(s.street, s.rel, s.n_active, bucket, s.hist));
        double out[16];
        std::vector<double> big;
        double* o = out;
        if (n_legal > 16) {
            big.resize((size_t)n_legal);
            o = big.data();
        }
        if (i < 0 || !game_->blueprint->policy_at(i, s.legal.data(), n_legal, o)) return 1.0 / n_legal;
        return s.a_idx >= 0 ? o[s.a_idx] : 0.0;
    }

    void compute_ranges() {
        const auto t0 = std::chrono::steady_clock::now();
        const int n = root_.n;
        reach_.assign((size_t)n, std::vector<double>());
        for (int s = 0; s < n; s++) if (!root_.players[s].folded) reach_[(size_t)s].assign((size_t)N_COMBOS, 0.0);
        // overrides by (street, seat)
        std::vector<const LikelihoodOverride*> ov((size_t)(5 * MAX_PLAYERS), nullptr);
        for (const LikelihoodOverride& o : hand_.overrides) {
            if (o.street < 0 || o.street >= root_street_ || o.seat < 0 || o.seat >= n)
                throw std::invalid_argument("override: a street before the root's and a seat of the game");
            if (o.w.size() != (size_t)N_COMBOS) throw std::invalid_argument("override: 1326 likelihoods expected");
            ov[(size_t)(o.street * MAX_PLAYERS + o.seat)] = &o;
        }
        const ComboTable& ct = combo_table();
        const int T = std::max(1, std::min(params_.threads, 64));
        auto work = [&](int tid) {
            for (int c = tid; c < N_COMBOS; c += T) {
                if (class_of_[(size_t)c] < 0) continue;
                const int hole[2] = {ct.c0[c], ct.c1[c]};
                int bucket_of_street[4] = {-1, -1, -1, -1};
                for (int s = 0; s < n; s++) {
                    if (root_.players[s].folded) continue;
                    double w = 1.0;
                    // street by street, in order: an override replaces the seat's actions on its street
                    for (int street = PREFLOP; street < root_street_; street++) {
                        const LikelihoodOverride* o = ov[(size_t)(street * MAX_PLAYERS + s)];
                        if (o) {
                            w *= o->w[(size_t)c];
                            continue;
                        }
                        for (const PreRootStep& st : pre_) {
                            if (st.seat != s || st.street != street) continue;
                            int& b = bucket_of_street[st.street];
                            if (b < 0) b = game_->bucketer->bucket(hole, root_.board, st.n_board);
                            w *= std::max(blueprint_sigma(st, b), params_.min_prob);
                        }
                    }
                    reach_[(size_t)s][(size_t)c] = w;
                }
            }
        };
        if (T == 1) {
            work(0);
        } else {
            std::vector<std::thread> pool;
            for (int t = 1; t < T; t++) pool.emplace_back(work, t);
            work(0);
            for (auto& th : pool) th.join();
        }
        build_cdfs();
        range_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

public:
    // replace a seat's range (tests, later parts); weights for all 1326 combos
    void set_reach(int seat, const std::vector<double>& w) {
        if (seat < 0 || seat >= root_.n || root_.players[seat].folded) throw std::invalid_argument("set_reach: a live seat");
        if (w.size() != (size_t)N_COMBOS) throw std::invalid_argument("set_reach: 1326 weights");
        for (int c = 0; c < N_COMBOS; c++) reach_[(size_t)seat][(size_t)c] = class_of_[(size_t)c] >= 0 ? std::max(0.0, w[(size_t)c]) : 0.0;
        build_cdfs();
    }

private:
    void build_cdfs() {
        cdf_.assign(reach_.size(), std::vector<double>());
        for (size_t s = 0; s < reach_.size(); s++) {
            if (reach_[s].empty()) continue;
            std::vector<double>& c = cdf_[s];
            c.resize((size_t)N_COMBOS);
            double acc = 0.0;
            for (int i = 0; i < N_COMBOS; i++) {
                acc += reach_[s][(size_t)i];
                c[(size_t)i] = acc;
            }
            if (!(acc > 0.0)) throw std::runtime_error("the range of seat " + std::to_string(s) + " is empty");
        }
    }

    int draw_combo(int seat, FastRng& rng) const {
        const std::vector<double>& c = cdf_[(size_t)seat];
        const double u = rng.uniform() * c.back();
        const int i = (int)(std::upper_bound(c.begin(), c.end(), u) - c.begin());
        return i < N_COMBOS ? i : N_COMBOS - 1;
    }

    // every live hole from its range, rejected on any shared card (focused: ours is our actual
    // hole), then the rest of the board uniformly
    void deal(Ctx& ctx, bool focused) const {
        const ComboTable& ct = combo_table();
        const int n = root_.n;
        uint64_t used = 0;
        for (int attempt = 0;; attempt++) {
            if (attempt > 100000) throw std::runtime_error("cannot deal non-overlapping holes from these ranges");
            used = board_mask_;
            bool ok = true;
            if (focused) used |= (1ULL << hand_.our_hole[0]) | (1ULL << hand_.our_hole[1]);
            for (int s = 0; s < n && ok; s++) {
                if (reach_[(size_t)s].empty()) continue;  // folded before the root
                int c;
                if (focused && s == hand_.our_seat) {
                    c = our_combo_;
                } else {
                    c = draw_combo(s, ctx.rng);
                    const uint64_t m = (1ULL << ct.c0[c]) | (1ULL << ct.c1[c]);
                    if (used & m) {
                        ok = false;
                        break;
                    }
                    used |= m;
                }
                ctx.combo[s] = c;
                ctx.holes[s][0] = ct.c0[c];
                ctx.holes[s][1] = ct.c1[c];
            }
            if (ok) break;
            ctx.redeals++;
        }
        for (int s = 0; s < n; s++) {
            if (!reach_[(size_t)s].empty()) continue;
            ctx.combo[s] = -1;  // folded: never used by the engine
            ctx.holes[s][0] = root_.players[s].hole[0];
            ctx.holes[s][1] = root_.players[s].hole[1];
        }
        for (int i = 0; i < 5 - root_.n_board; i++) {
            int card;
            do card = ctx.rng.below(52); while (used >> card & 1);
            used |= 1ULL << card;
            ctx.deck[root_.deck_pos + i] = card;
        }
        ctx.actual = ctx.combo[hand_.our_seat] == our_combo_;
        for (int s = 0; s < MAX_PLAYERS; s++)
            for (int b = 0; b < 6; b++) ctx.bucket_memo[s][b] = -1;
    }

    int card_part(const HandState& st, int seat, Ctx& ctx) const {
        if (st.street == root_street_) return class_of_[(size_t)ctx.combo[seat]];
        int& m = ctx.bucket_memo[seat][st.n_board];
        if (m < 0) m = game_->bucketer->bucket(st.players[seat].hole, st.board, st.n_board);
        return m;
    }

    static int sample(const double* p, int n, FastRng& rng) {
        const double r = rng.uniform();
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += p[i];
            if (r < acc) return i;
        }
        return n - 1;
    }

    // external sampling; `k` = actions since the root, `on_path` = they were the real ones
    double traverse(HandState& st, PathHash ph, int k, bool on_path, int traverser, double weight, double w_imp, bool focused, Ctx& ctx) {
        if (st.terminal) return (double)st.net(traverser) / (double)game_->spec.bb;
        const int seat = st.to_act;
        const Obs obs = observe(st, seat);
        const bool path_node = on_path && k < (int)path_.size();
        NodeActions na;
        build_actions(st, obs, path_node ? &path_[(size_t)k] : nullptr, na);
        const NodeKey key = search_key(st.street, seat, card_part(st, seat, ctx), ph);
        Node* node = table_->get_or_create(key, ctx.tid, [&](Node& nd, NodeArena&) {
            nd.init(na.id, na.n);
            return "";
        }).node;
        ctx.nodes++;
        if (path_node && seat == hand_.our_seat && ctx.actual) {  // our action already taken: fixed for our actual hole
            const int a = path_[(size_t)k].index;
            ctx.forced++;
            st.apply(na.type[a], na.amount[a]);
            ph.step(a);
            return traverse(st, ph, k + 1, true, traverser, weight, w_imp, focused, ctx);
        }
        double sigma[MAX_ACTIONS];
        node->lock.lock();
        node->current_strategy(sigma);
        node->lock.unlock();
        if (path_node && focused && seat != traverser) {  // an opponent's real action, weighted by its probability
            const int a = path_[(size_t)k].index;
            const double w2 = w_imp * sigma[a];
            if (!(w2 > 0.0)) return 0.0;
            st.apply(na.type[a], na.amount[a]);
            ph.step(a);
            return traverse(st, ph, k + 1, true, traverser, weight, w2, focused, ctx);
        }
        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            for (int i = 0; i < na.n; i++) {
                HandState child(st);
                child.apply(na.type[i], na.amount[i]);
                PathHash ch = ph;
                ch.step(i);
                utils[i] = traverse(child, ch, k + 1, path_node && i == path_[(size_t)k].index, traverser, weight, w_imp, focused, ctx);
            }
            double u = 0.0;
            for (int i = 0; i < na.n; i++) u += sigma[i] * utils[i];
            const double w = weight * w_imp;
            node->lock.lock();
            for (int i = 0; i < na.n; i++) node->regret[i] += w * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        if (!focused) {
            node->lock.lock();
            for (int i = 0; i < na.n; i++) node->strategy_sum[i] += weight * sigma[i];
            node->visits += 1;
            node->lock.unlock();
        }
        const int a = sample(sigma, na.n, ctx.rng);
        st.apply(na.type[a], na.amount[a]);
        ph.step(a);
        return traverse(st, ph, k + 1, path_node && a == path_[(size_t)k].index, traverser, weight, w_imp, focused, ctx);
    }
};

}  // namespace negp
