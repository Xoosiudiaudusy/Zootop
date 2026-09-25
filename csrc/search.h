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

// Depth rules (part 2).  END: to the end of the hand.  PLURIBUS: a preflop root searches to the end
// of the preflop; a flop root with more than two players at its start to the start of the turn or
// right after the second raise of the flop, whichever comes first; anything else to the end of the
// hand.  HU_FLOP (Modicum, Depth-Limited Solving 2018): preflop and flop roots to the end of their
// round, turn and river roots to the end of the hand.  NEXT_STREET (experiments): every root but the
// river to the end of its round.
constexpr int DEPTH_END = 0, DEPTH_PLURIBUS = 1, DEPTH_HU_FLOP = 2, DEPTH_NEXT_STREET = 3;
constexpr int LEAF_STREET = 7;       // street field of the keys of leaf choices
constexpr int N_CONTINUATIONS = 4;   // the blueprint; fold, call, every raise x bias (renormalised)
constexpr int CONT_BP = 0, CONT_FOLD = 1, CONT_CALL = 2, CONT_RAISE = 3;

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

// the continuation `choice` applied to blueprint probabilities: the probability of fold, of call or
// of every raise (all-in included) multiplied by `bias`, then renormalised; kind per action:
// 0 fold, 1 call, 2 raise
inline void apply_continuation(const int* kind, int n, int choice, double bias, double* p) {
    if (choice == CONT_BP) return;
    const int want = choice == CONT_FOLD ? 0 : choice == CONT_CALL ? 1 : 2;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        if (kind[i] == want) p[i] *= bias;
        s += p[i];
    }
    if (s > 0.0)
        for (int i = 0; i < n; i++) p[i] /= s;
}

// ------------------------------------------------------------------ inputs and outputs
struct SearchGame {
    Spec spec;
    BetGrid grid;
    std::shared_ptr<Bucketer> bucketer;
    std::shared_ptr<const BlueprintTable> blueprint;  // null: no prior (uniform ranges)
    // optional (Pluribus' compression of the continuations): per blueprint record and continuation,
    // the index among the record's actions of one action drawn in advance; empty: off
    std::vector<uint8_t> presampled;

    SearchGame(const Spec& s, std::shared_ptr<Bucketer> bk, std::shared_ptr<const BlueprintTable> bp)
        : spec(s), grid(s.grid()), bucketer(std::move(bk)), blueprint(std::move(bp)) {
        if (!bucketer) throw std::invalid_argument("a bucketer is needed");
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        if (blueprint && blueprint->codec.n_players() != spec.n_players)
            throw std::invalid_argument("the blueprint's numeric keys are for " + std::to_string(blueprint->codec.n_players()) +
                                        " players, the game has " + std::to_string(spec.n_players));
    }

    // draw the actions (call before the searches; not concurrent with them)
    void presample(uint64_t seed, double bias) {
        if (!blueprint) throw std::runtime_error("presample: no blueprint");
        const BlueprintTable& b = *blueprint;
        std::vector<int> name_kind(b.names.size(), 2);
        for (size_t i = 0; i < b.names.size(); i++) name_kind[i] = b.names[i] == "f" ? 0 : b.names[i] == "c" ? 1 : 2;
        std::vector<uint8_t> out(b.size() * N_CONTINUATIONS, 0);
        FastRng rng(seed);
        for (size_t i = 0; i < b.size(); i++) {
            const uint32_t a = b.off[i], e = b.off[i + 1];
            const int n = (int)(e - a);
            if (n <= 0) continue;
            if (n > MAX_ACTIONS) throw std::runtime_error("presample: a record with more than 8 actions");
            int kind[MAX_ACTIONS];
            double base[MAX_ACTIONS];
            for (int t = 0; t < n; t++) {
                kind[t] = name_kind[b.ids[a + (uint32_t)t]];
                base[t] = b.prob(a + (uint32_t)t);
            }
            for (int k = 0; k < N_CONTINUATIONS; k++) {
                double p[MAX_ACTIONS];
                std::memcpy(p, base, sizeof(double) * (size_t)n);
                apply_continuation(kind, n, k, bias, p);
                double s = 0.0;
                for (int t = 0; t < n; t++) s += p[t];
                const double u = rng.uniform() * (s > 0.0 ? s : 1.0);
                double acc = 0.0;
                int pick = n - 1;
                for (int t = 0; t < n; t++) {
                    acc += s > 0.0 ? p[t] : 1.0 / n;
                    if (u < acc) { pick = t; break; }
                }
                out[i * N_CONTINUATIONS + (size_t)k] = (uint8_t)pick;
            }
        }
        presampled = std::move(out);
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
    int depth = DEPTH_END;     // depth rule (DEPTH_*)
    int rollouts = 3;          // rollouts per leaf value (Depth-Limited Solving 2018: three)
    double bias = 5.0;         // the continuations' factor (Pluribus: 5)
    int debug_leaves = 0;      // tests: log this many leaf choices made inside the solver
};

struct LeafInfo {  // tests: one leaf of the public tree
    std::vector<std::pair<int, int>> actions;  // from the root
    int street = 0, raises = 0, n_board = 0, reason = 0, choosers = 0;  // reason 0: next street, 1: raise limit
};

struct LeafLogEntry {  // tests: one continuation choice made inside the solver
    int seat = 0, combo = 0, cls = 0, n_board = 0;
    uint64_t k1 = 0, k2 = 0, path = 0;
    int board[5] = {0, 0, 0, 0, 0};
    int combos[MAX_PLAYERS] = {0};
};

struct SearchResult {
    std::vector<int> types, amounts, ids;       // the actions at our current decision
    std::vector<double> final_strategy;         // our actual hole, after the last iteration
    std::vector<double> average_strategy;       // our actual hole, average over the iterations
    // the node's accumulated average (the profile's, as likelihood() and the evaluator read it): for
    // our class it accumulates only when the opponents' traversals deal it and follow the real path,
    // so for a hole of small range weight it can stay empty (uniform)
    std::vector<double> table_average;
    bool visited = false;                       // our current infoset was reached
    long long iterations = 0, traversals = 0, focused = 0, nodes_touched = 0, forced = 0, redeals = 0;
    long long leaves = 0, leaf_evals = 0, rollouts = 0, rollout_steps = 0;
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
        if (params_.rollouts < 1) throw std::invalid_argument("rollouts >= 1");
        build_deck();
        replay();
        compute_classes();
        compute_ranges();
        set_depth_limits();
        grid_to_bp_.assign(game_->grid.names.size(), -1);
        if (game_->blueprint)
            for (size_t i = 0; i < grid_to_bp_.size(); i++)
                grid_to_bp_[i] = game_->blueprint->name_index(game_->grid.names[i].data(), game_->grid.names[i].size());
        build_river_table();
        init_turn_table();
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
    double river_table_seconds() const { return river_seconds_; }
    int river_table_boards() const { return (int)(river_.b.size() / N_COMBOS); }
    int limit_street() const { return limit_street_; }
    int raise_limit() const { return raise_limit_; }
    NodeActions actions_at(const HandState& st, int k) const {
        NodeActions na;
        build_actions(st, observe(st, st.to_act), k < (int)path_.size() ? &path_[k] : nullptr, na);
        return na;
    }

    // Measurement: every player plays the root's round with the strategy of `src` (kind 0 its
    // average, 1 its final iteration), frozen, so that solve() learns only the later rounds -- the
    // river re-solved after a turn search, for instance.  The evaluator and solve()'s result read the
    // frozen round from `src` as well.  `src` is a solved search of the same spot (game, hand, path),
    // must outlive this one, and must not be solved again meanwhile; nullptr unfreezes.
    void freeze_round(const SubgameSearch* src, int kind) {
        if (src == nullptr) {
            frozen_ = nullptr;
            return;
        }
        if (kind != 0 && kind != 1) throw std::invalid_argument("freeze_round: kind 0 (average) or 1 (final iteration)");
        if (src == this) throw std::invalid_argument("freeze_round: the source must be another search");
        if (!src->table_) throw std::runtime_error("freeze_round: solve the source first");
        bool same = src->game_ == game_ && src->root_street_ == root_street_ && src->hand_.our_seat == hand_.our_seat &&
                    src->our_combo_ == our_combo_ && src->class_of_ == class_of_ && src->reach_ == reach_ &&
                    src->path_.size() == path_.size() && std::memcmp(src->deck_, deck_, sizeof deck_) == 0;
        for (size_t i = 0; same && i < path_.size(); i++)
            same = src->path_[i].index == path_[i].index && src->path_[i].type == path_[i].type && src->path_[i].amount == path_[i].amount;
        if (!same) throw std::invalid_argument("freeze_round: the source searched another spot");
        frozen_ = src;
        frozen_kind_ = kind;
    }
    bool frozen() const { return frozen_ != nullptr; }

    // the strategy (kind 0 average, 1 final iteration) at a key of this search's table; uniform if absent
    void strategy_row(const NodeKey& key, int n, int kind, double* out) const {
        const FlatNodeTable::Found f = table_ ? table_->find(key) : FlatNodeTable::Found();
        if (!f.node || f.node->n != n) {
            for (int i = 0; i < n; i++) out[i] = 1.0 / n;
        } else if (kind == 0) {
            f.node->average_strategy(out);
        } else {
            f.node->current_strategy(out);
        }
    }

    // Measurement: at real-path node k (k = len(path): our decision now), the strategy (kind 0
    // average, 1 final iteration) of every combo of the node's actor; empty for combos on the board
    // and for infosets the table does not hold.  A frozen round reads its source.
    std::vector<std::vector<double>> path_strategies(int k, int kind) const {
        if (!table_) throw std::runtime_error("solve first");
        if (k < 0 || k > (int)path_.size()) throw std::invalid_argument("k: 0..len(path)");
        if (kind != 0 && kind != 1) throw std::invalid_argument("kind 0 (average) or 1 (final iteration)");
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
        const SubgameSearch& src = frozen_ ? *frozen_ : *this;
        const int kd = frozen_ ? frozen_kind_ : kind;
        std::vector<std::vector<double>> out((size_t)N_COMBOS);
        for (int c = 0; c < N_COMBOS; c++) {
            if (class_of_[(size_t)c] < 0) continue;
            if (k < (int)path_.size() && seat == hand_.our_seat && c == our_combo_) {
                out[(size_t)c].assign((size_t)na.n, 0.0);
                out[(size_t)c][(size_t)path_[(size_t)k].index] = 1.0;  // fixed
                continue;
            }
            const FlatNodeTable::Found f = src.table_->find(search_key(root_street_, seat, class_of_[(size_t)c], ph));
            if (!f.node || f.node->n != na.n) continue;
            out[(size_t)c].assign((size_t)na.n, 0.0);
            if (kd == 0) f.node->average_strategy(out[(size_t)c].data());
            else f.node->current_strategy(out[(size_t)c].data());
        }
        return out;
    }

    // ---------------------------------------------------------- solve
    SearchResult solve() {
        if (current_.terminal || current_.to_act != hand_.our_seat) throw std::runtime_error("it is not our turn to act");
        if (params_.iterations <= 0 && params_.time_budget <= 0) throw std::invalid_argument("set iterations or time_budget");
        const int T = params_.threads;
        {
            std::lock_guard<std::mutex> lk(log_mu_);
            leaf_log_.clear();
        }
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
        // our decision (our actual hole's class at the end of the real path): its strategy is added up
        // once per iteration, which is the average our hole plays (its own reach there is fixed)
        NodeActions our_na;
        build_actions(current_, observe(current_, hand_.our_seat), nullptr, our_na);
        PathHash our_ph;
        for (const PathStep& s : path_) our_ph.step(s.index);
        const NodeKey our_key = search_key(root_street_, hand_.our_seat, class_of_[our_combo_], our_ph);
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
                    if (!frozen_) {
                        Node* node = table_->get_or_create(our_key, ctx.tid, our_na.n, [&](Node& nd, NodeArena&) {
                            nd.init(our_na.id, our_na.n);
                            return "";
                        }).node;
                        if (node->n == our_na.n) {
                            double sg[MAX_ACTIONS];
                            node->lock.lock();
                            node->current_strategy(sg);
                            node->lock.unlock();
                            for (int i = 0; i < our_na.n; i++) ctx.ours[i] += weight * sg[i];
                        }
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
            r.leaves += c.leaves;
            r.leaf_evals += c.leaf_evals;
            r.rollouts += c.rollouts;
            r.rollout_steps += c.rollout_steps;
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
        const FlatNodeTable* tab = frozen_ ? frozen_->table_.get() : table_.get();
        const FlatNodeTable::Found f = tab->find(search_key(root_street_, hand_.our_seat, class_of_[our_combo_], ph));
        r.final_strategy.assign((size_t)na.n, 1.0 / na.n);
        r.average_strategy.assign((size_t)na.n, 1.0 / na.n);
        r.table_average.assign((size_t)na.n, 1.0 / na.n);
        if (f.node && f.node->n == na.n && frozen_) {  // the frozen round's strategy, as played
            r.visited = true;
            double buf[MAX_ACTIONS];
            if (frozen_kind_ == 0) f.node->average_strategy(buf);
            else f.node->current_strategy(buf);
            r.final_strategy.assign(buf, buf + na.n);
            r.average_strategy.assign(buf, buf + na.n);
            r.table_average.assign(buf, buf + na.n);
        } else if (f.node && f.node->n == na.n) {
            r.visited = true;
            double buf[MAX_ACTIONS];
            f.node->current_strategy(buf);
            r.final_strategy.assign(buf, buf + na.n);
            f.node->average_strategy(buf);
            r.table_average.assign(buf, buf + na.n);
            double ours[MAX_ACTIONS] = {0.0}, s = 0.0;
            for (const Ctx& c : ctxs)
                for (int i = 0; i < na.n; i++) ours[i] += c.ours[i];
            for (int i = 0; i < na.n; i++) s += ours[i];
            if (s > 0.0) {
                for (int i = 0; i < na.n; i++) r.average_strategy[(size_t)i] = ours[i] / s;
            } else {
                r.average_strategy = r.table_average;
            }
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
        return subgame_exploitability(kind, RIVER);
    }

    // The same for a root on the turn or the river, with the river card as a chance node (every
    // card, 1/44 for each pair of holes).  If the search stopped at the start of the river (depth
    // next_street), its river play is what the subgame contains: each player's continuation mix at
    // its leaf infoset, then that continuation (the blueprint as the agent plays it, biased).  The
    // best responder deviates only on streets >= br_street: the root's street for the whole
    // subgame, RIVER for the exploitability of the river play alone.  br_street = LEAF_STREET (7), for a
    // search with leaves at the river start: the exploitability inside the search's own model, the
    // best responder deviating on the turn and choosing its best continuation at a leaf (per hole,
    // before the river card), the river then played by the continuations.
    std::vector<double> subgame_exploitability(int kind, int br_street) const {
        if (root_street_ < TURN) throw std::runtime_error("subgame_exploitability: the root must be on the turn or the river");
        std::vector<int> live;
        for (int s = 0; s < root_.n; s++) if (!root_.players[s].folded) live.push_back(s);
        if (live.size() != 2) throw std::runtime_error("subgame_exploitability: exactly two live players");
        if (kind < 2 && !table_) throw std::runtime_error("solve first");
        if (kind == 2 && !game_->blueprint) throw std::runtime_error("no blueprint");
        Ex ex;
        ex.kind = kind;
        ex.br_street = std::max(br_street, root_street_);
        ex.leaf_river = kind < 2 && limit_street_ < RIVER;
        if (br_street == LEAF_STREET) {  // the search's own model: deviations on the root's round and in the leaf choice
            if (!ex.leaf_river) throw std::runtime_error("subgame_exploitability: the leaf game needs a search with leaves at the river start");
            ex.leaf_game = true;
            ex.br_street = root_street_;
        }
        const ComboTable& ct = combo_table();
        ex.boards.assign(53, Board5());
        const bool need_buckets = kind == 2 || ex.leaf_river || root_street_ < RIVER;
        auto fill_board = [&](int idx, const int* b, int n) {
            Board5& B = ex.boards[(size_t)idx];
            bool on[52] = {false};
            for (int i = 0; i < n; i++) on[b[i]] = true;
            if (n == 5) {
                B.strength.assign((size_t)N_COMBOS, 0);
                for (int c = 0; c < N_COMBOS; c++) {
                    if (on[ct.c0[c]] || on[ct.c1[c]]) continue;
                    int cards[7] = {ct.c0[c], ct.c1[c], b[0], b[1], b[2], b[3], b[4]};
                    B.strength[(size_t)c] = evaluate(cards, 7);
                    B.order.push_back(c);
                }
                std::sort(B.order.begin(), B.order.end(), [&](int a, int c2) { return B.strength[(size_t)a] < B.strength[(size_t)c2]; });
            }
            if (need_buckets) {
                B.bucket.assign((size_t)N_COMBOS, -1);
                for (int c = 0; c < N_COMBOS; c++) {
                    if (on[ct.c0[c]] || on[ct.c1[c]]) continue;
                    const int hole[2] = {ct.c0[c], ct.c1[c]};
                    B.bucket[(size_t)c] = later_bucket(hole, b, n);
                }
            }
        };
        int b5[5];
        for (int i = 0; i < root_.n_board; i++) b5[i] = root_.board[i];
        fill_board(52, b5, root_.n_board);  // the root's board (turn or river)
        if (root_.n_board == 4) {
            bool on[52] = {false};
            for (int i = 0; i < 4; i++) on[root_.board[i]] = true;
            for (int r = 0; r < 52; r++) {
                if (on[r]) continue;
                b5[4] = r;
                fill_board(r, b5, 5);
            }
        }
        double z = 0.0;  // total weight of the disjoint pairs
        {
            const std::vector<double>& ra = reach_[(size_t)live[0]];
            const std::vector<double>& rb = reach_[(size_t)live[1]];
            double total = 0.0, card[52] = {0.0};
            for (int d = 0; d < N_COMBOS; d++) {
                total += rb[(size_t)d];
                card[ct.c0[d]] += rb[(size_t)d];
                card[ct.c1[d]] += rb[(size_t)d];
            }
            for (int c = 0; c < N_COMBOS; c++)
                if (ra[(size_t)c] != 0.0) z += ra[(size_t)c] * (total - card[ct.c0[c]] - card[ct.c1[c]] + rb[(size_t)c]);
        }
        std::vector<double> out(3, 0.0);
        for (int i = 0; i < 2; i++) {
            ex.p = live[(size_t)i];
            ex.o = live[(size_t)(1 - i)];
            const std::vector<double>& rp = reach_[(size_t)ex.p];
            double br = 0.0, val = 0.0;
            for (int mode = 0; mode < 2; mode++) {
                HandState st(root_);
                st.deck = deck_;
                HistHash hh;
                std::string tok;
                hh.catch_up(st, game_->grid, tok);
                const std::vector<double> v = ex_walk(st, PathHash(), 0, true, reach_[(size_t)ex.o], 1, -1, hh, mode == 1, ex);
                double tot = 0.0;
                for (int c = 0; c < N_COMBOS; c++) tot += rp[(size_t)c] * v[(size_t)c];
                (mode == 1 ? br : val) = tot / z;
            }
            out[(size_t)(1 + i)] = br - val;
            out.push_back(val);  // the profile's value for seat a, then b (they sum to zero heads-up)
        }
        out[0] = 0.5 * (out[1] + out[2]);
        return out;
    }

    // tests: the river exploitability through pairwise showdowns (the fast evaluator's reference)
    std::vector<double> river_exploitability_slow(int kind) const {
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
        regret.assign(f.node->regret(), f.node->regret() + f.node->n);
        ssum.assign(f.node->strategy_sum(), f.node->strategy_sum() + f.node->n);
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
        // the best responder knows its exact hole: the best action per combo (per suit-isomorphism
        // class would be weaker when our forced actual hole makes the opponent's reach asymmetric)
        for (int c = 0; c < N_COMBOS; c++) {
            if (class_of_[(size_t)c] < 0) continue;
            double best = vs[0][(size_t)c];
            for (int a = 1; a < na.n; a++) best = std::max(best, vs[(size_t)a][(size_t)c]);
            v[(size_t)c] = best;
        }
        return v;
    }

    // ---- the general evaluator (turn or river roots)
    struct Board5 {                     // one board: hand values and the combos in increasing value (river), buckets
        std::vector<int64_t> strength;
        std::vector<int> order;
        std::vector<int> bucket;        // per combo, -1 on the board
    };
    struct Ex {
        int kind = 0, br_street = RIVER, p = 0, o = 1;
        bool leaf_river = false;        // the search stopped at the river start: its river play is the continuations
        bool leaf_game = false;         // the best responder picks its best continuation at a leaf instead of any river play
        std::vector<Board5> boards;     // [river card] from a turn root, [52] for a river root
    };

    // rows[x][a] for every combo x of the actor q: its probabilities under the profile, or continuation `cont`
    void ex_rows(const HandState& st, const PathHash& ph, int k, bool path_node, const NodeActions& na, int cont, const HistHash& hh,
                 const Ex& ex, std::vector<double>& rows) const {
        const int q = st.to_act;
        rows.assign((size_t)N_COMBOS * MAX_ACTIONS, 0.0);
        const bool by_bucket = cont >= 0 || ex.kind == 2;
        const Board5& B = ex.boards[(size_t)(st.n_board == 5 && root_.n_board == 4 ? st.board[4] : 52)];
        double memo[256][MAX_ACTIONS];
        bool have[256] = {false};
        for (int x = 0; x < N_COMBOS; x++) {
            const int bx = B.bucket.empty() ? 0 : B.bucket[(size_t)x];
            if (B.bucket.empty() ? class_of_[(size_t)x] < 0 : bx < 0) continue;  // on the board
            double* row = &rows[(size_t)x * MAX_ACTIONS];
            if (path_node && q == hand_.our_seat && x == our_combo_) {  // our actual hole plays the action taken
                row[path_[(size_t)k].index] = 1.0;
                continue;
            }
            if (by_bucket) {
                const int b = bx;
                if (b < 0 || b > 255) throw std::runtime_error("ex_rows: bucket out of range");
                if (!have[b]) {  // the policy depends on the hole only through its bucket
                    rollout_probs(st, hh, b, na, cont >= 0 ? cont : CONT_BP, memo[b]);
                    have[b] = true;
                }
                for (int a = 0; a < na.n; a++) row[a] = memo[b][a];
                continue;
            }
            const int card = st.street == root_street_ ? class_of_[(size_t)x] : bx;
            const bool fz = frozen_ != nullptr && st.street == root_street_;  // the round frozen to another search's play
            (fz ? *frozen_ : *this).strategy_row(search_key(st.street, q, card, ph), na.n, fz ? frozen_kind_ : ex.kind, row);
        }
    }

    // the leaf choice distribution of seat q holding x at the leaf with public path ph
    void ex_leaf_mix(int q, int x, const PathHash& ph, const Ex& ex, double* w) const {
        const FlatNodeTable::Found f = table_->find(search_key(LEAF_STREET, q, class_of_[(size_t)x], ph));
        if (!f.node || f.node->n != N_CONTINUATIONS) {
            for (int k = 0; k < N_CONTINUATIONS; k++) w[k] = 1.0 / N_CONTINUATIONS;
        } else if (ex.kind == 0) {
            f.node->average_strategy(w);
        } else {
            f.node->current_strategy(w);
        }
    }

    std::vector<double> ex_terminal(const HandState& st, const std::vector<double>& ro, int nto, int bidx, const Ex& ex) const {
        const ComboTable& ct = combo_table();
        std::vector<double> eff((size_t)N_COMBOS, 0.0), v((size_t)N_COMBOS, 0.0);
        for (int t = 0; t < nto; t++)
            for (int d = 0; d < N_COMBOS; d++) eff[(size_t)d] += ro[(size_t)t * N_COMBOS + d];
        if (st.n_active() == 1) {  // a fold: the same result for every pair of holes
            const double net = (double)st.net(ex.p) / (double)game_->spec.bb;
            double total = 0.0, card[52] = {0.0};
            for (int d = 0; d < N_COMBOS; d++) {
                total += eff[(size_t)d];
                card[ct.c0[d]] += eff[(size_t)d];
                card[ct.c1[d]] += eff[(size_t)d];
            }
            for (int c = 0; c < N_COMBOS; c++) v[(size_t)c] = net * (total - card[ct.c0[c]] - card[ct.c1[c]] + eff[(size_t)c]);
            return v;
        }
        // showdown: the winner nets the smaller investment, a tie nets 0; wins and ties by prefix sums
        // over the combos in increasing value, card removal through per-card sums
        const Board5& B = ex.boards[(size_t)bidx];
        const double m = (double)std::min(st.players[ex.p].invested, st.players[ex.o].invested) / (double)game_->spec.bb;
        std::vector<double> win((size_t)N_COMBOS, 0.0), tie((size_t)N_COMBOS, 0.0);
        double total = 0.0, card[52] = {0.0}, gcard[52] = {0.0};
        size_t i = 0;
        while (i < B.order.size()) {
            size_t j = i;
            const int64_t s = B.strength[(size_t)B.order[i]];
            double gt = 0.0;
            while (j < B.order.size() && B.strength[(size_t)B.order[j]] == s) {
                const int d = B.order[j];
                gt += eff[(size_t)d];
                gcard[ct.c0[d]] += eff[(size_t)d];
                gcard[ct.c1[d]] += eff[(size_t)d];
                j++;
            }
            for (size_t t = i; t < j; t++) {
                const int c = B.order[t];
                win[(size_t)c] = total - card[ct.c0[c]] - card[ct.c1[c]];
                tie[(size_t)c] = gt - gcard[ct.c0[c]] - gcard[ct.c1[c]] + eff[(size_t)c];
            }
            for (size_t t = i; t < j; t++) {
                const int d = B.order[t];
                card[ct.c0[d]] += eff[(size_t)d];
                card[ct.c1[d]] += eff[(size_t)d];
                gcard[ct.c0[d]] = 0.0;
                gcard[ct.c1[d]] = 0.0;
            }
            total += gt;
            i = j;
        }
        for (int c : B.order) {
            const double all = total - card[ct.c0[c]] - card[ct.c1[c]] + eff[(size_t)c];
            const double lose = all - win[(size_t)c] - tie[(size_t)c];
            v[(size_t)c] = m * (win[(size_t)c] - lose);
        }
        return v;
    }

    // after action a at st: the child's values, averaged over the river card when the action ends the
    // turn (a new river node, or an all-in showdown run out)
    std::vector<double> ex_after(const HandState& st, int a, const NodeActions& na, const PathHash& ph, int k, bool child_path,
                                 const std::vector<double>& ro, int nto, int pk, const HistHash& hh, bool br, const Ex& ex) const {
        HandState child(st);
        child.apply(na.type[a], na.amount[a]);
        PathHash ch = ph;
        ch.step(a);
        const bool to_river = st.n_board == 4 && (child.terminal ? child.n_active() > 1 : child.street == RIVER);
        if (!to_river) return ex_walk(child, ch, k + 1, child_path, ro, nto, pk, hh, br, ex);
        const ComboTable& ct = combo_table();
        std::vector<int> cards;
        {
            bool on[52] = {false};
            for (int i = 0; i < 4; i++) on[st.board[i]] = true;
            for (int r = 0; r < 52; r++) if (!on[r]) cards.push_back(r);
        }
        // the river cards' subtrees are independent: on the search's threads, summed in card order
        std::vector<std::vector<double>> per((size_t)cards.size());
        std::atomic<size_t> next{0};
        std::string err;
        std::mutex err_mu;
        auto work = [&]() {
            try {
                std::vector<double> ror(ro.size());
                for (size_t i = next.fetch_add(1); i < cards.size(); i = next.fetch_add(1)) {
                    const int r = cards[i];
                    for (int t = 0; t < nto; t++)
                        for (int d = 0; d < N_COMBOS; d++)
                            ror[(size_t)t * N_COMBOS + d] = (ct.c0[d] == r || ct.c1[d] == r) ? 0.0 : ro[(size_t)t * N_COMBOS + d];
                    HandState cr(child);
                    cr.board[4] = r;
                    if (cr.terminal) per[i] = ex_terminal(cr, ror, nto, r, ex);
                    else if (ex.leaf_river) per[i] = ex_leaf(cr, ch, ror, nto, br, ex);
                    else per[i] = ex_walk(cr, ch, k + 1, false, ror, nto, pk, hh, br, ex);
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                err = e.what();
            }
        };
        const int T = std::max(1, std::min(params_.threads, (int)cards.size()));
        std::vector<std::thread> pool;
        for (int t = 1; t < T; t++) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
        if (!err.empty()) throw std::runtime_error(err);
        // the leaf game's best responder: per river card its values under each continuation, chosen
        // (the best one per hole) only after the average over the river card it had not seen
        const int nt = ex.leaf_game && br && ex.leaf_river && !child.terminal ? N_CONTINUATIONS : 1;
        std::vector<double> v((size_t)nt * N_COMBOS, 0.0);
        for (size_t i = 0; i < cards.size(); i++) {
            const int r = cards[i];
            if (per[i].size() != v.size()) throw std::runtime_error("ex_after: river values of another shape");
            for (int t = 0; t < nt; t++)
                for (int c = 0; c < N_COMBOS; c++)
                    if (ct.c0[c] != r && ct.c1[c] != r) v[(size_t)t * N_COMBOS + c] += per[i][(size_t)t * N_COMBOS + c];
        }
        for (double& x : v) x /= 44.0;  // 52 - 4 board cards - 4 hole cards
        if (nt > 1) {
            for (int c = 0; c < N_COMBOS; c++) {
                double best = v[(size_t)c];
                for (int t = 1; t < nt; t++) best = std::max(best, v[(size_t)t * N_COMBOS + c]);
                v[(size_t)c] = best;
            }
            v.resize((size_t)N_COMBOS);
        }
        return v;
    }

    // the leaf at the start of the river: each player's continuation mix (o: 4 types of its reach;
    // p in value mode: the mix of its 4 values; p best-responding: no continuation)
    std::vector<double> ex_leaf(HandState& st, const PathHash& ph, const std::vector<double>& ro, int nto, bool br, const Ex& ex) const {
        if (nto != 1) throw std::runtime_error("ex_leaf: a second leaf");
        std::vector<double> ro4((size_t)N_CONTINUATIONS * N_COMBOS, 0.0);
        double w[MAX_ACTIONS];
        for (int d = 0; d < N_COMBOS; d++) {
            if (ro[(size_t)d] == 0.0) continue;
            ex_leaf_mix(ex.o, d, ph, ex, w);
            for (int t = 0; t < N_CONTINUATIONS; t++) ro4[(size_t)t * N_COMBOS + d] = ro[(size_t)d] * w[t];
        }
        HistHash hh;
        std::string tok;
        hh.catch_up(st, game_->grid, tok);
        if (br && !ex.leaf_game) return ex_walk(st, ph, 1 << 20, false, ro4, N_CONTINUATIONS, -1, hh, true, ex);
        std::vector<double> v((size_t)N_COMBOS, 0.0);
        std::vector<std::vector<double>> vk((size_t)N_CONTINUATIONS);
        for (int t = 0; t < N_CONTINUATIONS; t++) vk[(size_t)t] = ex_walk(st, ph, 1 << 20, false, ro4, N_CONTINUATIONS, t, hh, false, ex);
        if (br) {  // the leaf game: p's values per continuation (ex_after takes the best one after the river card)
            v.assign((size_t)N_CONTINUATIONS * N_COMBOS, 0.0);
            for (int t = 0; t < N_CONTINUATIONS; t++) std::copy(vk[(size_t)t].begin(), vk[(size_t)t].end(), v.begin() + (size_t)t * N_COMBOS);
            return v;
        }
        for (int c = 0; c < N_COMBOS; c++) {
            if (class_of_[(size_t)c] < 0) continue;
            ex_leaf_mix(ex.p, c, ph, ex, w);
            for (int t = 0; t < N_CONTINUATIONS; t++) v[(size_t)c] += w[t] * vk[(size_t)t][(size_t)c];
        }
        return v;
    }

    // values of p's combos; ro: o's reach (nto types x 1326; types = o's continuations after a leaf);
    // pk: p's continuation after a leaf (-1: none); br: p best-responds on streets >= ex.br_street
    std::vector<double> ex_walk(HandState& st, const PathHash& ph, int k, bool on_path, const std::vector<double>& ro, int nto, int pk,
                                HistHash hh, bool br, const Ex& ex) const {
        if (st.terminal) return ex_terminal(st, ro, nto, st.n_board == 5 ? (root_.n_board == 5 ? 52 : st.board[4]) : 52, ex);
        const int seat = st.to_act;
        const bool path_node = on_path && k < (int)path_.size();
        NodeActions na;
        build_actions(st, observe(st, seat), path_node ? &path_[(size_t)k] : nullptr, na);
        std::string tok;
        hh.catch_up(st, game_->grid, tok);
        std::vector<double> v((size_t)N_COMBOS, 0.0);
        if (seat == ex.o) {
            std::vector<std::vector<double>> rows((size_t)nto);
            for (int t = 0; t < nto; t++) ex_rows(st, ph, k, path_node, na, nto > 1 ? t : -1, hh, ex, rows[(size_t)t]);
            std::vector<double> r2(ro.size());
            for (int a = 0; a < na.n; a++) {
                for (int t = 0; t < nto; t++)
                    for (int d = 0; d < N_COMBOS; d++)
                        r2[(size_t)t * N_COMBOS + d] = ro[(size_t)t * N_COMBOS + d] * rows[(size_t)t][(size_t)d * MAX_ACTIONS + a];
                const std::vector<double> va = ex_after(st, a, na, ph, k, path_node && a == path_[(size_t)k].index, r2, nto, pk, hh, br, ex);
                for (int c = 0; c < N_COMBOS; c++) v[(size_t)c] += va[(size_t)c];
            }
            return v;
        }
        std::vector<std::vector<double>> vs((size_t)na.n);
        for (int a = 0; a < na.n; a++) vs[(size_t)a] = ex_after(st, a, na, ph, k, path_node && a == path_[(size_t)k].index, ro, nto, pk, hh, br, ex);
        if (br && st.street >= ex.br_street) {  // the best responder knows its hole: the best action per combo
            for (int c = 0; c < N_COMBOS; c++) {
                double best = vs[0][(size_t)c];
                for (int a = 1; a < na.n; a++) best = std::max(best, vs[(size_t)a][(size_t)c]);
                v[(size_t)c] = best;
            }
            return v;
        }
        std::vector<double> rows;
        ex_rows(st, ph, k, path_node, na, pk, hh, ex, rows);
        for (int c = 0; c < N_COMBOS; c++)
            for (int a = 0; a < na.n; a++) v[(size_t)c] += rows[(size_t)c * MAX_ACTIONS + a] * vs[(size_t)a][(size_t)c];
        return v;
    }

    struct Ctx {
        FastRng rng;
        int tid = 0;
        int deck[52];
        int rdeck[52];  // a rollout's deck (the rest of the board redrawn)
        int holes[MAX_PLAYERS][2];
        int combo[MAX_PLAYERS];
        int bucket_memo[MAX_PLAYERS][6];
        bool actual = false;
        std::string tok;
        long long iterations = 0, traversals = 0, focused = 0, nodes = 0, forced = 0, redeals = 0;
        long long leaves = 0, leaf_evals = 0, rollouts = 0, rollout_steps = 0;
        double ours[MAX_ACTIONS] = {0.0};  // sum over this thread's iterations of weight x our decision's strategy
    };

    // a leaf: who chooses a continuation, what of the board they saw, the blueprint history there
    struct LeafCtx {
        int ch[MAX_PLAYERS];
        int nch = 0;
        int visible = 0;
        HistHash hh;
    };

    // river buckets of every board the subgame can reach from a flop or turn root, computed at once
    // (Bucketer::river_buckets_all; empty when the abstraction has no batch)
    struct RiverTable {
        int base = -1;                // the root board's size (3 or 4)
        std::vector<int32_t> slot;    // 52 x 52: base 4 the fifth card, base 3 the smaller x 52 + the larger extra card
        std::vector<uint8_t> b;       // boards x 1326
    };
    RiverTable river_;
    double river_seconds_ = 0.0;

    // flop roots: turn buckets of every combo per turn card, filled on first use from the bucketer
    // (its cache when warm): a lookup instead of a canonical form and a cache probe per rollout step
    struct TurnTable {
        bool on = false;
        std::vector<uint8_t> b;                       // 52 x 1326
        std::unique_ptr<std::once_flag[]> once;       // per turn card
    };
    mutable TurnTable turn_;

    void init_turn_table() {
        turn_ = TurnTable();
        if (root_.n_board != 3 || game_->spec.max_street < TURN) return;
        turn_.b.assign((size_t)52 * N_COMBOS, 255);
        turn_.once.reset(new std::once_flag[52]);
        turn_.on = true;
    }
    int turn_bucket(const int* hole, const int* board4) const {
        const int t = board4[3];
        std::call_once(turn_.once[(size_t)t], [&]() {
            const ComboTable& ct = combo_table();
            bool on[52] = {false};
            for (int i = 0; i < 4; i++) on[board4[i]] = true;
            uint8_t* row = &turn_.b[(size_t)t * N_COMBOS];
            for (int c = 0; c < N_COMBOS; c++) {
                if (on[ct.c0[c]] || on[ct.c1[c]]) continue;
                const int h[2] = {ct.c0[c], ct.c1[c]};
                row[c] = (uint8_t)game_->bucketer->bucket(h, board4, 4);
            }
        });
        return turn_.b[(size_t)t * N_COMBOS + (size_t)combo_index(hole[0], hole[1])];
    }
    int limit_street_ = RIVER;   // the last street searched; the start of the next one is a leaf
    int raise_limit_ = 0;        // > 0: right after this many raises on the root street is a leaf too
    std::vector<int> grid_to_bp_;  // grid action id -> index in the blueprint's names (-1 unknown)

    void set_depth_limits() {
        limit_street_ = RIVER;
        raise_limit_ = 0;
        const int d = params_.depth;
        int in_hand = 0;
        for (int s = 0; s < root_.n; s++) if (!root_.players[s].folded) in_hand++;
        if (d == DEPTH_END) return;
        if (d == DEPTH_PLURIBUS) {
            if (root_street_ == PREFLOP) {
                limit_street_ = PREFLOP;
            } else if (root_street_ == FLOP && in_hand > 2) {
                limit_street_ = FLOP;
                raise_limit_ = 2;
            }
        } else if (d == DEPTH_HU_FLOP) {
            if (root_street_ <= FLOP) limit_street_ = root_street_;
        } else if (d == DEPTH_NEXT_STREET) {
            if (root_street_ < RIVER) limit_street_ = root_street_;
        } else {
            throw std::invalid_argument("depth: 0 end, 1 pluribus, 2 hu_flop_limit, 3 next_street");
        }
    }

    // a non-terminal state where the subgame stops: past the last street searched, or (Pluribus,
    // more than two players on the flop) right after the second raise; nodes of the real path up to
    // our decision never are
    bool is_leaf(const HandState& st, int k, bool on_path) const {
        if (st.street > limit_street_) return true;
        return raise_limit_ > 0 && st.raises_this_street >= raise_limit_ && !(on_path && k <= (int)path_.size());
    }

    void build_river_table() {
        const auto t0 = std::chrono::steady_clock::now();
        river_ = RiverTable();
        const int base = root_.n_board;
        if (base != 3 && base != 4) return;
        if (game_->spec.max_street < RIVER) return;  // no river decisions, in the subgame or in rollouts
        std::vector<std::pair<int, int>> extras;
        bool on[52] = {false};
        for (int i = 0; i < base; i++) on[root_.board[i]] = true;
        if (base == 4) {
            for (int r = 0; r < 52; r++) if (!on[r]) extras.emplace_back(-1, r);
        } else {
            for (int t = 0; t < 52; t++)
                for (int r = t + 1; r < 52; r++)
                    if (!on[t] && !on[r]) extras.emplace_back(t, r);
        }
        std::vector<uint8_t> b(extras.size() * (size_t)N_COMBOS, 255);
        std::atomic<size_t> next{0};
        std::atomic<bool> unsupported{false};
        auto work = [&]() {
            int board5[5];
            for (int i = 0; i < base; i++) board5[i] = root_.board[i];
            for (size_t i = next.fetch_add(1); i < extras.size() && !unsupported.load(); i = next.fetch_add(1)) {
                if (base == 4) {
                    board5[4] = extras[i].second;
                } else {
                    board5[3] = extras[i].first;
                    board5[4] = extras[i].second;
                }
                if (!game_->bucketer->river_buckets_all(board5, combo_table().idx, &b[i * (size_t)N_COMBOS])) unsupported.store(true);
            }
        };
        const int T = std::max(1, std::min(params_.threads, 64));
        std::vector<std::thread> pool;
        for (int t = 1; t < T; t++) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
        if (unsupported.load()) return;
        river_.base = base;
        river_.slot.assign(52 * 52, -1);
        for (size_t i = 0; i < extras.size(); i++) {
            const int key = base == 4 ? extras[i].second : extras[i].first * 52 + extras[i].second;
            river_.slot[(size_t)key] = (int32_t)i;
        }
        river_.b = std::move(b);
        river_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // bucket of `hole` on a board past the root's street: the river table on the river when it
    // exists, else the bucketer (and its cache)
    int later_bucket(const int* hole, const int* board, int n_board) const {
        if (n_board == 5 && river_.base > 0) {
            int key;
            if (river_.base == 4) {
                key = board[4];
            } else {
                const int t = std::min(board[3], board[4]), r = std::max(board[3], board[4]);
                key = t * 52 + r;
            }
            const int32_t i = river_.slot[(size_t)key];
            if (i >= 0) return river_.b[(size_t)i * N_COMBOS + (size_t)combo_index(hole[0], hole[1])];
        }
        if (n_board == 4 && turn_.on) return turn_bucket(hole, board);
        return game_->bucketer->bucket(hole, board, n_board);
    }

    // the blueprint as the agent plays it at `st` for its actor (bucket `bucket`, blueprint history
    // `hh`), then the continuation `choice`; unknown keys play check / call, as BlueprintAgent does
    void rollout_probs(const HandState& st, const HistHash& hh, int bucket, const NodeActions& na, int choice, double* p) const {
        const int seat = st.to_act;
        bool have = false;
        int kind[MAX_ACTIONS];
        for (int a = 0; a < na.n; a++) kind[a] = na.id[a] == 0 ? 0 : na.id[a] == 1 ? 1 : 2;
        if (game_->blueprint) {
            const BlueprintTable& bp = *game_->blueprint;
            const int rel = ((seat - st.button) % st.n + st.n) % st.n;
            const long long i = bp.find(node_key(st.street, rel, st.n_active(), bucket, hh));
            if (i >= 0) {
                int legal[MAX_ACTIONS];
                for (int a = 0; a < na.n; a++) legal[a] = grid_to_bp_[na.id[a]];
                if (!game_->presampled.empty()) {  // the action drawn in advance for this infoset and continuation
                    const int name = bp.ids[bp.off[(size_t)i] + game_->presampled[(size_t)i * N_CONTINUATIONS + (size_t)choice]];
                    for (int a = 0; a < na.n; a++) {
                        if (legal[a] == name) {
                            for (int b = 0; b < na.n; b++) p[b] = b == a ? 1.0 : 0.0;
                            return;
                        }
                    }
                }
                have = bp.policy_at(i, legal, na.n, p);
            }
        }
        if (!have) {
            int calls = 0;
            for (int a = 0; a < na.n; a++) calls += kind[a] == 1;
            for (int a = 0; a < na.n; a++) p[a] = calls ? (kind[a] == 1 ? 1.0 : 0.0) : 1.0 / na.n;
        }
        apply_continuation(kind, na.n, choice, params_.bias, p);
    }

    // one rollout from a leaf: the board cards the choosers did not see redrawn, then every
    // player plays its chosen continuation to the end of the hand
    double rollout(const HandState& leaf, const LeafCtx& L, const int* choice, int traverser, Ctx& ctx) const {
        HandState s(leaf);
        uint64_t used = 0;
        for (int q = 0; q < s.n; q++)
            if (!s.players[q].folded) used |= (1ULL << s.players[q].hole[0]) | (1ULL << s.players[q].hole[1]);
        for (int i = 0; i < L.visible; i++) used |= 1ULL << s.board[i];
        for (int i = L.visible; i < s.n_board; i++) {
            int c;
            do c = ctx.rng.below(52); while (used >> c & 1);
            used |= 1ULL << c;
            s.board[i] = c;
        }
        std::memcpy(ctx.rdeck, s.deck, sizeof ctx.rdeck);
        for (int j = 0; j < 5 - s.n_board; j++) {
            int c;
            do c = ctx.rng.below(52); while (used >> c & 1);
            used |= 1ULL << c;
            ctx.rdeck[s.deck_pos + j] = c;
        }
        s.deck = ctx.rdeck;
        HistHash hh = L.hh;
        int memo[MAX_PLAYERS][6];
        for (int q = 0; q < MAX_PLAYERS; q++)
            for (int j = 0; j < 6; j++) memo[q][j] = -1;
        while (!s.terminal) {
            const int seat = s.to_act;
            NodeActions na;
            build_actions(s, observe(s, seat), nullptr, na);
            hh.catch_up(s, game_->grid, ctx.tok);
            int& b = memo[seat][s.n_board];
            if (b < 0) b = s.n_board == 0 ? game_->bucketer->bucket(s.players[seat].hole, s.board, 0)
                                          : later_bucket(s.players[seat].hole, s.board, s.n_board);
            double p[MAX_ACTIONS];
            rollout_probs(s, hh, b, na, choice[seat], p);
            const int a = sample(p, na.n, ctx.rng);
            s.apply(na.type[a], na.amount[a]);
            ctx.rollout_steps++;
        }
        return (double)s.net(traverser) / (double)game_->spec.bb;
    }

    mutable std::mutex log_mu_;
    std::vector<LeafLogEntry> leaf_log_;

    void log_leaf(const HandState& st, const PathHash& ph, int p, const NodeKey& key, const Ctx& ctx) {
        std::lock_guard<std::mutex> lk(log_mu_);
        if ((int)leaf_log_.size() >= params_.debug_leaves) return;
        LeafLogEntry e;
        e.seat = p;
        e.combo = ctx.combo[p];
        e.cls = class_of_[(size_t)ctx.combo[p]];
        e.n_board = st.n_board;
        e.k1 = key.k1;
        e.k2 = key.k2;
        e.path = ph.a;
        for (int j = 0; j < st.n_board; j++) e.board[j] = st.board[j];
        for (int q = 0; q < st.n; q++) e.combos[q] = ctx.combo[q];
        leaf_log_.push_back(e);
    }

    void dfs_leaves(HandState& st, int k, bool on_path, std::vector<std::pair<int, int>>& acts, std::vector<LeafInfo>& out,
                    long long max_nodes, long long& terminals, long long& decisions) const {
        if (terminals + decisions + (long long)out.size() > max_nodes) throw std::runtime_error("leaves: more than max_nodes nodes");
        if (st.terminal) {
            terminals++;
            return;
        }
        if (is_leaf(st, k, on_path)) {
            LeafInfo li;
            li.actions = acts;
            li.street = st.street;
            li.raises = st.raises_this_street;
            li.n_board = st.n_board;
            li.reason = st.street > limit_street_ ? 0 : 1;
            for (int s = 0; s < st.n; s++) li.choosers += st.players[s].can_act() ? 1 : 0;
            out.push_back(std::move(li));
            return;
        }
        decisions++;
        const bool path_node = on_path && k < (int)path_.size();
        NodeActions na;
        build_actions(st, observe(st, st.to_act), path_node ? &path_[(size_t)k] : nullptr, na);
        for (int a = 0; a < na.n; a++) {
            HandState child(st);
            child.apply(na.type[a], na.amount[a]);
            acts.emplace_back(na.type[a], na.amount[a]);
            dfs_leaves(child, k + 1, path_node && a == path_[(size_t)k].index, acts, out, max_nodes, terminals, decisions);
            acts.pop_back();
        }
    }

public:
    // tests: every leaf of the public tree under the depth rule (the base deal's cards)
    std::vector<LeafInfo> leaf_list(long long max_nodes, long long& terminals, long long& decisions) const {
        std::vector<LeafInfo> out;
        HandState st(root_);
        st.deck = deck_;
        std::vector<std::pair<int, int>> acts;
        terminals = decisions = 0;
        dfs_leaves(st, 0, true, acts, out, max_nodes, terminals, decisions);
        return out;
    }
    // tests: the rollout policy at the state reached by `actions` from the root, its actor holding
    // (h0, h1), continuation `choice`; `ids` gets the actions' grid ids
    std::vector<double> rollout_policy(const std::vector<std::pair<int, int>>& actions, int h0, int h1, int choice, std::vector<int>& ids) const {
        HandState st(root_);
        st.deck = deck_;
        for (const auto& a : actions) {
            if (st.terminal) throw std::invalid_argument("rollout_policy: the hand ended");
            st.apply(a.first, a.first == RAISE ? a.second : 0);
        }
        if (st.terminal) throw std::invalid_argument("rollout_policy: the hand ended");
        const int seat = st.to_act;
        st.players[seat].hole[0] = h0;
        st.players[seat].hole[1] = h1;
        NodeActions na;
        build_actions(st, observe(st, seat), nullptr, na);
        HistHash hh;
        std::string tok;
        hh.catch_up(st, game_->grid, tok);
        const int b = st.n_board == 0 ? game_->bucketer->bucket(st.players[seat].hole, st.board, 0)
                                      : later_bucket(st.players[seat].hole, st.board, st.n_board);
        double p[MAX_ACTIONS];
        rollout_probs(st, hh, b, na, choice, p);
        ids.assign(na.id, na.id + na.n);
        return std::vector<double>(p, p + na.n);
    }
    std::vector<LeafLogEntry> leaf_log() const {
        std::lock_guard<std::mutex> lk(log_mu_);
        return leaf_log_;
    }
    // tests: the bucket the search uses for a hole on a board extending the root's (tables or bucketer)
    int later_bucket_of(int h0, int h1, const std::vector<int>& board) const {
        const int hole[2] = {h0, h1};
        return later_bucket(hole, board.data(), (int)board.size());
    }

private:
    double leaf_value(const HandState& st, const PathHash& ph, int traverser, double weight, double w_imp, bool focused, Ctx& ctx) {
        ctx.leaves++;
        LeafCtx L;
        for (int s = 0; s < st.n; s++) if (st.players[s].can_act()) L.ch[L.nch++] = s;
        L.visible = st.street > limit_street_ ? board_cards_by_street(limit_street_) : st.n_board;
        L.hh.catch_up(st, game_->grid, ctx.tok);
        int choice[MAX_PLAYERS] = {0};
        return choose(st, ph, L, 0, choice, traverser, weight, w_imp, focused, ctx);
    }

    // each chooser in seat order picks a continuation at its infoset of the leaf (its hole's class
    // on the root street and the public path: the cards dealt after the root round and the other
    // players' holes are not part of it); the traverser tries all four on the same random numbers
    double choose(const HandState& st, const PathHash& ph, const LeafCtx& L, int i, int* choice, int traverser, double weight,
                  double w_imp, bool focused, Ctx& ctx) {
        if (i == L.nch) {
            double tot = 0.0;
            for (int r = 0; r < params_.rollouts; r++) tot += rollout(st, L, choice, traverser, ctx);
            ctx.leaf_evals++;
            ctx.rollouts += params_.rollouts;
            return tot / params_.rollouts;
        }
        const int p = L.ch[i];
        static const uint8_t ids[N_CONTINUATIONS] = {0, 1, 2, 3};
        const NodeKey key = search_key(LEAF_STREET, p, class_of_[(size_t)ctx.combo[p]], ph);
        Node* node = table_->get_or_create(key, ctx.tid, N_CONTINUATIONS, [&](Node& nd, NodeArena&) {
                                               nd.init(ids, N_CONTINUATIONS);
                                               return "";
                                           }).node;
        ctx.nodes++;
        if (params_.debug_leaves > 0) log_leaf(st, ph, p, key, ctx);
        double sigma[MAX_ACTIONS];
        node->lock.lock();
        node->current_strategy(sigma);
        node->lock.unlock();
        if (p == traverser) {
            double u[N_CONTINUATIONS];
            const FastRng saved = ctx.rng;
            FastRng after = ctx.rng;
            for (int k = 0; k < N_CONTINUATIONS; k++) {
                ctx.rng = saved;  // common random numbers: the four continuations meet the same cards
                choice[p] = k;
                u[k] = choose(st, ph, L, i + 1, choice, traverser, weight, w_imp, focused, ctx);
                if (k == N_CONTINUATIONS - 1) after = ctx.rng;
            }
            ctx.rng = after;
            double v = 0.0;
            for (int k = 0; k < N_CONTINUATIONS; k++) v += sigma[k] * u[k];
            const double w = weight * w_imp;
            node->lock.lock();
            for (int k = 0; k < N_CONTINUATIONS; k++) node->regret()[k] += w * (u[k] - v);
            node->lock.unlock();
            return v;
        }
        if (!focused) {
            node->lock.lock();
            for (int k = 0; k < N_CONTINUATIONS; k++) node->strategy_sum()[k] += weight * sigma[k];
            node->visits += 1;
            node->lock.unlock();
        }
        choice[p] = sample(sigma, N_CONTINUATIONS, ctx.rng);
        return choose(st, ph, L, i + 1, choice, traverser, weight, w_imp, focused, ctx);
    }

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
    const SubgameSearch* frozen_ = nullptr;  // measurement: the root's round played as this search says
    int frozen_kind_ = 0;

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
        if (m < 0) m = later_bucket(st.players[seat].hole, st.board, st.n_board);
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
        if (is_leaf(st, k, on_path)) return leaf_value(st, ph, traverser, weight, w_imp, focused, ctx);
        const int seat = st.to_act;
        const Obs obs = observe(st, seat);
        const bool path_node = on_path && k < (int)path_.size();
        NodeActions na;
        build_actions(st, obs, path_node ? &path_[(size_t)k] : nullptr, na);
        const NodeKey key = search_key(st.street, seat, card_part(st, seat, ctx), ph);
        const bool frozen = frozen_ != nullptr && st.street == root_street_;  // played as another search says, not learned
        Node* node = frozen ? nullptr : table_->get_or_create(key, ctx.tid, na.n, [&](Node& nd, NodeArena&) {
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
        if (frozen) {
            frozen_->strategy_row(key, na.n, frozen_kind_, sigma);
        } else {
            node->lock.lock();
            node->current_strategy(sigma);
            node->lock.unlock();
        }
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
            if (frozen) return u;
            const double w = weight * w_imp;
            node->lock.lock();
            for (int i = 0; i < na.n; i++) node->regret()[i] += w * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        if (!focused && !frozen) {
            node->lock.lock();
            for (int i = 0; i < na.n; i++) node->strategy_sum()[i] += weight * sigma[i];
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
