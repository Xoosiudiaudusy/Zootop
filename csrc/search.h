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
//   cards    lossless on the current round (infoset = canonical form of hole + board); later
//            rounds on the subgame's own abstraction (SearchGame::search_bucketer: the blueprint's
//            buckets unless another bucketer is given, e.g. finer ones, as Pluribus' 500 per round)
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

#if defined(_MSC_VER)
#define NEGP_NOINLINE __declspec(noinline)
#else
#define NEGP_NOINLINE __attribute__((noinline))
#endif

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
// The bucket tables of a search root's board: the river buckets of every board the subgame can reach
// from a flop or turn root (Bucketer::river_buckets_all, built at once; base -1 when the abstraction has
// no batch), and, for a flop root, the turn buckets of every combo per turn card (filled on first use
// from the bucketer).  They depend only on the bucketer and the board, so SearchGame keeps those of its
// last boards, one set per abstraction (the blueprint's, the subgame's own), and every search on one of
// those boards shares them (a lookup instead of a rebuild: the agent searches a board several times,
// and every flop search used to rebuild 1,176 river boards).
struct SearchBoardTables {
    std::vector<int> board;                       // the root board, in the order given
    std::once_flag river_once;
    int river_base = -1;                          // the root board's size (3 or 4); -1: no river table
    std::vector<int32_t> river_slot;              // 52 x 52: base 4 the fifth card, base 3 the smaller x 52 + the larger extra card
    std::vector<uint8_t> river_b;                 // boards x 1326
    std::once_flag turn_init;
    bool turn_on = false;
    std::vector<uint8_t> turn_b;                  // 52 x 1326 (flop roots)
    std::unique_ptr<std::once_flag[]> turn_once;  // per turn card
};

struct SearchGame {
    Spec spec;
    BetGrid grid;
    // the blueprint's card abstraction: the keys of every blueprint lookup (the ranges at the root,
    // rollouts from leaves, the blueprint rows of the evaluator) and the preflop classes
    std::shared_ptr<Bucketer> bucketer;
    // the subgame's own abstraction for the rounds after the root's (its infosets there and their bucket
    // tables): the blueprint's bucketer itself unless another one is given
    std::shared_ptr<Bucketer> search_bucketer;
    std::shared_ptr<const BlueprintTable> blueprint;  // null: no prior (uniform ranges)
    // optional (Pluribus' compression of the continuations): per blueprint record and continuation,
    // the index among the record's actions of one action drawn in advance; empty: off
    std::vector<uint8_t> presampled;

    // the subgame has an abstraction of its own after the root's round
    bool separate_buckets() const { return search_bucketer.get() != bucketer.get(); }

    // the bucket tables of the last boards searched (SearchBoardTables), one set per abstraction:
    // `search` asks for the subgame's own, which is the blueprint's set when both are one bucketer;
    // tables_keep boards are kept in each
    size_t tables_keep = 8;
    std::shared_ptr<SearchBoardTables> board_tables(const std::vector<int>& board, bool search = false) const {
        std::lock_guard<std::mutex> lk(tables_mu_);
        std::vector<std::shared_ptr<SearchBoardTables>>& tables = search && separate_buckets() ? search_tables_ : tables_;
        for (size_t i = 0; i < tables.size(); i++)
            if (tables[i]->board == board) {
                std::shared_ptr<SearchBoardTables> t = tables[i];
                tables.erase(tables.begin() + (long)i);
                tables.push_back(t);  // most recent last
                return t;
            }
        std::shared_ptr<SearchBoardTables> t = std::make_shared<SearchBoardTables>();
        t->board = board;
        tables.push_back(t);
        while (tables.size() > std::max<size_t>(1, tables_keep)) tables.erase(tables.begin());
        return t;
    }
    void clear_board_tables() const {
        std::lock_guard<std::mutex> lk(tables_mu_);
        tables_.clear();
        search_tables_.clear();
    }

    SearchGame(const Spec& s, std::shared_ptr<Bucketer> bk, std::shared_ptr<const BlueprintTable> bp,
               std::shared_ptr<Bucketer> search_bk = nullptr)
        : spec(s), grid(s.grid()), bucketer(std::move(bk)), blueprint(std::move(bp)) {
        if (!bucketer) throw std::invalid_argument("a bucketer is needed");
        search_bucketer = search_bk ? std::move(search_bk) : bucketer;
        if (spec.max_street > PREFLOP && !bucketer->fitted()) throw std::invalid_argument("this spec bets postflop: pass a fitted bucketer");
        if (separate_buckets()) {
            if (spec.max_street > PREFLOP && !search_bucketer->fitted()) throw std::invalid_argument("the search bucketer is not fitted");
            if (search_bucketer->identity().n_buckets > 255) throw std::invalid_argument("the search's bucket tables hold at most 255 buckets");
        }
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

private:
    mutable std::mutex tables_mu_;
    mutable std::vector<std::shared_ptr<SearchBoardTables>> tables_;         // the blueprint's abstraction
    mutable std::vector<std::shared_ptr<SearchBoardTables>> search_tables_;  // the subgame's own (separate_buckets())
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
    bool legacy_traverse = false;  // tests: traverse with the engine at every node instead of the public tree (same numbers)
    // vector Linear CFR (Pluribus's search for small subgames): every iteration walks the public tree
    // with the reach of all 1326 holes of both players (card removal), one river card sampled per
    // iteration on a turn root; the same infosets (class on the root's round, the subgame's bucket on the
    // river: SearchGame::search_bucketer, the blueprint's unless another is given),
    // table, outputs and exploitability as the MCCFR.  Only 2 live players, a turn or river root, no
    // leaves, no frozen round; anywhere else the MCCFR runs.  Off by default (the MCCFR, bit for bit).
    bool vector_cfr = false;
    // vector CFR on a turn root: river infosets without the subgame's buckets, a hole's class on a river board
    // being its hand strength there (equal strength, one class); rows per river card, kept as floats in the
    // public tree (never in the search's table).  Off: the vector CFR's numbers unchanged.
    bool river_exact = false;
    // vector CFR on a turn root, river_exact off: > 0, the river's infosets by K buckets of hand strength on the
    // river board (the bucket of a hole = the quantile of its strength among the board's holes, equal strengths
    // together), shared by all river cards as the subgame's buckets are (Pluribus: 500 per round); 0: the
    // subgame's buckets (search_bucketer, the blueprint's unless another is given; the vector CFR's numbers unchanged)
    int river_buckets = 0;
    // vector CFR's regret / average weighting: 0 Linear CFR (Pluribus; the numbers unchanged), 1 CFR+ (regrets
    // floored at 0, linear average), 2 DCFR(1.5, 0, 2) (Brown & Sandholm 2019: positive regrets x t^1.5/(t^1.5+1),
    // negative x 1/2 after each iteration, the average weighted by t^2); the MCCFR is always Linear
    int vector_discount = 0;
    // river_exact with river_buckets K > 0: warm start (Brown & Sandholm 2016, from the solution of a coarser
    // abstraction).  The first `river_warm` share of the budget (time, or iterations if given) learns the river by
    // the K shared strength buckets; then every exact river row starts from its bucket row's average strategy, its
    // regrets and strategy sums set as if that strategy had been played the T_w iterations so far (Linear weights
    // 1..T_w), and learns exactly from there.  0: off
    double river_warm = 0.0;
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
        bt_ = game_->board_tables(hand_.board);
        sbt_ = game_->board_tables(hand_.board, true);  // bt_ itself when the subgame uses the blueprint's abstraction
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
    // boards in the river bucket table of the subgame's own abstraction (0: none, e.g. a bucketer that answers
    // from precomputed bucket tables, which the search then reads directly) and of the blueprint's
    int river_table_boards() const { return (int)(sbt_->river_b.size() / N_COMBOS); }
    int blueprint_river_table_boards() const { return (int)(bt_->river_b.size() / N_COMBOS); }
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
        if (!frozen_ && !params_.legacy_traverse) build_troot();  // the public tree of this solve (its node caches point into table_)
        // our decision (our actual hole's class at the end of the real path): its strategy is added up
        // once per iteration, which is the average our hole plays (its own reach there is fixed)
        NodeActions our_na;
        build_actions(current_, observe(current_, hand_.our_seat), nullptr, our_na);
        PathHash our_ph;
        for (const PathStep& s : path_) our_ph.step(s.index);
        const NodeKey our_key = search_key(root_street_, hand_.our_seat, class_of_[our_combo_], our_ph);
        const bool use_vector = vector_eligible();
        vexact_ = false;
        vriver_k_ = 0;
        vriver_nodes_.clear();
        if (use_vector) prepare_vector();
        group_->begin(T);
        std::mutex err_mu;
        std::string err;
        auto work = [&](int tid) {
            Ctx& ctx = ctxs[(size_t)tid];
            ctx.tid = tid;
            ctx.rng.reseed(params_.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)tid * 0xD1B54A32D192ED03ULL + 1);
            std::memcpy(ctx.deck, deck_, sizeof deck_);
            try {
                if (use_vector) {
                    vector_loop(ctx, next_t, stop, deadline, our_key, our_na.n);
                } else
                while (!stop.load(std::memory_order_relaxed)) {
                    const long long t = next_t.fetch_add(1);
                    if (params_.iterations > 0 && t > params_.iterations) break;
                    const double weight = params_.linear ? (double)t : 1.0;
                    for (int trav : traversers) {
                        const bool focused = trav == hand_.our_seat && ctx.rng.uniform() < params_.focus;
                        deal(ctx, focused);
                        if (frozen_ || params_.legacy_traverse) {
                            HandState st(root_);
                            st.deck = ctx.deck;
                            for (int s = 0; s < st.n; s++) {
                                st.players[s].hole[0] = ctx.holes[s][0];
                                st.players[s].hole[1] = ctx.holes[s][1];
                            }
                            traverse(st, PathHash(), 0, true, trav, weight, 1.0, focused, ctx);
                        } else {
                            traverse_tree(troot_, trav, weight, 1.0, focused, ctx);
                        }
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
            // no exception may leave while a worker runs (a joinable std::thread destroyed: std::terminate):
            // a thread that cannot be started stops the solve, its slot leaves the group, the others are joined
            std::vector<std::thread> pool;
            try {
                pool.reserve((size_t)T - 1);
                for (int t = 1; t < T; t++) pool.emplace_back(work, t);
            } catch (const std::exception& e) {
                {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (err.empty()) err = std::string("starting the search threads: ") + e.what();
                }
                stop.store(true);
                for (int t = (int)pool.size() + 1; t < T; t++) group_->leave();
            }
            work(0);
            for (auto& th : pool) th.join();
        }
        group_->end();
        if (err.empty() && group_->failed()) err = group_->failure();
        if (!err.empty()) throw std::runtime_error("search failed: " + err);
        if (use_vector) vector_write_back();
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
            // the vector CFR plays its table's average (the row of our hole's class, with the solve's own weights:
            // t, or t^2 under DCFR), the strategy the exploitability is measured on.  (It added up the class's
            // current strategy per iteration before: read after the iteration's update, i.e. sigma(t+1), and
            // weighted by t under every discount -- 0.02..0.05 off the table under DCFR, 1e-3 under Linear.)
            if (use_vector) {
                r.average_strategy = r.table_average;
            } else if (s > 0.0) {
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
    // (no information); `missing` counts those (step, combo) lookups.  Each factor is at least
    // `floor` (as the blueprint's Bayes floors sigma at min_prob), so no combo drops out entirely.
    std::vector<double> likelihood(int seat, const std::vector<std::pair<int, int>>& actions, long long& missing,
                                   double floor = 0.0) const {
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
                    w[(size_t)c] *= std::max(avg[a], floor);
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
        // the blueprint's rows (kind 2; the continuations after leaves) are keyed by the blueprint's buckets, the
        // search's own rows after the root's round (none when it stopped at the river start) by the subgame's
        const bool need_bp = kind == 2 || ex.leaf_river;
        const bool need_search = kind < 2 && root_street_ < RIVER && !ex.leaf_river;
        if (need_bp) ensure_blueprint_river_table();
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
            const bool later = need_search && n > root_.n_board;  // a board of a later round (the root's: its classes)
            if (need_bp) B.bucket.assign((size_t)N_COMBOS, -1);
            if (later) B.sbucket.assign((size_t)N_COMBOS, -1);
            if (!need_bp && !later) return;
            for (int c = 0; c < N_COMBOS; c++) {
                if (on[ct.c0[c]] || on[ct.c1[c]]) continue;
                const int hole[2] = {ct.c0[c], ct.c1[c]};
                if (need_bp) B.bucket[(size_t)c] = later_bucket(hole, b, n);
                if (later) B.sbucket[(size_t)c] = search_bucket(hole, b, n);
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
        std::vector<int> bucket;        // per combo, the blueprint's bucket (blueprint rows), -1 on the board
        std::vector<int> sbucket;       // per combo, the subgame's bucket (its own rows after the root's round), -1 on the board
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
        // -1 marks the combos on this board; neither list filled: the root's board, where the classes tell
        const std::vector<int>& marks = !B.bucket.empty() ? B.bucket : B.sbucket;
        double memo[256][MAX_ACTIONS];
        bool have[256] = {false};
        // river_exact: this river node's rows of the dealt river card, by strength rank (uniform if never reached)
        const bool vx = !by_bucket && vexact_ && st.street == RIVER && root_street_ == TURN;
        const TNode::VRiver* vb = nullptr;
        const int* vrank = nullptr;
        if (vx) {
            const auto it = vriver_nodes_.find(ph.a);
            const TNode* t = it != vriver_nodes_.end() && it->second->ph.b == ph.b ? it->second : nullptr;
            vb = t && t->vriv ? t->vriv[(size_t)st.board[4]].load(std::memory_order_acquire) : nullptr;
            vrank = vboards_[(size_t)st.board[4]].rank.data();
        }
        for (int x = 0; x < N_COMBOS; x++) {
            if (marks.empty() ? class_of_[(size_t)x] < 0 : marks[(size_t)x] < 0) continue;  // on the board
            double* row = &rows[(size_t)x * MAX_ACTIONS];
            if (path_node && q == hand_.our_seat && x == our_combo_) {  // our actual hole plays the action taken
                row[path_[(size_t)k].index] = 1.0;
                continue;
            }
            if (by_bucket) {
                const int b = B.bucket.empty() ? 0 : B.bucket[(size_t)x];  // the blueprint's bucket
                if (b < 0 || b > 255) throw std::runtime_error("ex_rows: bucket out of range");
                if (!have[b]) {  // the policy depends on the hole only through its bucket
                    rollout_probs(st, hh, b, na, cont >= 0 ? cont : CONT_BP, memo[b]);
                    have[b] = true;
                }
                for (int a = 0; a < na.n; a++) row[a] = memo[b][a];
                continue;
            }
            // (river_warm: a block never made or never updated plays its bucket rows, read below by their keys)
            if (vx && ((vb && !vb->warm.load()) || vriver_k_ == 0)) {
                const int k = vrank[x];
                if (!vb || k < 0 || k >= vb->nc) {
                    for (int a = 0; a < na.n; a++) row[a] = 1.0 / na.n;
                    continue;
                }
                double w[MAX_ACTIONS];
                for (int a = 0; a < na.n; a++) w[a] = (double)(ex.kind == 0 ? vb->ss : vb->reg)[(size_t)k * na.n + a];
                if (ex.kind == 0) {
                    double s = 0.0;
                    for (int a = 0; a < na.n; a++) s += w[a];
                    for (int a = 0; a < na.n; a++) row[a] = s > 0.0 ? w[a] / s : 1.0 / na.n;
                } else {
                    v_regret_matching(w, na.n, row);
                }
                continue;
            }
            if (st.street != root_street_ && B.sbucket.empty()) throw std::runtime_error("ex_rows: no subgame buckets on this board");
            // later rounds: the subgame's bucket; the river of a turn root with river_buckets K: its K strength buckets
            const int card = st.street == root_street_ ? class_of_[(size_t)x]
                             : (vriver_k_ > 0 && st.street == RIVER) ? vboards_[(size_t)st.board[4]].qb[(size_t)x] : B.sbucket[(size_t)x];
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
        // the deal's board (the root's cards, then the deck from the root's position) and the showdown
        // strengths of the live seats, computed once per deal (the public tree's terminals)
        int board5[5];
        int str_board = -1;  // board size the strengths are for (-1: not computed for this deal)
        int64_t str[MAX_PLAYERS];
    };

    // a leaf: who chooses a continuation, what of the board they saw, the blueprint history there
    struct LeafCtx {
        int ch[MAX_PLAYERS];
        int nch = 0;
        int visible = 0;
        HistHash hh;
    };

    // the bucket tables of the root's board, shared through the game (SearchBoardTables): the blueprint's
    // (rollouts, the evaluator's blueprint rows) and the subgame's own (its infosets after the root's
    // round); one object when the subgame uses the blueprint's abstraction
    std::shared_ptr<SearchBoardTables> bt_, sbt_;
    double river_seconds_ = 0.0;  // seconds this search spent building river tables (0: shared)

    void init_turn_table() {
        if (root_.n_board != 3 || game_->spec.max_street < TURN) return;
        init_turn_table_of(*bt_);
        if (sbt_ != bt_) init_turn_table_of(*sbt_);
    }
    static void init_turn_table_of(SearchBoardTables& t) {
        std::call_once(t.turn_init, [&t]() {
            t.turn_b.assign((size_t)52 * N_COMBOS, 255);
            t.turn_once.reset(new std::once_flag[52]);
            t.turn_on = true;
        });
    }
    // the turn bucket of `hole` in the abstraction of `tb` / `bk` (a flop root's table, filled per turn card on first use)
    int turn_bucket(SearchBoardTables& tb, Bucketer& bk, const int* hole, const int* board4) const {
        const int t = board4[3];
        std::call_once(tb.turn_once[(size_t)t], [&]() {
            const ComboTable& ct = combo_table();
            bool on[52] = {false};
            for (int i = 0; i < 4; i++) on[board4[i]] = true;
            uint8_t* row = &tb.turn_b[(size_t)t * N_COMBOS];
            for (int c = 0; c < N_COMBOS; c++) {
                if (on[ct.c0[c]] || on[ct.c1[c]]) continue;
                const int h[2] = {ct.c0[c], ct.c1[c]};
                row[c] = (uint8_t)bk.bucket(h, board4, 4);
            }
        });
        return tb.turn_b[(size_t)t * N_COMBOS + (size_t)combo_index(hole[0], hole[1])];
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
        river_seconds_ = 0.0;
        if (sbt_ == bt_) {  // one abstraction: the subgame's river keys and the rollouts read one table
            std::call_once(bt_->river_once, [&]() { river_seconds_ = build_river_table_once(*bt_, *game_->bucketer); });
            return;
        }
        // the subgame's own river keys when it searches the river; the blueprint's river buckets when rollouts from
        // leaves reach the river (the evaluator builds the latter when it needs them: ensure_blueprint_river_table)
        if (limit_street_ == RIVER)
            std::call_once(sbt_->river_once, [&]() { river_seconds_ += build_river_table_once(*sbt_, *game_->search_bucketer); });
        else
            std::call_once(bt_->river_once, [&]() { river_seconds_ += build_river_table_once(*bt_, *game_->bucketer); });
    }
    void ensure_blueprint_river_table() const {
        if (sbt_ == bt_) return;  // built with the search
        std::call_once(bt_->river_once, [&]() { build_river_table_once(*bt_, *game_->bucketer); });
    }
    // the river buckets of every board the subgame can reach, in the abstraction of `bk`, into `rt`; returns the
    // seconds spent (0 when the bucketer has no batch: bucket() then answers)
    double build_river_table_once(SearchBoardTables& rt, const Bucketer& bk) const {
        const auto t0 = std::chrono::steady_clock::now();
        const int base = root_.n_board;
        if (base != 3 && base != 4) return 0.0;
        if (game_->spec.max_street < RIVER) return 0.0;  // no river decisions, in the subgame or in rollouts
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
                if (!bk.river_buckets_all(board5, combo_table().idx, &b[i * (size_t)N_COMBOS])) unsupported.store(true);
            }
        };
        const int T = std::max(1, std::min(params_.threads, 64));
        std::vector<std::thread> pool;
        for (int t = 1; t < T; t++) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
        if (unsupported.load()) return 0.0;
        rt.river_slot.assign(52 * 52, -1);
        for (size_t i = 0; i < extras.size(); i++) {
            const int key = base == 4 ? extras[i].second : extras[i].first * 52 + extras[i].second;
            rt.river_slot[(size_t)key] = (int32_t)i;
        }
        rt.river_b = std::move(b);
        rt.river_base = base;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // bucket of `hole` on a board past the root's street in the abstraction of `rt` / `bk`: its river table on
    // the river when it exists, its turn table on the turn of a flop root, else the bucketer (and its cache)
    int table_bucket(SearchBoardTables& rt, Bucketer& bk, const int* hole, const int* board, int n_board) const {
        if (n_board == 5 && rt.river_base > 0) {
            int key;
            if (rt.river_base == 4) {
                key = board[4];
            } else {
                const int t = std::min(board[3], board[4]), r = std::max(board[3], board[4]);
                key = t * 52 + r;
            }
            const int32_t i = rt.river_slot[(size_t)key];
            if (i >= 0) return rt.river_b[(size_t)i * N_COMBOS + (size_t)combo_index(hole[0], hole[1])];
        }
        if (n_board == 4 && rt.turn_on) return turn_bucket(rt, bk, hole, board);
        return bk.bucket(hole, board, n_board);
    }
    // the blueprint's bucket (the keys of blueprint lookups: rollouts, the evaluator's blueprint rows)
    int later_bucket(const int* hole, const int* board, int n_board) const {
        return table_bucket(*bt_, *game_->bucketer, hole, board, n_board);
    }
    // the subgame's own bucket (its infosets after the root's round)
    int search_bucket(const int* hole, const int* board, int n_board) const {
        return table_bucket(*sbt_, *game_->search_bucketer, hole, board, n_board);
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
    // tests: the bucket the search uses for a hole on a board extending the root's (tables or bucketer): the
    // subgame's own abstraction (its infosets), or with `blueprint` the blueprint's (rollouts)
    int later_bucket_of(int h0, int h1, const std::vector<int>& board, bool blueprint) const {
        const int hole[2] = {h0, h1};
        return blueprint ? later_bucket(hole, board.data(), (int)board.size()) : search_bucket(hole, board.data(), (int)board.size());
    }
    // tests: the subgame's infosets after the root's round in the table, per street (index FLOP..RIVER): the
    // number of (public node, bucket) keys found among buckets 0..255 at every public node the last solve
    // created, and the largest bucket among them (-1: none); a solve on the public tree (not legacy, not frozen)
    std::vector<std::pair<long long, int>> later_infosets() const {
        if (!table_) throw std::runtime_error("solve first");
        std::vector<std::pair<long long, int>> out(4, std::make_pair(0LL, -1));
        for (const std::unique_ptr<TNode>& t : tnodes_) {
            if (t->terminal || t->leaf || t->street <= root_street_) continue;
            for (int b = 0; b < 256; b++)
                if (table_->find(search_key(t->street, t->seat, b, t->ph)).node) {
                    out[(size_t)t->street].first++;
                    out[(size_t)t->street].second = std::max(out[(size_t)t->street].second, b);
                }
        }
        return out;
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
        for (int i = 0; i < 5; i++) ctx.board5[i] = i < root_.n_board ? root_.board[i] : ctx.deck[root_.deck_pos + (i - root_.n_board)];
        ctx.str_board = -1;
    }

    int card_part(const HandState& st, int seat, Ctx& ctx) const {
        if (st.street == root_street_) return class_of_[(size_t)ctx.combo[seat]];
        int& m = ctx.bucket_memo[seat][st.n_board];
        if (m < 0) m = search_bucket(st.players[seat].hole, st.board, st.n_board);
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

    // ---- the subgame's public tree: what traverse() recomputes from the engine at every visit (the
    // actions, the path hash, whether the node is on the real path or a leaf, the pot at a terminal),
    // stored once per public node and solve.  Nodes are created on first visit by replaying the path
    // from the root; a decision node caches, per card part (class on the root's street, bucket later),
    // the table node traverse() would look up.  traverse_tree() makes the same random draws, the same
    // table updates and the same arithmetic in the same order as traverse(): the same search, bit for bit.
    struct TNode {
        TNode* parent = nullptr;
        int parent_action = 0;
        bool terminal = false, leaf = false, on_path = false, path_node = false;
        int k = 0, seat = -1, street = 0, n_board = 0, path_index = -1;
        PathHash ph;
        NodeActions na;
        // terminal: what HandState::finish needs besides the cards, prepared once: the contribution
        // levels (sorted, distinct), the portion of each and its eligible seats in the order the odd
        // chips go (seat after the button first); a pot without showdown is a fixed value per seat
        int n_act = 0;
        bool folded[MAX_PLAYERS];
        int32_t invested[MAX_PLAYERS];
        double fixed_value[MAX_PLAYERS];  // n_act <= 1: the net result per seat in bb
        int n_levels = 0;
        int portion[MAX_PLAYERS];
        int n_el[MAX_PLAYERS];
        int8_t el[MAX_PLAYERS][MAX_PLAYERS];
        std::unique_ptr<std::atomic<TNode*>[]> child;
        std::unique_ptr<std::atomic<Node*>[]> cache;
        int n_cache = 0;
        // vector CFR: regrets, strategy sums and visits of every card part, [cp * na + a] (allocated on first use)
        std::vector<double> vreg, vss;
        std::vector<int64_t> vvis;
        std::vector<int32_t> vlast;  // DCFR: the iteration of the row's last update (its discounts are applied lazily)
        SpinLock vlock;
        std::once_flag vonce;
        // river_exact, a river node of a turn root: the rows of each river card, [class * na + a], made on first use
        struct VRiver {
            int nc = 0;
            std::vector<float> reg, ss;
            std::vector<int32_t> vis, last;
            std::atomic<bool> warm{false};  // warm start pending: sigma from the bucket rows until the first update
        };
        std::unique_ptr<std::atomic<VRiver*>[]> vriv;  // [river card]
        std::vector<std::unique_ptr<VRiver>> vriv_own;
        std::once_flag vriv_once;
    };
    std::vector<std::unique_ptr<TNode>> tnodes_;
    std::mutex tree_mu_;
    TNode* troot_ = nullptr;

    TNode* make_tnode(TNode* parent, int a, const HandState& st, const PathHash& ph, int k, bool on_path) {
        std::unique_ptr<TNode> up(new TNode);
        TNode* t = up.get();
        t->parent = parent;
        t->parent_action = a;
        t->ph = ph;
        t->k = k;
        t->on_path = on_path;
        t->street = st.street;
        t->n_board = st.n_board;
        if (st.terminal) {
            t->terminal = true;
            for (int s = 0; s < st.n; s++) {
                t->folded[s] = st.players[s].folded;
                t->invested[s] = st.players[s].invested;
                if (!st.players[s].folded) t->n_act++;
            }
            prepare_terminal(*t);
        } else if (is_leaf(st, k, on_path)) {
            t->leaf = true;
        } else {
            t->seat = st.to_act;
            t->path_node = on_path && k < (int)path_.size();
            build_actions(st, observe(st, t->seat), t->path_node ? &path_[(size_t)k] : nullptr, t->na);
            if (t->path_node) t->path_index = path_[(size_t)k].index;
            t->child.reset(new std::atomic<TNode*>[(size_t)t->na.n]);
            for (int i = 0; i < t->na.n; i++) t->child[(size_t)i].store(nullptr, std::memory_order_relaxed);
            t->n_cache = st.street == root_street_ ? n_classes_ : tree_buckets_;
            if (st.street == PREFLOP && st.street != root_street_) t->n_cache = 169;
            t->cache.reset(new std::atomic<Node*>[(size_t)t->n_cache]);
            for (int i = 0; i < t->n_cache; i++) t->cache[(size_t)i].store(nullptr, std::memory_order_relaxed);
        }
        tnodes_.push_back(std::move(up));
        return t;
    }

    // the public state at `t` for the deck `deck` (and the holes of `holes`, if given): the root, then
    // the path's actions
    HandState replay_tnode(const TNode* t, const int* deck, const int (*holes)[2]) const {
        const TNode* chain[512];
        int m = 0;
        for (const TNode* p = t; p->parent; p = p->parent) chain[m++] = p;
        HandState st(root_);
        st.deck = deck;
        if (holes)
            for (int s = 0; s < st.n; s++) {
                st.players[s].hole[0] = holes[s][0];
                st.players[s].hole[1] = holes[s][1];
            }
        for (int i = m - 1; i >= 0; i--) {
            const TNode* par = chain[i]->parent;
            const int a = chain[i]->parent_action;
            st.apply(par->na.type[a], par->na.amount[a]);
        }
        return st;
    }

    TNode* tchild(TNode* t, int a) {
        TNode* c = t->child[(size_t)a].load(std::memory_order_acquire);
        return c ? c : new_tchild(t, a);
    }
    // (out of line: the replay's state and path buffers would otherwise be set up at every tchild call)
    NEGP_NOINLINE TNode* new_tchild(TNode* t, int a) {
        TNode* c;
        std::lock_guard<std::mutex> lk(tree_mu_);
        c = t->child[(size_t)a].load(std::memory_order_relaxed);
        if (c) return c;
        HandState st = replay_tnode(t, deck_, nullptr);
        st.apply(t->na.type[a], t->na.amount[a]);
        PathHash ph = t->ph;
        ph.step(a);
        c = make_tnode(t, a, st, ph, t->k + 1, t->path_node && a == t->path_index);
        t->child[(size_t)a].store(c, std::memory_order_release);
        return c;
    }

    int tree_buckets_ = 1;  // card parts cached per node past the root's street: the subgame bucketer's bucket count

    // the card-independent part of HandState::finish at terminal t (see TNode)
    void prepare_terminal(TNode& t) const {
        const int n = root_.n;
        const int button = root_.button;
        int active[MAX_PLAYERS];
        int n_act = 0;
        for (int i = 0; i < n; i++) if (!t.folded[i]) active[n_act++] = i;
        int levels[MAX_PLAYERS];
        int n_levels = 0;
        for (int i = 0; i < n; i++) if (t.invested[i] > 0) levels[n_levels++] = t.invested[i];
        std::sort(levels, levels + n_levels);
        n_levels = (int)(std::unique(levels, levels + n_levels) - levels);
        t.n_levels = n_levels;
        int prev = 0;
        int won_fixed[MAX_PLAYERS] = {0};
        for (int li = 0; li < n_levels; li++) {
            const int lvl = levels[li];
            int portion = 0;
            for (int i = 0; i < n; i++) portion += std::max(0, std::min(t.invested[i], lvl) - prev);
            int eligible[MAX_PLAYERS];
            int n_el = 0;
            for (int j = 0; j < n_act; j++) if (t.invested[active[j]] >= lvl) eligible[n_el++] = active[j];
            if (n_el == 0) for (int j = 0; j < n_act; j++) eligible[n_el++] = active[j];
            // winners are a subsequence of the eligible seats; HandState::finish stable-sorts them by
            // (seat - button - 1) mod n, distinct per seat, so sorting the eligible seats once gives the same order
            std::stable_sort(eligible, eligible + n_el, [&](int x, int y) { return ((x - button - 1) % n + n) % n < ((y - button - 1) % n + n) % n; });
            t.portion[li] = portion;
            t.n_el[li] = n_el;
            for (int j = 0; j < n_el; j++) t.el[li][j] = (int8_t)eligible[j];
            if (n_el == 1) won_fixed[eligible[0]] += portion;
            prev = lvl;
        }
        for (int s = 0; s < n; s++) t.fixed_value[s] = (double)(won_fixed[s] - t.invested[s]) / (double)game_->spec.bb;
    }

    void build_troot() {
        tree_buckets_ = std::max(1, game_->search_bucketer->identity().n_buckets);
        tnodes_.clear();
        HandState st(root_);
        troot_ = make_tnode(nullptr, 0, st, PathHash(), 0, true);
    }

    // HandState::finish + net for `seat`, from the prepared terminal and the deal's cards
    double tree_net(const TNode* t, int seat, Ctx& ctx) const {
        if (t->n_act <= 1) return t->fixed_value[seat];
        const int n = root_.n;
        if (ctx.str_board != t->n_board) {
            for (int s = 0; s < n; s++) {  // every seat: with 3+ players another terminal of this board has other folds
                int cards[7];
                cards[0] = ctx.holes[s][0];
                cards[1] = ctx.holes[s][1];
                for (int i = 0; i < t->n_board; i++) cards[2 + i] = ctx.board5[i];
                ctx.str[s] = evaluate(cards, 2 + t->n_board);
            }
            ctx.str_board = t->n_board;
        }
        int won = 0;
        for (int li = 0; li < t->n_levels; li++) {
            const int n_el = t->n_el[li];
            const int8_t* el = t->el[li];
            if (n_el == 1) {
                if (el[0] == seat) won += t->portion[li];
                continue;
            }
            int64_t best = -1;
            for (int j = 0; j < n_el; j++) best = std::max(best, ctx.str[el[j]]);
            int n_ws = 0, mine = -1;
            for (int j = 0; j < n_el; j++)
                if (ctx.str[el[j]] == best) {
                    if (el[j] == seat) mine = n_ws;
                    n_ws++;
                }
            if (mine >= 0) {
                const int share = t->portion[li] / n_ws, odd = t->portion[li] % n_ws;
                won += share + (mine < odd ? 1 : 0);
            }
        }
        return (double)(won - t->invested[seat]) / (double)game_->spec.bb;
    }

    int tree_card_part(const TNode* t, int seat, Ctx& ctx) const {
        if (t->street == root_street_) return class_of_[(size_t)ctx.combo[seat]];
        int& m = ctx.bucket_memo[seat][t->n_board];
        if (m < 0) m = search_bucket(ctx.holes[seat], ctx.board5, t->n_board);
        return m;
    }

    NEGP_NOINLINE double tree_leaf_value(const TNode* t, int traverser, double weight, double w_imp, bool focused, Ctx& ctx) {
        HandState st = replay_tnode(t, ctx.deck, ctx.holes);
        return leaf_value(st, t->ph, traverser, weight, w_imp, focused, ctx);
    }

    // traverse() on the public tree (no frozen round)
    double traverse_tree(TNode* t, int traverser, double weight, double w_imp, bool focused, Ctx& ctx) {
        if (t->terminal) return tree_net(t, traverser, ctx);
        if (t->leaf) return tree_leaf_value(t, traverser, weight, w_imp, focused, ctx);
        const int seat = t->seat;
        const NodeActions& na = t->na;
        const int card = tree_card_part(t, seat, ctx);
        Node* node = card < t->n_cache ? t->cache[(size_t)card].load(std::memory_order_acquire) : nullptr;
        if (!node) {
            node = table_->get_or_create(search_key(t->street, seat, card, t->ph), ctx.tid, na.n, [&](Node& nd, NodeArena&) {
                nd.init(na.id, na.n);
                return "";
            }).node;
            if (card < t->n_cache) t->cache[(size_t)card].store(node, std::memory_order_release);
        }
        ctx.nodes++;
        if (t->path_node && seat == hand_.our_seat && ctx.actual) {  // our action already taken: fixed for our actual hole
            ctx.forced++;
            return traverse_tree(tchild(t, t->path_index), traverser, weight, w_imp, focused, ctx);
        }
        double sigma[MAX_ACTIONS];
        node->lock.lock();
        node->current_strategy(sigma);
        node->lock.unlock();
        if (t->path_node && focused && seat != traverser) {  // an opponent's real action, weighted by its probability
            const int a = t->path_index;
            const double w2 = w_imp * sigma[a];
            if (!(w2 > 0.0)) return 0.0;
            return traverse_tree(tchild(t, a), traverser, weight, w2, focused, ctx);
        }
        if (seat == traverser) {
            double utils[MAX_ACTIONS];
            for (int i = 0; i < na.n; i++) utils[i] = traverse_tree(tchild(t, i), traverser, weight, w_imp, focused, ctx);
            double u = 0.0;
            for (int i = 0; i < na.n; i++) u += sigma[i] * utils[i];
            const double w = weight * w_imp;
            node->lock.lock();
            for (int i = 0; i < na.n; i++) node->regret()[i] += w * (utils[i] - u);
            node->lock.unlock();
            return u;
        }
        if (!focused) {
            node->lock.lock();
            for (int i = 0; i < na.n; i++) node->strategy_sum()[i] += weight * sigma[i];
            node->visits += 1;
            node->lock.unlock();
        }
        const int a = sample(sigma, na.n, ctx.rng);
        return traverse_tree(tchild(t, a), traverser, weight, w_imp, focused, ctx);
    }

    // ---- vector Linear CFR (SearchParams::vector_cfr)
    struct VBoard {                      // one river board: strengths, combos in increasing strength, buckets
        std::vector<int64_t> strength;
        std::vector<int> order;
        std::vector<int> off;            // the combos holding a board card (not in order)
        std::vector<int> bucket;         // turn root: the subgame's river bucket (search_bucket); -1: the combo holds a board card
        std::vector<int> rank;           // river_exact: the combo's class, the index of its strength among the board's (-1: on the board)
        int n_rank = 0;                  // distinct strengths
        std::vector<int> qb;             // river_buckets K: the combo's strength bucket 0..K-1 (-1: on the board)
        std::vector<int> rank_qb;        // river_warm: exact class -> its strength bucket
    };
    int vriver_k_ = 0;                   // river_buckets on this turn root (0: off)
    bool vwarm_ = false;                 // river_warm: bucket phase, then exact rows warm-started
    std::atomic<bool> vswitched_{false}; // river_warm: the exact phase has begun
    std::atomic<long long> vwarm_t_{0};  // river_warm: iterations of the bucket phase (T_w)
    std::vector<VBoard> vboards_;        // [river card] on a turn root, [52] on a river root
    bool vexact_ = false;                // river_exact on this turn root
    std::unordered_map<uint64_t, TNode*> vriver_nodes_;  // river_exact, after the solve: river decision nodes by path hash (a)
    std::vector<int> vriver_;            // the river cards a turn root samples from
    int vseat_[2] = {0, 1};              // the two live seats
    TNode* vour_ = nullptr;              // our decision's node in the public tree

  public:
    // river_exact, after a solve on a turn root: per river card (52 entries, 0 for board cards) the number of
    // classes (distinct strengths), and the class of a combo on a river card (-1: holds a board card)
    std::vector<int> river_classes() const {
        std::vector<int> out(52, 0);
        if (!vexact_) return out;
        for (int r : vriver_) out[(size_t)r] = vboards_[(size_t)r].n_rank;
        return out;
    }
    int river_class(int r, int combo) const {
        if (!vexact_ || r < 0 || r >= 52 || combo < 0 || combo >= N_COMBOS || vboards_[(size_t)r].rank.empty()) return -1;
        return vboards_[(size_t)r].rank[(size_t)combo];
    }
    // river_exact: bytes of the river rows made in the last solve (floats and visit counters)
    size_t river_bytes() const {
        size_t b = 0;
        for (const auto& up : tnodes_)
            for (const auto& v : up->vriv_own) b += v->reg.capacity() * sizeof(float) * 2 + v->vis.capacity() * sizeof(int32_t);
        return b;
    }
    // the budget of the next solve() (e.g. vector-CFR iterations once vector_eligible() is known)
    void set_budget(long long iterations, double time_budget) {
        params_.iterations = iterations;
        params_.time_budget = time_budget;
    }
    bool vector_eligible() const {
        if (!params_.vector_cfr || frozen_ || params_.legacy_traverse) return false;
        if (root_street_ < TURN || limit_street_ < RIVER || raise_limit_ > 0) return false;
        int live = 0;
        for (int s = 0; s < root_.n; s++) if (!root_.players[s].folded) live++;
        return live == 2;
    }

  private:
    void prepare_vector() {
        if (params_.vector_discount < 0 || params_.vector_discount > 2) throw std::invalid_argument("vector_discount: 0 (Linear), 1 (CFR+), 2 (DCFR)");
        if (params_.vector_discount != 0 && !params_.linear) throw std::invalid_argument("vector_discount needs linear=True (the iteration count)");
        if (params_.river_warm > 0.0 && params_.vector_discount != 0) throw std::invalid_argument("river_warm: with Linear CFR only");
        vour_ = troot_;
        for (const PathStep& ps : path_) vour_ = tchild(vour_, ps.index);
        int k = 0;
        for (int s = 0; s < root_.n; s++) if (!root_.players[s].folded && k < 2) vseat_[k++] = s;
        const ComboTable& ct = combo_table();
        vboards_.assign(53, VBoard());
        vriver_.clear();
        auto fill = [&](int idx, const int* b) {
            VBoard& B = vboards_[(size_t)idx];
            bool on[52] = {false};
            for (int i = 0; i < 5; i++) on[b[i]] = true;
            B.strength.assign((size_t)N_COMBOS, 0);
            B.bucket.assign((size_t)N_COMBOS, -1);
            for (int c = 0; c < N_COMBOS; c++) {
                if (on[ct.c0[c]] || on[ct.c1[c]]) {
                    B.off.push_back(c);
                    continue;
                }
                int cards[7] = {ct.c0[c], ct.c1[c], b[0], b[1], b[2], b[3], b[4]};
                B.strength[(size_t)c] = evaluate(cards, 7);
                B.order.push_back(c);
                const int hole[2] = {ct.c0[c], ct.c1[c]};
                // the subgame's own river bucket, as the MCCFR's card parts (tree_card_part) and the evaluator's
                // own rows (Board5::sbucket) key the river; the blueprint's (later_bucket) only with the same bucketer
                if (root_.n_board == 4) B.bucket[(size_t)c] = search_bucket(hole, b, 5);
            }
            std::sort(B.order.begin(), B.order.end(), [&](int a, int c2) { return B.strength[(size_t)a] < B.strength[(size_t)c2]; });
            if (vexact_) {
                B.rank.assign((size_t)N_COMBOS, -1);
                int k = -1;
                for (size_t i = 0; i < B.order.size(); i++) {
                    if (i == 0 || B.strength[(size_t)B.order[i]] != B.strength[(size_t)B.order[i - 1]]) k++;
                    B.rank[(size_t)B.order[i]] = k;
                }
                B.n_rank = k + 1;
            }
            if (vriver_k_ > 0) {  // quantile of the strength among the board's holes: a tie group at its middle position
                const size_t m = B.order.size();
                B.qb.assign((size_t)N_COMBOS, -1);
                size_t i = 0;
                while (i < m) {
                    size_t j = i;
                    while (j < m && B.strength[(size_t)B.order[j]] == B.strength[(size_t)B.order[i]]) j++;
                    const double mid = 0.5 * (double)(i + j - 1);
                    const int q = std::min(vriver_k_ - 1, (int)((double)vriver_k_ * mid / (double)m));
                    for (size_t x = i; x < j; x++) B.qb[(size_t)B.order[x]] = q;
                    i = j;
                }
                if (vexact_) {
                    B.rank_qb.assign((size_t)B.n_rank, 0);
                    for (int c : B.order) B.rank_qb[(size_t)B.rank[(size_t)c]] = B.qb[(size_t)c];
                }
            }
        };
        vexact_ = params_.river_exact && root_.n_board == 4;
        vwarm_ = vexact_ && params_.river_warm > 0.0 && params_.river_buckets > 0;
        if (vexact_ && params_.river_warm > 0.0 && params_.river_buckets <= 0)
            throw std::invalid_argument("river_warm needs river_buckets > 0 (the coarser abstraction it starts from)");
        vswitched_.store(false);
        vwarm_t_.store(0);
        vriver_k_ = (!vexact_ || vwarm_) && root_.n_board == 4 && params_.river_buckets > 0 ? params_.river_buckets : 0;
        int b5[5];
        for (int i = 0; i < root_.n_board; i++) b5[i] = root_.board[i];
        if (root_.n_board == 5) {
            fill(52, b5);
        } else {
            bool on[52] = {false};
            for (int i = 0; i < 4; i++) on[root_.board[i]] = true;
            for (int r = 0; r < 52; r++) {
                if (on[r]) continue;
                b5[4] = r;
                fill(r, b5);
                vriver_.push_back(r);
            }
        }
    }

    // p's net in bb at showdown terminal t against o: cmp > 0 p's hand is better, 0 a tie, < 0 worse
    double v_showdown_net(const TNode* t, int p, int o, int cmp) const {
        int won = 0;
        for (int li = 0; li < t->n_levels; li++) {
            const int n_el = t->n_el[li];
            const int8_t* el = t->el[li];
            if (n_el == 1) {
                if (el[0] == p) won += t->portion[li];
                continue;
            }
            // the eligible seats are p and o (the others folded); winners in the prepared (odd chip) order
            int n_ws = 0, mine = -1;
            for (int j = 0; j < n_el; j++) {
                const int s = el[j];
                const bool wins = s == p ? cmp >= 0 : cmp <= 0;
                if (!wins) continue;
                if (s == p) mine = n_ws;
                n_ws++;
            }
            if (mine >= 0) won += t->portion[li] / n_ws + (mine < t->portion[li] % n_ws ? 1 : 0);
        }
        return (double)(won - t->invested[p]) / (double)game_->spec.bb;
    }

    // p's counterfactual values at terminal t (o's reach ro; river board index bidx)
    // the river board bidx was not filled (its combos in neither order nor off)
    bool B_invalid(int bidx) const {
        const VBoard& B = vboards_[(size_t)bidx];
        return B.order.size() + B.off.size() != (size_t)N_COMBOS;
    }
    void v_terminal(const TNode* t, int p, int o, const double* ro, int bidx, double* v) const {
        const ComboTable& ct = combo_table();
        if (t->n_act <= 1) {  // a fold: the same net for every disjoint pair
            const double net = t->fixed_value[p];
            double total = 0.0, card[52] = {0.0};
            for (int d = 0; d < N_COMBOS; d++) {
                const double w = ro[d];
                if (w == 0.0) continue;
                total += w;
                card[ct.c0[d]] += w;
                card[ct.c1[d]] += w;
            }
            for (int c = 0; c < N_COMBOS; c++) v[c] = net * (total - card[ct.c0[c]] - card[ct.c1[c]] + ro[c]);
            return;
        }
        // a showdown is on a 5-card board (an all-in before the river deals the rest as chance nodes): B.order and B.off
        // cover every combo, which the clearing below relies on
        if (t->n_board != 5 || B_invalid(bidx)) throw std::logic_error("vector CFR: a showdown terminal off a 5-card board");
        const VBoard& B = vboards_[(size_t)bidx];
        const double nw = v_showdown_net(t, p, o, 1), nt = v_showdown_net(t, p, o, 0), nl = v_showdown_net(t, p, o, -1);
        for (const int c : B.off) v[c] = 0.0;  // (every combo of B.order is written below)
        double total = 0.0, card[52] = {0.0}, gcard[52] = {0.0};
        size_t i = 0;
        const size_t m = B.order.size();
        // wins and ties by prefix sums over the combos in increasing strength, card removal by per-card sums
        // (win / tie of a combo of B.order are written before they are read: no clearing)
        static thread_local std::vector<double> win, tie;
        if (win.size() < (size_t)N_COMBOS) {
            win.resize((size_t)N_COMBOS);
            tie.resize((size_t)N_COMBOS);
        }
        while (i < m) {
            size_t j = i;
            const int64_t sv = B.strength[(size_t)B.order[i]];
            double gt = 0.0;
            while (j < m && B.strength[(size_t)B.order[j]] == sv) {
                const int d = B.order[j];
                gt += ro[d];
                gcard[ct.c0[d]] += ro[d];
                gcard[ct.c1[d]] += ro[d];
                j++;
            }
            for (size_t q = i; q < j; q++) {
                const int c = B.order[q];
                win[(size_t)c] = total - card[ct.c0[c]] - card[ct.c1[c]];
                tie[(size_t)c] = gt - gcard[ct.c0[c]] - gcard[ct.c1[c]] + ro[c];
            }
            for (size_t q = i; q < j; q++) {
                const int d = B.order[q];
                card[ct.c0[d]] += ro[d];
                card[ct.c1[d]] += ro[d];
                gcard[ct.c0[d]] = 0.0;
                gcard[ct.c1[d]] = 0.0;
            }
            total += gt;
            i = j;
        }
        for (size_t q = 0; q < m; q++) {
            const int c = B.order[q];
            const double all = total - card[ct.c0[c]] - card[ct.c1[c]] + ro[c];
            const double lose = all - win[(size_t)c] - tie[(size_t)c];
            v[c] = nw * win[(size_t)c] + nt * tie[(size_t)c] + nl * lose;
        }
    }

    struct VCtx {  // scratch of one thread, reused by depth: vectors of 1326, card-part rows
        std::vector<std::vector<double>> buf;
        size_t top = 0;
        std::vector<std::vector<double>> rows;   // per depth: sigma / regret / sum rows of the card parts
        std::vector<std::vector<int>> ints;      // per depth: counts of the card parts' combos
        int depth = 0;
        double* get() {
            if (top == buf.size()) buf.emplace_back((size_t)N_COMBOS, 0.0);
            return buf[top++].data();
        }
        void release(size_t n) { top -= n; }
    };

    static void v_regret_matching(const double* reg, int na, double* out) {
        double sum = 0.0;
        for (int a = 0; a < na; a++) sum += reg[a] > 0.0 ? reg[a] : 0.0;
        if (sum <= 0.0) {
            for (int a = 0; a < na; a++) out[a] = 1.0 / na;
            return;
        }
        for (int a = 0; a < na; a++) out[a] = reg[a] > 0.0 ? reg[a] / sum : 0.0;
    }

    // the rows of node t on river card r: the node's own (class on the root's round, bucket on the river; doubles),
    // or with river_exact at a river node the block of card r (class = strength rank there; floats, made on first use)
    struct VRef {
        int nc = 0;
        const int* cpv = nullptr;  // combo -> row (card part), < 0: not in the round
        double* dreg = nullptr;
        double* dss = nullptr;
        int64_t* dvis = nullptr;
        float* freg = nullptr;
        float* fss = nullptr;
        int32_t* fvis = nullptr;
        int32_t* last = nullptr;  // DCFR only
        TNode::VRiver* blk = nullptr;  // the exact block (river_exact)
        int r = -1;                    // its river card
    };
    VRef v_ref(TNode* t, int r) {
        VRef f;
        const int na = t->na.n;
        const bool river_part = t->street != root_street_;
        if (vexact_ && river_part && (!vwarm_ || vswitched_.load(std::memory_order_acquire))) {
            std::call_once(t->vriv_once, [&]() {
                t->vriv.reset(new std::atomic<TNode::VRiver*>[52]);
                for (int i = 0; i < 52; i++) t->vriv[(size_t)i].store(nullptr, std::memory_order_relaxed);
            });
            TNode::VRiver* b = t->vriv[(size_t)r].load(std::memory_order_acquire);
            if (!b) {
                t->vlock.lock();
                b = t->vriv[(size_t)r].load(std::memory_order_relaxed);
                if (!b) {
                    try {
                        std::unique_ptr<TNode::VRiver> nb(new TNode::VRiver);
                        nb->nc = vboards_[(size_t)r].n_rank;
                        nb->reg.assign((size_t)nb->nc * na, 0.0f);
                        nb->ss.assign((size_t)nb->nc * na, 0.0f);
                        nb->vis.assign((size_t)nb->nc, 0);
                        if (params_.vector_discount == 2) nb->last.assign((size_t)nb->nc, 0);
                        nb->warm.store(vwarm_);
                        b = nb.get();
                        t->vriv_own.push_back(std::move(nb));
                    } catch (...) {
                        t->vlock.unlock();
                        throw;  // (out of memory: the worker's handler, a Python exception)
                    }
                    t->vriv[(size_t)r].store(b, std::memory_order_release);
                }
                t->vlock.unlock();
            }
            f.nc = b->nc;
            f.cpv = vboards_[(size_t)r].rank.data();
            f.freg = b->reg.data();
            f.fss = b->ss.data();
            f.fvis = b->vis.data();
            f.last = b->last.empty() ? nullptr : b->last.data();
            f.blk = b;
            f.r = r;
            return f;
        }
        const bool kq = river_part && vriver_k_ > 0;  // river_buckets: K strength buckets shared by the river cards
        const int nc = kq ? vriver_k_ : t->n_cache;
        std::call_once(t->vonce, [&]() {
            t->vreg.assign((size_t)nc * na, 0.0);
            t->vss.assign((size_t)nc * na, 0.0);
            t->vvis.assign((size_t)nc, 0);
            if (params_.vector_discount == 2) t->vlast.assign((size_t)nc, 0);
        });
        f.last = t->vlast.empty() ? nullptr : t->vlast.data();
        f.nc = nc;
        f.cpv = kq ? vboards_[(size_t)r].qb.data()
                   : river_part ? vboards_[(size_t)(root_.n_board == 5 ? 52 : r)].bucket.data() : class_of_.data();
        f.dreg = t->vreg.data();
        f.dss = t->vss.data();
        f.dvis = t->vvis.data();
        return f;
    }
    // sigma of every row (a snapshot of the regrets)
    void v_sigma(TNode* t, const VRef& f, int na, double* sg) {
        if (f.blk && f.blk->warm.load(std::memory_order_acquire)) {
            // river_warm, a new exact block: its classes play their bucket rows' average strategy until its first update
            const VBoard& B = vboards_[(size_t)f.r];
            t->vlock.lock();
            for (int k = 0; k < f.nc; k++) {
                const size_t q = (size_t)B.rank_qb[(size_t)k];
                double s = 0.0;
                if (t->vss.size() >= (q + 1) * (size_t)na)
                    for (int a = 0; a < na; a++) s += t->vss[q * na + a];
                for (int a = 0; a < na; a++) sg[(size_t)k * na + a] = s > 0.0 ? t->vss[q * na + a] / s : 1.0 / na;
            }
            t->vlock.unlock();
            return;
        }
        t->vlock.lock();
        if (f.freg) {
            double tmp[MAX_ACTIONS];
            for (int cp = 0; cp < f.nc; cp++) {
                for (int a = 0; a < na; a++) tmp[a] = (double)f.freg[(size_t)cp * na + a];
                v_regret_matching(tmp, na, &sg[(size_t)cp * na]);
            }
        } else {
            for (int cp = 0; cp < f.nc; cp++) v_regret_matching(&f.dreg[(size_t)cp * na], na, &sg[(size_t)cp * na]);
        }
        t->vlock.unlock();
    }
    // weight x (regret increments, if given, and strategy-sum increments) into the rows with cnt > 0
    void v_add(TNode* t, const VRef& f, int na, const std::vector<int>& cnt, const double* reg, const double* ss, double weight) {
        if (params_.vector_discount != 0) {
            v_add_discounted(t, f, na, cnt, reg, ss, weight);
            return;
        }
        if (f.blk && f.blk->warm.load(std::memory_order_acquire)) {
            // the warm start: the bucket strategy's regrets and sums as if played in the bucket phase's T_w iterations
            // (Linear weights 1..T_w), in the share of them that would have dealt this block's river card (1 / 48)
            const double tw = (double)vwarm_t_.load();
            weight = (params_.linear ? 0.5 * tw * (tw + 1.0) : tw) / (double)std::max<size_t>(1, vriver_.size());
            if (reg) f.blk->warm.store(false, std::memory_order_release);
        }
        t->vlock.lock();
        for (int cp = 0; cp < f.nc; cp++) {
            if (!cnt[(size_t)cp]) continue;
            const size_t o = (size_t)cp * na;
            if (f.freg) {
                for (int a = 0; a < na; a++) {
                    if (reg) f.freg[o + a] += (float)(weight * reg[o + a]);
                    f.fss[o + a] += (float)(weight * ss[o + a]);
                }
                f.fvis[cp] += 1;
            } else {
                for (int a = 0; a < na; a++) {
                    if (reg) f.dreg[o + a] += weight * reg[o + a];
                    f.dss[o + a] += weight * ss[o + a];
                }
                f.dvis[cp] += 1;
            }
        }
        t->vlock.unlock();
    }

    // CFR+ / DCFR(1.5, 0, 2); `weight` is the iteration t (Linear weights on)
    void v_add_discounted(TNode* t, const VRef& f, int na, const std::vector<int>& cnt, const double* reg, const double* ss,
                          double weight) {
        const int it = (int)std::llround(weight);
        const bool dcfr = params_.vector_discount == 2;
        const double ws = dcfr ? weight * weight : weight;  // the average: t^2 (DCFR gamma = 2), t (CFR+)
        t->vlock.lock();
        for (int cp = 0; cp < f.nc; cp++) {
            if (!cnt[(size_t)cp]) continue;
            const size_t o = (size_t)cp * na;
            double R[MAX_ACTIONS];
            for (int a = 0; a < na; a++) R[a] = f.freg ? (double)f.freg[o + a] : f.dreg[o + a];
            if (reg && dcfr && f.last) {
                // the discounts of the iterations since the row's last update (last .. it - 1): positive regrets
                // x k^1.5 / (k^1.5 + 1) each, negative x 1/2 each (beta = 0)
                const int from = std::max(1, (int)f.last[cp]);
                if (f.last[cp] > 0 && from < it) {
                    double fp = 1.0;
                    for (int k = from; k < it; k++) fp *= dcfr_factor(k);
                    const double fn = std::ldexp(1.0, -(it - from));
                    for (int a = 0; a < na; a++) R[a] *= R[a] > 0.0 ? fp : fn;
                }
                f.last[cp] = it;
            }
            if (reg)
                for (int a = 0; a < na; a++) {
                    R[a] += reg[o + a];  // unweighted increments (CFR+ and DCFR discount instead)
                    if (!dcfr && R[a] < 0.0) R[a] = 0.0;  // CFR+: floored at zero
                }
            for (int a = 0; a < na; a++) {
                if (f.freg) {
                    f.freg[o + a] = (float)R[a];
                    f.fss[o + a] += (float)(ws * ss[o + a]);
                } else {
                    f.dreg[o + a] = R[a];
                    f.dss[o + a] += ws * ss[o + a];
                }
            }
            if (f.freg) f.fvis[cp] += 1;
            else f.dvis[cp] += 1;
        }
        t->vlock.unlock();
    }

    // DCFR's positive-regret discount of iteration k, k^1.5 / (k^1.5 + 1): the same double as computed in place,
    // from a table for the first 2^16 iterations (the pow per row and iteration was 5 % of a turn solve)
    static double dcfr_factor(int k) {
        static const std::vector<double> tab = [] {
            std::vector<double> v((size_t)1 << 16);
            for (size_t i = 0; i < v.size(); i++) {
                const double p = std::pow((double)i, 1.5);
                v[i] = p / (p + 1.0);
            }
            return v;
        }();
        if (k >= 0 && (size_t)k < tab.size()) return tab[(size_t)k];
        const double p = std::pow((double)k, 1.5);
        return p / (p + 1.0);
    }

    // the average strategy of p below t where o's reach is zero: no values, no regrets (both would be
    // zero), but p's strategy sums still add p's own reach x sigma, as at every node p reaches
    void v_avg(TNode* t, int p, const double* rp, int r, double weight, VCtx& vc) {
        if (t->terminal || t->leaf) return;
        {
            bool any = false;
            for (int d = 0; d < N_COMBOS && !any; d++) any = rp[d] != 0.0;
            if (!any) return;
        }
        const int q = t->seat;
        const int na = t->na.n;
        auto child = [&](int a, const double* rpa) {
            TNode* c = tchild(t, a);
            if (r >= 0 && t->n_board == 4 && c->n_board == 5) {
                const ComboTable& ct = combo_table();
                double* rp2 = vc.get();
                for (int d = 0; d < N_COMBOS; d++) rp2[d] = (ct.c0[d] == r || ct.c1[d] == r) ? 0.0 : rpa[d];
                v_avg(c, p, rp2, r, weight, vc);
                vc.release(1);
                return;
            }
            v_avg(c, p, rpa, r, weight, vc);
        };
        if (q != p) {
            for (int a = 0; a < na; a++) child(a, rp);
            return;
        }
        const VRef f = v_ref(t, r);
        const int nc = f.nc;
        const int* cpv = f.cpv;
        const int forced = t->path_node && q == hand_.our_seat ? our_combo_ : -1;
        const int dep = vc.depth++;
        if ((int)vc.rows.size() <= dep) {
            vc.rows.resize((size_t)dep + 1);
            vc.ints.resize((size_t)dep + 1);
        }
        std::vector<double>& rows = vc.rows[(size_t)dep];
        if (rows.size() < (size_t)nc * na * 3) rows.resize((size_t)nc * na * 3);
        double* sg = rows.data();
        double* ss = sg + (size_t)nc * na;
        v_sigma(t, f, na, sg);
        std::fill(ss, ss + (size_t)nc * na, 0.0);
        std::vector<int>& cnt = vc.ints[(size_t)dep];
        cnt.assign((size_t)nc, 0);
        for (int c = 0; c < N_COMBOS; c++) {
            const int cp = cpv[c];
            if (cp < 0 || cp >= nc || c == forced || rp[c] == 0.0) continue;
            cnt[(size_t)cp]++;
            for (int a = 0; a < na; a++) ss[(size_t)cp * na + a] += rp[c] * sg[(size_t)cp * na + a];
        }
        v_add(t, f, na, cnt, nullptr, ss, weight);
        double* rp2 = vc.get();
        for (int a = 0; a < na; a++) {
            for (int c = 0; c < N_COMBOS; c++) {
                const int cp = cpv[c];
                const double sgm = c == forced ? (a == t->path_index ? 1.0 : 0.0) : (cp < 0 || cp >= nc) ? 0.0 : sg[(size_t)cp * na + a];
                rp2[c] = rp[c] * sgm;
            }
            child(a, rp2);
        }
        vc.release(1);
        vc.depth--;
    }

    // the child of t through action a, crossing to the river card r when the action ends the turn
    void v_child(TNode* t, int a, int p, int o, const double* rp, const double* ro, int r, double weight, Ctx& ctx, VCtx& vc,
                 double* v) {
        TNode* c = tchild(t, a);
        if (r >= 0 && t->n_board == 4 && c->n_board == 5) {
            const ComboTable& ct = combo_table();
            double* rp2 = vc.get();
            double* ro2 = vc.get();
            for (int d = 0; d < N_COMBOS; d++) {
                const bool hit = ct.c0[d] == r || ct.c1[d] == r;
                rp2[d] = hit ? 0.0 : rp[d];
                ro2[d] = hit ? 0.0 : ro[d];
            }
            v_walk(c, p, o, rp2, ro2, r, weight, ctx, vc, v);
            const double scale = (double)vriver_.size() / 44.0;  // one card of the 48 sampled: x 48/44 over a pair's 44 cards
            for (int d = 0; d < N_COMBOS; d++) v[d] = (ct.c0[d] == r || ct.c1[d] == r) ? 0.0 : v[d] * scale;
            vc.release(2);
            return;
        }
        v_walk(c, p, o, rp, ro, r, weight, ctx, vc, v);
    }

    void v_walk(TNode* t, int p, int o, const double* rp, const double* ro, int r, double weight, Ctx& ctx, VCtx& vc, double* v) {
        if (t->terminal) {
            v_terminal(t, p, o, ro, t->n_board == 5 ? (root_.n_board == 5 ? 52 : r) : 52, v);
            return;
        }
        if (t->leaf) throw std::runtime_error("vector CFR: a leaf in the subgame");
        {
            bool any = false;
            for (int d = 0; d < N_COMBOS && !any; d++) any = ro[d] != 0.0;
            if (!any) {
                std::fill(v, v + N_COMBOS, 0.0);
                v_avg(t, p, rp, r, weight, vc);
                return;
            }
        }
        ctx.nodes++;
        const int q = t->seat;
        const int na = t->na.n;
        const VRef f = v_ref(t, r);
        const int nc = f.nc;
        const int* cpv = f.cpv;
        const int forced = t->path_node && q == hand_.our_seat ? our_combo_ : -1;  // our actual hole plays the real action
        const int dep = vc.depth++;
        if ((int)vc.rows.size() <= dep) {
            vc.rows.resize((size_t)dep + 1);
            vc.ints.resize((size_t)dep + 1);
        }
        // sigma of every card part (a snapshot of the node's regrets), then per combo and action
        std::vector<double>& rows = vc.rows[(size_t)dep];
        if (rows.size() < (size_t)nc * na * 3) rows.resize((size_t)nc * na * 3);
        double* sg = rows.data();
        v_sigma(t, f, na, sg);
        // S[a][c]: sigma of combo c (0 for combos not in the round: on the board, or holding the river card)
        const size_t base = vc.top;
        for (int a = 0; a < na; a++) vc.get();
        double* S[MAX_ACTIONS];
        for (int a = 0; a < na; a++) S[a] = vc.buf[base + (size_t)a].data();
        {  // one pass over the combos (a row per combo, a zero row outside the round), not one per action
            const double zero_row[MAX_ACTIONS] = {0.0};
            for (int c = 0; c < N_COMBOS; c++) {
                const int cp = cpv[c];
                const double* row = (cp >= 0 && cp < nc) ? &sg[(size_t)cp * na] : zero_row;
                for (int a = 0; a < na; a++) S[a][c] = row[a];
            }
        }
        if (forced >= 0)
            for (int a = 0; a < na; a++) S[a][forced] = a == t->path_index ? 1.0 : 0.0;
        // v = 0.0 + the first term + ... : the first term's pass writes 0.0 + x (not x: the same signed zeros)
        // instead of clearing v in a pass of its own
        if (q == o) {
            double* ro2 = vc.get();
            double* va = vc.get();
            bool first = true;
            for (int a = 0; a < na; a++) {
                const double* __restrict Sa = S[a];
                double mass = 0.0;
                for (int d = 0; d < N_COMBOS; d++) {
                    ro2[d] = ro[d] * Sa[d];
                    mass += ro2[d];
                }
                if (!(mass > 0.0)) {
                    TNode* c = tchild(t, a);
                    if (r >= 0 && t->n_board == 4 && c->n_board == 5) {
                        const ComboTable& ct = combo_table();
                        double* rp2 = vc.get();
                        for (int d = 0; d < N_COMBOS; d++) rp2[d] = (ct.c0[d] == r || ct.c1[d] == r) ? 0.0 : rp[d];
                        v_avg(c, p, rp2, r, weight, vc);
                        vc.release(1);
                    } else {
                        v_avg(c, p, rp, r, weight, vc);
                    }
                    continue;
                }
                v_child(t, a, p, o, rp, ro2, r, weight, ctx, vc, va);
                if (first) {
                    for (int c = 0; c < N_COMBOS; c++) v[c] = 0.0 + va[c];
                    first = false;
                } else {
                    for (int c = 0; c < N_COMBOS; c++) v[c] += va[c];
                }
            }
            if (first) std::fill(v, v + N_COMBOS, 0.0);
            vc.release((size_t)na + 2);
            vc.depth--;
            return;
        }
        // the traverser: values per action, then regrets and the average strategy per card part
        double* rp2 = vc.get();
        const size_t vbase = vc.top;
        for (int a = 0; a < na; a++) vc.get();
        double* vs[MAX_ACTIONS];
        for (int a = 0; a < na; a++) vs[a] = vc.buf[vbase + (size_t)a].data();
        for (int a = 0; a < na; a++) {
            for (int d = 0; d < N_COMBOS; d++) rp2[d] = rp[d] * S[a][d];
            v_child(t, a, p, o, rp2, ro, r, weight, ctx, vc, vs[a]);
        }
        for (int c = 0; c < N_COMBOS; c++) v[c] = 0.0 + S[0][c] * vs[0][c];
        for (int a = 1; a < na; a++)
            for (int c = 0; c < N_COMBOS; c++) v[c] += S[a][c] * vs[a][c];
        double* reg = sg + (size_t)nc * na;
        double* ss = reg + (size_t)nc * na;
        std::fill(reg, reg + (size_t)nc * na * 2, 0.0);
        std::vector<int>& cnt = vc.ints[(size_t)dep];
        cnt.assign((size_t)nc, 0);
        for (int c = 0; c < N_COMBOS; c++) {
            const int cp = cpv[c];
            if (cp < 0 || cp >= nc || c == forced) continue;
            cnt[(size_t)cp]++;
            double* rg = &reg[(size_t)cp * na];
            double* sm = &ss[(size_t)cp * na];
            const double vc_ = v[c], rpc = rp[c];
            for (int a = 0; a < na; a++) {
                rg[a] += vs[a][c] - vc_;
                sm[a] += rpc * S[a][c];
            }
        }
        v_add(t, f, na, cnt, reg, ss, weight);
        vc.release((size_t)na * 2 + 1);
        vc.depth--;
    }

    // after the vector solve: every node's rows into the search's table (the keys the MCCFR uses), so the
    // outputs, likelihood(), path_strategies() and the exact evaluator read them
    void vector_write_back() {
        for (const auto& up : tnodes_) {
            TNode* t = up.get();
            if (vexact_ && !t->terminal && !t->leaf && t->street == RIVER) vriver_nodes_[t->ph.a] = t;
            if (t->terminal || t->leaf || t->vvis.empty()) continue;
            const int na = t->na.n;
            for (int cp = 0; cp < (int)t->vvis.size(); cp++) {
                if (t->vvis[(size_t)cp] == 0) continue;
                Node* node = table_->get_or_create(search_key(t->street, t->seat, cp, t->ph), 0, na, [&](Node& nd, NodeArena&) {
                    nd.init(t->na.id, na);
                    return "";
                }).node;
                for (int a = 0; a < na; a++) {
                    node->regret()[a] = t->vreg[(size_t)cp * na + a];
                    node->strategy_sum()[a] = t->vss[(size_t)cp * na + a];
                }
                node->visits = t->vvis[(size_t)cp];
            }
        }
    }

    void vector_loop(Ctx& ctx, std::atomic<long long>& next_t, std::atomic<bool>& stop,
                     std::chrono::steady_clock::time_point deadline, const NodeKey& our_key, int our_n) {
        // a river root has no chance to sample: concurrent full-width iterations would only read each other's
        // stale regrets (measured: 4 threads converge 5x slower per second), so it runs on one thread
        if (vriver_.empty() && ctx.tid > 0) return;
        const auto t_start = deadline - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                            std::chrono::duration<double>(params_.time_budget > 0 ? params_.time_budget : 0.0));
        VCtx vc;
        std::vector<double> v((size_t)N_COMBOS);
        while (!stop.load(std::memory_order_relaxed)) {
            const long long t = next_t.fetch_add(1);
            if (params_.iterations > 0 && t > params_.iterations) break;
            const double weight = params_.linear ? (double)t : 1.0;
            if (vwarm_ && !vswitched_.load(std::memory_order_acquire)) {
                const bool done = params_.iterations > 0
                                      ? (double)t > params_.river_warm * (double)params_.iterations
                                      : std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() >=
                                            params_.river_warm * params_.time_budget;
                if (done) {
                    long long expected = 0;
                    if (vwarm_t_.compare_exchange_strong(expected, t - 1)) vswitched_.store(true, std::memory_order_release);
                }
            }
            const int r = vriver_.empty() ? -1 : vriver_[(size_t)ctx.rng.below((int)vriver_.size())];
            for (int i = 0; i < 2; i++) {
                const int p = vseat_[i], o = vseat_[1 - i];
                if (!root_.players[p].can_act()) continue;
                v_walk(troot_, p, o, reach_[(size_t)p].data(), reach_[(size_t)o].data(), r, weight, ctx, vc, v.data());
                ctx.traversals++;
            }
            (void)our_key;
            (void)our_n;
            ctx.iterations++;
            if (params_.time_budget > 0 && std::chrono::steady_clock::now() >= deadline) stop.store(true, std::memory_order_relaxed);
        }
    }
};

}  // namespace negp
