// The abstract betting tree of a trainer, built lazily: one HistNode per public action history.
//
// Every stack is the same at the start of every iteration and the abstract actions depend only
// on the public state (abstraction.h::abstract_actions reads pot, bets, stacks, raise counts),
// so a history, written in seats relative to the button, has the same successors, the same
// numeric key prefix and the same pot contributions in every deal.  The traversal therefore walks
// this tree instead of replaying the engine: a child is a pointer, the actions of a node are a
// stored list, a terminal stores what everybody put in, and a node's Node (nodetable.h) for a
// given bucket is a cached pointer.  Cards enter only through the buckets (memoised per
// iteration) and the showdown strengths (evaluated once per seat and iteration).
//
// Children are created on first visit: the parent's state is rebuilt by replaying the path from
// the root with the engine (HandState::apply), so every stored number comes from the engine
// itself.  Publication is one CAS per child pointer; a thread that loses the race drops its copy.
// The FlatNodeTable stays the owner of the regrets (checkpoints, exports unchanged); the tree only
// caches pointers into it and is rebuilt when the table is cleared (FlatNodeTable::epoch()).
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "abstraction.h"
#include "engine.h"
#include "nodetable.h"

namespace negp {

struct HistNode {
    // ---- both kinds
    HistNode* parent = nullptr;
    uint8_t parent_action = 0;  // index into parent's actions
    bool terminal = false;
    uint8_t depth = 0;          // number of events
    // ---- decision nodes
    uint8_t street = 0, rel = 0, n_active = 0, n_board = 0;
    uint8_t na = 0;
    uint8_t ids[MAX_ACTIONS];
    AbstractAction acts[MAX_ACTIONS];  // exactly what abstract_actions() produced (fractions unrounded)
    std::atomic<HistNode*> child[MAX_ACTIONS];
    HistHash hh;                     // hash of the history text (numeric key, nodetable.h)
    const char* hist = nullptr;      // history text, for the key string of new Nodes
    int n_cache = 0;                 // buckets cached: 169 preflop, n_buckets postflop
    std::atomic<Node*>* cache = nullptr;
    // ---- terminals: contributions and folds by relative seat
    int32_t invested[MAX_PLAYERS];
    uint16_t folded = 0;  // bit r: relative seat r folded
    uint8_t n_act = 0;    // players not folded
};

class HistTree {
public:
    // the game: equal starting stacks, blinds, ante, last street; `grid` must outlive the tree
    HistTree(std::vector<int> stacks, int sb, int bb, int ante, int max_street, const BetGrid& grid, int n_buckets)
        : stacks_(std::move(stacks)), sb_(sb), bb_(bb), ante_(ante), max_street_(max_street), grid_(grid), n_buckets_(n_buckets) {
        for (int i = 0; i < 52; i++) dummy_deck_[i] = i;
    }

    // the root for the current table epoch (rebuilt after FlatNodeTable::clear())
    HistNode* root(uint64_t table_epoch) {
        HistNode* r = root_.load(std::memory_order_acquire);
        if (r && epoch_ == table_epoch) return r;
        std::lock_guard<std::mutex> lk(mu_);
        r = root_.load(std::memory_order_relaxed);
        if (r && epoch_ == table_epoch) return r;
        // not concurrent with traversals (called at the start of train())
        chunks_.clear();
        strings_.clear();
        cache_chunks_.clear();
        used_ = 0;
        n_nodes_ = 0;
        HandState st = fresh_state();
        std::string s;
        r = make_node(nullptr, 0, st, HistHash(), s);
        epoch_ = table_epoch;
        root_.store(r, std::memory_order_release);
        return r;
    }

    // child `a` of `h` (created on first use)
    HistNode* child(HistNode* h, int a) {
        HistNode* c = h->child[a].load(std::memory_order_acquire);
        if (c) return c;
        std::lock_guard<std::mutex> lk(mu_);
        c = h->child[a].load(std::memory_order_relaxed);
        if (c) return c;
        HandState st = replay(h, dummy_deck_, 0);
        Obs obs = observe(st, st.to_act);
        int type, amount;
        grid_.to_concrete(obs, h->acts[a], type, amount);
        st.apply(type, amount);
        HistHash hh = h->hh;
        c = make_node(h, a, st, hh, scratch_);
        h->child[a].store(c, std::memory_order_release);
        return c;
    }

    // the engine state of `h` for a deal (`deck`, `button`): replays the path from the root
    HandState replay(const HistNode* h, const int* deck, int button) const {
        const HistNode* path[MAX_EVENTS + 1];
        int k = 0;
        for (const HistNode* p = h; p->parent; p = p->parent) path[k++] = p;
        HandState st(stacks_, button, sb_, bb_, ante_, deck, max_street_);
        for (int i = k - 1; i >= 0; i--) {
            const HistNode* parent = path[i]->parent;
            Obs obs = observe(st, st.to_act);
            int type, amount;
            grid_.to_concrete(obs, parent->acts[path[i]->parent_action], type, amount);
            st.apply(type, amount);
        }
        return st;
    }

    size_t nodes() const { return n_nodes_; }

private:
    HandState fresh_state() const {
        return HandState(stacks_, 0, sb_, bb_, ante_, dummy_deck_, max_street_);
    }

    HistNode* alloc() {
        if (chunks_.empty() || used_ == CHUNK) {
            chunks_.emplace_back(new HistNode[CHUNK]);
            used_ = 0;
        }
        n_nodes_++;
        return &chunks_.back()[used_++];
    }
    const char* keep(const std::string& s) {
        strings_.emplace_back(new char[s.size() + 1]);
        std::memcpy(strings_.back().get(), s.c_str(), s.size() + 1);
        return strings_.back().get();
    }

    // a node for state `st` (button 0, so seats are relative seats); `hh` = parent's hash
    HistNode* make_node(HistNode* parent, int a, const HandState& st, HistHash hh, std::string& tok) {
        HistNode* h = alloc();
        h->parent = parent;
        h->parent_action = (uint8_t)a;
        h->depth = (uint8_t)std::min(st.n_events, 255);
        for (int i = 0; i < MAX_ACTIONS; i++) h->child[i].store(nullptr, std::memory_order_relaxed);
        if (st.terminal) {
            h->terminal = true;
            h->folded = 0;
            h->n_act = 0;
            for (int r = 0; r < st.n; r++) {
                h->invested[r] = st.players[r].invested;
                if (st.players[r].folded) h->folded |= (uint16_t)(1u << r);
                else h->n_act++;
            }
            return h;
        }
        const int seat = st.to_act;
        Obs obs = observe(st, seat);
        ActionList actions;
        grid_.abstract_actions(obs, actions);
        if (actions.n > MAX_ACTIONS) throw std::runtime_error("too many abstract actions for the history tree");
        h->street = (uint8_t)obs.street;
        h->rel = (uint8_t)seat;  // button 0: absolute seat = relative seat
        h->n_active = (uint8_t)obs.n_active;
        h->n_board = (uint8_t)board_cards_by_street(obs.street);
        h->na = (uint8_t)actions.n;
        for (int i = 0; i < actions.n; i++) { h->ids[i] = (uint8_t)actions.a[i].id; h->acts[i] = actions.a[i]; }
        hh.catch_up(st, grid_, tok);
        h->hh = hh;
        std::string hist;
        grid_.history_string(st.events, st.n_events, hist);
        h->hist = keep(hist);
        h->n_cache = obs.street == PREFLOP ? 169 : n_buckets_;
        cache_chunks_.emplace_back(new std::atomic<Node*>[h->n_cache]);
        h->cache = cache_chunks_.back().get();
        for (int i = 0; i < h->n_cache; i++) h->cache[i].store(nullptr, std::memory_order_relaxed);
        return h;
    }

    static constexpr size_t CHUNK = 4096;
    std::vector<int> stacks_;
    int sb_, bb_, ante_, max_street_;
    const BetGrid& grid_;
    int n_buckets_;
    int dummy_deck_[52];
    std::mutex mu_;
    std::atomic<HistNode*> root_{nullptr};
    uint64_t epoch_ = ~0ULL;
    std::vector<std::unique_ptr<HistNode[]>> chunks_;
    std::vector<std::unique_ptr<char[]>> strings_;
    std::vector<std::unique_ptr<std::atomic<Node*>[]>> cache_chunks_;
    size_t used_ = 0, n_nodes_ = 0;
    std::string scratch_;
};

// net chips of relative seat `me` at terminal `h` (HandState::finish, in relative seats):
// `strength[r]` = the 7-card value of relative seat r (read only when two or more reach showdown)
inline int terminal_net(const HistNode* h, int n, int me, const int64_t* strength) {
    int active[MAX_PLAYERS];
    int n_act = 0;
    for (int r = 0; r < n; r++) if (!(h->folded >> r & 1)) active[n_act++] = r;
    int levels[MAX_PLAYERS];
    int n_levels = 0;
    for (int r = 0; r < n; r++) if (h->invested[r] > 0) levels[n_levels++] = h->invested[r];
    std::sort(levels, levels + n_levels);
    n_levels = (int)(std::unique(levels, levels + n_levels) - levels);
    int won = 0, prev = 0;
    for (int li = 0; li < n_levels; li++) {
        const int lvl = levels[li];
        int portion = 0;
        for (int r = 0; r < n; r++) portion += std::max(0, std::min(h->invested[r], lvl) - prev);
        int eligible[MAX_PLAYERS];
        int n_el = 0;
        for (int k = 0; k < n_act; k++) if (h->invested[active[k]] >= lvl) eligible[n_el++] = active[k];
        if (n_el == 0) for (int k = 0; k < n_act; k++) eligible[n_el++] = active[k];
        if (n_el == 1) {
            if (eligible[0] == me) won += portion;
        } else {
            int64_t best = -1;
            for (int k = 0; k < n_el; k++) best = std::max(best, strength[eligible[k]]);
            int ws[MAX_PLAYERS];
            int n_ws = 0;
            for (int k = 0; k < n_el; k++) if (strength[eligible[k]] == best) ws[n_ws++] = eligible[k];
            const int share = portion / n_ws, odd = portion % n_ws;
            // odd chips go first to the seat after the button: key (r - 1) mod n, stable
            std::stable_sort(ws, ws + n_ws, [&](int a, int b) { return ((a - 1) % n + n) % n < ((b - 1) % n + n) % n; });
            for (int i = 0; i < n_ws; i++) if (ws[i] == me) won += share + (i < odd ? 1 : 0);
        }
        prev = lvl;
    }
    return won - h->invested[me];
}

}  // namespace negp
