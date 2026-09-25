// The abstract betting tree of a trainer, built lazily: one node per public action history.
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
// itself.  Creation is serialised by a mutex (it is rare); publication is a release store of the
// child pointer.  The FlatNodeTable stays the owner of the regrets (checkpoints, exports
// unchanged); the tree caches pointers into it and is rebuilt when the table is cleared
// (FlatNodeTable::epoch()).  Nodes created by a tree traversal spell their key string from the
// tree on demand (Node::refer_to_tree), so the tree lives as long as those nodes.
//
// Memory: a node is allocated with exactly what it needs.  A terminal is a small header plus
// the contributions of the players; a decision node is a header plus `na` child pointers and
// n_cache * n_tables Node pointers inline.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "abstraction.h"
#include "engine.h"
#include "nodetable.h"

namespace negp {

struct HistNode {
    HistNode* parent = nullptr;
    uint8_t parent_action = 0;  // index into the parent's actions
    bool terminal = false;
    uint8_t n_players = 0;
};

// what HandState::finish needs, by relative seat
struct HistTerminal : HistNode {
    uint16_t folded = 0;  // bit r: relative seat r folded
    uint8_t n_act = 0;    // players not folded
    int32_t invested[MAX_PLAYERS];  // only n_players entries are allocated
    static size_t bytes_for(int n_players) {
        return sizeof(HistTerminal) - sizeof(int32_t) * (MAX_PLAYERS - (size_t)n_players);
    }
};

struct HistDecision : HistNode {
    uint8_t street = 0, rel = 0, n_active = 0, n_board = 0;
    uint8_t na = 0;
    uint8_t ids[MAX_ACTIONS];
    int32_t n_cache = 0;             // buckets cached per table: 169 preflop, n_buckets postflop
    HistHash hh;                     // hash of the history text (numeric key, nodetable.h)
    const char* hist = nullptr;      // history text, for key strings

    // inline arrays after the struct: child[na], then cache[n_tables * n_cache]
    std::atomic<HistNode*>* child() { return reinterpret_cast<std::atomic<HistNode*>*>(reinterpret_cast<char*>(this) + head()); }
    std::atomic<Node*>* cache() { return reinterpret_cast<std::atomic<Node*>*>(reinterpret_cast<char*>(this) + head() + sizeof(std::atomic<HistNode*>) * na); }
    static size_t head() { return (sizeof(HistDecision) + 7) & ~(size_t)7; }
    static size_t bytes_for(int na, int n_slots) {
        return head() + sizeof(std::atomic<HistNode*>) * (size_t)na + sizeof(std::atomic<Node*>) * (size_t)n_slots;
    }
};

// the key string infoset_key_for_bucket() spells for decision node `h` and bucket `b`
inline void tree_key(const HistDecision* h, int b, int n, std::string& key) {
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

// Node::key_string() of a node created by a tree traversal (nodetable.h)
inline void spell_tree_key(const void* hist_node, int bucket, std::string& out) {
    const HistDecision* h = static_cast<const HistDecision*>(hist_node);
    tree_key(h, bucket, h->n_players, out);
}

class HistTree {
public:
    // the game: equal starting stacks, blinds, ante, last street; `grid` must outlive the tree.
    // `n_tables`: node tables whose Node pointers are cached (MCCFR 1, RNR 2: hero / opponents)
    HistTree(std::vector<int> stacks, int sb, int bb, int ante, int max_street, const BetGrid& grid, int n_buckets, int n_tables = 1)
        : stacks_(std::move(stacks)), sb_(sb), bb_(bb), ante_(ante), max_street_(max_street), grid_(grid), n_buckets_(n_buckets),
          n_tables_(n_tables) {
        for (int i = 0; i < 52; i++) dummy_deck_[i] = i;
    }

    // the root for the current table epoch (rebuilt after FlatNodeTable::clear()); not concurrent
    // with traversals (called at the start of train())
    HistNode* root(uint64_t table_epoch) {
        std::lock_guard<std::mutex> lk(mu_);
        if (root_ && epoch_ == table_epoch) return root_;
        chunks_.clear();
        left_ = 0;
        n_nodes_ = 0;
        bytes_ = 0;
        HandState st = fresh_state();
        root_ = make_node(nullptr, 0, st, HistHash());
        epoch_ = table_epoch;
        return root_;
    }

    // child `a` of `h` (created on first use)
    HistNode* child(HistDecision* h, int a) {
        HistNode* c = h->child()[a].load(std::memory_order_acquire);
        if (c) return c;
        std::lock_guard<std::mutex> lk(mu_);
        c = h->child()[a].load(std::memory_order_relaxed);
        if (c) return c;
        HandState st = replay(h, dummy_deck_, 0);
        Obs obs = observe(st, st.to_act);
        int type, amount;
        grid_.to_concrete(obs, action_of(h->street, h->ids[a]), type, amount);
        st.apply(type, amount);
        c = make_node(h, a, st, h->hh);
        h->child()[a].store(c, std::memory_order_release);
        return c;
    }

    // the engine state of `h` for a deal (`deck`, `button`): replays the path from the root
    HandState replay(const HistNode* h, const int* deck, int button) const {
        const HistNode* path[MAX_EVENTS + 1];
        int k = 0;
        for (const HistNode* p = h; p->parent; p = p->parent) path[k++] = p;
        HandState st(stacks_, button, sb_, bb_, ante_, deck, max_street_);
        for (int i = k - 1; i >= 0; i--) {
            const HistDecision* parent = static_cast<const HistDecision*>(path[i]->parent);
            Obs obs = observe(st, st.to_act);
            int type, amount;
            grid_.to_concrete(obs, action_of(parent->street, parent->ids[path[i]->parent_action]), type, amount);
            st.apply(type, amount);
        }
        return st;
    }

    size_t nodes() const { return n_nodes_; }
    size_t bytes() const { return bytes_; }  // chunks allocated for nodes and history strings

private:
    AbstractAction action_of(int street, int id) const { return grid_.action_from_id(street, id); }

    HandState fresh_state() const { return HandState(stacks_, 0, sb_, bb_, ante_, dummy_deck_, max_street_); }

    void* alloc(size_t bytes) {
        bytes = (bytes + 7) & ~(size_t)7;
        if (bytes > left_) {
            const size_t sz = bytes > CHUNK_BYTES ? bytes : CHUNK_BYTES;
            chunks_.emplace_back(new uint64_t[sz / 8]);
            next_ = reinterpret_cast<char*>(chunks_.back().get());
            left_ = sz;
            bytes_ += sz;
        }
        void* p = next_;
        next_ += bytes;
        left_ -= bytes;
        return p;
    }
    const char* keep(const std::string& s) {
        char* p = static_cast<char*>(alloc(s.size() + 1));
        std::memcpy(p, s.c_str(), s.size() + 1);
        return p;
    }

    // a node for state `st` (button 0, so seats are relative seats); `hh` = parent's hash
    HistNode* make_node(HistDecision* parent, int a, const HandState& st, HistHash hh) {
        n_nodes_++;
        if (st.terminal) {
            HistTerminal* t = new (alloc(HistTerminal::bytes_for(st.n))) HistTerminal;
            t->parent = parent;
            t->parent_action = (uint8_t)a;
            t->terminal = true;
            t->n_players = (uint8_t)st.n;
            for (int r = 0; r < st.n; r++) {
                t->invested[r] = st.players[r].invested;
                if (st.players[r].folded) t->folded |= (uint16_t)(1u << r);
                else t->n_act++;
            }
            return t;
        }
        const int seat = st.to_act;
        Obs obs = observe(st, seat);
        ActionList actions;
        grid_.abstract_actions(obs, actions);
        if (actions.n > MAX_ACTIONS) throw std::runtime_error("too many abstract actions for the history tree");
        for (int i = 0; i < actions.n; i++) {  // the id is enough to rebuild the action exactly
            const AbstractAction b = action_of(obs.street, actions.a[i].id);
            if (b.kind != actions.a[i].kind || b.frac != actions.a[i].frac) throw std::logic_error("history tree: action not determined by its id");
        }
        const int n_cache = obs.street == PREFLOP ? 169 : n_buckets_;
        const int n_slots = n_cache * n_tables_;
        HistDecision* h = new (alloc(HistDecision::bytes_for(actions.n, n_slots))) HistDecision;
        h->parent = parent;
        h->parent_action = (uint8_t)a;
        h->n_players = (uint8_t)st.n;
        h->street = (uint8_t)obs.street;
        h->rel = (uint8_t)seat;  // button 0: absolute seat = relative seat
        h->n_active = (uint8_t)obs.n_active;
        h->n_board = (uint8_t)board_cards_by_street(obs.street);
        h->na = (uint8_t)actions.n;
        for (int i = 0; i < actions.n; i++) h->ids[i] = (uint8_t)actions.a[i].id;
        hh.catch_up(st, grid_, tok_);
        h->hh = hh;
        std::string hist;
        grid_.history_string(st.events, st.n_events, hist);
        h->hist = keep(hist);
        h->n_cache = n_cache;
        for (int i = 0; i < h->na; i++) new (&h->child()[i]) std::atomic<HistNode*>(nullptr);
        for (int i = 0; i < n_slots; i++) new (&h->cache()[i]) std::atomic<Node*>(nullptr);
        return h;
    }

    static constexpr size_t CHUNK_BYTES = 256 * 1024;
    std::vector<int> stacks_;
    int sb_, bb_, ante_, max_street_;
    const BetGrid& grid_;
    int n_buckets_;
    int n_tables_;
    int dummy_deck_[52];
    std::mutex mu_;
    HistNode* root_ = nullptr;
    uint64_t epoch_ = ~0ULL;
    std::vector<std::unique_ptr<uint64_t[]>> chunks_;
    char* next_ = nullptr;
    size_t left_ = 0, n_nodes_ = 0, bytes_ = 0;
    std::string tok_;
};

// net chips of relative seat `me` at terminal `h` (HandState::finish, in relative seats):
// `strength[r]` = the 7-card value of relative seat r (read only when two or more reach showdown)
// (the terminal given by its contributions `invested` and fold bits `folded`)
inline int terminal_net_of(const int32_t* invested, uint16_t folded, int n, int me, const int64_t* strength) {
    int active[MAX_PLAYERS];
    int n_act = 0;
    for (int r = 0; r < n; r++) if (!(folded >> r & 1)) active[n_act++] = r;
    int levels[MAX_PLAYERS];
    int n_levels = 0;
    for (int r = 0; r < n; r++) if (invested[r] > 0) levels[n_levels++] = invested[r];
    std::sort(levels, levels + n_levels);
    n_levels = (int)(std::unique(levels, levels + n_levels) - levels);
    int won = 0, prev = 0;
    for (int li = 0; li < n_levels; li++) {
        const int lvl = levels[li];
        int portion = 0;
        for (int r = 0; r < n; r++) portion += std::max(0, std::min(invested[r], lvl) - prev);
        int eligible[MAX_PLAYERS];
        int n_el = 0;
        for (int k = 0; k < n_act; k++) if (invested[active[k]] >= lvl) eligible[n_el++] = active[k];
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
    return won - invested[me];
}

inline int terminal_net(const HistTerminal* h, int n, int me, const int64_t* strength) {
    return terminal_net_of(h->invested, h->folded, n, me, strength);
}

}  // namespace negp
