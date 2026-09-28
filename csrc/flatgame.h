// The betting tree of a trainer as flat arrays (structure of arrays, CSR children): the layout the
// GPU trainer uploads to the device (GPU_DESIGN.md, step 2).
//
// Built by enumerating the history tree (histtree.h) completely, depth first, so every number
// (actions, contributions, history hashes, key strings) is the one the CPU trainers use.
//
//   decision d:  street, rel (seat relative to the button), n_board, na, the action ids,
//                children child[child_base[d] + a] (>= 0: decision id, < 0: ~terminal id),
//                history hash lanes hh_a / hh_b (the numeric key prefix and the sampling address),
//                n_cache buckets (169 preflop, n_buckets postflop)
//   terminal t:  invested[t * n + r], folded bits, players not folded
//   tables:      dense, one row per (decision, bucket): infoset i = info_base[d] + b,
//                cells cell_base[d] + b * na + a (regret and strategy sum), visits per infoset
//
// The table is dense: every (history, bucket) has its cells whether or not it is ever reached (an
// unreached row keeps zero regrets = the uniform strategy, as a node the sparse table never created).
// Its size is sum over decisions of n_cache * na doubles per array; the history tree of the game
// must be enumerable (heads-up blueprint games are; see n_cells() before uploading).
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "histtree.h"

namespace negp {

struct FlatGame {
    int n_players = 0;
    // decisions
    std::vector<uint8_t> street, rel, n_board, na;
    std::vector<uint8_t> ids;          // [d * MAX_ACTIONS + a]
    std::vector<uint32_t> child_base;  // children of d: child[child_base[d] .. + na[d])
    std::vector<int32_t> child;
    std::vector<uint64_t> hh_a, hh_b;
    std::vector<int32_t> n_cache;
    std::vector<uint64_t> info_base, cell_base;
    std::vector<const HistDecision*> src;  // the history-tree node (key strings)
    // terminals
    std::vector<int32_t> invested;     // [t * n_players + r]
    std::vector<uint16_t> folded;
    std::vector<uint8_t> n_act;
    int depth = 0;                     // decisions on the longest path
    uint64_t n_infosets = 0, n_cells = 0;

    size_t n_decisions() const { return street.size(); }
    size_t n_terminals() const { return folded.size(); }

    // enumerate every node of `tree` below `root` (creates the missing children)
    void build(HistTree& tree, HistNode* root, int n) {
        *this = FlatGame();
        n_players = n;
        if (root->terminal) throw std::invalid_argument("flat game: the root is terminal");
        struct Frame { HistDecision* h; int d; int level; };
        std::vector<Frame> stack;
        stack.push_back({static_cast<HistDecision*>(root), add_decision(static_cast<HistDecision*>(root)), 1});
        // depth first, children in action order; ids are assigned when a node is first pushed
        while (!stack.empty()) {
            const Frame f = stack.back();
            stack.pop_back();
            if (f.level > depth) depth = f.level;
            std::vector<Frame> kids;
            for (int a = 0; a < f.h->na; a++) {
                HistNode* c = tree.child(f.h, a);
                int32_t ref;
                if (c->terminal) {
                    ref = ~add_terminal(static_cast<const HistTerminal*>(c));
                } else {
                    HistDecision* hc = static_cast<HistDecision*>(c);
                    const int d = add_decision(hc);
                    ref = d;
                    kids.push_back({hc, d, f.level + 1});
                }
                child[child_base[(size_t)f.d] + (size_t)a] = ref;
            }
            for (size_t i = kids.size(); i-- > 0;) stack.push_back(kids[i]);
        }
    }

    // the key string of infoset (d, b) (the string the sparse tables store)
    void key(size_t d, int b, std::string& out) const { tree_key(src[d], b, n_players, out); }

private:
    int add_decision(HistDecision* h) {
        const int d = (int)street.size();
        street.push_back(h->street);
        rel.push_back(h->rel);
        n_board.push_back(h->n_board);
        na.push_back(h->na);
        for (int a = 0; a < MAX_ACTIONS; a++) ids.push_back(a < h->na ? h->ids[a] : 0);
        child_base.push_back((uint32_t)child.size());
        child.resize(child.size() + h->na, 0);
        hh_a.push_back(h->hh.a);
        hh_b.push_back(h->hh.b);
        n_cache.push_back(h->n_cache);
        info_base.push_back(n_infosets);
        cell_base.push_back(n_cells);
        n_infosets += (uint64_t)h->n_cache;
        n_cells += (uint64_t)h->n_cache * h->na;
        src.push_back(h);
        return d;
    }
    int add_terminal(const HistTerminal* h) {
        const int t = (int)folded.size();
        for (int r = 0; r < n_players; r++) invested.push_back(h->invested[r]);
        folded.push_back(h->folded);
        n_act.push_back(h->n_act);
        return t;
    }
};

}  // namespace negp
