// Port of negpluribus/equity.py::equity_vs_random.  Given the same ``random.Random`` state
// (PyRandom) it draws exactly the same cards as the Python loop and returns the same double.
// ``equity_won_vs_random`` exposes the accumulated ``won`` (a multiple of 1/(n_opponents+1);
// with one opponent a multiple of 1/2) so that a cache can store it as an exact integer.
//
// Speed (2026-09-24), with the same draws and the same sum: no heap allocation per call (deck,
// draws and hands are stack arrays, PyRandom::sample allocates nothing for a deck), and when the
// board is already complete our hand is evaluated once instead of once per sample (evaluate()
// is a pure function of the 7 cards, so the value compared in every sample is the same).
#pragma once
#include "evaluator.h"
#include "pyrandom.h"

namespace negp {

inline double equity_won_vs_random(const int* hole, int n_hole, const int* board, int n_board,
                                   int n_opponents, int samples, PyRandom& rng) {
    bool used[52] = {false};
    for (int i = 0; i < n_hole; i++) used[hole[i]] = true;
    for (int i = 0; i < n_board; i++) used[board[i]] = true;
    int deck[52];
    int n_deck = 0;
    for (int c = 0; c < 52; c++) if (!used[c]) deck[n_deck++] = c;
    int need_board = 5 - n_board;
    int n_draw = need_board + 2 * n_opponents;
    double won = 0.0;
    int drawn[52];  // n_draw <= n_deck <= 52: PyRandom::sample raises before writing otherwise
    int my[7], opp[7];
    int full_board[5];
    for (int i = 0; i < n_board; i++) full_board[i] = board[i];
    for (int i = 0; i < n_hole; i++) my[i] = hole[i];
    // complete board: our hand does not depend on the draws
    const bool board_complete = need_board == 0;
    int64_t mine_fixed = 0;
    if (board_complete && samples > 0) {
        for (int i = 0; i < 5; i++) my[n_hole + i] = full_board[i];
        mine_fixed = evaluate(my, n_hole + 5);
    }
    for (int s = 0; s < samples; s++) {
        rng.sample(deck, n_deck, n_draw, drawn);
        for (int i = 0; i < need_board; i++) full_board[n_board + i] = drawn[i];
        int64_t mine;
        if (board_complete) {
            mine = mine_fixed;
        } else {
            for (int i = 0; i < 5; i++) my[n_hole + i] = full_board[i];
            mine = evaluate(my, n_hole + 5);
        }
        int64_t best_opp = -1;
        int ties = 0;
        for (int o = 0; o < n_opponents; o++) {
            opp[0] = drawn[need_board + 2 * o];
            opp[1] = drawn[need_board + 2 * o + 1];
            for (int i = 0; i < 5; i++) opp[2 + i] = full_board[i];
            int64_t v = evaluate(opp, 7);
            if (v > best_opp) { best_opp = v; ties = 1; }
            else if (v == best_opp) ties++;
        }
        if (mine > best_opp) won += 1.0;
        else if (mine == best_opp) won += 1.0 / (ties + 1);
    }
    return won;
}

inline double equity_vs_random(const int* hole, int n_hole, const int* board, int n_board,
                               int n_opponents, int samples, PyRandom& rng) {
    double won = equity_won_vs_random(hole, n_hole, board, n_board, n_opponents, samples, rng);
    return won / samples;
}

}  // namespace negp
