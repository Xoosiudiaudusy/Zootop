// Port of negpluribus/engine.py (HandState) for 2..9 players: blinds/antes, heads-up button
// rule, min-raise tracking, all-ins, side pots, odd chip, max_street run-out.  Same decisions,
// same events, same net results as the Python engine (tested against it on random hands).
//
// The state is a flat struct with fixed-size arrays so that CFR can clone it with a memcpy
// of the used prefix (the Python version deep-copies lists per clone).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "evaluator.h"

namespace negp {

enum ActionType { FOLD = 0, CALL = 1, RAISE = 2 };
enum StreetId { PREFLOP = 0, FLOP = 1, TURN = 2, RIVER = 3, SHOWDOWN = 4 };

constexpr int MAX_PLAYERS = 9;
constexpr int MAX_EVENTS = 512;

inline int board_cards_by_street(int street) {
    static const int n[5] = {0, 3, 4, 5, 5};
    return n[street];
}

inline const char* street_letter(int street) {
    static const char* s[5] = {"P", "F", "T", "R", "S"};
    return s[street];
}

inline std::string position_name(int seat, int button, int n) {
    int rel = ((seat - button) % n + n) % n;
    if (n == 2) return rel == 0 ? "BTN/SB" : "BB";
    static const char* base[3] = {"BTN", "SB", "BB"};
    static const char* late[3] = {"UTG", "HJ", "CO"};
    if (rel < 3) return base[rel];
    // n >= 6: late[:n-3]; n < 6: late[6-n:]
    int offset = n < 6 ? 6 - n : 0;
    return late[offset + rel - 3];
}

struct Player {
    int seat = 0;
    int stack = 0;
    int hole[2] = {0, 0};
    int street_bet = 0;
    int invested = 0;
    bool folded = false;
    bool all_in = false;
    bool acted = false;
    bool can_act() const { return !folded && !all_in; }
};

struct Event {
    int8_t street;
    int8_t seat;
    int8_t type;
    bool facing_raise;
    bool all_in;
    int16_t raises_this_street;
    int32_t amount;       // RAISE: total committed on this street after acting
    int32_t to_call;
    int32_t pot_before;
    int32_t paid;
    // The actor's stack after acting; needed to translate off-grid raises against the actor's
    // all-in (Ganzfried & Sandholm 2013).  -1 would mean "unknown" (from_concrete then leaves the
    // all-in out), but HandState::apply, the only place that creates events, sets every field.
    // No default member initializer on purpose: it made Event non-trivial, so every HandState
    // construction - the copy `HandState child(st)` of each traversal step included - stored -1
    // into all MAX_EVENTS entries of events[] (14 KB) before copy_from() overwrote the used prefix.
    // Entries at or after n_events are never read.
    int32_t stack_after;
};
static_assert(std::is_trivially_default_constructible<Event>::value && std::is_trivially_copyable<Event>::value,
              "Event must stay trivial: HandState copies only the used prefix of events[] (see stack_after)");

struct HandState {
    int n = 0;
    int button = 0;
    int sb = 50, bb = 100, ante = 0;
    int max_street = RIVER;
    int starting_stacks[MAX_PLAYERS];
    Player players[MAX_PLAYERS];
    const int* deck = nullptr;  // 52-card order, never mutated (shared by clones)
    int deck_pos = 0;
    int board[5];
    int n_board = 0;
    int street = PREFLOP;
    int current_bet = 0;
    int min_raise = 100;
    int raises_this_street = 0;
    int last_aggressor = -1;
    int to_act = -1;
    bool terminal = false;
    bool saw_flop[MAX_PLAYERS];
    int winners[MAX_PLAYERS];
    int n_winners = 0;
    int showdown_seats[MAX_PLAYERS];
    int n_showdown = 0;
    int n_events = 0;
    Event events[MAX_EVENTS];

    HandState() = default;

    // copy only the used prefix of the event log
    HandState(const HandState& o) { copy_from(o); }
    HandState& operator=(const HandState& o) { if (this != &o) copy_from(o); return *this; }

    void copy_from(const HandState& o) {
        std::memcpy((void*)this, (const void*)&o, offsetof(HandState, events));
        if (o.n_events) std::memcpy((void*)events, (const void*)o.events, sizeof(Event) * o.n_events);
    }

    HandState(const std::vector<int>& stacks, int button_, int sb_, int bb_, int ante_,
              const int* deck_order, int max_street_) {
        n = (int)stacks.size();
        if (n < 2 || n > MAX_PLAYERS) throw std::invalid_argument("2..9 players");
        for (int s : stacks) if (s <= 0) throw std::invalid_argument("all stacks must be > 0");
        button = ((button_ % n) + n) % n;
        sb = sb_; bb = bb_; ante = ante_;
        max_street = std::min(max_street_, (int)RIVER);
        for (int i = 0; i < n; i++) {
            starting_stacks[i] = stacks[i];
            players[i] = Player();
            players[i].seat = i;
            players[i].stack = stacks[i];
            saw_flop[i] = false;
        }
        deck = deck_order;
        deck_pos = 0;
        n_board = 0;
        n_events = 0;
        n_winners = 0;
        n_showdown = 0;
        street = PREFLOP;
        current_bet = 0;
        min_raise = bb;
        raises_this_street = 0;
        last_aggressor = -1;
        to_act = -1;
        terminal = false;
        for (int i = 0; i < n; i++) { players[i].hole[0] = deck[deck_pos++]; players[i].hole[1] = deck[deck_pos++]; }
        if (ante) for (int i = 0; i < n; i++) commit(players[i], ante);
        int sb_seat, bb_seat;
        if (n == 2) { sb_seat = button; bb_seat = (button + 1) % n; }
        else { sb_seat = (button + 1) % n; bb_seat = (button + 2) % n; }
        commit(players[sb_seat], sb);
        commit(players[bb_seat], bb);
        current_bet = std::max(players[sb_seat].street_bet, players[bb_seat].street_bet);
        min_raise = bb;
        to_act = next_can_act(bb_seat % n);
        check_round_end(true);
    }

    // ------------------------------------------------------------------ helpers
    int commit(Player& p, int amount) {
        amount = std::min(amount, p.stack);
        p.stack -= amount;
        p.street_bet += amount;
        p.invested += amount;
        if (p.stack == 0) p.all_in = true;
        return amount;
    }

    int pot() const {
        int s = 0;
        for (int i = 0; i < n; i++) s += players[i].invested;
        return s;
    }

    bool is_terminal() const { return terminal; }
    int current_player() const { return terminal ? -1 : to_act; }

    int next_can_act(int after) const {
        for (int k = 1; k <= n; k++) {
            const Player& q = players[(after + k) % n];
            if (q.can_act()) return q.seat;
        }
        return -1;
    }

    int n_active() const {
        int c = 0;
        for (int i = 0; i < n; i++) if (!players[i].folded) c++;
        return c;
    }

    int to_call_for(int seat) const {
        const Player& p = players[seat];
        return std::min(current_bet - p.street_bet, p.stack);
    }

    // (can_raise, min_raise_to, max_raise_to)
    void raise_bounds(int seat, bool& can_raise, int& min_to, int& max_to) const {
        const Player& p = players[seat];
        int to_call = current_bet - p.street_bet;
        can_raise = false; min_to = 0; max_to = 0;
        if (p.stack <= to_call) return;
        bool others = false;
        for (int i = 0; i < n; i++) if (i != seat && players[i].can_act()) { others = true; break; }
        if (!others) return;
        max_to = p.street_bet + p.stack;
        min_to = current_bet > 0 ? current_bet + min_raise : bb;
        if (max_to < min_to) min_to = max_to;
        can_raise = true;
    }

    int street_bets_max() const {
        int m = 0;
        for (int i = 0; i < n; i++) m = std::max(m, players[i].street_bet);
        return m;
    }

    // ------------------------------------------------------------------ apply
    Event apply(int type, int amount) {
        if (terminal || to_act < 0) throw std::runtime_error("hand is over");
        if (n_events >= MAX_EVENTS) throw std::runtime_error("event log full");
        int seat = to_act;
        Player& p = players[seat];
        int to_call = to_call_for(seat);
        int pot_before = pot();
        bool facing = raises_this_street > 0;
        int n_raises = raises_this_street;
        int paid = 0;
        if (type == FOLD) {
            if (to_call == 0) throw std::invalid_argument("cannot fold when checking is free");
            p.folded = true;
        } else if (type == CALL) {
            paid = commit(p, to_call);
        } else if (type == RAISE) {
            bool can_raise; int min_to, max_to;
            raise_bounds(seat, can_raise, min_to, max_to);
            if (!can_raise) throw std::invalid_argument("raise not allowed here");
            int to = amount;
            if (to < min_to || to > max_to) throw std::invalid_argument("raise_to outside bounds");
            int raise_size = to - current_bet;
            paid = commit(p, to - p.street_bet);
            if (raise_size >= min_raise) min_raise = raise_size;
            current_bet = to;
            raises_this_street += 1;
            last_aggressor = seat;
            for (int i = 0; i < n; i++) if (i != seat) players[i].acted = false;
        } else {
            throw std::invalid_argument("bad action type");
        }
        p.acted = true;
        Event ev;
        ev.street = (int8_t)street; ev.seat = (int8_t)seat; ev.type = (int8_t)type;
        ev.amount = type == RAISE ? amount : 0;
        ev.to_call = to_call; ev.pot_before = pot_before; ev.facing_raise = facing;
        ev.raises_this_street = (int16_t)n_raises; ev.paid = paid; ev.all_in = p.all_in; ev.stack_after = p.stack;
        events[n_events++] = ev;
        check_round_end(false);
        return ev;
    }

    // ------------------------------------------------------- round transitions
    void check_round_end(bool initial) {
        int active_cnt = n_active();
        if (active_cnt == 1) { finish(); return; }
        int can_act_cnt = 0;
        bool all_matched = true;
        bool done = true;
        for (int i = 0; i < n; i++) {
            const Player& p = players[i];
            if (p.folded) continue;
            if (!(p.street_bet == current_bet || p.all_in)) all_matched = false;
            if (p.can_act()) {
                can_act_cnt++;
                if (!(p.acted && p.street_bet == current_bet)) done = false;
            }
        }
        if (initial) {
            if (can_act_cnt <= 1 && all_matched) run_out_and_showdown();
            else if (can_act_cnt == 0) run_out_and_showdown();
            return;
        }
        if (!done) { to_act = next_can_act(to_act); return; }
        if (can_act_cnt <= 1 || street >= max_street) { run_out_and_showdown(); return; }
        next_street();
    }

    void next_street() {
        street += 1;
        for (int i = 0; i < n; i++) { players[i].street_bet = 0; players[i].acted = false; }
        current_bet = 0;
        min_raise = bb;
        raises_this_street = 0;
        int need = board_cards_by_street(street) - n_board;
        for (int i = 0; i < need; i++) board[n_board++] = deck[deck_pos++];
        if (street == FLOP) for (int i = 0; i < n; i++) if (!players[i].folded) saw_flop[i] = true;
        to_act = next_can_act(button);
    }

    void run_out_and_showdown() {
        if (street < FLOP) for (int i = 0; i < n; i++) if (!players[i].folded) saw_flop[i] = true;
        int need = 5 - n_board;
        for (int i = 0; i < need; i++) board[n_board++] = deck[deck_pos++];
        street = RIVER;
        n_showdown = 0;
        for (int i = 0; i < n; i++) if (!players[i].folded) showdown_seats[n_showdown++] = i;
        finish();
    }

    void finish() {
        terminal = true;
        to_act = -1;
        int active[MAX_PLAYERS]; int n_act = 0;
        for (int i = 0; i < n; i++) if (!players[i].folded) active[n_act++] = i;
        int64_t strengths[MAX_PLAYERS] = {0};
        if (n_act > 1) {
            for (int k = 0; k < n_act; k++) {
                int s = active[k];
                int cards[7];
                cards[0] = players[s].hole[0]; cards[1] = players[s].hole[1];
                for (int i = 0; i < n_board; i++) cards[2 + i] = board[i];
                strengths[s] = evaluate(cards, 2 + n_board);
            }
        }
        bool won[MAX_PLAYERS] = {false};
        int levels[MAX_PLAYERS]; int n_levels = 0;
        for (int i = 0; i < n; i++) if (players[i].invested > 0) levels[n_levels++] = players[i].invested;
        std::sort(levels, levels + n_levels);
        n_levels = (int)(std::unique(levels, levels + n_levels) - levels);
        int prev = 0;
        for (int li = 0; li < n_levels; li++) {
            int lvl = levels[li];
            int portion = 0;
            for (int i = 0; i < n; i++) portion += std::max(0, std::min(players[i].invested, lvl) - prev);
            int eligible[MAX_PLAYERS]; int n_el = 0;
            for (int k = 0; k < n_act; k++) if (players[active[k]].invested >= lvl) eligible[n_el++] = active[k];
            if (n_el == 0) { for (int k = 0; k < n_act; k++) eligible[n_el++] = active[k]; }
            if (n_el == 1) {
                players[eligible[0]].stack += portion;
                won[eligible[0]] = true;
            } else {
                int64_t best = -1;
                for (int k = 0; k < n_el; k++) best = std::max(best, strengths[eligible[k]]);
                int ws[MAX_PLAYERS]; int n_ws = 0;
                for (int k = 0; k < n_el; k++) if (strengths[eligible[k]] == best) ws[n_ws++] = eligible[k];
                int share = portion / n_ws;
                int odd = portion % n_ws;
                std::stable_sort(ws, ws + n_ws, [&](int a, int b) {
                    return ((a - button - 1) % n + n) % n < ((b - button - 1) % n + n) % n;
                });
                for (int i = 0; i < n_ws; i++) {
                    players[ws[i]].stack += share + (i < odd ? 1 : 0);
                    won[ws[i]] = true;
                }
            }
            prev = lvl;
        }
        n_winners = 0;
        for (int s = 0; s < n; s++) if (won[s]) winners[n_winners++] = s;
        street = SHOWDOWN;
    }

    int net(int seat) const { return players[seat].stack - starting_stacks[seat]; }
};

}  // namespace negp
