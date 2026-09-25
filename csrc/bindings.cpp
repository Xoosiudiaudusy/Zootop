// pybind11 module ``negpluribus._fastcore``: the C++ core behind negpluribus.fast.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <thread>
#include <string>
#include <vector>

#include "abstraction.h"
#include "buckettable.h"
#include "engine.h"
#include "equity.h"
#include "evaluator.h"
#include "handindex.h"
#include "mccfr.h"
#include "persist.h"
#include "pyrandom.h"
#include "rnr.h"
#include "search.h"

namespace py = pybind11;
using namespace negp;

namespace negp {
const int SUIT_PERMS[24][4] = {
    {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
    {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
    {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
    {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0},
};
}

// ------------------------------------------------------------------ helpers
static std::vector<int> to_cards(const py::sequence& seq) {
    std::vector<int> out;
    out.reserve(py::len(seq));
    for (auto item : seq) {
        int c = item.cast<int>();
        if (c < 0 || c > 51) throw std::invalid_argument("card out of range");
        out.push_back(c);
    }
    return out;
}

// equity_won_vs_random keeps our hand and the board in fixed arrays (7 and 5 cards)
static void check_equity_cards(const std::vector<int>& hole, const std::vector<int>& board) {
    if (hole.size() > 2) throw std::invalid_argument("equity_vs_random: at most 2 hole cards");
    if (board.size() > 5) throw std::invalid_argument("equity_vs_random: at most 5 board cards");
}

static void state_from_py(const py::tuple& st, PyRandom& rng) {
    // random.Random.getstate()[1]: 624 words + index
    if (py::len(st) != PyRandom::N + 1) throw std::invalid_argument("bad MT state tuple");
    std::vector<uint32_t> words(PyRandom::N);
    for (int i = 0; i < PyRandom::N; i++) words[i] = st[i].cast<uint32_t>();
    int idx = st[PyRandom::N].cast<int>();
    rng.set_state(words, idx);
}

static py::tuple state_to_py(const PyRandom& rng) {
    std::vector<uint32_t> words; int idx;
    rng.get_state(words, idx);
    py::tuple t(PyRandom::N + 1);
    for (int i = 0; i < PyRandom::N; i++) t[i] = py::int_(words[i]);
    t[PyRandom::N] = py::int_(idx);
    return t;
}

static Spec spec_from_dict(const py::dict& d) {
    Spec s;
    s.n_players = d["n_players"].cast<int>();
    s.stack_bb = d["stack_bb"].cast<int>();
    s.sb = d["sb"].cast<int>();
    s.bb = d["bb"].cast<int>();
    s.ante = d["ante"].cast<int>();
    s.max_street = d["max_street"].cast<int>();
    s.preflop_fracs = d["preflop_fracs"].cast<std::vector<double>>();
    s.postflop_fracs = d["postflop_fracs"].cast<std::vector<double>>();
    s.max_raises_per_street = d["max_raises_per_street"].cast<int>();
    s.n_buckets = d["n_buckets"].cast<int>();
    s.forbid_open_limp = d["forbid_open_limp"].cast<bool>();
    s.allow_all_in = d.contains("allow_all_in") ? d["allow_all_in"].cast<bool>() : true;
    return s;
}

static py::dict event_to_dict(const Event& ev) {
    py::dict d;
    d["street"] = (int)ev.street;
    d["seat"] = (int)ev.seat;
    d["type"] = (int)ev.type;
    d["amount"] = ev.amount;
    d["to_call"] = ev.to_call;
    d["pot_before"] = ev.pot_before;
    d["facing_raise"] = ev.facing_raise;
    d["raises_this_street"] = (int)ev.raises_this_street;
    d["paid"] = ev.paid;
    d["all_in"] = ev.all_in;
    d["stack_after"] = ev.stack_after;
    return d;
}

static py::dict record_to_dict(const HandState& st) {
    py::dict d;
    std::vector<int> net, showdown, winners, board;
    for (int i = 0; i < st.n; i++) net.push_back(st.net(i));
    for (int i = 0; i < st.n_showdown; i++) showdown.push_back(st.showdown_seats[i]);
    for (int i = 0; i < st.n_winners; i++) winners.push_back(st.winners[i]);
    for (int i = 0; i < st.n_board; i++) board.push_back(st.board[i]);
    d["net"] = net;
    d["showdown_seats"] = showdown;
    d["winners"] = winners;
    d["board"] = board;
    d["n_events"] = st.n_events;
    d["terminal"] = st.terminal;
    d["street"] = st.street;
    py::list evs;
    for (int i = 0; i < st.n_events; i++) evs.append(event_to_dict(st.events[i]));
    d["events"] = evs;
    std::vector<bool> saw;
    for (int i = 0; i < st.n; i++) saw.push_back(st.saw_flop[i]);
    d["saw_flop"] = saw;
    return d;
}

static py::dict obs_to_dict(const HandState& st, int seat) {
    Obs o = observe(st, seat);
    py::dict d;
    d["seat"] = o.seat;
    d["street"] = o.street;
    d["pot"] = o.pot;
    d["to_call"] = o.to_call;
    d["stack"] = o.stack;
    d["min_raise_to"] = o.min_raise_to;
    d["max_raise_to"] = o.max_raise_to;
    d["can_raise"] = o.can_raise;
    d["can_fold"] = o.can_fold;
    d["raises_this_street"] = o.raises_this_street;
    d["n_active"] = o.n_active;
    d["position"] = position_name(seat, st.button, st.n);
    d["facing_raise"] = st.raises_this_street > 0;
    d["aggressor"] = st.last_aggressor;
    std::vector<int> stacks, bets; std::vector<bool> folded, all_in;
    for (int i = 0; i < st.n; i++) {
        stacks.push_back(st.players[i].stack); bets.push_back(st.players[i].street_bet);
        folded.push_back(st.players[i].folded); all_in.push_back(st.players[i].all_in);
    }
    d["stacks"] = stacks; d["street_bets"] = bets; d["folded"] = folded; d["all_in"] = all_in;
    std::vector<int> board; for (int i = 0; i < st.n_board; i++) board.push_back(st.board[i]);
    d["board"] = board;
    d["hole"] = std::vector<int>{st.players[seat].hole[0], st.players[seat].hole[1]};
    return d;
}

static StratTable strat_table_from_dict(const py::dict& d) {
    StratTable t;
    for (auto kv : d) {
        std::string key = kv.first.cast<std::string>();
        py::sequence row = kv.second.cast<py::sequence>();
        StratEntry e;
        e.names = row[0].cast<std::vector<std::string>>();
        e.probs = row[1].cast<std::vector<double>>();
        if (e.names.size() != e.probs.size()) throw std::invalid_argument("strategy entry names/probs length mismatch: " + key);
        t.emplace(std::move(key), std::move(e));
    }
    return t;
}

// ------------------------------------------------------------------ node tables <-> Python
// {key: (actions, regret, strategy_sum, visits)}, the checkpoint layout (order: the table's)
static py::dict export_table(FlatNodeTable& nodes, const BetGrid& grid) {
    py::dict out;
    nodes.for_each([&](const char* key, Node& n) {
        py::list acts, reg, ss;
        for (int i = 0; i < n.n; i++) {
            acts.append(grid.names[n.acts[i]]);
            reg.append(n.regret()[i]);
            ss.append(n.strategy_sum()[i]);
        }
        out[py::str(key)] = py::make_tuple(acts, reg, ss, n.visits);
    });
    return out;
}

struct NodeRow {
    std::vector<std::string> acts;
    std::vector<double> reg, ss;
    long long visits = 0;
    uint8_t ids[MAX_ACTIONS];
};

static NodeRow node_row(const std::string& key, const py::handle& value, BetGrid& grid) {
    py::sequence row = value.cast<py::sequence>();
    NodeRow r;
    r.acts = row[0].cast<std::vector<std::string>>();
    r.reg = row[1].cast<std::vector<double>>();
    r.ss = row[2].cast<std::vector<double>>();
    r.visits = row[3].cast<long long>();
    if (r.acts.size() > MAX_ACTIONS) throw std::invalid_argument("too many actions in node " + key);
    if (r.reg.size() != r.acts.size() || r.ss.size() != r.acts.size())
        throw std::invalid_argument("node " + key + ": regret / strategy_sum lengths differ from the action list");
    // the ids the traversal gives this node (per street: persist.h grid_action_id)
    const int street = street_of_key(key.c_str());
    for (size_t i = 0; i < r.acts.size(); i++) r.ids[i] = (uint8_t)grid_action_id(grid, street, r.acts[i]);
    return r;
}

// set the nodes of `d` (checkpoint rows); `clear` empties the table first
static void import_table(FlatNodeTable& nodes, const KeyCodec& codec, int arena, BetGrid& grid, const py::dict& d, bool clear) {
    if (clear) nodes.clear();
    for (auto kv : d) {
        std::string key = kv.first.cast<std::string>();
        NodeRow r = node_row(key, kv.second, grid);
        const int k = (int)r.acts.size();
        Node* n = get_or_create_by_string(nodes, codec, key, r.ids, k, arena).node;
        n->lock.lock();
        n->init(r.ids, k);
        for (int i = 0; i < k; i++) { n->regret()[i] = r.reg[i]; n->strategy_sum()[i] = r.ss[i]; }
        n->visits = r.visits;
        n->lock.unlock();
    }
}

// additive merge (multiprocess/multi-trainer sync): regrets, strategy sums, visits add up
static void add_table(FlatNodeTable& nodes, const KeyCodec& codec, int arena, BetGrid& grid, const py::dict& d) {
    for (auto kv : d) {
        std::string key = kv.first.cast<std::string>();
        NodeRow r = node_row(key, kv.second, grid);
        Node* n = get_or_create_by_string(nodes, codec, key, r.ids, (int)r.acts.size(), arena).node;
        n->lock.lock();
        for (size_t i = 0; i < r.acts.size() && i < n->n; i++) { n->regret()[i] += r.reg[i]; n->strategy_sum()[i] += r.ss[i]; }
        n->visits += r.visits;
        n->lock.unlock();
    }
}

static py::dict strategy_table(FlatNodeTable& nodes, const BetGrid& grid) {
    py::dict out;
    nodes.for_each([&](const char* key, Node& n) {
        double avg[MAX_ACTIONS];
        n.average_strategy(avg);
        py::list acts, probs;
        for (int i = 0; i < n.n; i++) { acts.append(grid.names[n.acts[i]]); probs.append(avg[i]); }
        out[py::str(key)] = py::make_tuple(acts, probs);
    });
    return out;
}

static std::vector<std::string> table_keys(FlatNodeTable& nodes) {
    std::vector<std::string> out;
    out.reserve(nodes.size());
    nodes.for_each([&](const char* key, Node&) { out.emplace_back(key); });
    return out;
}

static py::dict table_stats_dict(const FlatNodeTable& t) {
    py::dict d;
    d["size"] = t.size();
    d["capacity"] = t.capacity();
    d["slot_bytes"] = t.slot_bytes();
    d["node_bytes"] = t.node_bytes();
    d["key_bytes"] = t.key_bytes();
    return d;
}

static py::tuple key_to_py(const NodeKey& k) { return py::make_tuple(py::int_(k.k1), py::int_(k.k2)); }

// ------------------------------------------------------------------ table stress test (tests only)
// `threads` workers look up the same n_keys key strings in their own random orders, `rounds` times,
// on a table that starts at `initial_capacity` slots (so it grows under them).  Every key must be
// created exactly once and every lookup must return the same node with the key's own string.
static py::dict table_stress(int threads, int n_keys, int rounds, size_t initial_capacity, uint64_t seed) {
    if (threads < 1 || n_keys < 1 || rounds < 1) throw std::invalid_argument("table_stress: threads, n_keys, rounds >= 1");
    FlatNodeTable table(threads + 1, initial_capacity);
    TableGroup group;
    table.attach(&group);
    group.add(&table);
    KeyCodec codec(2);
    std::vector<std::string> keys((size_t)n_keys);
    std::vector<NodeKey> nks((size_t)n_keys);
    for (int i = 0; i < n_keys; i++) {
        keys[i] = std::string(i % 2 ? "T|BB|2|b" : "R|BTN/SB|2|b") + std::to_string(i % 97) + "|c r" + std::to_string(i / 97);
        nks[i] = codec.of_string(keys[i]);
    }
    std::vector<std::atomic<Node*>> owner((size_t)n_keys);
    for (auto& o : owner) o.store(nullptr);
    std::atomic<long long> created{0}, other_node{0}, wrong_key{0}, lookups{0};
    const uint8_t ids[1] = {1};
    group.begin(threads);
    auto work = [&](int tid) {
        PyRandom rng(seed + (uint64_t)tid);
        std::vector<int> order((size_t)n_keys);
        for (int round = 0; round < rounds; round++) {
            for (int i = 0; i < n_keys; i++) order[i] = i;
            rng.shuffle(order);
            for (int idx : order) {
                FlatNodeTable::Found f = table.get_or_create(nks[idx], tid, 1, [&](Node& n, NodeArena& a) {
                    n.init(ids, 1);
                    return a.copy_key(keys[idx].data(), keys[idx].size());
                });
                lookups.fetch_add(1, std::memory_order_relaxed);
                if (f.created) created.fetch_add(1, std::memory_order_relaxed);
                if (std::strcmp(f.key, keys[idx].c_str()) != 0) wrong_key.fetch_add(1, std::memory_order_relaxed);
                Node* expected = nullptr;
                if (!owner[idx].compare_exchange_strong(expected, f.node) && expected != f.node) other_node.fetch_add(1, std::memory_order_relaxed);
            }
        }
        group.leave();
    };
    {
        py::gil_scoped_release nogil;
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; t++) pool.emplace_back(work, t);
        work(0);
        for (auto& th : pool) th.join();
        group.end();
    }
    long long missing = 0, for_each_count = 0;
    for (int i = 0; i < n_keys; i++) {
        FlatNodeTable::Found f = table.find(nks[i]);
        if (f.node == nullptr || f.node != owner[i].load() || std::strcmp(f.key, keys[i].c_str()) != 0) missing++;
    }
    table.for_each([&](const char*, Node&) { for_each_count++; });
    py::dict d;
    d["created"] = created.load();
    d["lookups"] = lookups.load();
    d["other_node"] = other_node.load();
    d["wrong_key"] = wrong_key.load();
    d["missing"] = missing;
    d["size"] = table.size();
    d["for_each"] = for_each_count;
    d["capacity"] = table.capacity();
    d["resizes"] = group.resizes();
    return d;
}

static std::vector<size_t> caps_from_py(const py::object& caps) {
    // None -> defaults; an int -> the same cap for flop/turn/river; a sequence -> per street
    std::vector<size_t> out;
    if (caps.is_none()) return out;
    if (py::isinstance<py::int_>(caps)) {
        size_t c = caps.cast<size_t>();
        return {c, c, c};
    }
    for (auto item : caps.cast<py::sequence>()) out.push_back(item.cast<size_t>());
    if (out.size() > 3) throw std::invalid_argument("cache_caps: at most three values (flop, turn, river)");
    return out;
}

static py::dict cache_stats_dict(const Bucketer& b) {
    py::dict out;
    const char* names[4] = {"preflop", "flop", "turn", "river"};
    for (int s = FLOP; s <= RIVER; s++) {
        FormCacheStats st = b.cache_stats(s);
        py::dict d;
        d["capacity"] = st.capacity;
        d["size"] = st.size;
        d["computes"] = st.computes;
        d["evictions"] = st.evictions;
        out[names[s]] = d;
    }
    return out;
}

static py::object node_to_py(Node* n, const BetGrid& grid) {
    if (!n) return py::none();
    py::list acts, reg, ss;
    n->lock.lock();
    for (int i = 0; i < n->n; i++) {
        acts.append(grid.names[n->acts[i]]);
        reg.append(n->regret()[i]);
        ss.append(n->strategy_sum()[i]);
    }
    long long visits = n->visits;
    n->lock.unlock();
    return py::make_tuple(acts, reg, ss, visits);
}

template <class T>
static py::list rng_states_of(T& t) {
    py::list out;
    for (int i = 0; i < t.threads; i++) out.append(state_to_py(t.rng(i)));
    return out;
}

template <class T>
static void set_rng_states_of(T& t, const py::sequence& states) {
    int n = (int)py::len(states);
    for (int i = 0; i < n && i < t.threads; i++) state_from_py(states[i].cast<py::tuple>(), t.rng(i));
}

// ------------------------------------------------------------------ files (persist.h)
template <class T>
static std::vector<PyRandom*> rng_ptrs(T& t) {
    std::vector<PyRandom*> v;
    for (int i = 0; i < t.threads; i++) v.push_back(&t.rng(i));
    return v;
}

// what the trainer's own game is; the grid names are the spec's (imports may have appended
// foreign action names to the live grid)
template <class T>
static GameIdentity identity_of(const T& t) { return game_identity(t.spec, t.spec.grid(), *t.bucketer); }

// a loaded checkpoint's iteration, linear flag and RNG streams (as many as both have)
template <class T>
static void apply_scalars(T& t, const CheckpointScalars& sc) {
    t.set_iteration(sc.iteration);
    t.linear = sc.linear;
    const size_t n = std::min(sc.rng_words.size(), (size_t)t.threads);
    for (size_t i = 0; i < n; i++) t.rng((int)i).set_state(sc.rng_words[i], sc.rng_index[i]);
}

static py::dict identity_dict(const GameIdentity& id) {
    py::dict d;
    d["key_scheme"] = id.key_scheme;
    if (!id.has_game) return d;
    const Spec& s = id.spec;
    d["n_players"] = s.n_players;
    d["stack_bb"] = s.stack_bb;
    d["sb"] = s.sb;
    d["bb"] = s.bb;
    d["ante"] = s.ante;
    d["max_street"] = s.max_street;
    d["preflop_fracs"] = s.preflop_fracs;
    d["postflop_fracs"] = s.postflop_fracs;
    d["max_raises_per_street"] = s.max_raises_per_street;
    d["n_buckets"] = s.n_buckets;
    d["forbid_open_limp"] = s.forbid_open_limp;
    d["allow_all_in"] = s.allow_all_in;
    d["grid_names"] = id.grid_names;
    d["bucketer_kind"] = id.bucketer.kind;
    d["bucketer_n_buckets"] = id.bucketer.n_buckets;
    d["bucketer_samples"] = id.bucketer.samples;
    d["bucketer_bins"] = id.bucketer.bins;
    d["bucketer_fingerprint"] = id.bucketer.fingerprint;
    return d;
}

static long long count_keys(FlatNodeTable& t, const std::string& prefix, const std::string& suffix) {
    long long c = 0;
    t.for_each([&](const char* key, Node&) {
        const size_t len = std::strlen(key);
        if (len >= prefix.size() && len >= suffix.size() && std::memcmp(key, prefix.data(), prefix.size()) == 0 &&
            std::memcmp(key + len - suffix.size(), suffix.data(), suffix.size()) == 0)
            c++;
    });
    return c;
}

// BlueprintStrategy.policy(key, legal) on the C++ lookup: a list of floats or None
static py::object blueprint_policy_py(const BlueprintTable& b, py::handle key, py::handle legal) {
    if (!PyUnicode_Check(key.ptr())) return py::none();  // a str-keyed dict has no other keys
    Py_ssize_t klen = 0;
    const char* k = PyUnicode_AsUTF8AndSize(key.ptr(), &klen);
    if (!k) throw py::error_already_set();
    const long long i = b.find(k, (size_t)klen);
    if (i < 0) return py::none();
    PyObject* seq = PySequence_Fast(legal.ptr(), "legal must be a sequence of action names");
    if (!seq) throw py::error_already_set();
    py::object guard = py::reinterpret_steal<py::object>(seq);
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    int idx_buf[16];
    double out_buf[16];
    std::vector<int> idx_vec;
    std::vector<double> out_vec;
    int* idx = idx_buf;
    double* out = out_buf;
    if (n > 16) {
        idx_vec.resize((size_t)n);
        out_vec.resize((size_t)n);
        idx = idx_vec.data();
        out = out_vec.data();
    }
    for (Py_ssize_t j = 0; j < n; j++) {
        idx[j] = -1;
        if (!PyUnicode_Check(items[j])) continue;  // lookup.get(a, 0.0) of a non-name: 0.0
        Py_ssize_t len = 0;
        const char* s = PyUnicode_AsUTF8AndSize(items[j], &len);
        if (!s) throw py::error_already_set();
        idx[j] = b.name_index(s, (size_t)len);
    }
    if (!b.policy_at(i, idx, (int)n, out)) return py::none();
    PyObject* list = PyList_New(n);
    if (!list) throw py::error_already_set();
    for (Py_ssize_t j = 0; j < n; j++) PyList_SET_ITEM(list, j, PyFloat_FromDouble(out[j]));
    return py::reinterpret_steal<py::object>(list);
}

static py::tuple blueprint_entry_py(const BlueprintTable& b, long long i) {
    py::list names, probs;
    for (uint32_t t = b.off[(size_t)i]; t < b.off[(size_t)i + 1]; t++) {
        names.append(py::str(b.names[b.ids[t]]));
        probs.append(py::float_(b.prob(t)));
    }
    return py::make_tuple(names, probs);
}

static std::string file_kind(const std::string& path) {
    const std::string mg = file_magic(path, 8);
    if (mg == std::string(CHECKPOINT_MAGIC, 8)) return "checkpoint";
    if (mg == std::string(BLUEPRINT_MAGIC, 8)) return "blueprint";
    for (char c : mg) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        return c == '{' ? "json" : "unknown";
    }
    return mg.empty() ? "unreadable" : "unknown";
}

// ------------------------------------------------------------------ Hand: a driveable state
struct Hand {
    std::vector<int> order;
    HandState st;
    BetGrid grid;
    std::shared_ptr<Bucketer> bk;
    bool has_grid = false;

    Hand(const std::vector<int>& stacks, int button, int sb, int bb, int ante, const std::vector<int>& deck, int max_street)
        : order(deck) {
        if (order.size() != 52) throw std::invalid_argument("deck order must have 52 cards");
        st = HandState(stacks, button, sb, bb, ante, order.data(), max_street);
    }

    py::dict apply(int type, int amount) { return event_to_dict(st.apply(type, amount)); }
    bool is_terminal() const { return st.terminal; }
    int current_player() const { return st.current_player(); }
    py::dict observe_(int seat) const { return obs_to_dict(st, seat < 0 ? st.to_act : seat); }
    py::dict record() const { return record_to_dict(st); }

    std::vector<std::string> legal() const {
        if (!has_grid) throw std::runtime_error("no grid attached");
        Obs o = observe(st, st.to_act);
        ActionList al; grid.abstract_actions(o, al);
        std::vector<std::string> out;
        for (int i = 0; i < al.n; i++) out.push_back(grid.names[al.a[i].id]);
        return out;
    }
    std::string key() {
        if (!has_grid || !bk) throw std::runtime_error("no grid/bucketer attached");
        Obs o = observe(st, st.to_act);
        std::string k, h;
        infoset_key(st, o, *bk, grid, k, h);
        return k;
    }
    // the numeric key the trainers compute for this state: fields + history hashed event by event
    py::tuple numeric_key() {
        if (!has_grid || !bk) throw std::runtime_error("no grid/bucketer attached");
        Obs o = observe(st, st.to_act);
        HistHash hh;
        std::string tok;
        hh.catch_up(st, grid, tok);
        const Player& p = st.players[o.seat];
        int b = bk->bucket(p.hole, st.board, st.n_board);
        return key_to_py(KeyCodec(st.n).key(o, st.button, b, hh));
    }
    std::string history() const {
        std::string h; grid.history_string(st.events, st.n_events, h); return h;
    }
    py::tuple concrete(const std::string& name) {
        Obs o = observe(st, st.to_act);
        AbstractAction a = grid.action_from_name(name);
        int type, amount; grid.to_concrete(o, a, type, amount);
        return py::make_tuple(type, amount);
    }
    py::dict apply_name(const std::string& name) {
        Obs o = observe(st, st.to_act);
        AbstractAction a = grid.action_from_name(name);
        int type, amount; grid.to_concrete(o, a, type, amount);
        return apply(type, amount);
    }
};

// ------------------------------------------------------------------ module
PYBIND11_MODULE(_fastcore, m) {
    m.doc() = "NegativePluribus C++ core: evaluator, equity, engine, abstraction, MCCFR";

    m.def("evaluate", [](const py::sequence& cards) {
        std::vector<int> c = to_cards(cards);
        return evaluate(c.data(), (int)c.size());
    }, "7-card evaluator, identical values to negpluribus.evaluator.evaluate");

    m.def("evaluate_many", [](const py::sequence& hands) {
        std::vector<int64_t> out;
        out.reserve(py::len(hands));
        for (auto h : hands) {
            std::vector<int> c = to_cards(h.cast<py::sequence>());
            out.push_back(evaluate(c.data(), (int)c.size()));
        }
        return out;
    });

    m.def("equity_vs_random", [](const py::sequence& hole, const py::sequence& board, int n_opp, int samples, const py::tuple& state) {
        std::vector<int> h = to_cards(hole), b = to_cards(board);
        check_equity_cards(h, b);
        PyRandom rng;
        state_from_py(state, rng);
        double eq;
        {
            py::gil_scoped_release nogil;
            eq = equity_vs_random(h.data(), (int)h.size(), b.data(), (int)b.size(), n_opp, samples, rng);
        }
        return py::make_tuple(eq, state_to_py(rng));
    }, "equity_vs_random driven by a random.Random state; returns (equity, new_state)");

    m.def("equity_vs_random_seeded", [](const py::sequence& hole, const py::sequence& board, int n_opp, int samples, uint64_t seed) {
        std::vector<int> h = to_cards(hole), b = to_cards(board);
        check_equity_cards(h, b);
        PyRandom rng(seed);
        py::gil_scoped_release nogil;
        return equity_vs_random(h.data(), (int)h.size(), b.data(), (int)b.size(), n_opp, samples, rng);
    });

    m.def("canonical_form", [](const py::sequence& hole, const py::sequence& board) {
        std::vector<int> h = to_cards(hole), b = to_cards(board);
        if (h.size() != 2 || b.size() > 5) throw std::invalid_argument("canonical_form expects 2 hole cards and <= 5 board cards");
        CanonicalForm cf;
        canonical_form(h.data(), b.data(), (int)b.size(), cf);
        py::tuple ht = py::make_tuple(cf.hole[0], cf.hole[1]);
        py::tuple bt(cf.n_board);
        for (int i = 0; i < cf.n_board; i++) bt[i] = py::int_(cf.board[i]);
        return py::make_tuple(ht, bt);
    });

    m.def("canonical_key", [](const py::sequence& hole, const py::sequence& board) {
        std::vector<int> h = to_cards(hole), b = to_cards(board);
        if (h.size() != 2 || b.size() > 5) throw std::invalid_argument("canonical_key expects 2 hole cards and <= 5 board cards");
        CanonicalForm cf;
        canonical_form(h.data(), b.data(), (int)b.size(), cf);
        py::tuple t(3 + cf.n_board);
        t[0] = py::int_(cf.hole[0]); t[1] = py::int_(cf.hole[1]); t[2] = py::int_(-1);
        for (int i = 0; i < cf.n_board; i++) t[3 + i] = py::int_(cf.board[i]);
        return t;
    });

    // ---- test hooks for the CPython ports
    m.def("tuple_hash", [](const std::vector<int64_t>& items) { return py_tuple_hash(items.data(), (int)items.size()); });
    m.def("py_sum", [](const std::vector<double>& xs) { return py_sum(xs.data(), (int)xs.size()); });
    m.def("random_floats", [](uint64_t seed, int n) {
        PyRandom r(seed); std::vector<double> out; for (int i = 0; i < n; i++) out.push_back(r.random()); return out;
    });
    m.def("random_shuffle", [](uint64_t seed, int n) {
        PyRandom r(seed); std::vector<int> v(n); for (int i = 0; i < n; i++) v[i] = i; r.shuffle(v); return v;
    });
    m.def("random_sample", [](uint64_t seed, const std::vector<int>& pop, int k) {
        PyRandom r(seed); std::vector<int> out; r.sample(pop, k, out); return out;
    });
    m.def("random_sample_seq", [](uint64_t seed, const std::vector<int>& pop, int k, int reps) {
        // `reps` consecutive samples from one generator, then its state: checks the draws and the
        // number of MT words every call consumes against random.Random(seed).sample
        PyRandom r(seed);
        std::vector<std::vector<int>> out;
        for (int i = 0; i < reps; i++) {
            std::vector<int> s((size_t)std::max(k, 0));
            r.sample(pop.data(), (int)pop.size(), k, s.data());
            out.push_back(std::move(s));
        }
        return py::make_tuple(out, state_to_py(r));
    });
    m.def("random_getrandbits", [](uint64_t seed, int k) { PyRandom r(seed); return r.getrandbits(k); });
    m.def("_table_stress", &table_stress, py::arg("threads"), py::arg("n_keys"), py::arg("rounds") = 2,
          py::arg("initial_capacity") = 16, py::arg("seed") = 0,
          "tests only: concurrent lookups/inserts of the same keys on a growing FlatNodeTable (counts of anomalies)");
    m.def("position_name", &position_name);
    m.def("pseudo_harmonic", [](double x, std::vector<double> grid) { return BetGrid::pseudo_harmonic(x, grid); });
    m.def("raise_name", &raise_name);

    // ---- files (persist.h): formats, conversions, test hooks of the Python-identical spellings
    m.def("file_kind", &file_kind, "'checkpoint' / 'blueprint' (binary, by magic), 'json', 'unknown' or 'unreadable'");
    m.def("checkpoint_bin_to_json", [](const std::string& src, const std::string& dst) {
        py::gil_scoped_release nogil;
        checkpoint_bin_to_json(src, dst);
    }, py::arg("src"), py::arg("dst"), "a binary checkpoint as the JSON checkpoint of the C++ trainer (streamed; nodes in file order)");
    m.def("py_float_repr", [](double x) { std::string s; py_float_repr(x, s); return s; },
          "tests only: the spelling json.dump gives the float x");
    m.def("py_json_string", [](const std::string& s) { std::string o; py_json_string(s.data(), s.size(), o); return o; },
          "tests only: the spelling json.dump gives the str s");
    m.def("round5", &round5, "tests only: round(x, 5) as the blueprint writers compute it (fast path + exact fallback)");
    m.def("round5_exact", &round5_exact, "tests only: round(x, 5) through std::to_chars / std::from_chars");

    // ---- engine
    py::class_<Hand>(m, "Hand")
        .def(py::init<const std::vector<int>&, int, int, int, int, const std::vector<int>&, int>(),
             py::arg("stacks"), py::arg("button"), py::arg("sb"), py::arg("bb"), py::arg("ante"), py::arg("deck_order"), py::arg("max_street"))
        .def("apply", &Hand::apply, py::arg("type"), py::arg("amount") = 0)
        .def("apply_name", &Hand::apply_name)
        .def("concrete", &Hand::concrete)
        .def("legal", &Hand::legal)
        .def("key", &Hand::key)
        .def("numeric_key", &Hand::numeric_key, "(k1, k2): the key the trainers compute for this state without the string")
        .def("history", &Hand::history)
        .def("observe", &Hand::observe_, py::arg("seat") = -1)
        .def("record", &Hand::record)
        .def_property_readonly("is_terminal", &Hand::is_terminal)
        .def_property_readonly("current_player", &Hand::current_player)
        .def_property_readonly("pot", [](const Hand& h) { return h.st.pot(); })
        .def_property_readonly("street", [](const Hand& h) { return h.st.street; });

    // ---- potential-aware features (abstraction/potential.py hooks; same numbers as the reference)
    m.def("potential_histogram", [](const py::sequence& hole, const py::sequence& board, int samples, int bins) {
        std::vector<int> h = to_cards(hole), bd = to_cards(board);
        if (h.size() != 2 || bd.size() < 3 || bd.size() > 4) throw std::invalid_argument("potential_histogram expects 2 hole cards and a 3..4 card board");
        if (bins < 1 || bins > MAX_BINS) throw std::invalid_argument("bins must be 1..64");
        int counts[MAX_BINS];
        double mean;
        {
            py::gil_scoped_release nogil;
            potential_histogram(h.data(), bd.data(), (int)bd.size(), samples, bins, counts, mean);
        }
        return py::make_tuple(std::vector<int>(counts, counts + bins), mean);
    }, "next-street E[HS] histogram (counts over `bins`, mean equity) as abstraction/potential.py computes it");

    m.def("count_betting_tree", [](const py::dict& spec_d, long long limit) {
        // size of the abstract betting tree (button 0): decision histories and actions per street,
        // terminals; stops after `limit` decision histories (then "complete" is False)
        Spec spec = spec_from_dict(spec_d);
        BetGrid grid = spec.grid();
        int deck[52];
        for (int i = 0; i < 52; i++) deck[i] = i;
        long long dec[4] = {0, 0, 0, 0}, acts[4] = {0, 0, 0, 0}, term = 0, total = 0;
        bool complete = true;
        {
            py::gil_scoped_release nogil;
            std::vector<HandState> stack;
            stack.emplace_back(spec.stacks(), 0, spec.sb, spec.bb, spec.ante, deck, spec.max_street);
            while (!stack.empty()) {
                HandState st = stack.back();
                stack.pop_back();
                if (st.terminal) { term++; continue; }
                if (total >= limit) { complete = false; break; }
                Obs obs = observe(st, st.to_act);
                ActionList al;
                grid.abstract_actions(obs, al);
                dec[obs.street]++;
                acts[obs.street] += al.n;
                total++;
                for (int i = 0; i < al.n; i++) {
                    HandState c(st);
                    int type, amount;
                    grid.to_concrete(obs, al.a[i], type, amount);
                    c.apply(type, amount);
                    stack.push_back(c);
                }
            }
        }
        py::dict d;
        d["decisions"] = std::vector<long long>(dec, dec + 4);
        d["actions"] = std::vector<long long>(acts, acts + 4);
        d["terminals"] = term;
        d["complete"] = complete;
        return d;
    }, py::arg("spec"), py::arg("limit") = 200000000LL);

    m.def("river_equity_exact", [](const py::sequence& hole, const py::sequence& board) {
        std::vector<int> h = to_cards(hole), bd = to_cards(board);
        if (h.size() != 2 || bd.size() != 5) throw std::invalid_argument("river_equity_exact expects 2 hole cards and a 5-card board");
        py::gil_scoped_release nogil;
        return river_equity_exact(h.data(), bd.data(), (int)bd.size());
    }, "exact equity vs one random hand on the river (all 990 opponent combos)");

    // ---- bucketers (a trainer takes either kind)
    py::class_<Bucketer, std::shared_ptr<Bucketer>>(m, "BucketerBase")
        .def("bucket", [](Bucketer& b, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2) throw std::invalid_argument("two hole cards");
            return b.bucket(h.data(), bd.data(), (int)bd.size());
        })
        .def("cache_size", &Bucketer::cache_size)
        .def("cache_stats", [](const Bucketer& b) { return cache_stats_dict(b); },
             "per street: capacity (slots), size (occupied), computes (misses), evictions")
        .def("set_cache_caps", [](Bucketer& b, const py::object& caps) { b.set_cache_caps(caps_from_py(caps)); },
             "capacities in entries for (flop, turn, river); call before training")
        .def_property_readonly("cache_caps", [](const Bucketer& b) { return b.cache_caps(); })
        .def_property_readonly("fitted", &Bucketer::fitted)
        .def_property_readonly("identity", [](const Bucketer& b) {
            const BucketerIdentity id = b.identity();
            py::dict d;
            d["kind"] = id.kind; d["n_buckets"] = id.n_buckets; d["samples"] = id.samples; d["bins"] = id.bins;
            d["fingerprint"] = id.fingerprint;
            return d;
        }, "kind, n_buckets, samples, bins and the fingerprint of the fitted parameters");

    py::class_<PotentialBucketer, Bucketer, std::shared_ptr<PotentialBucketer>>(m, "PotentialBucketer")
        .def(py::init([](int n_buckets, int samples, int bins, const py::dict& centroids, const py::dict& boundaries, const py::object& cache_caps) {
            auto b = std::make_shared<PotentialBucketer>(n_buckets, samples, bins);
            b->set_cache_caps(caps_from_py(cache_caps));
            for (auto kv : centroids) {
                int street = kv.first.cast<int>();
                if (street != FLOP && street != TURN) throw std::invalid_argument("centroids keyed by street 1 (flop) or 2 (turn)");
                auto cens = kv.second.cast<std::vector<std::vector<double>>>();
                for (const auto& c : cens) if ((int)c.size() != bins) throw std::invalid_argument("centroid length must equal bins");
                b->centroids[street] = std::move(cens);
            }
            for (auto kv : boundaries) {
                int street = kv.first.cast<int>();
                if (street != RIVER) throw std::invalid_argument("boundaries keyed by street 3 (river)");
                b->boundaries[street] = kv.second.cast<std::vector<double>>();
            }
            return b;
        }), py::arg("n_buckets"), py::arg("samples"), py::arg("bins"), py::arg("centroids"), py::arg("boundaries"), py::arg("cache_caps") = py::none())
        .def("feature", [](const PotentialBucketer& b, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2 || bd.size() < 3 || bd.size() > 4) throw std::invalid_argument("feature expects 2 hole cards and a 3..4 card board");
            std::vector<double> cdf; double mean;
            {
                py::gil_scoped_release nogil;
                b.feature(h.data(), bd.data(), (int)bd.size(), cdf, mean);
            }
            return py::make_tuple(cdf, mean);
        }, "(CDF of the next-street equity histogram, mean equity) on the canonical representative")
        .def("river_ehs", [](const PotentialBucketer& b, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2 || bd.size() != 5) throw std::invalid_argument("river_ehs expects 2 hole cards and a 5-card board");
            py::gil_scoped_release nogil;
            return b.river_ehs(h.data(), bd.data(), (int)bd.size());
        })
        .def_property_readonly("n_buckets", [](const PotentialBucketer& b) { return b.n_buckets; })
        .def_property_readonly("samples", [](const PotentialBucketer& b) { return b.samples; })
        .def_property_readonly("bins", [](const PotentialBucketer& b) { return b.bins; });

    py::class_<EquityBucketer, Bucketer, std::shared_ptr<EquityBucketer>>(m, "Bucketer")
        .def(py::init([](int n_buckets, int samples, const py::dict& boundaries, const py::object& cache_caps) {
            auto b = std::make_shared<EquityBucketer>(n_buckets, samples);
            b->set_cache_caps(caps_from_py(cache_caps));
            for (auto kv : boundaries) {
                int street = kv.first.cast<int>();
                if (street < 1 || street > 3) throw std::invalid_argument("boundaries keyed by street 1..3");
                b->boundaries[street] = kv.second.cast<std::vector<double>>();
            }
            return b;
        }), py::arg("n_buckets"), py::arg("samples"), py::arg("boundaries"), py::arg("cache_caps") = py::none())
        .def("ehs", [](EquityBucketer& b, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2) throw std::invalid_argument("two hole cards");
            return b.ehs(h.data(), bd.data(), (int)bd.size());
        })
        .def("bucket", [](EquityBucketer& b, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2) throw std::invalid_argument("two hole cards");
            return b.bucket(h.data(), bd.data(), (int)bd.size());
        })
        .def("cache_size", &EquityBucketer::cache_size)
        .def_property_readonly("n_buckets", [](const EquityBucketer& b) { return b.n_buckets; })
        .def_property_readonly("samples", [](const EquityBucketer& b) { return b.samples; });

    // ---- hand classes and precomputed bucket tables (handindex.h, buckettable.h)
    py::class_<HandIndexer, std::shared_ptr<HandIndexer>>(m, "HandIndexer")
        .def(py::init<int>(), py::arg("n_board"))
        .def_property_readonly("size", &HandIndexer::size)
        .def_property_readonly("n_board", &HandIndexer::n_board)
        .def("index", [](const HandIndexer& ix, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2 || (int)bd.size() != ix.n_board()) throw std::invalid_argument("2 hole cards and n_board board cards");
            return ix.index(h.data(), bd.data());
        })
        .def("unindex", [](const HandIndexer& ix, uint64_t idx) {
            int h[2], bd[5];
            ix.unindex(idx, h, bd);
            return py::make_tuple(std::vector<int>(h, h + 2), std::vector<int>(bd, bd + ix.n_board()));
        });

    py::class_<BucketTables, std::shared_ptr<BucketTables>>(m, "BucketTables")
        .def(py::init<>())
        .def("build", [](BucketTables& t, const Bucketer& bk, int street, int threads, const py::object& progress, double every) {
            // runs without the GIL; `progress(done, total)` is called from this thread every `every` s
            std::atomic<uint64_t> done{0};
            std::atomic<bool> finished{false};
            std::exception_ptr err;
            std::thread worker([&] {
                try { t.build(bk, street, threads, &done); } catch (...) { err = std::current_exception(); }
                finished.store(true);
            });
            const uint64_t total = t.size(street);
            {
                py::gil_scoped_release nogil;
                while (!finished.load()) {
                    for (int i = 0; i < (int)(every * 20) && !finished.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    if (!finished.load() && !progress.is_none()) {
                        py::gil_scoped_acquire g;
                        progress(done.load(), total);
                    }
                }
                worker.join();
            }
            if (err) std::rethrow_exception(err);
        }, py::arg("bucketer"), py::arg("street"), py::arg("threads") = 1, py::arg("progress") = py::none(), py::arg("every") = 10.0,
           "tabulate one street (1 flop, 2 turn, 3 river) of a fitted core bucketer")
        .def("has", &BucketTables::has)
        .def("size", &BucketTables::size)
        .def("lookup", [](const BucketTables& t, const py::sequence& hole, const py::sequence& board) {
            std::vector<int> h = to_cards(hole), bd = to_cards(board);
            if (h.size() != 2 || bd.size() < 3 || bd.size() > 5) throw std::invalid_argument("2 hole cards and a 3..5 card board");
            if (!t.has(street_of_board((int)bd.size()))) throw std::invalid_argument("street not tabulated");
            return t.lookup(h.data(), bd.data(), (int)bd.size());
        })
        .def("save", [](const BucketTables& t, const std::string& path) { py::gil_scoped_release nogil; t.save(path); })
        .def("load", [](BucketTables& t, const std::string& path, std::shared_ptr<Bucketer> expect) {
            py::gil_scoped_release nogil;
            t.load(path, expect.get());
        }, py::arg("path"), py::arg("expect") = nullptr);

    py::class_<TabulatedBucketer, Bucketer, std::shared_ptr<TabulatedBucketer>>(m, "TabulatedBucketer")
        .def(py::init([](std::shared_ptr<Bucketer> inner, std::shared_ptr<BucketTables> tables) {
            return std::make_shared<TabulatedBucketer>(std::move(inner), std::shared_ptr<const BucketTables>(tables));
        }), py::arg("inner"), py::arg("tables"))
        .def_property_readonly("inner", &TabulatedBucketer::inner);

    // ---- game = spec + grid + bucketer, hands driven from Python (tests, key checks)
    struct Game {
        Spec spec; BetGrid grid; std::shared_ptr<Bucketer> bk;
    };
    py::class_<Game>(m, "Game")
        .def(py::init([](const py::dict& spec, std::shared_ptr<Bucketer> bk) {
            Game g; g.spec = spec_from_dict(spec); g.grid = g.spec.grid(); g.bk = std::move(bk); return g;
        }), py::arg("spec"), py::arg("bucketer"))
        .def("new_hand", [](const Game& g, const std::vector<int>& deck_order, int button) {
            Hand h(g.spec.stacks(), button, g.spec.sb, g.spec.bb, g.spec.ante, deck_order, g.spec.max_street);
            h.grid = g.grid; h.bk = g.bk; h.has_grid = true;
            return h;
        })
        .def("numeric_key", [](const Game& g, const std::string& key) { return key_to_py(KeyCodec(g.spec.n_players).of_string(key)); },
             "(k1, k2) of a key string, as a checkpoint import computes it")
        .def_property_readonly("action_names", [](const Game& g) { return g.grid.names; });

    // ---- blueprint lookup (persist.h): the average strategy as flat arrays, numeric keys
    py::class_<BlueprintTable, std::shared_ptr<BlueprintTable>>(m, "BlueprintTable")
        .def_static("load", [](const std::string& path, bool keys, int n_players) {
            auto b = std::make_shared<BlueprintTable>();
            const std::string kind = file_kind(path);
            if (kind == "checkpoint") throw std::invalid_argument(path + " is a checkpoint, not a blueprint");
            {
                py::gil_scoped_release nogil;
                if (kind == "blueprint") load_blueprint_bin(path, *b, keys);
                else load_blueprint_json(path, *b, keys, n_players);
            }
            return b;
        }, py::arg("path"), py::arg("keys") = false, py::arg("n_players") = 0,
           "a binary blueprint or the JSON of BlueprintStrategy.save; keys: also keep the key strings (exports, items)")
        .def("policy", &blueprint_policy_py, py::arg("key"), py::arg("legal"),
             "BlueprintStrategy.policy(key, legal): probabilities aligned with legal, or None")
        .def("get", [](const BlueprintTable& b, const std::string& key) -> py::object {
            const long long i = b.find(key.data(), key.size());
            return i < 0 ? py::object(py::none()) : py::object(blueprint_entry_py(b, i));
        }, py::arg("key"), "(names, probs) of a key, or None (BlueprintStrategy.table.get)")
        .def("__len__", &BlueprintTable::size)
        .def("__contains__", [](const BlueprintTable& b, const py::object& key) {
            if (!py::isinstance<py::str>(key)) return false;
            const std::string k = key.cast<std::string>();
            return b.find(k.data(), k.size()) >= 0;
        })
        .def("items", [](const BlueprintTable& b) {
            if (!b.has_keys()) throw std::runtime_error("this blueprint was loaded without its key strings (keys=True)");
            py::list out;
            for (size_t i = 0; i < b.size(); i++) {
                py::tuple e = blueprint_entry_py(b, (long long)i);
                out.append(py::make_tuple(py::str(b.key_at(i)), e[0], e[1]));
            }
            return out;
        }, "[(key, names, probs)] in record order (needs the key strings)")
        .def("save", [](const BlueprintTable& b, const std::string& path) {
            py::gil_scoped_release nogil;
            save_lookup_blueprint(path, b);
        }, py::arg("path"), "binary blueprint file (needs the key strings)")
        .def("save_json", [](const BlueprintTable& b, const std::string& path) {
            py::gil_scoped_release nogil;
            write_lookup_blueprint_json(path, b);
        }, py::arg("path"), "the JSON of BlueprintStrategy.save (round(p, 5)), keys in record order (needs the key strings)")
        .def("stats", [](const BlueprintTable& b) {
            py::dict d;
            d["infosets"] = b.size();
            d["actions"] = b.n_actions();
            d["bytes"] = b.memory_bytes();
            d["key_bytes"] = b.keys.capacity() * sizeof(NodeKey);
            d["offset_bytes"] = b.off.capacity() * 4;
            d["id_bytes"] = b.ids.capacity();
            d["prob_bytes"] = b.prob_bytes();
            d["packed"] = b.packed;
            d["directory_bytes"] = b.dir_bytes();
            d["key_string_bytes"] = b.key_off.capacity() * 8 + b.key_chars.capacity();
            return d;
        }, "memory of the lookup by part (bytes)")
        .def_property_readonly("has_keys", &BlueprintTable::has_keys)
        .def_property_readonly("rounded", [](const BlueprintTable& b) { return b.rounded; })
        .def_property_readonly("iteration", [](const BlueprintTable& b) { return b.iteration; })
        .def_property_readonly("n_players", [](const BlueprintTable& b) { return b.codec.n_players(); })
        .def_property_readonly("action_names", [](const BlueprintTable& b) { return b.names; })
        .def_property_readonly("identity", [](const BlueprintTable& b) -> py::object {
            if (!b.identity.has_game) return py::none();
            return identity_dict(b.identity);
        });

    // Every accessor of a node table takes the trainer's api_mu, which train() holds for its whole
    // run (without the GIL): a Python thread reading the table while another trains waits instead
    // of racing a table resize.
    using ApiLock = std::lock_guard<std::mutex>;

    // ---- trainer
    py::class_<Trainer>(m, "Trainer")
        .def(py::init([](const py::dict& spec, std::shared_ptr<Bucketer> bk, uint64_t seed, bool linear, int threads, bool verify_keys) {
            return new Trainer(spec_from_dict(spec), std::move(bk), seed, linear, threads, verify_keys);
        }), py::arg("spec"), py::arg("bucketer"), py::arg("seed") = 0, py::arg("linear") = true, py::arg("threads") = 1,
            py::arg("verify_keys") = false)
        .def("train", [](Trainer& t, long long iterations) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            t.train(iterations);
        }, py::arg("iterations"))
        .def_property("iteration", &Trainer::iteration, &Trainer::set_iteration)
        .def_property("nodes_touched", &Trainer::nodes_touched, &Trainer::set_nodes_touched)
        .def_property_readonly("threads", [](const Trainer& t) { return t.threads; })
        .def_property("linear", [](const Trainer& t) { return t.linear; }, [](Trainer& t, bool v) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            t.linear = v;
        })
        .def_property_readonly("verify_keys", [](const Trainer& t) { return t.verify_keys; })
        .def_property_readonly("n_nodes", [](const Trainer& t) { return t.nodes.size(); })
        .def_property_readonly("action_names", [](const Trainer& t) { return t.grid.names; })
        .def_property_readonly("cache_size", [](const Trainer& t) { return t.bucketer->cache_size(); })
        .def("cache_stats", [](const Trainer& t) { return cache_stats_dict(*t.bucketer); })
        .def("table_stats", [](Trainer& t) {
            ApiLock lk(t.api_mu);
            py::dict d = table_stats_dict(t.nodes);
            d["resizes"] = t.table_resizes();
            d["tree_nodes"] = t.tree_nodes();
            d["tree_bytes"] = t.tree_bytes();
            return d;
        }, "node table: size, capacity (slots), slot / node / key-string bytes, resizes; history tree: nodes, bytes")
        .def("rng_state", [](Trainer& t) { return state_to_py(t.rng0()); })
        .def("set_rng_state", [](Trainer& t, const py::tuple& st) { state_from_py(st, t.rng0()); })
        .def("rng_states", [](Trainer& t) { return rng_states_of(t); }, "MT state of every thread's generator")
        .def("set_rng_states", [](Trainer& t, const py::sequence& states) { set_rng_states_of(t, states); })
        .def("get_node", [](Trainer& t, const std::string& key) {
            ApiLock lk(t.api_mu);
            return node_to_py(find_by_string(t.nodes, t.codec, key).node, t.grid);
        }, "[actions, regret, strategy_sum, visits] of one key, or None")
        .def("keys", [](Trainer& t) { ApiLock lk(t.api_mu); return table_keys(t.nodes); })
        .def("export_nodes", [](Trainer& t) {
            // {key: [actions, regret, strategy_sum, visits]} - the checkpoint layout of MCCFRTrainer
            ApiLock lk(t.api_mu);
            return export_table(t.nodes, t.grid);
        })
        .def("import_nodes", [](Trainer& t, const py::dict& d, bool clear) {
            ApiLock lk(t.api_mu);
            import_table(t.nodes, t.codec, t.main_arena(), t.grid, d, clear);
        }, py::arg("nodes"), py::arg("clear") = true)
        .def("add_nodes", [](Trainer& t, const py::dict& d) {
            ApiLock lk(t.api_mu);
            add_table(t.nodes, t.codec, t.main_arena(), t.grid, d);
        })
        .def("strategy", [](Trainer& t) {
            // {key: (actions, average_strategy)} - BlueprintStrategy table layout
            ApiLock lk(t.api_mu);
            return strategy_table(t.nodes, t.grid);
        })
        .def("numeric_key", [](const Trainer& t, const std::string& key) { return key_to_py(t.codec.of_string(key)); },
             "(k1, k2) of a key string")
        .def("_debug_relabel", [](Trainer& t, const std::string& key, const std::string& stored) {
            ApiLock lk(t.api_mu);
            return t.nodes.debug_set_key(t.codec.of_string(key), t.main_arena(), stored);
        }, "tests only: give the node of `key` another stored key string (verify_keys must then fail)")
        // ---- files (persist.h); every call streams between the table and the file
        .def("save_checkpoint_bin", [](Trainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            write_checkpoint_bin(path, identity_of(t), CKPT_MCCFR, t.iteration(), t.linear, rng_ptrs(t), t.grid.names, {{"nodes", &t.nodes}});
        }, py::arg("path"), "binary checkpoint (nodes, iteration, linear flag, every thread's RNG state, game identity)")
        .def("load_checkpoint_bin", [](Trainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            CheckpointScalars sc = read_checkpoint_bin(path, identity_of(t), CKPT_MCCFR, t.grid, t.codec, {{"nodes", &t.nodes}},
                                                       t.main_arena(), t.verify_keys);
            apply_scalars(t, sc);
        }, py::arg("path"), "replace the table, iteration, linear flag and RNG states; refuses a checkpoint of another game")
        .def("save_checkpoint_json", [](Trainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            write_checkpoint_json(path, t.iteration(), t.linear, t.threads, rng_ptrs(t), t.grid, {{"nodes", &t.nodes}});
        }, py::arg("path"), "the JSON checkpoint, byte for byte what json.dump writes for the same contents")
        .def("load_checkpoint_json", [](Trainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            CheckpointScalars sc = read_checkpoint_json(path, t.grid, t.codec, {{"nodes", &t.nodes}}, t.main_arena());
            apply_scalars(t, sc);
        }, py::arg("path"), "a JSON checkpoint of either trainer, streamed (no Python objects)")
        .def("save_blueprint_bin", [](Trainer& t, const std::string& path, bool rounded) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            save_table_blueprint(path, t.nodes, identity_of(t), t.iteration(), rounded, t.spec.n_players, t.grid.names);
        }, py::arg("path"), py::arg("rounded") = true,
           "binary blueprint of the average strategy; rounded: round(p, 5) like BlueprintStrategy.save")
        .def("save_blueprint_json", [](Trainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            write_table_blueprint_json(path, t.nodes, t.grid);
        }, py::arg("path"), "the JSON of trainer.strategy().save(path), byte for byte")
        .def("blueprint_table", [](Trainer& t, bool rounded, bool keys) {
            auto b = std::make_shared<BlueprintTable>();
            {
                py::gil_scoped_release nogil;
                ApiLock lk(t.api_mu);
                blueprint_from_table(t.nodes, t.grid, identity_of(t), t.iteration(), rounded, keys, *b);
            }
            return b;
        }, py::arg("rounded") = false, py::arg("keys") = false, "the average strategy as a C++ lookup (BlueprintTable)")
        .def("strategy_change", [](Trainer& t, const BlueprintTable& prev) -> py::object {
            if (prev.codec.n_players() != t.codec.n_players())
                throw std::invalid_argument("the previous blueprint's numeric keys are for " + std::to_string(prev.codec.n_players()) + " players");
            std::pair<double, long long> r;
            {
                py::gil_scoped_release nogil;
                ApiLock lk(t.api_mu);
                r = strategy_change(t.nodes, t.grid, prev);
            }
            return py::make_tuple(r.second ? py::object(py::float_(r.first)) : py::object(py::none()), r.second);
        }, py::arg("prev"), "(mean L1 change of the average strategy vs prev or None, number of shared infosets)")
        .def("count_keys", [](Trainer& t, const std::string& prefix, const std::string& suffix) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            return count_keys(t.nodes, prefix, suffix);
        }, py::arg("prefix"), py::arg("suffix") = "", "number of key strings with this prefix and suffix")
        .def("identity", [](const Trainer& t) { return identity_dict(identity_of(t)); }, "what a binary checkpoint records about the game");

    // ---- restricted Nash response
    py::class_<RNRTrainer>(m, "RNRTrainer")
        .def(py::init([](const py::dict& spec, std::shared_ptr<Bucketer> bk, uint64_t seed, bool linear, int threads,
                         double p_model, const py::dict& model, const py::dict& warm_start, double warm_visits, double regret_scale_bb,
                         bool verify_keys) {
            auto* t = new RNRTrainer(spec_from_dict(spec), std::move(bk), seed, linear, threads, p_model, warm_visits, regret_scale_bb,
                                     verify_keys);
            t->set_tables(strat_table_from_dict(model), strat_table_from_dict(warm_start));
            return t;
        }), py::arg("spec"), py::arg("bucketer"), py::arg("seed"), py::arg("linear"), py::arg("threads"), py::arg("p_model"),
            py::arg("model"), py::arg("warm_start"), py::arg("warm_visits"), py::arg("regret_scale_bb"), py::arg("verify_keys") = false)
        .def("train", [](RNRTrainer& t, long long iterations) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            t.train(iterations);
        }, py::arg("iterations"))
        .def_property("iteration", &RNRTrainer::iteration, &RNRTrainer::set_iteration)
        .def_property("nodes_touched", &RNRTrainer::nodes_touched, &RNRTrainer::set_nodes_touched)
        .def_property("planned_iters", [](const RNRTrainer& t) { return t.planned_iters; }, [](RNRTrainer& t, long long v) { t.planned_iters = v; })
        .def_property_readonly("threads", [](const RNRTrainer& t) { return t.threads; })
        .def_property("linear", [](const RNRTrainer& t) { return t.linear; }, [](RNRTrainer& t, bool v) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            t.linear = v;
        })
        .def("save_checkpoint_bin", [](RNRTrainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            write_checkpoint_bin(path, identity_of(t), CKPT_RNR, t.iteration(), t.linear, rng_ptrs(t), t.grid.names,
                                 {{"nodes", &t.hero_nodes}, {"opp_nodes", &t.opp_nodes}});
        }, py::arg("path"))
        .def("load_checkpoint_bin", [](RNRTrainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            CheckpointScalars sc = read_checkpoint_bin(path, identity_of(t), CKPT_RNR, t.grid, t.codec,
                                                       {{"nodes", &t.hero_nodes}, {"opp_nodes", &t.opp_nodes}}, t.main_arena(), t.verify_keys);
            apply_scalars(t, sc);
        }, py::arg("path"))
        .def("save_checkpoint_json", [](RNRTrainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            write_checkpoint_json(path, t.iteration(), t.linear, t.threads, rng_ptrs(t), t.grid,
                                  {{"nodes", &t.hero_nodes}, {"opp_nodes", &t.opp_nodes}});
        }, py::arg("path"))
        .def("load_checkpoint_json", [](RNRTrainer& t, const std::string& path) {
            py::gil_scoped_release nogil;
            ApiLock lk(t.api_mu);
            CheckpointScalars sc = read_checkpoint_json(path, t.grid, t.codec, {{"nodes", &t.hero_nodes}, {"opp_nodes", &t.opp_nodes}},
                                                        t.main_arena());
            apply_scalars(t, sc);
        }, py::arg("path"))
        .def_property_readonly("verify_keys", [](const RNRTrainer& t) { return t.verify_keys; })
        .def_property_readonly("n_hero", [](const RNRTrainer& t) { return t.hero_nodes.size(); })
        .def_property_readonly("n_opp", [](const RNRTrainer& t) { return t.opp_nodes.size(); })
        .def_property_readonly("cache_size", [](const RNRTrainer& t) { return t.bucketer->cache_size(); })
        .def("rng_state", [](RNRTrainer& t) { return state_to_py(t.rng0()); })
        .def("set_rng_state", [](RNRTrainer& t, const py::tuple& st) { state_from_py(st, t.rng0()); })
        .def("rng_states", [](RNRTrainer& t) { return rng_states_of(t); })
        .def("set_rng_states", [](RNRTrainer& t, const py::sequence& states) { set_rng_states_of(t, states); })
        .def("cache_stats", [](const RNRTrainer& t) { return cache_stats_dict(*t.bucketer); })
        .def("table_stats", [](RNRTrainer& t) {
            ApiLock lk(t.api_mu);
            py::dict d;
            d["hero"] = table_stats_dict(t.hero_nodes);
            d["opp"] = table_stats_dict(t.opp_nodes);
            d["resizes"] = t.table_resizes();
            d["tree_nodes"] = t.tree_nodes();
            d["tree_bytes"] = t.tree_bytes();
            return d;
        })
        .def("get_hero", [](RNRTrainer& t, const std::string& key) {
            ApiLock lk(t.api_mu);
            return node_to_py(find_by_string(t.hero_nodes, t.codec, key).node, t.grid);
        })
        .def("get_opp", [](RNRTrainer& t, const std::string& key) {
            ApiLock lk(t.api_mu);
            return node_to_py(find_by_string(t.opp_nodes, t.codec, key).node, t.grid);
        })
        .def("hero_keys", [](RNRTrainer& t) { ApiLock lk(t.api_mu); return table_keys(t.hero_nodes); })
        .def("opp_keys", [](RNRTrainer& t) { ApiLock lk(t.api_mu); return table_keys(t.opp_nodes); })
        .def("export_hero", [](RNRTrainer& t) { ApiLock lk(t.api_mu); return export_table(t.hero_nodes, t.grid); })
        .def("export_opp", [](RNRTrainer& t) { ApiLock lk(t.api_mu); return export_table(t.opp_nodes, t.grid); })
        .def("import_hero", [](RNRTrainer& t, const py::dict& d, bool clear) {
            ApiLock lk(t.api_mu);
            import_table(t.hero_nodes, t.codec, t.main_arena(), t.grid, d, clear);
        }, py::arg("nodes"), py::arg("clear") = true)
        .def("import_opp", [](RNRTrainer& t, const py::dict& d, bool clear) {
            ApiLock lk(t.api_mu);
            import_table(t.opp_nodes, t.codec, t.main_arena(), t.grid, d, clear);
        }, py::arg("nodes"), py::arg("clear") = true)
        .def("strategy", [](RNRTrainer& t) { ApiLock lk(t.api_mu); return strategy_table(t.hero_nodes, t.grid); })
        .def("rational_opponent", [](RNRTrainer& t) { ApiLock lk(t.api_mu); return strategy_table(t.opp_nodes, t.grid); })
        .def("_debug_relabel", [](RNRTrainer& t, const std::string& key, const std::string& stored, bool hero) {
            ApiLock lk(t.api_mu);
            FlatNodeTable& tab = hero ? t.hero_nodes : t.opp_nodes;
            return tab.debug_set_key(t.codec.of_string(key), t.main_arena(), stored);
        }, py::arg("key"), py::arg("stored"), py::arg("hero") = true);

    // ---- real-time search, part 1 (search.h)
    m.def("combo_index", &combo_index, "index of the hole (a, b) among the 1326 combos (a < b order), -1 if invalid");
    py::class_<SearchGame, std::shared_ptr<SearchGame>>(m, "SearchGame")
        .def(py::init([](const py::dict& spec, std::shared_ptr<Bucketer> bk, const py::object& blueprint) {
            std::shared_ptr<const BlueprintTable> bp;
            if (!blueprint.is_none()) bp = blueprint.cast<std::shared_ptr<BlueprintTable>>();
            return std::make_shared<SearchGame>(spec_from_dict(spec), std::move(bk), bp);
        }), py::arg("spec"), py::arg("bucketer"), py::arg("blueprint") = py::none(),
            "game + bucketer + blueprint lookup (BlueprintTable, or None: uniform ranges) shared by the searches of a match")
        .def_property_readonly("action_names", [](const SearchGame& g) { return g.grid.names; });

    py::class_<SubgameSearch>(m, "SubgameSearch")
        .def(py::init([](std::shared_ptr<SearchGame> game, const std::vector<int>& stacks, int button,
                         const std::vector<std::pair<int, int>>& actions, const std::vector<int>& board, int seat,
                         const std::vector<int>& hole, long long iterations, double time_budget, int threads, uint64_t seed,
                         double focus, double min_prob, bool linear, const py::object& overrides) {
            HandInput h;
            h.stacks = stacks;
            h.button = button;
            h.actions = actions;
            h.board = board;
            h.our_seat = seat;
            if (hole.size() != 2) throw std::invalid_argument("hole: two cards");
            h.our_hole[0] = hole[0];
            h.our_hole[1] = hole[1];
            if (!overrides.is_none()) {
                for (auto item : overrides.cast<py::sequence>()) {
                    py::sequence t = item.cast<py::sequence>();
                    LikelihoodOverride o;
                    o.street = t[0].cast<int>();
                    o.seat = t[1].cast<int>();
                    o.w = t[2].cast<std::vector<double>>();
                    h.overrides.push_back(std::move(o));
                }
            }
            SearchParams p;
            p.iterations = iterations;
            p.time_budget = time_budget;
            p.threads = threads;
            p.seed = seed;
            p.focus = focus;
            p.min_prob = min_prob;
            p.linear = linear;
            py::gil_scoped_release nogil;
            return new SubgameSearch(std::shared_ptr<const SearchGame>(game), h, p);
        }), py::arg("game"), py::arg("stacks"), py::arg("button"), py::arg("actions"), py::arg("board"), py::arg("seat"),
            py::arg("hole"), py::arg("iterations") = 0, py::arg("time_budget") = 2.0, py::arg("threads") = 15, py::arg("seed") = 0,
            py::arg("focus") = 0.5, py::arg("min_prob") = 1e-3, py::arg("linear") = true, py::arg("overrides") = py::none(),
            "the subgame of the hand so far: root at the start of the current round, ranges by Bayes over the blueprint")
        .def("solve", [](SubgameSearch& s) {
            SearchResult r;
            {
                py::gil_scoped_release nogil;
                r = s.solve();
            }
            py::dict d;
            py::list names;
            for (size_t i = 0; i < r.ids.size(); i++)
                names.append(r.ids[i] == INSERTED_ID ? "x" + std::to_string(r.amounts[i]) : s.game().grid.names[(size_t)r.ids[i]]);
            d["actions"] = names;
            d["types"] = r.types;
            d["amounts"] = r.amounts;
            d["ids"] = r.ids;
            d["final"] = r.final_strategy;
            d["average"] = r.average_strategy;
            d["visited"] = r.visited;
            d["iterations"] = r.iterations;
            d["traversals"] = r.traversals;
            d["focused"] = r.focused;
            d["nodes_touched"] = r.nodes_touched;
            d["forced"] = r.forced;
            d["redeals"] = r.redeals;
            d["table_size"] = r.table_size;
            d["seconds"] = r.seconds;
            d["threads"] = r.threads;
            return d;
        }, "run the solver; the strategy of our actual hole at our decision (final iteration and average)")
        .def("root_info", [](const SubgameSearch& s) {
            const HandState& st = s.root();
            py::dict d;
            std::vector<int> stacks, bets, invested;
            std::vector<bool> folded, all_in;
            for (int i = 0; i < st.n; i++) {
                stacks.push_back(st.players[i].stack);
                bets.push_back(st.players[i].street_bet);
                invested.push_back(st.players[i].invested);
                folded.push_back(st.players[i].folded);
                all_in.push_back(st.players[i].all_in);
            }
            d["street"] = st.street;
            d["pot"] = st.pot();
            d["stacks"] = stacks;
            d["street_bets"] = bets;
            d["invested"] = invested;
            d["folded"] = folded;
            d["all_in"] = all_in;
            d["to_act"] = st.to_act;
            d["n_active"] = st.n_active();
            d["board"] = std::vector<int>(st.board, st.board + st.n_board);
            d["current_bet"] = st.current_bet;
            d["min_raise"] = st.min_raise;
            d["raises_this_street"] = st.raises_this_street;
            d["n_events"] = st.n_events;
            d["n_classes"] = s.n_classes();
            d["range_seconds"] = s.range_seconds();
            return d;
        }, "the root: the public state at the start of the current betting round")
        .def("path", [](const SubgameSearch& s) {
            py::list out;
            HandState st(s.root());
            for (size_t k = 0; k < s.path().size(); k++) {
                const PathStep& p = s.path()[k];
                const NodeActions na = s.actions_at(st, (int)k);
                py::list types, amounts, ids, names;
                for (int i = 0; i < na.n; i++) {
                    types.append(na.type[i]);
                    amounts.append(na.amount[i]);
                    ids.append((int)na.id[i]);
                    names.append(na.id[i] == INSERTED_ID ? "x" + std::to_string(na.amount[i]) : s.game().grid.names[(size_t)na.id[i]]);
                }
                py::dict d;
                d["actions"] = names;
                d["actor"] = p.actor;
                d["index"] = p.index;
                d["inserted"] = p.inserted;
                d["type"] = p.type;
                d["amount"] = p.amount;
                d["types"] = types;
                d["amounts"] = amounts;
                d["ids"] = ids;
                out.append(d);
                st.apply(p.type, p.amount);
            }
            return out;
        }, "the real actions of this round: actor, index in the node's action list, inserted or on the grid, the node's actions")
        .def("ranges", [](const SubgameSearch& s, bool conditioned, int our_seat, const std::vector<int>& hole) {
            py::list out;
            const std::vector<std::vector<double>>& r = s.reach();
            const ComboTable& ct = combo_table();
            for (size_t seat = 0; seat < r.size(); seat++) {
                if (r[seat].empty()) { out.append(py::none()); continue; }
                std::vector<double> w = r[seat];
                if (conditioned && (int)seat != our_seat) {
                    for (int c = 0; c < N_COMBOS; c++) {
                        const int a = ct.c0[c], b = ct.c1[c];
                        for (int h : hole) if (a == h || b == h) w[(size_t)c] = 0.0;
                    }
                }
                out.append(w);
            }
            return out;
        }, py::arg("conditioned") = false, py::arg("our_seat") = -1, py::arg("hole") = std::vector<int>(),
           "per seat: 1326 weights (None if folded); conditioned: the other seats' combos holding `hole` removed")
        .def("set_reach", &SubgameSearch::set_reach, py::arg("seat"), py::arg("weights"),
             "replace a live seat's range (1326 weights)")
        .def("likelihood", [](const SubgameSearch& s, int seat, const py::object& actions) {
            std::vector<std::pair<int, int>> acts;
            if (actions.is_none()) {
                for (const PathStep& p : s.path()) acts.emplace_back(p.type, p.amount);
            } else {
                acts = actions.cast<std::vector<std::pair<int, int>>>();
            }
            long long missing = 0;
            std::vector<double> w = s.likelihood(seat, acts, missing);
            return py::make_tuple(w, missing);
        }, py::arg("seat"), py::arg("actions") = py::none(),
           "(1326 likelihoods of `seat`'s round actions under the average strategy, missing lookups)")
        .def("_probe_path", [](const SubgameSearch& s, int k, int h0, int h1) -> py::object {
            std::vector<double> v = s.probe_path(k, h0, h1);
            if (v.empty()) return py::none();
            return py::cast(v);
        }, "tests: the strategy used at real-path node k for that hole of the node's actor")
        .def("river_exploitability", [](const SubgameSearch& s, int kind) {
            std::vector<double> r;
            {
                py::gil_scoped_release nogil;
                r = s.river_exploitability(kind);
            }
            return py::make_tuple(r[0], r[1], r[2]);
        }, py::arg("kind") = 0,
           "2 players, river root: (exploitability, BR gains of the two live seats) in bb per deal of the subgame, of "
           "0 the search's average strategy, 1 its final iteration, 2 the blueprint as the agent plays it")
        .def("_node_at_path", [](const SubgameSearch& s, int k, int h0, int h1) -> py::object {
            std::vector<double> regret, ssum;
            long long visits = 0;
            if (!s.node_at_path(k, h0, h1, regret, ssum, visits)) return py::none();
            py::dict d;
            d["regret"] = regret;
            d["strategy_sum"] = ssum;
            d["visits"] = visits;
            return d;
        }, "tests: the raw node (regret, strategy_sum, visits) of real-path node k's actor holding that hole, or None")
        .def("_sample_deals", [](const SubgameSearch& s, int n, bool focused, uint64_t seed) {
            return s.sample_deals(n, focused, seed);
        }, py::arg("n"), py::arg("focused") = false, py::arg("seed") = 1,
           "tests: deals as the solver draws them: 2 cards per seat (in seat order), then the rest of the board");
}
