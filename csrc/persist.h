// Checkpoints and blueprints on disk, read and written by the C++ core without Python objects
// (2026-09-25).  Formats: docs/backends.md, "Binary checkpoints and blueprints".
//
//   * binary checkpoint  (magic "NPCKPT01")  every node of the trainer's table(s) with its numeric
//                        key, key string, action ids, regrets, strategy sums and visits; iteration,
//                        linear flag, every thread's RNG state; the identity of the game (spec,
//                        grid, bucketer fingerprint): a resume into another game is refused
//   * binary blueprint   (magic "NPBLUE01")  the average strategy as flat arrays sorted by numeric
//                        key (what BlueprintTable, the lookup of the agents, loads in a few reads),
//                        then the key strings (for exports; the lookup can skip them)
//   * JSON               the old formats, read by a streaming parser and written with the exact
//                        bytes json.dump writes (py_float_repr), so they stay interchangeable with
//                        the Python code
//
// Files are written to <path>.tmp and renamed over <path> when complete, and carry checksums.
// Records are written in increasing numeric-key order, so the same contents always give the same
// bytes (whatever the insertion history of the table).
#pragma once
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "binio.h"
#include "mccfr.h"
#include "rnr.h"

namespace negp {

// Version of the numeric keys (nodetable.h: HistHash, node_key, KeyCodec).  Bump it whenever they
// change: files with another scheme are then re-keyed from their key strings when loaded.
constexpr uint32_t KEY_SCHEME = 1;
constexpr uint32_t CHECKPOINT_VERSION = 1;
constexpr uint32_t BLUEPRINT_VERSION = 1;
static const char CHECKPOINT_MAGIC[9] = "NPCKPT01";
static const char BLUEPRINT_MAGIC[9] = "NPBLUE01";
constexpr uint8_t CKPT_MCCFR = 0, CKPT_RNR = 1;

// ------------------------------------------------------------------ rounding like round(x, 5)
// Python's round(x, 5): the decimal with 5 digits after the point nearest to the exact binary
// value of x (ties to even), read back as the nearest double.  std::to_chars(fixed, 5) rounds the
// exact value the same way and std::from_chars reads correctly rounded (checked against Python on
// 403,370 values with every exact tie of the form odd/64: no difference).  Fast path for
// probabilities: k = the nearest integer to x * 1e5 whenever the product is not within 1e-6 of a
// half (its rounding error is below 2e-11 for x <= 1), and k / 1e5 is the double nearest to the
// decimal k * 10^-5, exactly what Python returns (no ties possible for k / 10^5).
inline double round5_exact(double x) {
    if (!std::isfinite(x)) return x;
    char buf[400];
    auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::fixed, 5);
    double y = 0.0;
    std::from_chars(buf, r.ptr, y);
    return y;
}
inline double round5(double x) {
    if (x >= 0.0 && x <= 1.0) {
        const double y = x * 100000.0;
        const double f = std::floor(y);
        const double d = y - f;
        if (std::fabs(d - 0.5) > 1e-6) return (d < 0.5 ? f : f + 1.0) / 100000.0;
    }
    return round5_exact(x);
}

// Rounded probabilities are stored as k (probability = k / 100000.0, the very double round() gave,
// since no ties exist for k / 10^5 and the division is correctly rounded).  `v` packs iff it is
// bit for bit such a value (so -0.0, NaN, negatives and unrounded values do not).
inline bool pack5(double v, uint32_t& k) {
    if (!(v >= 0.0) || v > 42949.0 || std::signbit(v)) return false;
    const double kk = std::nearbyint(v * 100000.0);
    const double back = kk / 100000.0;
    if (std::memcmp(&back, &v, sizeof(double)) != 0) return false;
    k = (uint32_t)kk;
    return true;
}
inline double unpack5(uint32_t k) { return (double)k / 100000.0; }

// ------------------------------------------------------------------ game identity
struct GameIdentity {
    uint32_t key_scheme = KEY_SCHEME;
    bool has_game = false;  // false: unknown (a blueprint converted from JSON)
    Spec spec;
    std::vector<std::string> grid_names;
    BucketerIdentity bucketer;
};

inline GameIdentity game_identity(const Spec& spec, const BetGrid& grid, const Bucketer& bk) {
    GameIdentity id;
    id.has_game = true;
    id.spec = spec;
    id.grid_names = grid.names;
    id.bucketer = bk.identity();
    return id;
}

inline void write_identity(BinWriter& w, const GameIdentity& id) {
    w.u32(id.key_scheme);
    w.u8(id.has_game ? 1 : 0);
    if (!id.has_game) return;
    const Spec& s = id.spec;
    for (int v : {s.n_players, s.stack_bb, s.sb, s.bb, s.ante, s.max_street, s.max_raises_per_street, s.n_buckets}) w.i32(v);
    w.u8(s.forbid_open_limp ? 1 : 0);
    w.u8(s.allow_all_in ? 1 : 0);
    for (const std::vector<double>* fr : {&s.preflop_fracs, &s.postflop_fracs}) {
        w.u16((uint16_t)fr->size());
        for (double f : *fr) w.f64(f);
    }
    w.u16((uint16_t)id.grid_names.size());
    for (const std::string& n : id.grid_names) w.str16(n);
    w.str16(id.bucketer.kind);
    w.i32(id.bucketer.n_buckets);
    w.i32(id.bucketer.samples);
    w.i32(id.bucketer.bins);
    w.u64(id.bucketer.fingerprint);
}

inline GameIdentity read_identity(BinReader& r) {
    GameIdentity id;
    id.key_scheme = r.u32();
    id.has_game = r.u8() != 0;
    if (!id.has_game) return id;
    Spec& s = id.spec;
    s.n_players = r.i32();
    s.stack_bb = r.i32();
    s.sb = r.i32();
    s.bb = r.i32();
    s.ante = r.i32();
    s.max_street = r.i32();
    s.max_raises_per_street = r.i32();
    s.n_buckets = r.i32();
    s.forbid_open_limp = r.u8() != 0;
    s.allow_all_in = r.u8() != 0;
    for (std::vector<double>* fr : {&s.preflop_fracs, &s.postflop_fracs}) {
        fr->resize(r.u16());
        for (double& f : *fr) f = r.f64();
    }
    id.grid_names.resize(r.u16());
    for (std::string& n : id.grid_names) n = r.str16();
    id.bucketer.kind = r.str16();
    id.bucketer.n_buckets = r.i32();
    id.bucketer.samples = r.i32();
    id.bucketer.bins = r.i32();
    id.bucketer.fingerprint = r.u64();
    return id;
}

inline std::string fracs_text(const std::vector<double>& v) {
    std::string s = "(";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += ", ";
        py_float_repr(v[i], s);
    }
    return s + ")";
}

// "" when a table saved in game `file` may be loaded into a trainer of game `mine`, else what
// differs.  The bucketer is not compared for preflop-only games (their keys use the 169 classes).
inline std::string identity_diff(const GameIdentity& file, const GameIdentity& mine) {
    if (!file.has_game) return "";
    std::string d;
    auto add = [&](const std::string& what, const std::string& a, const std::string& b) {
        if (a == b) return;
        if (!d.empty()) d += "; ";
        d += what + " " + a + " in the file, " + b + " here";
    };
    auto i2s = [](long long v) { return std::to_string(v); };
    const Spec &a = file.spec, &b = mine.spec;
    add("players", i2s(a.n_players), i2s(b.n_players));
    add("stack_bb", i2s(a.stack_bb), i2s(b.stack_bb));
    add("blinds", i2s(a.sb) + "/" + i2s(a.bb), i2s(b.sb) + "/" + i2s(b.bb));
    add("ante", i2s(a.ante), i2s(b.ante));
    add("max_street", i2s(a.max_street), i2s(b.max_street));
    add("max_raises_per_street", i2s(a.max_raises_per_street), i2s(b.max_raises_per_street));
    add("n_buckets", i2s(a.n_buckets), i2s(b.n_buckets));
    add("forbid_open_limp", i2s(a.forbid_open_limp), i2s(b.forbid_open_limp));
    add("allow_all_in", i2s(a.allow_all_in), i2s(b.allow_all_in));
    add("preflop_fracs", fracs_text(a.preflop_fracs), fracs_text(b.preflop_fracs));
    add("postflop_fracs", fracs_text(a.postflop_fracs), fracs_text(b.postflop_fracs));
    if (b.max_street > PREFLOP) {
        const BucketerIdentity &x = file.bucketer, &y = mine.bucketer;
        add("bucketer", x.kind + "/" + i2s(x.n_buckets) + " buckets/" + i2s(x.samples) + " samples/" + i2s(x.bins) + " bins",
            y.kind + "/" + i2s(y.n_buckets) + " buckets/" + i2s(y.samples) + " samples/" + i2s(y.bins) + " bins");
        if (x.kind == y.kind && x.n_buckets == y.n_buckets && x.samples == y.samples && x.bins == y.bins &&
            x.fingerprint != y.fingerprint)
            add("bucketer fit (fingerprint of the cut points / centroids)", i2s((long long)x.fingerprint), i2s((long long)y.fingerprint));
    }
    return d;
}

// ------------------------------------------------------------------ action ids
// The grid id of `name` in a node of `street`: ids are per street (the narrow grid has "r0.5" and
// "r1" preflop AND postflop, with different ids), and the traversals compare ids, so an imported
// node must get the ids the traversal gives it.  Unknown street (a key not in canonical form):
// the first id with that name.
inline int grid_action_id(BetGrid& grid, int street, const std::string& name) {
    if (name == "f") return 0;
    if (name == "c") return 1;
    if (name == "a") return 2;
    if (street >= PREFLOP && street <= RIVER) {
        for (int id : grid.ids_for(street)) if (grid.names[id] == name) return id;
    }
    return grid.id_of(name);
}

// the street of a canonical key string ("F|BB|2|b3|..." -> FLOP), -1 when it has none
inline int street_of_key(const char* key) {
    if (key[0] == '\0' || key[1] != '|') return -1;
    switch (key[0]) {
        case 'P': return PREFLOP;
        case 'F': return FLOP;
        case 'T': return TURN;
        case 'R': return RIVER;
        default: return -1;
    }
}

// file action index -> grid id, per street (-1 .. 3), for a names table read from a file
class IdMap {
public:
    IdMap(BetGrid& grid, const std::vector<std::string>& names) {
        if (names.size() > 256) throw std::runtime_error("more than 256 action names");
        for (int s = 0; s < 5; s++) {
            map_[s].resize(names.size());
            for (size_t i = 0; i < names.size(); i++) {
                const int id = grid_action_id(grid, s - 1, names[i]);
                if (id > 255) throw std::runtime_error("more than 256 action names");
                map_[s][i] = (uint8_t)id;
            }
        }
    }
    // false when an index is out of range
    bool map(int street, const uint8_t* in, int n, uint8_t* out) const {
        const std::vector<uint8_t>& m = map_[street + 1];
        for (int i = 0; i < n; i++) {
            if (in[i] >= m.size()) return false;
            out[i] = m[in[i]];
        }
        return true;
    }

private:
    std::vector<uint8_t> map_[5];
};

// ------------------------------------------------------------------ table order
struct OrderedSlot {
    uint64_t k1;
    uint32_t slot;
};

// the slots of `t` in increasing (k1, k2) order: the order of every file written from a table
inline std::vector<OrderedSlot> ordered_slots(const FlatNodeTable& t) {
    const size_t cap = t.capacity();
    if (cap > 0xFFFFFFFFULL) throw std::runtime_error("node table too large for ordered_slots");
    std::vector<OrderedSlot> out;
    out.reserve(t.size());
    for (size_t i = 0; i < cap; i++) {
        NodeKey k;
        const char* key;
        Node* n;
        if (t.at(i, k, key, n)) out.push_back({k.k1, (uint32_t)i});
    }
    std::sort(out.begin(), out.end(), [&](const OrderedSlot& a, const OrderedSlot& b) {
        if (a.k1 != b.k1) return a.k1 < b.k1;
        return t.k2_at(a.slot) < t.k2_at(b.slot);
    });
    return out;
}

// ------------------------------------------------------------------ binary checkpoint
struct NamedTable {
    const char* name;  // "nodes" / "opp_nodes" (the JSON member names)
    FlatNodeTable* table;
};

inline void write_rng_states(BinWriter& w, const std::vector<PyRandom*>& rngs) {
    w.u32((uint32_t)rngs.size());
    std::vector<uint32_t> words;
    for (PyRandom* g : rngs) {
        int idx = 0;
        g->get_state(words, idx);
        w.array(words);
        w.i32(idx);
    }
}

// Layout (little-endian):
//   "NPCKPT01" u32 version u32 flags | identity | u8 kind (0 MCCFR, 1 RNR) | i64 iteration | u8 linear
//   | u32 n_rng, n_rng x (624 x u32 words, i32 index) | u16 n_names, n_names x str16 (the names the
//   action ids index) | u32 n_tables, per table: str16 name, u64 n_nodes, n_nodes x record
//   | u64 checksum of every byte before it
//   record = u64 k1, u64 k2, str16 key, u8 n, n x u8 ids, n x f64 regret, n x f64 strategy_sum, i64 visits
inline void write_checkpoint_bin(const std::string& path, const GameIdentity& ident, uint8_t kind, long long iteration,
                                 bool linear, const std::vector<PyRandom*>& rngs, const std::vector<std::string>& names,
                                 const std::vector<NamedTable>& tables) {
    BinWriter w(path);
    w.bytes(CHECKPOINT_MAGIC, 8);
    w.u32(CHECKPOINT_VERSION);
    w.u32(0);
    write_identity(w, ident);
    w.u8(kind);
    w.i64(iteration);
    w.u8(linear ? 1 : 0);
    write_rng_states(w, rngs);
    w.u16((uint16_t)names.size());
    for (const std::string& n : names) w.str16(n);
    w.u32((uint32_t)tables.size());
    for (const NamedTable& nt : tables) {
        w.str16(nt.name, std::strlen(nt.name));
        const std::vector<OrderedSlot> order = ordered_slots(*nt.table);
        w.u64(order.size());
        for (const OrderedSlot& o : order) {
            NodeKey k;
            const char* key;
            Node* n;
            nt.table->at(o.slot, k, key, n);
            w.u64(k.k1);
            w.u64(k.k2);
            w.str16(key, std::strlen(key));
            w.u8(n->n);
            w.bytes(n->acts, n->n);
            w.bytes(n->regret(), sizeof(double) * n->n);
            w.bytes(n->strategy_sum(), sizeof(double) * n->n);
            w.i64(n->visits);
        }
    }
    w.u64(w.checksum());
    w.finish();
}

struct CheckpointScalars {
    long long iteration = 0;
    bool linear = true;
    std::vector<std::vector<uint32_t>> rng_words;
    std::vector<int> rng_index;
};

inline void read_rng_states(BinReader& r, CheckpointScalars& sc) {
    const uint32_t n = r.u32();
    if ((uint64_t)n * (PyRandom::N * 4 + 4) > r.remaining()) throw std::runtime_error("corrupt file (RNG states): " + r.path());
    sc.rng_words.resize(n);
    sc.rng_index.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        r.array(sc.rng_words[i], PyRandom::N);
        sc.rng_index[i] = r.i32();
        if (sc.rng_index[i] < 0 || sc.rng_index[i] > PyRandom::N) throw std::runtime_error("corrupt file (RNG index): " + r.path());
    }
}

// Load a binary checkpoint into `tables` (matched by name, every table of the file must exist
// here) after checking the game identity; the scalars are returned for the caller to apply.  On any
// error the tables are left empty and the exception propagates.
inline CheckpointScalars read_checkpoint_bin(const std::string& path, const GameIdentity& mine, uint8_t kind, BetGrid& grid,
                                             const KeyCodec& codec, const std::vector<NamedTable>& tables, int arena,
                                             bool verify_keys) {
    BinReader r(path);
    char magic[8];
    r.bytes(magic, 8);
    if (std::memcmp(magic, CHECKPOINT_MAGIC, 8) != 0) throw std::runtime_error(path + " is not a binary checkpoint (NPCKPT01)");
    const uint32_t version = r.u32();
    if (version != CHECKPOINT_VERSION) throw std::runtime_error(path + ": checkpoint format version " + std::to_string(version) + " is not supported");
    r.u32();  // flags
    const GameIdentity file = read_identity(r);
    const std::string diff = identity_diff(file, mine);
    if (!diff.empty()) throw std::runtime_error("refusing to resume from " + path + ": it was saved for another game (" + diff + ")");
    const uint8_t k = r.u8();
    if (k != kind) throw std::runtime_error(path + " is a checkpoint of " + (k == CKPT_RNR ? "an RNR" : "an MCCFR") + " trainer");
    CheckpointScalars sc;
    sc.iteration = r.i64();
    sc.linear = r.u8() != 0;
    read_rng_states(r, sc);
    std::vector<std::string> names(r.u16());
    for (std::string& n : names) n = r.str16();
    const IdMap ids(grid, names);
    const bool rekey = file.key_scheme != KEY_SCHEME;
    for (const NamedTable& nt : tables) nt.table->clear();
    try {
        const uint32_t n_tables = r.u32();
        std::vector<bool> seen(tables.size(), false);
        std::string key;
        for (uint32_t ti = 0; ti < n_tables; ti++) {
            const std::string tname = r.str16();
            int which = -1;
            for (size_t j = 0; j < tables.size(); j++) if (tname == tables[j].name) which = (int)j;
            if (which < 0 || seen[which]) throw std::runtime_error(path + ": unexpected table '" + tname + "'");
            seen[which] = true;
            FlatNodeTable& t = *tables[which].table;
            const uint64_t n = r.u64();
            if (n > r.remaining() / 27) throw std::runtime_error("corrupt file (node count): " + path);  // a record has >= 27 bytes
            t.reserve((size_t)n);
            for (uint64_t i = 0; i < n; i++) {
                NodeKey nk;
                nk.k1 = r.u64();
                nk.k2 = r.u64();
                r.str16(key);
                const int na = r.u8();
                if (na > MAX_ACTIONS) throw std::runtime_error("corrupt file (a node with " + std::to_string(na) + " actions): " + path);
                uint8_t raw[MAX_ACTIONS], acts[MAX_ACTIONS];
                double reg[MAX_ACTIONS], ss[MAX_ACTIONS];
                r.bytes(raw, (size_t)na);
                r.bytes(reg, sizeof(double) * na);
                r.bytes(ss, sizeof(double) * na);
                const long long visits = r.i64();
                if (!ids.map(street_of_key(key.c_str()), raw, na, acts)) throw std::runtime_error("corrupt file (action index): " + path);
                if (rekey || verify_keys) {
                    const NodeKey c = codec.of_string(key);
                    if (!rekey && c != nk) throw std::runtime_error(path + ": the numeric key stored for '" + key + "' is not the key of that string");
                    nk = c;
                }
                FlatNodeTable::Found f = t.get_or_create(nk, arena, na, [&](Node& node, NodeArena& a) {
                    node.init(acts, na);
                    return a.copy_key(key.data(), key.size());
                });
                if (!f.created) {
                    if (std::strcmp(f.key, key.c_str()) == 0) throw std::runtime_error("corrupt file (key '" + key + "' twice): " + path);
                    throw std::runtime_error("numeric infoset key collision: '" + key + "' and '" + std::string(f.key) + "'");
                }
                Node* node = f.node;
                for (int a = 0; a < na; a++) { node->regret()[a] = reg[a]; node->strategy_sum()[a] = ss[a]; }
                node->visits = visits;
            }
        }
        r.expect_checksum("checkpoint");
    } catch (...) {
        for (const NamedTable& nt : tables) nt.table->clear();
        throw;
    }
    return sc;
}

// ------------------------------------------------------------------ JSON checkpoint
// The JSON the Python trainers write, with the same bytes: {"iteration": I, "linear": B, "nodes":
// {key: [[names], [regret], [strategy_sum], visits], ...}[, "opp_nodes": {...}], "backend": "cpp",
// "threads": T, "rng_states": [[624 words, index], ...]}; keys in the table's slot order (the order
// export_nodes() hands to json.dump).
inline void json_node(const char* key, const Node& n, const BetGrid& grid, std::string& o) {
    py_json_string(key, std::strlen(key), o);
    o += ": [[";
    for (int i = 0; i < n.n; i++) {
        if (i) o += ", ";
        const std::string& nm = grid.names[n.acts[i]];
        py_json_string(nm.data(), nm.size(), o);
    }
    o += "], [";
    for (int i = 0; i < n.n; i++) {
        if (i) o += ", ";
        py_float_repr(n.regret()[i], o);
    }
    o += "], [";
    for (int i = 0; i < n.n; i++) {
        if (i) o += ", ";
        py_float_repr(n.strategy_sum()[i], o);
    }
    o += "], ";
    o += std::to_string(n.visits);
    o += ']';
}

inline void write_rng_json(const std::vector<PyRandom*>& rngs, std::string& o) {
    o += "[";
    std::vector<uint32_t> words;
    for (size_t t = 0; t < rngs.size(); t++) {
        if (t) o += ", ";
        int idx = 0;
        rngs[t]->get_state(words, idx);
        o += '[';
        for (size_t i = 0; i < words.size(); i++) {
            o += std::to_string(words[i]);
            o += ", ";
        }
        o += std::to_string(idx);
        o += ']';
    }
    o += "]";
}

inline void write_checkpoint_json(const std::string& path, long long iteration, bool linear, int threads,
                                  const std::vector<PyRandom*>& rngs, const BetGrid& grid, const std::vector<NamedTable>& tables) {
    TextWriter w(path);
    std::string& o = w.buf();
    o += "{\"iteration\": ";
    o += std::to_string(iteration);
    o += ", \"linear\": ";
    o += linear ? "true" : "false";
    for (const NamedTable& nt : tables) {
        o += ", \"";
        o += nt.name;
        o += "\": {";
        bool first = true;
        nt.table->for_each([&](const char* key, Node& n) {
            if (!first) o += ", ";
            first = false;
            json_node(key, n, grid, o);
            w.maybe_flush();
        });
        o += '}';
    }
    o += ", \"backend\": \"cpp\", \"threads\": ";
    o += std::to_string(threads);
    o += ", \"rng_states\": ";
    write_rng_json(rngs, o);
    o += '}';
    w.finish();
}

// one member value of "nodes": [names, regret, strategy_sum, visits] (extra items ignored, as the
// Python import ignores them)
struct JsonNodeRow {
    std::vector<std::string> names;
    std::vector<double> reg, ss;
    long long visits = 0;
};

inline void read_json_row(JsonReader& j, JsonNodeRow& row, const std::string& key) {
    row.names.clear();
    row.reg.clear();
    row.ss.clear();
    j.expect('[');
    bool first = true;
    int item = 0;
    while (j.next_member(first, ']')) {
        if (item == 0) {
            j.expect('[');
            bool f = true;
            while (j.next_member(f, ']')) row.names.push_back(j.string());
        } else if (item == 1 || item == 2) {
            std::vector<double>& v = item == 1 ? row.reg : row.ss;
            j.expect('[');
            bool f = true;
            while (j.next_member(f, ']')) v.push_back(j.number());
        } else if (item == 3) {
            row.visits = j.integer();
        } else {
            j.skip_value();
        }
        item++;
    }
    if (item < 4) j.fail("node '" + key + "' has fewer than 4 fields");
    if (row.names.size() > MAX_ACTIONS) j.fail("node '" + key + "' has more than 8 actions");
    if (row.reg.size() != row.names.size() || row.ss.size() != row.names.size())
        j.fail("node '" + key + "': regret / strategy_sum lengths differ from the action list");
}

inline void read_json_nodes(JsonReader& j, FlatNodeTable& t, BetGrid& grid, const KeyCodec& codec, int arena) {
    t.clear();  // a repeated member replaces the earlier one, as in json.load
    j.expect('{');
    bool first = true;
    JsonNodeRow row;
    while (j.next_member(first, '}')) {
        const std::string key = j.string();
        j.expect(':');
        read_json_row(j, row, key);
        const int street = street_of_key(key.c_str());
        const int k = (int)row.names.size();
        uint8_t ids[MAX_ACTIONS];
        for (int i = 0; i < k; i++) ids[i] = (uint8_t)grid_action_id(grid, street, row.names[i]);
        Node* n = get_or_create_by_string(t, codec, key, ids, k, arena).node;
        n->init(ids, k);  // a key given twice: the last row wins, as in json.load
        for (int i = 0; i < k; i++) { n->regret()[i] = row.reg[i]; n->strategy_sum()[i] = row.ss[i]; }
        n->visits = row.visits;
    }
}

// Load a JSON checkpoint (either trainer's) into `tables` by member name.  Required: "iteration",
// "linear" and the first table ("nodes"); other tables default to empty; "rng_states" is optional
// (the Python trainer does not write it).  On any error the tables are left empty.
inline CheckpointScalars read_checkpoint_json(const std::string& path, BetGrid& grid, const KeyCodec& codec,
                                              const std::vector<NamedTable>& tables, int arena) {
    JsonReader j(path);
    CheckpointScalars sc;
    bool have_it = false, have_lin = false, have_nodes = false;
    for (const NamedTable& nt : tables) nt.table->clear();
    try {
        j.expect('{');
        bool first = true;
        while (j.next_member(first, '}')) {
            const std::string name = j.string();
            j.expect(':');
            int which = -1;
            for (size_t t = 0; t < tables.size(); t++) if (name == tables[t].name) which = (int)t;
            if (name == "iteration") {
                sc.iteration = j.integer();
                have_it = true;
            } else if (name == "linear") {
                sc.linear = j.boolean();
                have_lin = true;
            } else if (which >= 0) {
                read_json_nodes(j, *tables[which].table, grid, codec, arena);
                if (which == 0) have_nodes = true;
            } else if (name == "rng_states") {
                sc.rng_words.clear();
                sc.rng_index.clear();
                if (j.peek() == 'n') { j.skip_value(); continue; }
                j.expect('[');
                bool f = true;
                while (j.next_member(f, ']')) {
                    std::vector<int64_t> st;
                    j.expect('[');
                    bool g = true;
                    while (j.next_member(g, ']')) st.push_back(j.integer());
                    if (st.size() != PyRandom::N + 1) j.fail("bad MT state (" + std::to_string(st.size()) + " values)");
                    std::vector<uint32_t> words(PyRandom::N);
                    for (int i = 0; i < PyRandom::N; i++) {
                        if (st[i] < 0 || st[i] > 0xFFFFFFFFLL) j.fail("bad MT state word");
                        words[i] = (uint32_t)st[i];
                    }
                    if (st[PyRandom::N] < 0 || st[PyRandom::N] > PyRandom::N) j.fail("bad MT state index");
                    sc.rng_words.push_back(std::move(words));
                    sc.rng_index.push_back((int)st[PyRandom::N]);
                }
            } else {
                j.skip_value();
            }
        }
        if (!have_it || !have_lin || !have_nodes)
            throw std::runtime_error(path + ": not a checkpoint (needs \"iteration\", \"linear\" and \"" + std::string(tables[0].name) + "\")");
    } catch (...) {
        for (const NamedTable& nt : tables) nt.table->clear();
        throw;
    }
    return sc;
}

// ------------------------------------------------------------------ blueprint lookup
// The average strategy as flat arrays: numeric keys sorted by (k1, k2), a directory over the top
// bits of k1, per infoset an offset into the action names / probabilities.  About 16 + 4 + 4 +
// 9 x (actions) bytes per infoset, plus the key strings only when asked for (exports, `items`).
//
// policy(key, legal) is BlueprintStrategy.policy: the entry of `key` (None if absent), the
// probability of each legal name (the last one if a name appears twice, 0.0 if absent), their sum
// with CPython's sum() (py_sum), None if it is <= 0, else each value divided by the sum.  Keys are
// found by numeric key: equal strings, equal keys; a different string with the same 128-bit key
// is the (never observed) collision case, refused at load when the key strings are there.
class BlueprintTable {
public:
    GameIdentity identity;  // has_game false when unknown (a table loaded from JSON)
    long long iteration = -1;
    bool rounded = false;   // probabilities are round(p, 5), as in the JSON of BlueprintStrategy.save
    bool packed = false;    // stored as k in kprobs (probability = k / 100000.0), else as doubles in probs
    KeyCodec codec{2};      // the codec of the numeric keys (its player count is stored in the file)
    std::vector<std::string> names;  // distinct action names
    std::vector<NodeKey> keys;
    std::vector<uint32_t> off;       // size() + 1 offsets into ids / probabilities
    std::vector<uint8_t> ids;        // index into names
    std::vector<double> probs;       // !packed
    std::vector<uint32_t> kprobs;    // packed
    std::vector<uint64_t> key_off;   // optional: size() + 1 offsets into key_chars
    std::vector<char> key_chars;

    size_t size() const { return keys.size(); }
    size_t n_actions() const { return ids.size(); }
    bool has_keys() const { return key_off.size() == keys.size() + 1; }
    double prob(size_t t) const { return packed ? unpack5(kprobs[t]) : probs[t]; }

    // doubles -> k when every value packs (rounded tables); else keep the doubles
    void try_pack() {
        std::vector<uint32_t> k(probs.size());
        for (size_t t = 0; t < probs.size(); t++) if (!pack5(probs[t], k[t])) return;
        kprobs = std::move(k);
        probs.clear();
        probs.shrink_to_fit();
        packed = true;
    }

    long long find(const NodeKey& k) const {
        if (keys.empty()) return -1;
        const size_t j = (size_t)(k.k1 >> shift_);
        for (uint32_t i = dir_[j], e = dir_[j + 1]; i < e; i++) {
            if (keys[i].k1 == k.k1 && keys[i].k2 == k.k2) return (long long)i;
            if (keys[i].k1 > k.k1) break;
        }
        return -1;
    }
    long long find(const char* key, size_t len) const {
        const long long i = find(codec.of_string(key, len));
        if (i >= 0 && has_keys()) {  // key strings present: the string must match too
            const size_t a = (size_t)key_off[(size_t)i], b = (size_t)key_off[(size_t)i + 1];
            if (b - a != len || std::memcmp(key_chars.data() + a, key, len) != 0) return -1;
        }
        return i;
    }
    std::string key_at(size_t i) const {
        return std::string(key_chars.data() + key_off[i], (size_t)(key_off[i + 1] - key_off[i]));
    }
    int name_index(const char* s, size_t len) const {
        for (size_t i = 0; i < names.size(); i++)
            if (names[i].size() == len && std::memcmp(names[i].data(), s, len) == 0) return (int)i;
        return -1;
    }

    // BlueprintStrategy.policy for record i and the legal names given by their index in `names`
    // (-1: not a name of this table); false = None
    bool policy_at(long long i, const int* legal, int n_legal, double* out) const {
        const uint32_t a = off[(size_t)i], b = off[(size_t)i + 1];
        for (int l = 0; l < n_legal; l++) {
            double v = 0.0;
            if (legal[l] >= 0)
                for (uint32_t t = a; t < b; t++) if (ids[t] == legal[l]) v = prob(t);
            out[l] = v;
        }
        const double s = py_sum(out, n_legal);
        if (s <= 0) return false;
        for (int l = 0; l < n_legal; l++) out[l] = out[l] / s;
        return true;
    }

    size_t prob_bytes() const { return probs.capacity() * 8 + kprobs.capacity() * 4; }
    size_t memory_bytes() const {
        return keys.capacity() * sizeof(NodeKey) + off.capacity() * 4 + ids.capacity() + prob_bytes() + dir_.capacity() * 4 +
               key_off.capacity() * 8 + key_chars.capacity();
    }
    size_t dir_bytes() const { return dir_.capacity() * 4; }

    // after keys / off / ids / probs are set (keys sorted and distinct): the directory
    void build_index() {
        const size_t n = keys.size();
        int bits = 1;
        while (bits < 40 && ((size_t)1 << bits) < n / 2 + 1) bits++;
        shift_ = 64 - bits;
        const size_t nb = (size_t)1 << bits;
        dir_.assign(nb + 1, 0);
        size_t i = 0;
        for (size_t j = 0; j <= nb; j++) {
            while (i < n && (size_t)(keys[i].k1 >> shift_) < j) i++;
            dir_[j] = (uint32_t)i;
        }
        dir_.shrink_to_fit();
    }
    // names deduplicated (a grid has "r1" preflop and postflop), ids rewritten to the first index
    void dedupe_names() {
        std::vector<std::string> uniq;
        std::vector<uint8_t> canon(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            size_t j = 0;
            while (j < uniq.size() && uniq[j] != names[i]) j++;
            if (j == uniq.size()) uniq.push_back(names[i]);
            canon[i] = (uint8_t)j;
        }
        if (uniq.size() == names.size()) return;
        for (uint8_t& id : ids) id = canon[id];
        names = std::move(uniq);
    }
    void check_sorted(const std::string& what) const {
        for (size_t i = 1; i < keys.size(); i++) {
            const NodeKey &a = keys[i - 1], &b = keys[i];
            if (!(a.k1 < b.k1 || (a.k1 == b.k1 && a.k2 < b.k2))) throw std::runtime_error(what + ": numeric keys not sorted or not distinct");
        }
    }

private:
    std::vector<uint32_t> dir_;
    int shift_ = 63;
};

// Layout (little-endian):
//   "NPBLUE01" u32 version u32 flags | identity | i64 iteration (-1 unknown) | u8 rounded
//   | u8 probability format (1: u32 k, probability = k / 100000.0; 0: f64) | u32 codec players
//   | u16 n_names, n_names x str16 | u64 n, u64 m | n x (u64 k1, u64 k2), sorted
//   | (n + 1) x u32 offsets | m x u8 action index | m x (u32 k or f64) probability
//   | u64 checksum (all above) | n x str16 key strings (record order) | u64 checksum (whole file)
template <class RecordSource>
inline void write_blueprint_bin(const std::string& path, const GameIdentity& ident, long long iteration, bool rounded,
                                int codec_players, const std::vector<std::string>& names, uint64_t n, uint64_t m,
                                RecordSource&& src) {
    BinWriter w(path);
    w.bytes(BLUEPRINT_MAGIC, 8);
    w.u32(BLUEPRINT_VERSION);
    w.u32(0);
    write_identity(w, ident);
    w.i64(iteration);
    w.u8(rounded ? 1 : 0);
    w.u8(src.packed() ? 1 : 0);
    w.u32((uint32_t)codec_players);
    w.u16((uint16_t)names.size());
    for (const std::string& s : names) w.str16(s);
    w.u64(n);
    w.u64(m);
    src.keys(w);
    src.offsets(w);
    src.ids(w);
    src.probs(w);
    w.u64(w.checksum());
    src.strings(w);
    w.u64(w.checksum());
    w.finish();
}

// the blueprint of a trainer's table straight from its nodes (average strategy, rounded or not)
struct TableBlueprintSource {
    FlatNodeTable& t;
    std::vector<OrderedSlot> order;
    bool rounded;
    uint64_t m = 0;       // actions
    bool pack = false;    // rounded and every rounded value packs (always, for average strategies)

    TableBlueprintSource(FlatNodeTable& t_, bool rounded_) : t(t_), order(ordered_slots(t_)), rounded(rounded_) {
        NodeKey k;
        const char* key;
        double avg[MAX_ACTIONS];
        uint32_t kk;
        pack = rounded;
        for (size_t i = 0; i < order.size(); i++) {
            Node* n = node(i, k, key);
            m += n->n;
            if (pack) {
                n->average_strategy(avg);
                for (int a = 0; a < n->n; a++) if (!pack5(round5(avg[a]), kk)) pack = false;
            }
        }
    }
    Node* node(size_t i, NodeKey& k, const char*& key) const {
        Node* n = nullptr;
        t.at(order[i].slot, k, key, n);
        return n;
    }
    bool packed() const { return pack; }
    uint64_t actions() const { return m; }
    void keys(BinWriter& w) const {
        NodeKey k;
        const char* key;
        for (size_t i = 0; i < order.size(); i++) { node(i, k, key); w.u64(k.k1); w.u64(k.k2); }
    }
    void offsets(BinWriter& w) const {
        uint64_t acc = 0;
        NodeKey k;
        const char* key;
        w.u32(0);
        for (size_t i = 0; i < order.size(); i++) {
            acc += node(i, k, key)->n;
            if (acc > 0xFFFFFFFFULL) throw std::runtime_error("blueprint too large (more than 2^32 actions)");
            w.u32((uint32_t)acc);
        }
    }
    void ids(BinWriter& w) const {
        NodeKey k;
        const char* key;
        for (size_t i = 0; i < order.size(); i++) { Node* n = node(i, k, key); w.bytes(n->acts, n->n); }
    }
    void probs(BinWriter& w) const {
        NodeKey k;
        const char* key;
        double avg[MAX_ACTIONS];
        uint32_t ks[MAX_ACTIONS];
        for (size_t i = 0; i < order.size(); i++) {
            Node* n = node(i, k, key);
            n->average_strategy(avg);
            if (rounded) for (int a = 0; a < n->n; a++) avg[a] = round5(avg[a]);
            if (pack) {
                for (int a = 0; a < n->n; a++) pack5(avg[a], ks[a]);
                w.bytes(ks, sizeof(uint32_t) * n->n);
            } else {
                w.bytes(avg, sizeof(double) * n->n);
            }
        }
    }
    void strings(BinWriter& w) const {
        NodeKey k;
        const char* key;
        for (size_t i = 0; i < order.size(); i++) { node(i, k, key); w.str16(key, std::strlen(key)); }
    }
};

// a BlueprintTable (with its key strings) as a file
struct LookupBlueprintSource {
    const BlueprintTable& b;
    bool packed() const { return b.packed; }
    void keys(BinWriter& w) const { w.array(b.keys); }
    void offsets(BinWriter& w) const { w.array(b.off); }
    void ids(BinWriter& w) const { w.array(b.ids); }
    void probs(BinWriter& w) const {
        if (b.packed) w.array(b.kprobs);
        else w.array(b.probs);
    }
    void strings(BinWriter& w) const {
        for (size_t i = 0; i < b.size(); i++) w.str16(b.key_chars.data() + b.key_off[i], (size_t)(b.key_off[i + 1] - b.key_off[i]));
    }
};

inline void save_table_blueprint(const std::string& path, FlatNodeTable& t, const GameIdentity& ident, long long iteration,
                                 bool rounded, int codec_players, const std::vector<std::string>& grid_names) {
    TableBlueprintSource src(t, rounded);
    write_blueprint_bin(path, ident, iteration, rounded, codec_players, grid_names, src.order.size(), src.actions(), src);
}

inline void save_lookup_blueprint(const std::string& path, const BlueprintTable& b) {
    if (!b.has_keys()) throw std::runtime_error("this blueprint was loaded without its key strings (load it with keys=True to save it)");
    LookupBlueprintSource src{b};
    write_blueprint_bin(path, b.identity, b.iteration, b.rounded, b.codec.n_players(), b.names, b.size(), b.n_actions(), src);
}

// a BlueprintTable from a binary blueprint file; with `with_keys` the key strings too
inline void load_blueprint_bin(const std::string& path, BlueprintTable& b, bool with_keys) {
    BinReader r(path);
    char magic[8];
    r.bytes(magic, 8);
    if (std::memcmp(magic, BLUEPRINT_MAGIC, 8) != 0) throw std::runtime_error(path + " is not a binary blueprint (NPBLUE01)");
    const uint32_t version = r.u32();
    if (version != BLUEPRINT_VERSION) throw std::runtime_error(path + ": blueprint format version " + std::to_string(version) + " is not supported");
    r.u32();
    b.identity = read_identity(r);
    b.iteration = r.i64();
    b.rounded = r.u8() != 0;
    const uint8_t format = r.u8();
    if (format > 1) throw std::runtime_error("corrupt file (probability format): " + path);
    b.packed = format == 1;
    b.codec.set_players((int)r.u32());
    b.names.resize(r.u16());
    for (std::string& s : b.names) s = r.str16();
    const uint64_t n = r.u64(), m = r.u64();
    r.array(b.keys, n);
    r.array(b.off, n + 1);
    r.array(b.ids, m);
    b.probs.clear();
    b.kprobs.clear();
    if (b.packed) r.array(b.kprobs, m);
    else r.array(b.probs, m);
    r.expect_checksum("blueprint records");
    if (b.off.front() != 0 || b.off.back() != m) throw std::runtime_error("corrupt file (offsets): " + path);
    for (size_t i = 0; i < n; i++) if (b.off[i] > b.off[i + 1]) throw std::runtime_error("corrupt file (offsets): " + path);
    for (uint8_t id : b.ids) if (id >= b.names.size()) throw std::runtime_error("corrupt file (action index): " + path);
    b.check_sorted(path);
    const bool rekey = b.identity.key_scheme != KEY_SCHEME;
    b.key_off.clear();
    b.key_chars.clear();
    if (with_keys || rekey) {
        b.key_off.reserve(n + 1);
        b.key_off.push_back(0);
        std::string s;
        for (uint64_t i = 0; i < n; i++) {
            r.str16(s);
            b.key_chars.insert(b.key_chars.end(), s.begin(), s.end());
            b.key_off.push_back(b.key_chars.size());
        }
        r.expect_checksum("blueprint key strings");
        b.key_chars.shrink_to_fit();
    }
    b.dedupe_names();
    if (rekey) {
        // numeric keys of another scheme: recompute them from the strings and re-sort
        std::vector<size_t> perm(n);
        std::vector<NodeKey> nk(n);
        for (size_t i = 0; i < n; i++) {
            perm[i] = i;
            nk[i] = b.codec.of_string(b.key_chars.data() + b.key_off[i], (size_t)(b.key_off[i + 1] - b.key_off[i]));
        }
        std::sort(perm.begin(), perm.end(), [&](size_t x, size_t y) { return nk[x].k1 != nk[y].k1 ? nk[x].k1 < nk[y].k1 : nk[x].k2 < nk[y].k2; });
        BlueprintTable c;
        for (size_t i : perm) {
            c.keys.push_back(nk[i]);
            c.off.push_back((uint32_t)c.ids.size());
            for (uint32_t t = b.off[i]; t < b.off[i + 1]; t++) {
                c.ids.push_back(b.ids[t]);
                if (b.packed) c.kprobs.push_back(b.kprobs[t]);
                else c.probs.push_back(b.probs[t]);
            }
        }
        c.off.push_back((uint32_t)c.ids.size());
        if (with_keys) {
            c.key_off.push_back(0);
            for (size_t i : perm) {
                c.key_chars.insert(c.key_chars.end(), b.key_chars.begin() + (long long)b.key_off[i], b.key_chars.begin() + (long long)b.key_off[i + 1]);
                c.key_off.push_back(c.key_chars.size());
            }
        }
        b.keys = std::move(c.keys);
        b.off = std::move(c.off);
        b.ids = std::move(c.ids);
        b.probs = std::move(c.probs);
        b.kprobs = std::move(c.kprobs);
        b.key_off = std::move(c.key_off);
        b.key_chars = std::move(c.key_chars);
        b.identity.key_scheme = KEY_SCHEME;
        b.check_sorted(path + " (re-keyed)");
    }
    b.build_index();
}

// the n_active field of a canonical key (0 if none): a JSON blueprint's player count is the largest
inline int key_players(const char* s, size_t len) {
    int bars = 0;
    size_t i = 0;
    for (; i < len && bars < 2; i++) if (s[i] == '|') bars++;
    int v = 0, digits = 0;
    for (; i < len && s[i] >= '0' && s[i] <= '9' && digits < 3; i++, digits++) v = v * 10 + (s[i] - '0');
    return (i < len && s[i] == '|' && digits > 0) ? v : 0;
}

// a BlueprintTable from the JSON of BlueprintStrategy.save ({"table": {key: [names, probs]}});
// numbers as Python's float() reads them.  Keys given twice: the last entry wins (json.load).
// `n_players` <= 0: the codec's player count from the keys (largest n_active field).
inline void load_blueprint_json(const std::string& path, BlueprintTable& b, bool with_keys, int n_players) {
    JsonReader j(path);
    struct Entry {
        uint64_t key_at, act_at;
        uint32_t key_len, n_act;
    };
    std::vector<Entry> entries;
    std::vector<char> chars;
    std::vector<uint8_t> acts;
    std::vector<double> ps;
    std::unordered_map<std::string, int> name_idx;
    std::vector<std::string> names;
    std::vector<std::string> row_names;
    std::vector<double> row_probs;
    int max_players = 0;
    bool have_table = false;
    j.expect('{');
    bool first = true;
    while (j.next_member(first, '}')) {
        const std::string member = j.string();
        j.expect(':');
        if (member != "table") { j.skip_value(); continue; }
        have_table = true;
        entries.clear(); chars.clear(); acts.clear(); ps.clear();
        j.expect('{');
        bool f = true;
        while (j.next_member(f, '}')) {
            const std::string key = j.string();
            j.expect(':');
            row_names.clear();
            row_probs.clear();
            j.expect('[');
            bool g = true;
            int item = 0;
            while (j.next_member(g, ']')) {
                if (item == 0 || item == 1) {
                    j.expect('[');
                    bool h = true;
                    while (j.next_member(h, ']')) {
                        if (item == 0) row_names.push_back(j.string());
                        else row_probs.push_back(j.number());
                    }
                } else {
                    j.skip_value();
                }
                item++;
            }
            if (item < 2) j.fail("entry '" + key + "' needs [names, probabilities]");
            const size_t k = std::min(row_names.size(), row_probs.size());  // zip() stops at the shorter
            Entry e{chars.size(), acts.size(), (uint32_t)key.size(), (uint32_t)k};
            chars.insert(chars.end(), key.begin(), key.end());
            for (size_t i = 0; i < k; i++) {
                auto it = name_idx.find(row_names[i]);
                int id;
                if (it == name_idx.end()) {
                    if (names.size() >= 256) j.fail("more than 256 action names");
                    id = (int)names.size();
                    name_idx.emplace(row_names[i], id);
                    names.push_back(row_names[i]);
                } else {
                    id = it->second;
                }
                acts.push_back((uint8_t)id);
                ps.push_back(row_probs[i]);
            }
            entries.push_back(e);
            max_players = std::max(max_players, key_players(key.data(), key.size()));
        }
    }
    if (!have_table) throw std::runtime_error(path + ": not a blueprint (no \"table\")");
    b = BlueprintTable();
    b.identity.has_game = false;
    b.codec.set_players(n_players > 0 ? n_players : (max_players >= 2 ? max_players : 2));
    b.names = names;
    std::vector<NodeKey> nk(entries.size());
    std::vector<size_t> perm(entries.size());
    for (size_t i = 0; i < entries.size(); i++) {
        nk[i] = b.codec.of_string(chars.data() + entries[i].key_at, entries[i].key_len);
        perm[i] = i;
    }
    std::stable_sort(perm.begin(), perm.end(), [&](size_t x, size_t y) { return nk[x].k1 != nk[y].k1 ? nk[x].k1 < nk[y].k1 : nk[x].k2 < nk[y].k2; });
    auto same_string = [&](size_t x, size_t y) {
        return entries[x].key_len == entries[y].key_len &&
               std::memcmp(chars.data() + entries[x].key_at, chars.data() + entries[y].key_at, entries[x].key_len) == 0;
    };
    b.off.push_back(0);
    if (with_keys) b.key_off.push_back(0);
    for (size_t p = 0; p < perm.size(); p++) {
        const size_t i = perm[p];
        if (p + 1 < perm.size() && nk[perm[p + 1]] == nk[i]) {  // the same key again later in sorted order
            if (!same_string(i, perm[p + 1]))
                throw std::runtime_error("numeric infoset key collision in " + path + ": '" +
                                         std::string(chars.data() + entries[i].key_at, entries[i].key_len) + "'");
            continue;  // stable sort: the later entry of the file follows and wins
        }
        b.keys.push_back(nk[i]);
        for (uint32_t t = 0; t < entries[i].n_act; t++) {
            b.ids.push_back(acts[entries[i].act_at + t]);
            b.probs.push_back(ps[entries[i].act_at + t]);
        }
        if (b.ids.size() > 0xFFFFFFFFULL) throw std::runtime_error("blueprint too large (more than 2^32 actions)");
        b.off.push_back((uint32_t)b.ids.size());
        if (with_keys) {
            b.key_chars.insert(b.key_chars.end(), chars.begin() + (long long)entries[i].key_at,
                               chars.begin() + (long long)(entries[i].key_at + entries[i].key_len));
            b.key_off.push_back(b.key_chars.size());
        }
    }
    b.dedupe_names();
    b.try_pack();  // the JSON of BlueprintStrategy.save holds round(p, 5) values: 4 bytes each instead of 8
    b.rounded = b.packed;
    // no growth slack in the lookup (the vectors grew by push_back)
    b.keys.shrink_to_fit();
    b.off.shrink_to_fit();
    b.ids.shrink_to_fit();
    b.probs.shrink_to_fit();
    b.key_off.shrink_to_fit();
    b.key_chars.shrink_to_fit();
    b.build_index();
}

// a BlueprintTable from a trainer's table (average strategy; rounded like the JSON or not)
inline void blueprint_from_table(FlatNodeTable& t, const BetGrid& grid, const GameIdentity& ident, long long iteration,
                                 bool rounded, bool with_keys, BlueprintTable& b) {
    TableBlueprintSource src(t, rounded);
    b = BlueprintTable();
    b.identity = ident;
    b.iteration = iteration;
    b.rounded = rounded;
    b.packed = src.packed();
    b.codec.set_players(ident.spec.n_players);
    b.names = grid.names;
    const size_t n = src.order.size();
    b.keys.resize(n);
    b.off.reserve(n + 1);
    b.off.push_back(0);
    const uint64_t m = src.actions();
    if (m > 0xFFFFFFFFULL) throw std::runtime_error("blueprint too large (more than 2^32 actions)");
    b.ids.reserve((size_t)m);
    if (b.packed) b.kprobs.reserve((size_t)m);
    else b.probs.reserve((size_t)m);
    if (with_keys) b.key_off.push_back(0);
    double avg[MAX_ACTIONS];
    for (size_t i = 0; i < n; i++) {
        const char* key;
        Node* node = src.node(i, b.keys[i], key);
        node->average_strategy(avg);
        for (int a = 0; a < node->n; a++) {
            b.ids.push_back(node->acts[a]);
            const double p = rounded ? round5(avg[a]) : avg[a];
            if (b.packed) {
                uint32_t k = 0;
                pack5(p, k);
                b.kprobs.push_back(k);
            } else {
                b.probs.push_back(p);
            }
        }
        b.off.push_back((uint32_t)b.ids.size());
        if (with_keys) {
            const size_t len = std::strlen(key);
            b.key_chars.insert(b.key_chars.end(), key, key + len);
            b.key_off.push_back(b.key_chars.size());
        }
    }
    b.key_off.shrink_to_fit();
    b.key_chars.shrink_to_fit();
    b.dedupe_names();
    b.build_index();
}

// the JSON of BlueprintStrategy.save ({"table": {key: [names, [round(p, 5), ...]]}}) with the
// bytes json.dump writes; from a trainer's table in slot order (the order of trainer.strategy())
inline void write_table_blueprint_json(const std::string& path, FlatNodeTable& t, const BetGrid& grid) {
    TextWriter w(path);
    std::string& o = w.buf();
    o += "{\"table\": {";
    bool first = true;
    double avg[MAX_ACTIONS];
    t.for_each([&](const char* key, Node& n) {
        if (!first) o += ", ";
        first = false;
        py_json_string(key, std::strlen(key), o);
        o += ": [[";
        for (int i = 0; i < n.n; i++) {
            if (i) o += ", ";
            const std::string& nm = grid.names[n.acts[i]];
            py_json_string(nm.data(), nm.size(), o);
        }
        o += "], [";
        n.average_strategy(avg);
        for (int i = 0; i < n.n; i++) {
            if (i) o += ", ";
            py_float_repr(round5(avg[i]), o);
        }
        o += "]]";
        w.maybe_flush();
    });
    o += "}}";
    w.finish();
}

// ... and from a BlueprintTable (record order)
inline void write_lookup_blueprint_json(const std::string& path, const BlueprintTable& b) {
    if (!b.has_keys()) throw std::runtime_error("this blueprint was loaded without its key strings (load it with keys=True to export it)");
    TextWriter w(path);
    std::string& o = w.buf();
    o += "{\"table\": {";
    for (size_t i = 0; i < b.size(); i++) {
        if (i) o += ", ";
        py_json_string(b.key_chars.data() + b.key_off[i], (size_t)(b.key_off[i + 1] - b.key_off[i]), o);
        o += ": [[";
        for (uint32_t t = b.off[i]; t < b.off[i + 1]; t++) {
            if (t > b.off[i]) o += ", ";
            const std::string& nm = b.names[b.ids[t]];
            py_json_string(nm.data(), nm.size(), o);
        }
        o += "], [";
        for (uint32_t t = b.off[i]; t < b.off[i + 1]; t++) {
            if (t > b.off[i]) o += ", ";
            py_float_repr(round5(b.prob(t)), o);
        }
        o += "]]";
        w.maybe_flush();
    }
    o += "}}";
    w.finish();
}

// Binary checkpoint -> JSON, streamed (no trainer needed): the layout of the C++ trainer's JSON,
// nodes in file order.
inline void checkpoint_bin_to_json(const std::string& src, const std::string& dst) {
    BinReader r(src);
    char magic[8];
    r.bytes(magic, 8);
    if (std::memcmp(magic, CHECKPOINT_MAGIC, 8) != 0) throw std::runtime_error(src + " is not a binary checkpoint (NPCKPT01)");
    const uint32_t version = r.u32();
    if (version != CHECKPOINT_VERSION) throw std::runtime_error(src + ": checkpoint format version " + std::to_string(version) + " is not supported");
    r.u32();
    read_identity(r);
    r.u8();
    CheckpointScalars sc;
    sc.iteration = r.i64();
    sc.linear = r.u8() != 0;
    read_rng_states(r, sc);
    std::vector<std::string> names(r.u16());
    for (std::string& n : names) n = r.str16();
    TextWriter w(dst);
    std::string& o = w.buf();
    o += "{\"iteration\": ";
    o += std::to_string(sc.iteration);
    o += ", \"linear\": ";
    o += sc.linear ? "true" : "false";
    const uint32_t n_tables = r.u32();
    std::string key;
    for (uint32_t ti = 0; ti < n_tables; ti++) {
        const std::string tname = r.str16();
        o += ", ";
        py_json_string(tname.data(), tname.size(), o);
        o += ": {";
        const uint64_t n = r.u64();
        for (uint64_t i = 0; i < n; i++) {
            r.u64();
            r.u64();
            r.str16(key);
            Node node;
            node.n = r.u8();
            if (node.n > MAX_ACTIONS) throw std::runtime_error("corrupt file (node actions): " + src);
            r.bytes(node.acts, node.n);
            r.bytes(node.regret(), sizeof(double) * node.n);
            r.bytes(node.strategy_sum(), sizeof(double) * node.n);
            node.visits = r.i64();
            if (i) o += ", ";
            py_json_string(key.data(), key.size(), o);
            o += ": [[";
            for (int a = 0; a < node.n; a++) {
                if (a) o += ", ";
                if (node.acts[a] >= names.size()) throw std::runtime_error("corrupt file (action index): " + src);
                py_json_string(names[node.acts[a]].data(), names[node.acts[a]].size(), o);
            }
            o += "], [";
            for (int a = 0; a < node.n; a++) { if (a) o += ", "; py_float_repr(node.regret()[a], o); }
            o += "], [";
            for (int a = 0; a < node.n; a++) { if (a) o += ", "; py_float_repr(node.strategy_sum()[a], o); }
            o += "], ";
            o += std::to_string(node.visits);
            o += ']';
            w.maybe_flush();
        }
        o += '}';
    }
    r.expect_checksum("checkpoint");
    o += ", \"backend\": \"cpp\", \"threads\": ";
    o += std::to_string(sc.rng_words.size());
    o += ", \"rng_states\": [";
    for (size_t t = 0; t < sc.rng_words.size(); t++) {
        if (t) o += ", ";
        o += '[';
        for (uint32_t v : sc.rng_words[t]) { o += std::to_string(v); o += ", "; }
        o += std::to_string(sc.rng_index[t]);
        o += ']';
    }
    o += "]}";
    w.finish();
}

// ------------------------------------------------------------------ L1 change
// scripts/train_blueprint.py strategy_change(prev, cur) with cur = the table's average strategy:
// over the table's nodes in slot order (the order of trainer.strategy()), those whose key is in
// `prev` with the same action names add sum(|cur - prev|) (CPython sum: py_sum) to a running
// total; the mean over them, or n = 0 (None).  Same order of additions, so the same double.
inline std::pair<double, long long> strategy_change(FlatNodeTable& t, const BetGrid& grid, const BlueprintTable& prev) {
    std::vector<int> to_prev(grid.names.size(), -1);  // grid id -> index in prev.names
    for (size_t g = 0; g < grid.names.size(); g++) to_prev[g] = prev.name_index(grid.names[g].data(), grid.names[g].size());
    double tot = 0.0;
    long long n = 0;
    t.for_each_keyed([&](const NodeKey& k, const char*, Node& node) {
        const long long i = prev.find(k);
        if (i < 0) return;
        const uint32_t a = prev.off[(size_t)i], b = prev.off[(size_t)i + 1];
        if (b - a != node.n) return;
        for (int j = 0; j < node.n; j++) if (to_prev[node.acts[j]] != prev.ids[a + j]) return;
        double avg[MAX_ACTIONS], d[MAX_ACTIONS];
        node.average_strategy(avg);
        for (int j = 0; j < node.n; j++) d[j] = std::fabs(avg[j] - prev.prob(a + j));
        tot += py_sum(d, node.n);
        n++;
    });
    return {n ? tot / (double)n : 0.0, n};
}

}  // namespace negp
