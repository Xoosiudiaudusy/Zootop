// Precomputed postflop buckets: one byte per hand class (handindex.h), per street.
//
// A bucket is a pure function of the canonical form (EquityBucketer seeds its Monte-Carlo with
// the form's hash, PotentialBucketer likewise, and its river is exact), so computing it once for
// every class gives exactly the values bucket() would compute during training.  In training a
// lookup is then one index computation and one byte load instead of a Monte-Carlo run on every
// cache miss (the bounded caches of abstraction.h miss almost always on the turn and river:
// 14M / 123M classes).  Sizes: flop 1.29 MB, turn 13.96 MB, river 123.16 MB.
//
// File (little-endian): "NPBT" magic, u32 version = 1, identity of the bucketer (kind string,
// n_buckets, samples, bins, fingerprint), then per street FLOP..RIVER: u32 n_board, u64 size
// (0 = street not tabulated), size bytes, u64 FNV-1a of those bytes.  A table is only valid
// for the bucketer whose identity it carries; load() checks it.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "binio.h"
#include "handindex.h"

namespace negp {

class BucketTables {
public:
    BucketTables() {
        for (int s = FLOP; s <= RIVER; s++) ix_[s].reset(new HandIndexer(s == FLOP ? 3 : (s == TURN ? 4 : 5)));
    }

    const HandIndexer& indexer(int street) const { return *ix_[street]; }
    bool has(int street) const { return !t_[street].empty(); }
    uint64_t size(int street) const { return ix_[street]->size(); }
    const BucketerIdentity& identity() const { return id_; }

    int lookup(const int* hole, const int* board, int n_board) const {
        const int s = street_of_board(n_board);
        return t_[s][ix_[s]->index(hole, board)];
    }
    uint8_t at(int street, uint64_t idx) const { return t_[street][idx]; }

    // compute street `street` with `threads` threads (chunks handed out dynamically, so slow and
    // fast cores both stay busy); `done` (optional) counts finished classes for progress reports
    void build(const Bucketer& bk, int street, int threads, std::atomic<uint64_t>* done = nullptr) {
        if (street < FLOP || street > RIVER) throw std::invalid_argument("street must be flop, turn or river");
        if (!bk.fitted()) throw std::invalid_argument("bucketer not fitted");
        const BucketerIdentity id = bk.identity();
        if (id.n_buckets > 256) throw std::invalid_argument("bucket tables hold at most 256 buckets");
        if (any() && !same_identity(id_, id)) throw std::invalid_argument("tables of another bucketer: build into a fresh object");
        id_ = id;
        if (street == RIVER && build_river_by_board(bk, threads, done)) return;
        const HandIndexer& ix = *ix_[street];
        const uint64_t n = ix.size();
        std::vector<uint8_t> out(n);
        std::atomic<uint64_t> next{0};
        std::atomic<bool> failed{false};
        std::string error;
        std::mutex err_mu;
        const uint64_t CHUNK = 4096;
        auto work = [&]() {
            try {
                int hole[2], board[5];
                CanonicalForm cf;
                for (;;) {
                    const uint64_t lo = next.fetch_add(CHUNK, std::memory_order_relaxed);
                    if (lo >= n || failed.load(std::memory_order_relaxed)) return;
                    const uint64_t hi = lo + CHUNK < n ? lo + CHUNK : n;
                    for (uint64_t i = lo; i < hi; i++) {
                        ix.unindex(i, hole, board);
                        canonical_form(hole, board, ix.n_board(), cf);
                        const int b = bk.bucket_of_form(cf);
                        if (b < 0 || b > 255) throw std::runtime_error("bucket out of 0..255");
                        out[i] = (uint8_t)b;
                    }
                    if (done) done->fetch_add(hi - lo, std::memory_order_relaxed);
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!failed.exchange(true)) error = e.what();
            }
        };
        const int T = threads < 1 ? 1 : threads;
        std::vector<std::thread> pool;
        for (int t = 1; t < T; t++) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
        if (failed.load()) throw std::runtime_error("bucket table build failed: " + error);
        t_[street].swap(out);
    }

    void save(const std::string& path) const {
        FILE* f = open_file(path, "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        auto w = [&](const void* p, size_t n) {
            if (n && std::fwrite(p, 1, n, f) != n) { std::fclose(f); throw std::runtime_error("write failed: " + path); }
        };
        auto u32 = [&](uint32_t v) { w(&v, 4); };
        auto u64 = [&](uint64_t v) { w(&v, 8); };
        w("NPBT", 4);
        u32(1);
        u32((uint32_t)id_.kind.size());
        w(id_.kind.data(), id_.kind.size());
        u32((uint32_t)id_.n_buckets);
        u32((uint32_t)id_.samples);
        u32((uint32_t)id_.bins);
        u64(id_.fingerprint);
        for (int s = FLOP; s <= RIVER; s++) {
            u32((uint32_t)ix_[s]->n_board());
            u64(t_[s].size());
            w(t_[s].data(), t_[s].size());
            u64(fnv1a(t_[s]));
        }
        if (std::fclose(f) != 0) throw std::runtime_error("write failed: " + path);
    }

    // load a table file; with `expect` (the bucketer that will use it) the identities must match
    void load(const std::string& path, const Bucketer* expect = nullptr) {
        FILE* f = open_file(path, "rb");
        if (!f) throw std::runtime_error("cannot read " + path);
        auto r = [&](void* p, size_t n) {
            if (n && std::fread(p, 1, n, f) != n) { std::fclose(f); throw std::runtime_error("truncated bucket table: " + path); }
        };
        auto u32 = [&]() { uint32_t v; r(&v, 4); return v; };
        auto u64 = [&]() { uint64_t v; r(&v, 8); return v; };
        char magic[4];
        r(magic, 4);
        if (std::memcmp(magic, "NPBT", 4) != 0) { std::fclose(f); throw std::runtime_error("not a bucket table: " + path); }
        if (u32() != 1) { std::fclose(f); throw std::runtime_error("unsupported bucket table version: " + path); }
        BucketerIdentity id;
        const uint32_t kl = u32();
        if (kl > 64) { std::fclose(f); throw std::runtime_error("corrupt bucket table: " + path); }
        id.kind.resize(kl);
        r(&id.kind[0], kl);
        id.n_buckets = (int)u32();
        id.samples = (int)u32();
        id.bins = (int)u32();
        id.fingerprint = u64();
        std::vector<uint8_t> tabs[4];
        for (int s = FLOP; s <= RIVER; s++) {
            const uint32_t nb = u32();
            const uint64_t n = u64();
            if ((int)nb != ix_[s]->n_board() || (n != 0 && n != ix_[s]->size())) { std::fclose(f); throw std::runtime_error("corrupt bucket table: " + path); }
            tabs[s].resize(n);
            r(tabs[s].data(), n);
            if (u64() != fnv1a(tabs[s])) { std::fclose(f); throw std::runtime_error("bucket table checksum mismatch: " + path); }
        }
        std::fclose(f);
        if (expect && !same_identity(id, expect->identity()))
            throw std::runtime_error("bucket table " + path + " was built for another bucketer (" + id.kind + ", fingerprint differs)");
        id_ = id;
        for (int s = FLOP; s <= RIVER; s++) t_[s].swap(tabs[s]);
    }

    static bool same_identity(const BucketerIdentity& a, const BucketerIdentity& b) {
        return a.kind == b.kind && a.n_buckets == b.n_buckets && a.samples == b.samples && a.bins == b.bins &&
               a.fingerprint == b.fingerprint;
    }

    // the canonical 5-card boards: the lexicographically smallest sorted image under the 24 suit
    // permutations (every board is a relabelling of exactly one of them)
    static std::vector<std::array<int, 5>> canonical_boards() {
        std::vector<std::array<int, 5>> out;
        int b[5];
        for (b[0] = 0; b[0] < 52; b[0]++)
            for (b[1] = b[0] + 1; b[1] < 52; b[1]++)
                for (b[2] = b[1] + 1; b[2] < 52; b[2]++)
                    for (b[3] = b[2] + 1; b[3] < 52; b[3]++)
                        for (b[4] = b[3] + 1; b[4] < 52; b[4]++) {
                            bool smallest = true;
                            for (int p = 1; p < 24 && smallest; p++) {
                                int q[5];
                                for (int i = 0; i < 5; i++) q[i] = (b[i] >> 2) * 4 + SUIT_PERMS[p][b[i] & 3];
                                std::sort(q, q + 5);
                                if (std::lexicographical_compare(q, q + 5, b, b + 5)) smallest = false;
                            }
                            if (smallest) out.push_back({b[0], b[1], b[2], b[3], b[4]});
                        }
        return out;
    }

private:
    // River by board, for bucketers with a per-board batch (Bucketer::river_buckets_all: the
    // potential-aware exact river equity, all 1,081 holes of a board from one sorted pass instead of
    // 990 evaluations per hand).  Every class (hole, board) is a relabelling of a hand on a canonical
    // board, so the canonical boards cover the table; a class met on several boards (or twice on one)
    // gets the same bucket each time, because the batch gives bucket()'s own number (a pure function
    // of the class).  Returns false (nothing built) when the bucketer has no batch.
    bool build_river_by_board(const Bucketer& bk, int threads, std::atomic<uint64_t>* done) {
        {
            const int probe[5] = {0, 5, 10, 15, 20};
            std::vector<uint8_t> tmp(1326);
            if (!bk.river_buckets_all(probe, pair_index(), tmp.data())) return false;
        }
        const std::vector<std::array<int, 5>> boards = canonical_boards();
        const HandIndexer& ix = *ix_[RIVER];
        const uint64_t n = ix.size();
        std::vector<uint8_t> out(n, 0);
        std::vector<uint8_t> seen((n + 7) / 8, 0);  // coverage check: every class written at least once
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::mutex seen_mu;
        const int (*idx)[52] = pair_index();
        auto work = [&]() {
            std::vector<uint8_t> b(1326);
            std::vector<uint64_t> mine;
            for (size_t i = next.fetch_add(1); i < boards.size() && !failed.load(); i = next.fetch_add(1)) {
                const int* board = boards[i].data();
                if (!bk.river_buckets_all(board, idx, b.data())) { failed.store(true); return; }
                bool on[52] = {false};
                for (int k = 0; k < 5; k++) on[board[k]] = true;
                mine.clear();
                for (int c = 0; c < 52; c++) {
                    if (on[c]) continue;
                    for (int d = c + 1; d < 52; d++) {
                        if (on[d]) continue;
                        const int hole[2] = {c, d};
                        const uint64_t id = ix.index(hole, board);
                        out[id] = b[(size_t)idx[c][d]];  // equal values for equal classes: a benign race
                        mine.push_back(id);
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(seen_mu);
                    for (uint64_t id : mine) seen[id >> 3] |= (uint8_t)(1u << (id & 7));
                }
                if (done) done->fetch_add(mine.size(), std::memory_order_relaxed);
            }
        };
        const int T = threads < 1 ? 1 : threads;
        std::vector<std::thread> pool;
        for (int t = 1; t < T; t++) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
        if (failed.load()) throw std::runtime_error("bucket table build failed: river batch refused a board");
        for (uint64_t id = 0; id < n; id++)
            if (!(seen[id >> 3] >> (id & 7) & 1)) throw std::logic_error("bucket table: a river class was not covered by the canonical boards");
        t_[RIVER].swap(out);
        return true;
    }

    // pair (c, d) -> 0..1325, c < d in card order (the convention of river_buckets_all's callers)
    static const int (*pair_index())[52] {
        static const struct Idx {
            int v[52][52];
            Idx() {
                int k = 0;
                for (int a = 0; a < 52; a++) for (int b = 0; b < 52; b++) v[a][b] = -1;
                for (int a = 0; a < 52; a++)
                    for (int b = a + 1; b < 52; b++) { v[a][b] = v[b][a] = k; k++; }
            }
        } t;
        return t.v;
    }

    bool any() const { return has(FLOP) || has(TURN) || has(RIVER); }
    static uint64_t fnv1a(const std::vector<uint8_t>& v) {
        uint64_t h = 0xCBF29CE484222325ULL;
        for (uint8_t c : v) { h ^= c; h *= 0x100000001B3ULL; }
        return h;
    }

    std::unique_ptr<HandIndexer> ix_[4];
    std::vector<uint8_t> t_[4];
    BucketerIdentity id_;
};

// A bucketer that answers from the tables where a street is tabulated and asks the wrapped
// bucketer otherwise.  Same identity as the wrapped one: the buckets are the same numbers, so
// blueprints and checkpoints stay interchangeable.
class TabulatedBucketer : public Bucketer {
public:
    TabulatedBucketer(std::shared_ptr<Bucketer> inner, std::shared_ptr<const BucketTables> tables)
        : inner_(std::move(inner)), tab_(std::move(tables)) {
        if (!BucketTables::same_identity(tab_->identity(), inner_->identity()))
            throw std::invalid_argument("bucket tables were built for another bucketer");
        for (int s = FLOP; s <= RIVER; s++) on_[s] = tab_->has(s);
    }
    bool fitted() const override { return inner_->fitted(); }
    BucketerIdentity identity() const override { return inner_->identity(); }
    int bucket_of_form(const CanonicalForm& cf) const override { return inner_->bucket_of_form(cf); }
    int bucket(const int* hole, const int* board, int n_board) override {
        if (n_board == 0) return classes_.of(hole[0], hole[1]);
        const int s = street_of_board(n_board);
        if (on_[s]) return tab_->at(s, tab_->indexer(s).index(hole, board));
        return inner_->bucket(hole, board, n_board);
    }
    const std::shared_ptr<Bucketer>& inner() const { return inner_; }
    const std::shared_ptr<const BucketTables>& tables() const { return tab_; }

private:
    std::shared_ptr<Bucketer> inner_;
    std::shared_ptr<const BucketTables> tab_;
    bool on_[4] = {false, false, false, false};
};

}  // namespace negp
