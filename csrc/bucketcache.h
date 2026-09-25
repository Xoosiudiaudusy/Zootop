// Warm bucket caches (2026-09-25): compute the bucket of every canonical (hole, board) form of a
// street once, and save / load the bucketer's cache contents, so a search on a new flop does not
// pay for potential-aware histograms (about 0.4 ms per flop or turn form, 1.29M flop and 13.96M
// turn forms in all).
//
//   precompute_buckets(bk, street, threads)  every canonical board of the street (one per suit
//       isomorphism class: 1,755 flops, 16,432 turns) times every hole: every canonical form
//   save_bucket_cache / load_bucket_cache    "NPBKCH01" | the bucketer's identity (kind,
//       parameters, fingerprint of the fitted numbers: a cache of other buckets is refused) |
//       per street: u8 street, u64 n, n x u64 words (key << 20 | value) | u64 checksum
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "abstraction.h"
#include "binio.h"
#include "engine.h"

namespace negp {

static const char BUCKET_CACHE_MAGIC[9] = "NPBKCH01";

// the boards of n cards that are their own canonical representative under the 24 suit permutations
inline std::vector<std::array<int, 5>> canonical_boards(int n) {
    std::vector<std::array<int, 5>> out;
    int c[5];
    std::array<int, 5> cur{};
    auto consider = [&]() {
        for (int p = 0; p < 24; p++) {
            int b[5];
            for (int i = 0; i < n; i++) b[i] = (c[i] >> 2) * 4 + SUIT_PERMS[p][c[i] & 3];
            std::sort(b, b + n);
            for (int i = 0; i < n; i++) {
                if (b[i] != c[i]) {
                    if (b[i] < c[i]) return;  // a smaller representative exists
                    break;
                }
            }
        }
        for (int i = 0; i < n; i++) cur[(size_t)i] = c[i];
        out.push_back(cur);
    };
    std::function<void(int, int)> rec = [&](int depth, int start) {
        if (depth == n) { consider(); return; }
        for (int x = start; x < 52; x++) {
            c[depth] = x;
            rec(depth + 1, x + 1);
        }
    };
    rec(0, 0);
    return out;
}

// every canonical form of `street` through bk.bucket() (computed into the cache); returns the
// number of (board, hole) pairs visited
inline long long precompute_buckets(Bucketer& bk, int street, int threads) {
    if (street < FLOP || street > RIVER) throw std::invalid_argument("precompute: street 1..3");
    const int n = board_cards_by_street(street);
    const std::vector<std::array<int, 5>> boards = canonical_boards(n);
    std::atomic<size_t> next{0};
    std::atomic<long long> visited{0};
    std::string err;
    std::mutex err_mu;
    auto work = [&]() {
        try {
            long long mine = 0;
            for (size_t i = next.fetch_add(1); i < boards.size(); i = next.fetch_add(1)) {
                const std::array<int, 5>& b = boards[i];
                bool on[52] = {false};
                for (int k = 0; k < n; k++) on[b[(size_t)k]] = true;
                for (int x = 0; x < 52; x++) {
                    if (on[x]) continue;
                    for (int y = x + 1; y < 52; y++) {
                        if (on[y]) continue;
                        const int hole[2] = {x, y};
                        bk.bucket(hole, b.data(), n);
                        mine++;
                    }
                }
            }
            visited.fetch_add(mine);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(err_mu);
            err = e.what();
        }
    };
    const int T = std::max(1, threads);
    std::vector<std::thread> pool;
    for (int t = 1; t < T; t++) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    if (!err.empty()) throw std::runtime_error("precompute: " + err);
    return visited.load();
}

inline void write_bucketer_identity(BinWriter& w, const BucketerIdentity& id) {
    w.str16(id.kind);
    w.i32(id.n_buckets);
    w.i32(id.samples);
    w.i32(id.bins);
    w.u64(id.fingerprint);
}

inline void save_bucket_cache(const std::string& path, const Bucketer& bk, const std::vector<int>& streets) {
    BinWriter w(path);
    w.bytes(BUCKET_CACHE_MAGIC, 8);
    write_bucketer_identity(w, bk.identity());
    w.u32((uint32_t)streets.size());
    for (int s : streets) {
        if (s < FLOP || s > RIVER) throw std::invalid_argument("save_bucket_cache: streets 1..3");
        const std::vector<uint64_t> words = bk.cache_words(s);
        w.u8((uint8_t)s);
        w.u64(words.size());
        w.array(words);
    }
    w.u64(w.checksum());
    w.finish();
}

// returns per loaded street (street, entries, entries that did not fit without an eviction)
inline std::vector<std::array<long long, 3>> load_bucket_cache(const std::string& path, Bucketer& bk) {
    BinReader r(path);
    char magic[8];
    r.bytes(magic, 8);
    if (std::memcmp(magic, BUCKET_CACHE_MAGIC, 8) != 0) throw std::runtime_error(path + " is not a bucket cache (NPBKCH01)");
    BucketerIdentity f;
    f.kind = r.str16();
    f.n_buckets = r.i32();
    f.samples = r.i32();
    f.bins = r.i32();
    f.fingerprint = r.u64();
    const BucketerIdentity mine = bk.identity();
    if (f.kind != mine.kind || f.n_buckets != mine.n_buckets || f.samples != mine.samples || f.bins != mine.bins ||
        f.fingerprint != mine.fingerprint)
        throw std::runtime_error(path + " holds the cache of other buckets (" + f.kind + ", " + std::to_string(f.n_buckets) +
                                 " buckets): refusing to load it");
    const uint32_t ns = r.u32();
    std::vector<std::pair<int, std::vector<uint64_t>>> parts;
    for (uint32_t i = 0; i < ns; i++) {
        const int s = r.u8();
        if (s < FLOP || s > RIVER) throw std::runtime_error("corrupt file (street): " + path);
        std::vector<uint64_t> words;
        r.array(words, r.u64());
        parts.emplace_back(s, std::move(words));
    }
    r.expect_checksum("bucket cache");
    std::vector<std::array<long long, 3>> out;
    for (auto& p : parts) {
        const size_t evicted = bk.load_cache_words(p.first, p.second.data(), p.second.size());
        out.push_back({(long long)p.first, (long long)p.second.size(), (long long)evicted});
    }
    return out;
}

}  // namespace negp
