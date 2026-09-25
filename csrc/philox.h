// Philox4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC 2011),
// the counter-based generator of cuRAND (curand_Philox4x32_10): a pure function of a 128-bit counter
// and a 64-bit key, so every draw of the batched trainer is addressed by (seed, iteration, purpose,
// index) and a CPU and a GPU implementation give the same numbers whatever the thread schedule.
//
// Draws used by the batched trainer (batch mode of mccfr.h):
//   deal of iteration t:        counter (t_lo, t_hi, 0, k), k = 0, 1, ... (4 words per call)
//   sample of traverser p at the history with hash lanes (a, b) (HistHash, nodetable.h):
//                               counter (t_lo, t_hi, (uint32)b ^ golden * (1 + p), (uint32)a) -- a node is
//                               reached at most once per (iteration, traverser), and the address does not
//                               depend on the traversal order (depth-first on the CPU, by levels on the GPU)
// uniform01 = (hi 26 bits, lo 27 bits of two words) / 2^53 (the numpy / Python recipe); an index below
// n = (word * n) >> 32 (multiply-shift; bias < n / 2^32).
#pragma once
#include <cstdint>

namespace negp {

struct Philox4x32 {
    uint32_t v[4];
};

inline uint32_t philox_mulhilo(uint32_t a, uint32_t b, uint32_t& hi) {
    const uint64_t p = (uint64_t)a * (uint64_t)b;
    hi = (uint32_t)(p >> 32);
    return (uint32_t)p;
}

// ten rounds of Philox4x32 on `ctr` with `key`
inline Philox4x32 philox4x32_10(Philox4x32 ctr, uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; r++) {
        uint32_t hi0, hi1;
        const uint32_t lo0 = philox_mulhilo(M0, ctr.v[0], hi0);
        const uint32_t lo1 = philox_mulhilo(M1, ctr.v[2], hi1);
        Philox4x32 n;
        n.v[0] = hi1 ^ ctr.v[1] ^ k0;
        n.v[1] = lo1;
        n.v[2] = hi0 ^ ctr.v[3] ^ k1;
        n.v[3] = lo0;
        ctr = n;
        k0 += W0;
        k1 += W1;
    }
    return ctr;
}

// the generator of one (seed, iteration, purpose) stream
struct PhiloxStream {
    uint32_t k0, k1, t0, t1, purpose;
    uint32_t idx = 0;           // next counter word 3
    uint32_t buf[4];
    int left = 0;

    PhiloxStream(uint64_t seed, uint64_t t, uint32_t purpose_)
        : k0((uint32_t)seed), k1((uint32_t)(seed >> 32)), t0((uint32_t)t), t1((uint32_t)(t >> 32)), purpose(purpose_) {}

    uint32_t next() {
        if (left == 0) {
            Philox4x32 c;
            c.v[0] = t0;
            c.v[1] = t1;
            c.v[2] = purpose;
            c.v[3] = idx++;
            const Philox4x32 r = philox4x32_10(c, k0, k1);
            for (int i = 0; i < 4; i++) buf[i] = r.v[i];
            left = 4;
        }
        return buf[4 - left--];
    }
    uint32_t below(uint32_t n) { return (uint32_t)(((uint64_t)next() * (uint64_t)n) >> 32); }
    double uniform01() {
        const uint32_t a = next() >> 5, b = next() >> 6;
        return ((double)a * 67108864.0 + (double)b) * (1.0 / 9007199254740992.0);
    }
};

// the uniform draw of traverser p at the history (a, b) in iteration t (see the header)
inline double philox_sample_u01(uint64_t seed, uint64_t t, int p, uint64_t a, uint64_t b) {
    Philox4x32 c;
    c.v[0] = (uint32_t)t;
    c.v[1] = (uint32_t)(t >> 32);
    c.v[2] = (uint32_t)b ^ (0x9E3779B9u * (uint32_t)(1 + p));
    c.v[3] = (uint32_t)a;
    const Philox4x32 r = philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
    const uint32_t x = r.v[0] >> 5, y = r.v[1] >> 6;
    return ((double)x * 67108864.0 + (double)y) * (1.0 / 9007199254740992.0);
}

// the deal of iteration t: Fisher-Yates from the top over 0..51
inline void philox_deal(uint64_t seed, uint64_t t, int* order) {
    for (int i = 0; i < 52; i++) order[i] = i;
    PhiloxStream s(seed, t, 0);
    for (int i = 51; i > 0; i--) {
        const int j = (int)s.below((uint32_t)(i + 1));
        const int x = order[i];
        order[i] = order[j];
        order[j] = x;
    }
}

}  // namespace negp
