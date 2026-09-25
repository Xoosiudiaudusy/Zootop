// CPython-compatible pseudo-randomness, so the C++ core can reproduce the pure-Python
// reference bit for bit:
//
//   * PyRandom      - MT19937 seeded exactly like ``random.Random(int)`` (init_by_array over the
//                     32-bit words of |seed|), with ``random()`` (53-bit), ``getrandbits``,
//                     ``_randbelow_with_getrandbits``, ``shuffle`` and ``sample`` (both the
//                     pool and the set branch of CPython's algorithm; no heap allocation for
//                     populations of up to 64 items).
//   * py_tuple_hash - CPython 3.8+ tuple hash (xxHash-derived) over small ints, used by
//                     ``EquityBucketer.ehs`` to seed the per-canonical-form Monte-Carlo.
//   * py_sum        - CPython 3.12+ ``sum()`` of floats (Neumaier compensated summation).
//
// Only what the reference code paths need is ported; nothing here is a general RNG library.
#pragma once
#if defined(_MSC_VER)
#pragma fp_contract(off)
#endif
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace negp {

struct PyRandom {
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfU;
    static constexpr uint32_t UPPER_MASK = 0x80000000U;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffU;

    uint32_t mt[N];
    int index = N + 1;

    PyRandom() { seed(0); }
    explicit PyRandom(uint64_t s) { seed(s); }

    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < N; i++) {
            mt[i] = (1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i);
        }
        index = N;
    }

    void init_by_array(const uint32_t* key, int key_length) {
        init_genrand(19650218U);
        int i = 1, j = 0;
        int k = (N > key_length ? N : key_length);
        for (; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint32_t)j;
            i++; j++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
            if (j >= key_length) j = 0;
        }
        for (k = N - 1; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
            i++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
        }
        mt[0] = 0x80000000U;
        index = N;
    }

    // random.Random(seed) for a non-negative int seed that fits in 64 bits
    // (CPython: key = 32-bit little-endian words of abs(seed); zero -> one word).
    void seed(uint64_t s) {
        uint32_t key[2];
        int n = 0;
        if (s == 0) { key[0] = 0; n = 1; }
        else {
            key[0] = (uint32_t)(s & 0xffffffffU);
            n = 1;
            if (s >> 32) { key[1] = (uint32_t)(s >> 32); n = 2; }
        }
        init_by_array(key, n);
    }

    // seed from an arbitrary word list (used to derive distinct per-thread streams)
    void seed_words(const std::vector<uint32_t>& words) {
        if (words.empty()) { seed(0); return; }
        init_by_array(words.data(), (int)words.size());
    }

    uint32_t genrand_int32() {
        static const uint32_t mag01[2] = {0x0U, MATRIX_A};
        uint32_t y;
        if (index >= N) {
            int kk;
            for (kk = 0; kk < N - M; kk++) {
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; kk++) {
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index = 0;
        }
        y = mt[index++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    // random.random(): 53-bit float in [0, 1)
    double random() {
        uint32_t a = genrand_int32() >> 5, b = genrand_int32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // random.getrandbits(k) for 0 < k <= 32
    uint32_t getrandbits(int k) {
        if (k <= 0) return 0;
        return genrand_int32() >> (32 - k);
    }

    // random._randbelow_with_getrandbits(n), n in [1, 2^32)
    uint32_t randbelow(uint32_t n) {
        int k = 0;
        for (uint32_t t = n; t; t >>= 1) k++;
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // random.shuffle(x)
    template <class T>
    void shuffle(std::vector<T>& x) {
        for (int i = (int)x.size() - 1; i >= 1; i--) {
            uint32_t j = randbelow((uint32_t)i + 1);
            T tmp = x[i]; x[i] = x[j]; x[j] = tmp;
        }
    }

    // populations up to this size are sampled without touching the heap (every call the
    // reference makes: decks of at most 52 cards)
    static constexpr int SAMPLE_STACK_N = 64;

    // random.sample(population, k) into out[0..k): identical branch selection and draws as
    // CPython (Lib/random.py): the pool branch when n <= setsize, else the "selected" set
    // branch; the same randbelow calls in the same order.  k outside 0..n raises like CPython
    // ("Sample larger than population or is negative") instead of looping forever.
    void sample(const int* population, int n, int k, int* out) {
        if (k < 0 || k > n) throw std::invalid_argument("Sample larger than population or is negative");
        int setsize = 21;
        if (k > 5) {
            double e = std::ceil(std::log((double)(k * 3)) / std::log(4.0));
            setsize += (int)std::pow(4.0, e);
        }
        if (n <= setsize) {
            if (n <= SAMPLE_STACK_N) {
                int pool[SAMPLE_STACK_N];
                for (int i = 0; i < n; i++) pool[i] = population[i];
                sample_pool(pool, n, k, out);
            } else {
                std::vector<int> pool(population, population + n);
                sample_pool(pool.data(), n, k, out);
            }
        } else if (n <= SAMPLE_STACK_N) {
            // the "selected" set as a 64-bit mask (j < n <= 64)
            uint64_t selected = 0;
            for (int i = 0; i < k; i++) {
                uint32_t j = randbelow((uint32_t)n);
                while ((selected >> j) & 1U) j = randbelow((uint32_t)n);
                selected |= (uint64_t)1 << j;
                out[i] = population[j];
            }
        } else {
            std::vector<unsigned char> selected(n, 0);
            for (int i = 0; i < k; i++) {
                uint32_t j = randbelow((uint32_t)n);
                while (selected[j]) j = randbelow((uint32_t)n);
                selected[j] = 1;
                out[i] = population[j];
            }
        }
    }

    void sample(const std::vector<int>& population, int k, std::vector<int>& out) {
        int n = (int)population.size();
        if (k < 0 || k > n) throw std::invalid_argument("Sample larger than population or is negative");
        out.resize(k);
        sample(population.data(), n, k, out.data());
    }

    // pool branch: the non-selected items are pool[0 : n - i]; the last of them fills the vacancy
    void sample_pool(int* pool, int n, int k, int* out) {
        for (int i = 0; i < k; i++) {
            uint32_t j = randbelow((uint32_t)(n - i));
            out[i] = pool[j];
            pool[j] = pool[n - i - 1];
        }
    }

    // state interop with random.Random.getstate()/setstate(): 624 words + index
    void get_state(std::vector<uint32_t>& words, int& idx) const {
        words.assign(mt, mt + N);
        idx = index;
    }
    void set_state(const std::vector<uint32_t>& words, int idx) {
        for (int i = 0; i < N; i++) mt[i] = words[i];
        index = idx;
    }
};

// CPython tuple hash for a tuple of small ints (hash(int i) == i for 0 <= i < 2^61-1, hash(-1) == -2).
inline int64_t py_tuple_hash(const int64_t* items, int len) {
    const uint64_t P1 = 11400714785074694791ULL;
    const uint64_t P2 = 14029467366897019727ULL;
    const uint64_t P5 = 2870177450012600261ULL;
    uint64_t acc = P5;
    for (int i = 0; i < len; i++) {
        int64_t h = items[i] == -1 ? -2 : items[i];
        uint64_t lane = (uint64_t)h;
        acc += lane * P2;
        acc = (acc << 31) | (acc >> 33);
        acc *= P1;
    }
    acc += (uint64_t)len ^ (P5 ^ 3527539ULL);
    if (acc == (uint64_t)-1) return 1546275796;
    return (int64_t)acc;
}

// CPython 3.12+ builtins.sum over a non-empty list of floats (start=0): the first item is
// added to the int 0 exactly, the rest with Neumaier compensation.
inline double py_sum(const double* x, int n) {
    if (n == 0) return 0.0;
    double f = x[0];
    double c = 0.0;
    for (int i = 1; i < n; i++) {
        double xi = x[i];
        double t = f + xi;
        if (std::fabs(f) >= std::fabs(xi)) c += (f - t) + xi;
        else c += (xi - t) + f;
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}

}  // namespace negp
