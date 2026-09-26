// The GPU trainer: see gpucfr.h for the algorithm and flatcfr.h for the CPU version it reproduces.
#include "gpucfr.h"

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "gpukernels.h"

namespace negp {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}
#define CK(x) check((x), #x)

constexpr int THREADS = 256;

// a device buffer that grows (keeping its contents when asked)
template <class T>
struct Buf {
    T* p = nullptr;
    size_t cap = 0;
    void reserve(size_t n, size_t keep = 0) {
        if (n <= cap) return;
        size_t c = std::max(n, cap + cap / 2);
        T* q = nullptr;
        CK(cudaMalloc(&q, c * sizeof(T)));
        if (keep && p) CK(cudaMemcpy(q, p, keep * sizeof(T), cudaMemcpyDeviceToDevice));
        if (p) cudaFree(p);
        p = q;
        cap = c;
    }
    ~Buf() { if (p) cudaFree(p); }
};

// one thread per item: the work is in gpukernels.h (shared with the host emulation)
__global__ void k_init(DevLevel L, uint32_t m, uint32_t job0) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) item_init(L, i, job0);
}

__global__ void k_count(DevGame g, DevLevel L, size_t off, uint32_t m, const FlatIter* iters, const double* regret, uint64_t seed, int bb,
                        int* err) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) item_count(g, L, off + i, iters, regret, seed, bb, err);
}

__global__ void k_emit(DevGame g, DevLevel L, size_t off, uint32_t m, size_t next_off) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) item_emit(g, L, off + i, next_off);
}

// an item's records go to the slots base + roff[x] (exclusive scan of the record counts of the pass)
__global__ void k_back(DevGame g, DevLevel L, size_t off, uint32_t m, size_t next_off, const FlatIter* iters, const double* regret,
                       KeyLayout kl, uint64_t* keys, double* vals, uint64_t base, int* err) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const size_t x = off + i;
    if (L.rcnt[x] == 0) return;
    item_back(g, L, x, next_off, iters, regret, kl, keys, vals, base + L.roff[x], err);
}

__global__ void k_apply(const uint64_t* keys, const double* vals, size_t m, KeyLayout kl, double* regret, double* ssum, int64_t* visits,
                        uint8_t* touched) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) apply_run(keys, vals, m, i, kl, regret, ssum, visits, touched);
}

inline unsigned blocks(size_t m) { return (unsigned)((m + THREADS - 1) / THREADS); }

template <class T>
T* upload_array(const T* src, size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    if (n) CK(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

}  // namespace

struct GpuFlatTrainer::Impl {
    int device = 0;
    int bb = 100;
    uint64_t seed = 0;
    uint64_t n_infosets = 0, n_cells = 0;
    DevGame g{};
    std::vector<void*> game_arrays;
    double *regret = nullptr, *ssum = nullptr;
    int64_t* visits = nullptr;
    uint8_t* touched = nullptr;
    // per batch
    Buf<FlatIter> iters;
    Buf<int32_t> node;
    Buf<uint32_t> job, first, cnt;
    Buf<uint8_t> choice;
    Buf<double> value;
    Buf<uint32_t> rcnt;
    Buf<uint64_t> roff;
    Buf<uint64_t> keys, keys_alt;
    Buf<double> vals, vals_alt;
    Buf<unsigned char> temp;
    int* err = nullptr;
    cudaEvent_t ev[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    GpuStats st;

    DevLevel level() { return DevLevel{node.p, job.p, first.p, cnt.p, choice.p, value.p, rcnt.p, roff.p}; }
    void reserve_items(size_t n, size_t keep) {
        node.reserve(n, keep);
        job.reserve(n, keep);
        first.reserve(n, keep);
        cnt.reserve(n, keep);
        choice.reserve(n, keep);
        value.reserve(n, keep);
        rcnt.reserve(n, keep);
        roff.reserve(n, keep);
    }
    template <class T>
    T* keep(T* p) { game_arrays.push_back(p); return p; }

    ~Impl() {
        for (void* p : game_arrays) cudaFree(p);
        if (regret) cudaFree(regret);
        if (ssum) cudaFree(ssum);
        if (visits) cudaFree(visits);
        if (touched) cudaFree(touched);
        if (err) cudaFree(err);
        for (auto& e : ev) if (e) cudaEventDestroy(e);
    }
};

bool GpuFlatTrainer::available(std::string& why) {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { why = cudaGetErrorString(e); return false; }
    if (n == 0) { why = "no CUDA device"; return false; }
    why.clear();
    return true;
}

GpuFlatTrainer::GpuFlatTrainer(const FlatGameView& v, int bb, uint64_t seed, int device) : impl_(new Impl) {
    Impl& m = *impl_;
    m.device = device;
    CK(cudaSetDevice(device));
    m.bb = bb;
    m.seed = seed;
    m.n_infosets = v.n_infosets;
    m.n_cells = v.n_cells;
    m.g.n = v.n_players;
    m.g.rel = m.keep(upload_array(v.rel, v.n_decisions));
    m.g.n_board = m.keep(upload_array(v.n_board, v.n_decisions));
    m.g.na = m.keep(upload_array(v.na, v.n_decisions));
    m.g.child_base = m.keep(upload_array(v.child_base, v.n_decisions));
    m.g.child = m.keep(upload_array(v.child, v.n_child));
    m.g.hh_a = m.keep(upload_array(v.hh_a, v.n_decisions));
    m.g.hh_b = m.keep(upload_array(v.hh_b, v.n_decisions));
    m.g.info_base = m.keep(upload_array(v.info_base, v.n_decisions));
    m.g.cell_base = m.keep(upload_array(v.cell_base, v.n_decisions));
    m.g.n_cache = m.keep(upload_array(v.n_cache, v.n_decisions));
    m.g.invested = m.keep(upload_array(v.invested, v.n_terminals * (size_t)v.n_players));
    m.g.folded = m.keep(upload_array(v.folded, v.n_terminals));
    CK(cudaMalloc(&m.regret, std::max<uint64_t>(1, v.n_cells) * sizeof(double)));
    CK(cudaMalloc(&m.ssum, std::max<uint64_t>(1, v.n_cells) * sizeof(double)));
    CK(cudaMalloc(&m.visits, std::max<uint64_t>(1, v.n_infosets) * sizeof(int64_t)));
    CK(cudaMalloc(&m.touched, std::max<uint64_t>(1, v.n_cells)));
    CK(cudaMemset(m.regret, 0, v.n_cells * sizeof(double)));
    CK(cudaMemset(m.ssum, 0, v.n_cells * sizeof(double)));
    CK(cudaMemset(m.visits, 0, v.n_infosets * sizeof(int64_t)));
    CK(cudaMemset(m.touched, 0, v.n_cells));
    CK(cudaMalloc(&m.err, sizeof(int)));
    CK(cudaMemset(m.err, 0, sizeof(int)));
    for (auto& e : m.ev) CK(cudaEventCreate(&e));
}

GpuFlatTrainer::~GpuFlatTrainer() = default;

void GpuFlatTrainer::run_batch(const FlatIter* its, int k, int pass) {
    Impl& m = *impl_;
    CK(cudaSetDevice(m.device));
    const int n = m.g.n;
    const KeyLayout kl = key_layout((uint64_t)k * (uint64_t)n, std::max(m.n_cells, m.n_infosets));
    if (kl.bits() == 0 || kl.jb == 0) throw std::invalid_argument("GPU trainer: batch x table too large for 64-bit update keys");
    if (pass < 1) pass = k;
    m.iters.reserve((size_t)k);
    CK(cudaMemcpy(m.iters.p, its, (size_t)k * sizeof(FlatIter), cudaMemcpyHostToDevice));
    m.st = GpuStats();
    size_t n_rec = 0;  // records of the batch so far
    auto elapsed = [&](int a, int b) {
        float t = 0;
        CK(cudaEventElapsedTime(&t, m.ev[a], m.ev[b]));
        return (double)t;
    };
    for (int lo = 0; lo < k; lo += pass) {
        CK(cudaEventRecord(m.ev[0]));
        const int it_n = std::min(pass, k - lo);
        const uint32_t jobs = (uint32_t)it_n * (uint32_t)n;
        // level 0: the roots of the pass's jobs (job ids are batch relative)
        std::vector<size_t> off{0}, size{jobs};
        m.reserve_items(jobs, 0);
        k_init<<<blocks(jobs), THREADS>>>(m.level(), jobs, (uint32_t)lo * (uint32_t)n);
        CK(cudaGetLastError());
        for (size_t L = 0; size[L] > 0; L++) {
            const uint32_t cur = (uint32_t)size[L];
            k_count<<<blocks(cur), THREADS>>>(m.g, m.level(), off[L], cur, m.iters.p, m.regret, m.seed, m.bb, m.err);
            CK(cudaGetLastError());
            size_t tb = 0;
            CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, m.cnt.p + off[L], m.first.p + off[L], cur));
            m.temp.reserve(tb);
            CK(cub::DeviceScan::ExclusiveSum(m.temp.p, tb, m.cnt.p + off[L], m.first.p + off[L], cur));
            uint32_t last_first = 0, last_cnt = 0;
            CK(cudaMemcpy(&last_first, m.first.p + off[L] + cur - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&last_cnt, m.cnt.p + off[L] + cur - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            const size_t nxt = (size_t)last_first + last_cnt;
            const size_t next_off = off[L] + cur;
            m.reserve_items(next_off + nxt, next_off);
            if (nxt) k_emit<<<blocks(cur), THREADS>>>(m.g, m.level(), off[L], cur, next_off);
            CK(cudaGetLastError());
            off.push_back(next_off);
            size.push_back(nxt);
        }
        m.st.items += off.back();
        // records of this pass: slots from one exclusive scan of the record counts of all its items
        const size_t items = off.back();
        size_t tb = 0;
        CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, m.rcnt.p, m.roff.p, (int64_t)items));
        m.temp.reserve(tb);
        CK(cub::DeviceScan::ExclusiveSum(m.temp.p, tb, m.rcnt.p, m.roff.p, (int64_t)items));
        uint64_t last_off = 0;
        uint32_t last_rc = 0;
        CK(cudaMemcpy(&last_off, m.roff.p + items - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&last_rc, m.rcnt.p + items - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const uint64_t cnt_rec = last_off + last_rc;
        CK(cudaEventRecord(m.ev[1]));
        m.keys.reserve(n_rec + cnt_rec, n_rec);
        m.vals.reserve(n_rec + cnt_rec, n_rec);
        for (size_t L = size.size() - 1; L-- > 0;) {
            if (size[L] == 0) continue;
            k_back<<<blocks(size[L]), THREADS>>>(m.g, m.level(), off[L], (uint32_t)size[L], off[L + 1], m.iters.p, m.regret, kl, m.keys.p, m.vals.p,
                                                 (uint64_t)n_rec, m.err);
            CK(cudaGetLastError());
        }
        CK(cudaEventRecord(m.ev[2]));
        CK(cudaEventSynchronize(m.ev[2]));
        m.st.ms_forward += elapsed(0, 1);
        m.st.ms_backward += elapsed(1, 2);
        n_rec += cnt_rec;
    }
    CK(cudaEventRecord(m.ev[2]));
    CK(cudaEventRecord(m.ev[3]));
    // sort and apply
    if (n_rec) {
        m.keys_alt.reserve(n_rec);
        m.vals_alt.reserve(n_rec);
        cub::DoubleBuffer<uint64_t> kb(m.keys.p, m.keys_alt.p);
        cub::DoubleBuffer<double> vb(m.vals.p, m.vals_alt.p);
        size_t tb = 0;
        // only the key bits in use: fewer radix passes
        CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, kb, vb, (int64_t)n_rec, 0, kl.bits()));
        m.temp.reserve(tb);
        CK(cub::DeviceRadixSort::SortPairs(m.temp.p, tb, kb, vb, (int64_t)n_rec, 0, kl.bits()));
        CK(cudaEventRecord(m.ev[3]));
        k_apply<<<blocks(n_rec), THREADS>>>(kb.Current(), vb.Current(), n_rec, kl, m.regret, m.ssum, m.visits, m.touched);
        CK(cudaGetLastError());
    }
    CK(cudaEventRecord(m.ev[4]));
    CK(cudaEventSynchronize(m.ev[4]));
    int e = 0;
    CK(cudaMemcpy(&e, m.err, sizeof(int), cudaMemcpyDeviceToHost));
    if (e) throw std::runtime_error("GPU trainer: a bucket outside its node's rows");
    m.st.ms_sort = elapsed(2, 3);
    m.st.ms_runs = elapsed(3, 4);
    m.st.ms_traverse = m.st.ms_forward + m.st.ms_backward;
    m.st.ms_apply = m.st.ms_sort + m.st.ms_runs;
    m.st.records = n_rec;
}

void GpuFlatTrainer::upload(const double* regret, const double* ssum, const int64_t* visits, const uint8_t* touched) {
    Impl& m = *impl_;
    CK(cudaSetDevice(m.device));
    CK(cudaMemcpy(m.regret, regret, m.n_cells * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.ssum, ssum, m.n_cells * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.visits, visits, m.n_infosets * sizeof(int64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.touched, touched, m.n_cells, cudaMemcpyHostToDevice));
}

void GpuFlatTrainer::download(double* regret, double* ssum, int64_t* visits, uint8_t* touched) const {
    const Impl& m = *impl_;
    CK(cudaSetDevice(m.device));
    CK(cudaMemcpy(regret, m.regret, m.n_cells * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ssum, m.ssum, m.n_cells * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(visits, m.visits, m.n_infosets * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(touched, m.touched, m.n_cells, cudaMemcpyDeviceToHost));
}

std::string GpuFlatTrainer::device_name() const {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, impl_->device));
    return std::string(p.name) + " (sm_" + std::to_string(p.major) + std::to_string(p.minor) + ")";
}

GpuStats GpuFlatTrainer::stats() const { return impl_->st; }

}  // namespace negp
