// The GPU trainer (gpucfr.cu): FlatTrainer's batched level-synchronous MCCFR (flatcfr.h) on a CUDA
// device.  The flat game and the dense tables live in device memory; the host hands it, per batch, the
// deals of the batch's iterations (FlatIter: buckets, showdown strengths, weights) and gets the tables
// back on request.  Same algorithm, same doubles as the CPU version (tests/test_flatcfr.py, run on a
// machine with a GPU):
//   forward, per level:  count kernel (terminal values; an opponent node's strategy and Philox sample),
//                        exclusive scan of the child counts (CUB), emit kernel (children of the next level);
//   backward, per level: values from the children; the updates (regret / strategy sum / visit) are written
//                        as (key, value) records, key = kind | cell | (iteration - batch start) * n + traverser;
//   after the batch:     radix sort of the records (CUB; keys are unique, so the order is a function of the
//                        keys), then one thread per (kind, cell) adds its values in key order, i.e. in the
//                        order (iteration, traverser) of the CPU trainers.
// The device code is compiled with --fmad=false: no a * b + c is fused, as on the CPU.
//
// This header has no CUDA types; without a CUDA build (NEGP_WITH_CUDA) only available() exists.
#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfrmath.h"

namespace negp {

// the flat game (flatgame.h) as raw arrays, for the upload
struct FlatGameView {
    int n_players = 0;
    size_t n_decisions = 0, n_terminals = 0, n_child = 0;
    const uint8_t *rel = nullptr, *n_board = nullptr, *na = nullptr;
    const uint32_t* child_base = nullptr;
    const int32_t* child = nullptr;
    const uint64_t *hh_a = nullptr, *hh_b = nullptr, *info_base = nullptr, *cell_base = nullptr;
    const int32_t* n_cache = nullptr;
    const int32_t* invested = nullptr;
    const uint16_t* folded = nullptr;
    uint64_t n_infosets = 0, n_cells = 0;
};

struct GpuStats {
    uint64_t items = 0;      // (node, job) items of the last batch, terminals included
    uint64_t records = 0;    // updates of the last batch
    double ms_traverse = 0;  // device time of the last batch: traversal ...
    double ms_apply = 0;     // ... and sort + application
};

class GpuFlatTrainer {
public:
    // a usable CUDA device (and a CUDA build); otherwise false and the reason
    static bool available(std::string& why);

    GpuFlatTrainer(const FlatGameView& g, int bb, uint64_t seed, int device);
    ~GpuFlatTrainer();
    GpuFlatTrainer(const GpuFlatTrainer&) = delete;
    GpuFlatTrainer& operator=(const GpuFlatTrainer&) = delete;

    // one batch: the iterations iters[0 .. k) (consecutive t, iters[0].t = batch start), every
    // traverser; `pass` iterations traversed at once (memory; never the result)
    void run_batch(const FlatIter* iters, int k, int pass);

    // tables: regret / strategy sum per cell, visits per infoset, touched per cell (an update was applied)
    void upload(const double* regret, const double* ssum, const int64_t* visits, const uint8_t* touched_cells);
    void download(double* regret, double* ssum, int64_t* visits, uint8_t* touched_cells) const;

    std::string device_name() const;
    GpuStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#if !defined(NEGP_WITH_CUDA)
// a core built without CUDA: no device, every call says so
struct GpuFlatTrainer::Impl {};
inline bool GpuFlatTrainer::available(std::string& why) {
    why = "the C++ core was built without CUDA (install the CUDA Toolkit 12.8+ and rebuild)";
    return false;
}
inline GpuFlatTrainer::GpuFlatTrainer(const FlatGameView&, int, uint64_t, int) { throw std::runtime_error("built without CUDA"); }
inline GpuFlatTrainer::~GpuFlatTrainer() = default;
inline void GpuFlatTrainer::run_batch(const FlatIter*, int, int) { throw std::runtime_error("built without CUDA"); }
inline void GpuFlatTrainer::upload(const double*, const double*, const int64_t*, const uint8_t*) { throw std::runtime_error("built without CUDA"); }
inline void GpuFlatTrainer::download(double*, double*, int64_t*, uint8_t*) const { throw std::runtime_error("built without CUDA"); }
inline std::string GpuFlatTrainer::device_name() const { return std::string(); }
inline GpuStats GpuFlatTrainer::stats() const { return GpuStats(); }
#endif

}  // namespace negp
