# GPU training (CUDA): the flat batched MCCFR trainer

Blueprint training on an NVIDIA GPU, reached through `scripts/train_blueprint.py --gpu 0 --batch B`.  It runs
the same algorithm as the CPU trainer's batched mode (`--batch B` without `--gpu`), and its tables are equal
to that mode's **bit for bit**.  It writes the ordinary checkpoints and blueprints, which the search, the
duels and the bots read unchanged.

Measured on an RTX 5070 against an i5-14400F on 16 threads.  Game: 3-max, 100bb, wide grid, potential-aware
64 buckets with exact features:

| | iterations / s | 767M iterations |
|---|---|---|
| CPU trainer, 16 threads | 64.5k | 3 h 18 min |
| GPU, batch 4096 | 442k (x6.8) | 29 min |
| GPU, batch 16384 | 662k (x10.3) | ~20 min |

Quality at equal numbers of iterations (duels of 1M deals, 3-max as above, ±1 bb/100):

| comparison | 79M iterations | 767M iterations |
|---|---|---|
| GPU vs CPU, batch 4096 | 0.0 | 0.0 |
| GPU vs CPU, batch 16384 | 0.0 (two seeds) | 0.0 |
| deep CPU vs shallow CPU (control) | | +4.3 |

In other words, the GPU learns exactly as well per iteration and is 7-10x faster.  At equal wall time the
GPU blueprint beats the CPU one by about +4 bb/100 on this game.  The reports are in branches
`opt/gpu-report5..7` and the summary is in `GPU_DESIGN.md` on the comms branch.

## Requirements and build

- An NVIDIA GPU: sm_89 (RTX 40xx) or sm_120 (RTX 50xx), which are the architectures built by default.
  Other GPUs: `-DCMAKE_CUDA_ARCHITECTURES=...`.
- CUDA Toolkit 12.8+ (13.x works), installed after Visual Studio 2022 on Windows.
- `python scripts/build_fast.py --clean`.  The output must contain `negpluribus: GPU trainer ON`.
  - Without a CUDA compiler, the core builds as before and `--gpu` reports why it cannot run.
  - `-DNEGP_CUDA=OFF` skips the GPU part.
- `python -c "import negpluribus._fastcore as f; print(f.cuda_available())"` must print `(True, '')`.
- **Bucket tables** are needed in practice (`NEGPLURIBUS_BUCKET_TABLES`, see `docs/buckets.md`).  The CPU
  computes the buckets of every deal for the GPU.  Without tables, Monte-Carlo or exact-feature buckets are
  far too slow, and the GPU waits.

## Usage

```
set NEGPLURIBUS_BUCKET_TABLES=data\bucket_tables
python scripts/train_blueprint.py <game flags> --backend cpp --gpu 0 --batch 16384 --iters 2000000000 --checkpoint-every 100000000 --tag T
python scripts/train_blueprint.py <game flags> --backend cpp --gpu 0 --batch 16384 --iters 1000000000 --resume --tag T
python scripts/train_blueprint.py <game flags> --backend cpp --gpu 0 --batch 16384 --seconds 3600 --tag T
```

- **`--batch B`** (required).  The number of iterations that share one strategy snapshot.
  - Batches are aligned on absolute iteration numbers: 1..B, B+1..2B, and so on.
  - 4096 and 16384 were measured equal in quality at 79M and 767M iterations; 16384 is 1.5x faster.
  - Larger batches gain little speed.  They also carry a quality risk on short runs: 65536 on HU at 20M
    iterations lost 11 bb/100.
- **`--checkpoint-every N`**: a checkpoint and a blueprint (plus the `.it<N>` copy) every N iterations.
  - N is rounded up to a multiple of B.
  - Each checkpoint copies the device tables into the ordinary trainer and frees that copy after writing.
- **`--resume`**: continues from `checkpoint_<tag>.bin`.
  - A GPU checkpoint also writes `checkpoint_<tag>.bin.gpu.json` (seed, batch, linear, linear_until), and
    resuming with other values is refused.
  - A run resumed from a GPU checkpoint of the same batch continues **bit for bit** as if it never stopped
    (test `test_checkpoint_and_resume_continue_bit_for_bit`).
  - A CPU checkpoint can be continued on the GPU, and a GPU checkpoint on the CPU.  The formats are the same;
    the results then differ from any single-device run, as any change of trainer does.
- **`--seconds S`**: trains for S seconds of wall time instead of `--iters`, for equal-time comparisons.
- **`--linear-until`** works as on the CPU.  Not supported on the GPU: pruning (`--prune-below`) and the L1
  strategy-change report of `--checkpoint-every`.
- **`--gpu-emulate`** is for tests only.  It runs the GPU kernels' code on the CPU: same numbers, slow, no GPU.

## Limits: what fits

The GPU keeps a **dense** table: every (history, bucket) row, visited or not.  Each cell of regret, strategy
sum and strategy costs ~25 bytes, plus work buffers.

| game (100bb) | cells at 64 buckets | GPU memory |
|---|---|---|
| HU, narrow grid | 1.3M | < 1 GB |
| 3-max, narrow grid | 24M | ~1 GB |
| 3-max, wide grid (preflop 0.5/1/3, postflop 0.5/1/2/4, 3 raises) | 150M | 6.2 GB (batch 4096) / 7.5 GB (16384), measured |
| 3-max wide, 128 buckets | ~300M | ~10 GB (estimate) |
| 3-max wide, 256 buckets | ~600M | does not fit in 12 GB |
| 6-max | 10^10+ | does not fit anywhere (an abstraction question, see `docs/scale_3max.md`) |

Host RAM during a 3-max wide run is ~16 GB for the whole PC.  The checkpoint of that game is 6 GB and its
blueprint 4.2 GB.  Games beyond this need a sparse table on the device, which does not exist yet.

## How it works

Everything is in `csrc/`:

- **`flatgame.h`**: the betting tree enumerated into flat arrays (CSR children, terminals with their
  contributions, history hashes) and the dense table layout `cell = cell_base[node] + bucket * n_actions + a`.
- **`flatcfr.h`**: `FlatTrainer`.
  - The batched algorithm, traversed level by level instead of depth first: a batch of iterations x
    traversers ("jobs") goes forward level by level and then backward.
  - The CPU prepares the deals, buckets and showdown strengths of batch i+1 while the device runs batch i.
  - The level-by-level CPU version, the host emulation of the kernels (`emulate_gpu`) and the device all
    give the same numbers.
  - `copy_to` / `copy_from` move the tables to and from an ordinary `Trainer` in C++, for checkpoints,
    blueprints and resume.
- **`gpukernels.h`**: the per-item work of every kernel as `__host__ __device__` functions shared by the
  device and the emulation:
  - the strategy of every row changed in the last batch, computed once per batch;
  - the forward count and emit;
  - the backward values and update records;
  - the application of the sorted records.
- **`gpucfr.cu`**: the CUDA side.
  - Levels, CUB scans for children and record slots, and a CUB stable radix sort of 32-bit keys
    (kind | cell).
  - Then one sequential sum per cell.  A cell's records are already in (iteration, traverser) order in the
    buffer, which is the order of the CPU trainer's additions.
  - Built with `--fmad=false`, so no a*b+c is fused, as on the CPU.
- **`cfrmath.h`**: the arithmetic both sides share: CPython-order sum, regret matching, terminal payoff.
- **Randomness**: Philox4x32-10, addressed by (seed, iteration, purpose), so draws do not depend on thread
  or traversal order.
  - The deal of iteration t: `philox_deal(seed, t)`.
  - An opponent's sample at a history: `philox_sample_u01(seed, t, traverser, history hash)`.

## Verification

- `python -m pytest tests/test_flatcfr.py tests/test_batched.py`.  On a machine with a GPU, every flat test
  runs in three modes, `[cpu]`, `[emu]` and `[gpu]`, each equal bit for bit to `Trainer` in batched mode.
  This includes checkpoint + resume.
- `python scripts/gpu_bench.py [--buckets JSON]` checks identity on push/fold and HU 100bb, then measures
  speed per batch size against the CPU trainer.  `--emulate` runs the same checks without a GPU.
- Duels: `scripts/compare_checkpoints.py` takes buckets from the C++ bucketer and the table (checked against
  the Python bucketer at start).  `--python-buckets` restores the old, slow path.
