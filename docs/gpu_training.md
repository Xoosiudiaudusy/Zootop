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

Quality per iteration: batched training is **delayed feedback**, because every iteration of a batch sees one
strategy snapshot.  It costs some quality per iteration.

Exact exploitability on a preflop game (HU 20bb, raises of 0.5/1/3 pots, 11.5k infosets;
`scripts/batch_epsilon.py`, 3 seeds; ratio = eps(batched) / eps(ordinary) at the same iteration count):

| batch | batches per run: 16-64 | 256-1024 | 4096-16384 | one batched iteration is worth |
|---|---|---|---|---|
| 4096 | 1.06-1.09 | 1.07-1.11 | 1.20 | ~0.8-0.9 ordinary iterations |
| 16384 | 1.29-1.36 | 1.20 | 1.21 | ~0.7-0.8 |
| 65536 | 3.2-4.7 | 2.6-3.1 | 1.66 (at 1024) | 0.1-0.5: do not use |

- In the tiny push/fold game there is no penalty.
- Duels (3-max, ±1 bb/100) cannot see a 10-20 % eps difference: at equal iterations GPU and CPU drew.
- Net, at equal wall time on 3-max wide: batch 4096 ~x5.5-6 and batch 16384 ~x7.5-8 against the CPU trainer.
  At equal time the GPU blueprint beat the CPU one by about +4 bb/100 (round 6, control CPU/CPU = 0.0).

**Rules.**
- Batched mode is for **blueprint training only**.  Never use it in real-time search, which runs tens of
  thousands of iterations per decision.
- Batch ≤ 16384, and **at least ~1000 batches per run**: ≥ 16M iterations at 16384, ≥ 4M at 4096.
- Prefer 16384 on long runs (it is 1.5x faster than 4096 on the GPU at a similar penalty) and 4096 on short
  ones.

The reports are in branches `opt/gpu-report5..8`, and the summaries on the comms branch are `GPU_DESIGN.md`
and `2026-09-27_batch-epsilon.md`.

Merged into the NegativePluribus master line on 28.09.2026.  The checks on the user's PC (bit identity of
the default trainer and the search, GPU = CPU batched reference and resume on HU 200bb wide, speeds) are in
`docs/backends.md`, "GPU trainer: batched MCCFR on CUDA".

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
  - See the rules above: B ≤ 16384 and at least ~1000 batches per run.
  - Larger batches gain little speed.  They also carry a quality risk on short runs: 65536 on HU at 20M
    iterations lost 11 bb/100.
- **`--checkpoint-every N`**: a checkpoint and a blueprint (plus the `.it<N>` copy) every N iterations.
  - N is rounded up to a multiple of B, so the `.it<N>` names of a GPU run are multiples of B.
  - The final iteration always gets its `.it<N>` copy too (a later `--resume` writes over the plain files).
  - Each checkpoint copies the device tables into the ordinary trainer and frees that copy after writing.
- **`--resume`**: continues from `checkpoint_<tag>.bin` (or `.json`: the one of the larger iteration).
  - No checkpoint is an error (not a new run over the tag's files).
  - Every checkpoint and blueprint has a passport `<file>.run.json`: seed, batch, Linear CFR and `--linear-until`,
    pruning, backend, threads, device, code commit, dates, and the run's segments across `--resume`
    (`scripts/blueprint_info.py FILE` prints it).  Resuming with other seed / batch / linear / pruning values is
    refused (`--resume-override` continues anyway, and the passport records the change).  Checkpoints of before
    29.09.2026 may have the older `checkpoint_<tag>.bin.gpu.json` instead, which is still checked.
  - A run resumed from a GPU checkpoint of the same batch continues **bit for bit** as if it never stopped
    (test `test_checkpoint_and_resume_continue_bit_for_bit`).
  - A CPU checkpoint can be continued on the GPU, and a GPU checkpoint on the CPU.  The formats are the same;
    the results then differ from any single-device run, as any change of trainer does.
- **`--seconds S`**: trains for S seconds of wall time instead of `--iters`, for equal-time comparisons.
- **`--linear-until`** works as on the CPU.  Not supported on the GPU: pruning (`--prune-below`) and the L1
  strategy-change report of `--checkpoint-every`.
- **`--gpu-emulate`** is for tests only.  It runs the GPU kernels' code on the CPU: same numbers, slow, no GPU.
- **Keep the CPU free while the GPU trains.**  The host prepares every batch (deals, buckets, strengths) on
  `--threads` threads.  With the CPU oversubscribed this part collapses.  On HU 200bb wide, a 12-thread CPU
  trainer and a 4-thread job next to it (16 logical CPUs) took the GPU run from ~735k to ~85k it/s
  (28.09.2026, `docs/backends.md`).  Keep the busy threads of all jobs at or below the logical CPUs.
- **`--threads 4` with `--gpu`.**  The default is all cores, and more than 4 host threads made the small
  HU 100bb game slower (389k it/s at 12 against ~1M at 4); on HU 200bb wide 2-12 threads gave the same speed.

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
