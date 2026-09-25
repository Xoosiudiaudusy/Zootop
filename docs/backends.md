# Fast backends: C++ core and multiprocess MCCFR

The pure-Python modules stay the **reference implementation** and the default.  This document
describes the two accelerations layered on top, how to build and select them, what is
bit-identical and what is only statistically equivalent, and the measured numbers.

```
csrc/                  C++17 sources of negpluribus._fastcore (pybind11, CMake, MSVC/GCC/Clang)
  pyrandom.h           random.Random-compatible MT19937, CPython tuple hash, CPython 3.12+ sum()
  evaluator.h          port of evaluator.py (identical integers)
  equity.h             port of equity.py::equity_vs_random (identical draws)
  engine.h             port of engine.py::HandState (flat struct, memcpy clone)
  abstraction.h        BetGrid + pseudo-harmonic, canonical form, FormCache (bounded, lock-free per-street
                       cache), E[HS] and potential-aware bucketers, infoset key
  nodetable.h          numeric infoset keys, flat lock-free node table
  mccfr.h              external-sampling MCCFR / Linear CFR, N threads, per-node spinlocks
  rnr.h                Restricted Nash Response trainer (two tables, tabular opponent model)
  binio.h, persist.h   binary checkpoints / blueprints, blueprint lookup, JSON read and written in C++
  bindings.cpp         the Python module
negpluribus/fast/      Python side: dispatch, phevaluator tier, CppMCCFRTrainer, CppRNRTrainer,
                       blueprint.py (C++ lookup for the agents), binfmt.py (pure-Python file reader)
negpluribus/cfr/parallel.py   multiprocess driver (any backend, spawn-safe)
scripts/build_fast.py  build the extension in place
scripts/bench_backends.py     the benchmark below
tests/test_backends.py        acceptance tests (skipped automatically when the core is not built)
```

## Build (Windows / MSVC)

Prerequisites: Visual Studio Build Tools (C++ workload; both VS 2022 and the "18" toolset work),
CMake >= 3.18 on `PATH`, Python 3.10+.  No developer prompt is needed: the script uses the
Visual Studio generator, which finds MSBuild itself.

```powershell
python -m pip install -e ".[dev,fast,fast-build]"    # pytest, phevaluator+numpy, pybind11+cmake+ninja
python scripts/build_fast.py                         # -> negpluribus\_fastcore.cp3XX-win_amd64.pyd
python -c "import negpluribus.fast as f; print(f.describe())"
python -m pytest -q                                  # 79 reference tests + tests/test_backends.py
```

Options: `--clean` wipes `build/fast`; `--generator Ninja` (needs `cl.exe` on `PATH`, i.e. a
"x64 Native Tools" prompt) for a faster incremental build; `--config Debug`.  The equivalent by
hand:

```powershell
cmake -S csrc -B build\fast -A x64 -DPython_EXECUTABLE=(python -c "import sys;print(sys.executable)") -Dpybind11_DIR=(python -c "import pybind11;print(pybind11.get_cmake_dir())")
cmake --build build\fast --config Release --parallel
```

Linux/macOS: the same script (`-O3 -ffp-contract=off`, system compiler).  The extension is
git-ignored (`*.pyd`, `*.so`); rebuild after pulling changes to `csrc/`.

## Selecting a backend

| what | how | default |
|---|---|---|
| evaluator / equity / canonical form | automatic: C++ core if built, else `phevaluator` (evaluator only), else Python; `NEGPLURIBUS_FAST_EVAL=0` forces Python | on |
| MCCFR traversal | `MCCFRTrainer(spec, bucketer, seed, backend="cpp", threads=T)` or `NEGPLURIBUS_BACKEND=cpp` (`NEGPLURIBUS_THREADS=T`) | `python` |
| RNR traversal | `RNRTrainer(..., backend="cpp", threads=T)` or the same env var; needs a tabular model (`OpponentModel` / `BlueprintStrategy`), otherwise stays Python | `python` |
| multiprocess (python traversal) | `trainer.train_parallel(iterations, workers=K, sync_every=N)` | - |

`negpluribus.fast.describe()` prints what is active.  `backend="cpp"` without the built core
raises a clear error; the env var does too.

The `backend` flag returns a subclass (`CppMCCFRTrainer` / `CppRNRTrainer`) with the same
interface: `train`, `iterate`, `iteration`, `nodes_touched`, `nodes` (a live view of the C++
table), `strategy()`, `save_checkpoint()` / `load_checkpoint()`.  A `*.json` checkpoint has the
JSON layout (and the bytes) of the Python trainer, so checkpoints and blueprints stay
interchangeable between backends (tested both ways); any other name gives the binary checkpoint
of "Binary checkpoints and blueprints" below, which `scripts/train_blueprint.py` writes by
default with the C++ backend.  Scripts such as `scripts/train_blueprint.py` and
`scripts/exploitability.py` work unchanged with `NEGPLURIBUS_BACKEND=cpp`.

## What is identical, what is equivalent

**Bit-identical to the reference** (tests in `tests/test_backends.py`):

* `evaluate()` returns the very same integers (100k random 5..7 card hands).  The `phevaluator`
  tier translates its 1..7462 rank through a table built once against the Python evaluator,
  so it is identical too, not merely order-preserving.
* `equity_vs_random(..., rng)` with a `random.Random`: the generator state is handed to C++, the
  identical Mersenne-Twister draws (`sample`, both branches of CPython's algorithm) are made
  there, and the advanced state is written back.  Same float, same subsequent random numbers.
* `EquityBucketer.ehs` / `bucket`: canonical form, CPython tuple hash seed, Monte-Carlo on the
  canonical representative, cut points: same values, same buckets, in every process and every
  backend (since 2026-09-23; see "Bounded bucket caches" for why the representative matters).
* Engine: identical observations, events, net chips, showdown seats and winners on 2000 random
  hands with random legal actions (2, 3, 6 players, short stacks, antes, every `max_street`).
* Infoset keys and legal abstract action lists: identical along random trajectories in four
  games (preflop-only, flop, 6-player river with a 3-size grid).
* **`MCCFRTrainer(backend="cpp", threads=1)` reproduces the Python trainer bit for bit** for the
  same seed: same deals, same sampled actions, same regrets / strategy sums / visit counts /
  `nodes_touched` (CPython's compensated `sum()` and `round()` are ported; FMA contraction is
  off).  The same holds for `RNRTrainer` with a tabular model and warm start.

**Statistically equivalent** (not seed-identical):

* `threads > 1`: iteration `t` is run by thread `(t-1) % T` with its own RNG stream, so the
  *deals* are deterministic but the interleaving of table updates is not.  Every node has a
  spinlock; the current strategy is read, regrets and strategy sums are added under it, so no
  update is lost.  What remains is *staleness*: a traverser computes its regret update with the
  strategy it read before descending while other threads may have updated the node meanwhile;
  the semantics of lock-free MCCFR (Pluribus).  Measured effect: none visible (table below).
* `train_parallel`: the same staleness for a whole round (`sync_every` iterations per worker on
  one broadcast strategy).  Regrets, strategy sums and visits are additive, so the master sums
  the workers' differences; Linear CFR weights stay consistent because worker `w` runs the
  iteration indices `base+1+w, base+1+w+K, ...` and the global counter advances by `K*sync_every`
  per round.  Larger `sync_every` = fewer merges but more staleness (curve below).
* `equity_vs_random(..., rng=None)`: an unseeded call was never reproducible; it now uses a
  C++ generator seeded from `random.getrandbits(64)`.

**Documented differences**

* The C++ grid accepts at most 5 raise fractions per street (8 actions per node); the
  reference has no limit.  Larger grids raise `ValueError` at construction.
* `BetGrid.from_concrete` on an empty grid: the reference would raise `IndexError` for a
  non-all-in raise (never reached in self-play); the port returns `"a"`.
* `RNRTrainer(backend="cpp")` evaluates `model.policy(key, names)` once per blueprint key in
  Python and uses that table; for keys outside the blueprint the model answers `None` in both
  versions.  A model whose `policy` is not a function of the key alone (none in the repo) could
  not be tabulated; then, and for any non-`OpponentModel`/`BlueprintStrategy` model or a
  `warm_start` that overrides `policy`, the trainer silently stays on the Python traversal
  (`trainer.backend == "python"`).
* Multithreaded runs are not reproducible run to run (measured spread at 100k push/fold
  iterations: 0.07 average absolute difference between two runs of the same seed).

## Bounded bucket caches (2026-09-23)

**The problem.** Both C++ bucketers cached their per-canonical-form values in unbounded
`std::unordered_map`s.  On the flop that is fine (1,286,792 canonical forms exist, all fit),
but 13,960,050 canonical turn forms and 123,156,254 river forms exist for our key, and a 4-street run
keeps meeting new ones: on the 2-player 100bb river game the cache grew by ~2.4M entries per
million iterations (~74 bytes each with the map overhead), the process reached 3 GB working set
at 16M iterations, and the maps' rehashes under the shard locks left 3 of 16 cores busy.

**The fix: `FormCache` (csrc/abstraction.h).** One fixed-capacity, lock-free, 8-way
set-associative table per street: 44-bit packed canonical form -> 20-bit value, one 64-bit
atomic word per slot (0 = empty), one set = one 64-byte cache line.  A lookup is up to 8
relaxed loads, an insert one relaxed store (first empty way, else a per-thread round-robin
victim); no locks, no rehashing, no per-entry metadata, no global LRU list, and the table is
allocated once on the first insert (preflop-only games never allocate).  Memory is flat by
construction: 8 bytes per slot.  The E[HS] value is stored as `2 * won` (an exact integer:
with one opponent each sample adds 1 or 1/2), which decodes to the very same double as the
reference's `won / samples`; the potential-aware bucketer stores the bucket index.

**Caps.** Per street, in entries, rounded up to whole 8-way power-of-two sets:

| how | flop | turn | river |
|---|---:|---:|---:|
| default (`fast.DEFAULT_CACHE_CAPS`, since 2026-09-24) | 4,000,000 -> 4,194,304 slots (32 MB) | 32,000,000 -> 33,554,432 slots (256 MB) | 4,000,000 -> 4,194,304 slots (32 MB) |
| default before 2026-09-24 | 4,000,000 (32 MB) | 4,000,000 (32 MB) | 4,000,000 (32 MB) |
| env `NEGPLURIBUS_BUCKET_CACHE="flop,turn,river"` (K/M/G suffixes; one value = all three; `0` = no cache) | | | |
| kwarg `MCCFRTrainer(..., backend="cpp", cache_caps=(f, t, r))` / `"4M"` / `int` (also `RNRTrainer` with `backend="cpp"`, `core_bucketer(bk, caps)`) | | | |
| script `train_blueprint.py --bucket-cache 4M,32M,4M` | | | |

320 MB per trainer at the defaults for a 4-street game (96 MB before 2026-09-24).  A street's
table is allocated on its first insert, so a flop game uses 32 MB and a preflop-only game
nothing.  All flop forms fit, and since 2026-09-24 all turn forms (42% load); on the river the
cache is a working set.  (The within-iteration reuse, every traverser node of one hand asking
for the same (hole, board), is served by the trainers' per-iteration memo since 2026-09-24,
before the cache is consulted; see "Speed-ups 2026-09-24".)  `trainer.cache_stats()` reports per
street `capacity`, `size`, `computes` (= misses) and `evictions`; the training script prints
them at every checkpoint.  A core bucketer built without caps (e.g. `core.Bucketer(...)`
directly) keeps 4M entries per street.

**Correction (2026-09-24, counted and measured).** Two statements above were wrong.

1. *The form counts.* Our key keeps the board as one sorted set, so the turn card is not told
   apart from the flop cards.  A Burnside count over the 24 suit permutations gives 1,286,792
   flop, 13,960,050 turn and 123,156,254 river forms (the flop number matches the 1,286,766
   entries the cache holds after a run).  The often-quoted 55,190,538 and 2,428,287,420 count
   forms that keep the turn and river cards apart, which this key does not do.
2. *"Bigger caps buy little."*  True for E[HS], false for the potential-aware bucketer on the
   turn.  One bucket computation, single thread, measured while a 16-thread run was going:

   | street | E[HS] | potential-aware |
   |---|---:|---:|
   | turn | 65 us | 1,425 us (46 estimates of 100 runouts) |
   | river | 59 us | 87 us (exact equity) |

   The 50M-iteration runs on the narrow 100bb game recomputed 69.7M turn and 96.6M river
   buckets with 4M slots per street.  For potential-aware buckets that is 73-86% of the run's
   CPU time on the turn alone (86% from the benchmark cost; 73% if the non-bucket work is taken
   equal to the E[HS] run's), which is why that run took 7,184 s against 2,048 s for E[HS].
   All turn forms fit in 32M slots (256 MB, 42% load): `--bucket-cache 4M,32M,4M` (the default
   since 2026-09-24) bounds the turn recomputations by 13.96M.  The river does not pay back:
   123M forms against 96.6M computes means few repeats.  Prediction before measuring
   (hypothesis): a potential-aware run of 40M iterations with a 32M-slot turn cache takes about
   45 minutes instead of about 96.

**Bit-identical, by construction and by test.** Every cached value is a pure function of the
canonical key, so an eviction only costs a recomputation.  `tests/test_backends.py::
test_bounded_bucket_cache_is_bit_identical` trains the 4-street game with a 64-entry cache
(constant eviction), with no cache and with a roomy cache, for both bucketer kinds, and asserts
identical regrets / strategy sums / visits / strategies, all equal to the Python reference.

*The one reference-semantics change this needed:* `EquityBucketer.ehs` used to run its
Monte-Carlo on whichever suit representative of a canonical form it met first, and cached that.
The value therefore depended on visiting order: for 141 of 300 random forms two representatives
gave different E[HS] (the deck lists differ, so the same MT draws pick different cards).  An
evict-and-recompute cache cannot be bit-identical under that rule, and multi-threaded runs
were not reproducible near cut points either.  Since 2026-09-23 the Monte-Carlo runs on the
canonical representative (`key[:2]`, `key[3:]`), in Python and in C++ - exactly what the
potential-aware bucketer already did.  Consequences: E[HS] values / buckets are the same in
every process and backend (`test_ehs_is_a_pure_function_of_the_canonical_key`); the fitted cut
points (`fit()`) are unchanged; blueprints trained before the change stay loadable, but a
flop-game training run started after it is statistically equivalent, not bit-identical, to one
started before (a small fraction of hands near a cut point changes bucket).

**Checkpoints and views (memory between exports).** `trainer.nodes` of the C++ trainers is a
live read-through view (`fast.trainer.NodeView`: `len`, `get`, `[]`, iteration and `items()`
go to the core; `nodes.snapshot()` for a dict copy), so `len(trainer.nodes)` or a few
`nodes.get(key)` between checkpoints no longer materialise the whole table, and
`strategy()` / `save_checkpoint()` build transient dicts.  Checkpoints written by the C++
trainers carry every thread's RNG state (`"rng_states"`), so a resumed run continues its
streams and the iteration counter (Linear CFR weight `t`): with one thread, resume + train is
bit-identical to an uninterrupted run (`test_cpp_resume_is_bit_identical_and_nodes_view_is_live`);
the Python trainer ignores the extra fields and re-seeds, as before.  `train_parallel` with a
C++ master merges the workers' deltas inside the core (`add_tables`) instead of through a
Python copy of the table.

**Measured (2p 100bb river, grid preflop (1.0) / postflop (0.5, 1.0), 2 raises, 8 E[HS]
buckets, `data/buckets_hunl100_ehs_s0.json`; i5-14400F, 10 cores / 16 threads, 31.8 GB).**

*Before* (unbounded `unordered_map` caches, the build of commit b9f4dba, 16 threads, 30M
iterations; working set sampled externally):

| iterations | elapsed | it/s (interval) | cache entries | working set / private bytes |
|---:|---:|---:|---:|---:|
| 1M | 36 s | 27,900 | 4.8M | |
| 5M | 189 s | 26,800 | 17.9M | |
| 10M | 372 s | 28,600 | 30.1M | |
| 16M | 577 s | 30,800 | 41.7M | 2,973 MB / 3,094 MB |
| 20M | 707 s | 30,800 | 48.2M | |
| 23M | 811 s | 29,300 | 52.7M | 3,571 MB / 3,722 MB |
| 30M | 1,051 s | 29,600 | 62.1M | (≈74 bytes per entry: ~4.6 GB) |

Memory grows linearly with the iterations (about 2M new turn/river forms per million
iterations); on this 32 GB machine the throughput had not yet collapsed at 30M (the reported
crawl at 3 GB came from a machine that started paging), but the trend is the same.  Rows from
20M on were measured while a test suite ran on the same cores.

*After* (bounded caches, default caps 4M/4M/4M = 12.6M slots, 96 MB; working set / private
bytes from `GetProcessMemoryInfo` inside the process; `scripts/`-equivalent probe: strategy()
+ save_checkpoint() every 5M iterations in the 16-thread run):

| threads | iterations | elapsed | it/s (interval) | nodes/s | cache entries | working set / private |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 250k | 83 s | 3,006 | 316k | 1.36M | 141 MB / 133 MB |
| 1 | 500k | 177 s | 2,654 | 388k | 2.60M | 141 MB / 133 MB |
| 1 | 1M | 361 s | 2,741 | 437k | 4.77M | 141 MB / 133 MB |
| 16 | 5M | 188 s | 26,600 | 4.32M | 9.64M | 153 MB / 144 MB |
| 16 | 10M | 383 s | 25,593 | 4.35M | 9.67M | 159 MB / 150 MB |
| 16 | 15M | 569 s | 26,973 | 4.62M | 9.68M | 168 MB / 159 MB |
| 16 | 20M | 761 s | 25,948 | 4.50M | 9.68M | 171 MB / 162 MB |

The cache fills to a steady ~9.7M of 12.6M slots (8-way random replacement) within the first
5M iterations and stays there; the working set is flat at ~0.16 GB instead of growing by
~0.15 GB per million iterations.  Per-thread throughput on this 6P+4E-core CPU: ~2,750 it/s on
one thread, ~26k it/s on 16 (9.5x; the E-cores and the shared node table account for the
rest), the same rate as the unbounded build had before its memory grew.

## Exact measurements 2026-09-24 18:47-18:50 (idle machine, MSVC /O2 /fp:precise)

**One turn bucket computation** (a cache miss), 7 repetitions, mean +/- standard error; "fast" = our
hand evaluated once on a complete board + card-draw flags on the stack; identical results on
16,000 situations.  "16 threads" = 16 threads computing at once, as in a 16-thread training.

| computation | 1 thread, us | 16 threads, us |
|---|---:|---:|
| potential-aware turn, current | 632.0 +/- 8.6 | 1295.5 +/- 16.9 |
| potential-aware turn, fast | 339.6 +/- 7.0 | 608.4 +/- 15.1 |
| E[HS] turn, current | 30.8 +/- 0.6 | 61.3 +/- 1.7 |
| E[HS] turn, fast | 25.4 +/- 0.6 | 45.6 +/- 0.5 |

**Where one MCCFR iteration goes** (a copy of `Trainer::traverse` with a cycle counter around one
operation per run, the others untimed; 2p 100bb river game, 16 E[HS] buckets, 1 thread, 8,000
replayed deals so every bucket lookup is a cache hit; 90,395 infosets in the table):

| operation | ns per call | calls per iteration | us per iteration |
|---|---:|---:|---:|
| node lookup, `NodeTable::get_or_create` | 2292.9 | 85.1 | 195.2 |
| history string | 269.8 | 80.5 | 21.7 |
| canonical form (postflop) | 289.1 | 70.6 | 20.4 |
| copy of the game state for a child | 187.8 | 91.6 | 17.2 |
| regret matching + node lock | 114.7 | 87.4 | 10.0 |
| key string assembly | 92.4 | 84.6 | 7.8 |
| to_concrete + apply | 48.6 | 156.9 | 7.6 |
| bucket value lookup (cache hit) | 35.7 | 71.3 | 2.5 |
| everything else | | | 5.4 |

Untimed baseline 269.4 us per iteration (80.6 nodes).  With 8 buckets (54,619 infosets) the node
lookup costs 997.5 ns per call (81.7 of 190.7 us): the lookup gets 2.3x dearer as the table grows,
which is why the 16-bucket training runs about 2x slower than the 8-bucket one.  The copy of the
game state costs 20.9 ns instead of 187.8 ns when `Event::stack_after` has no default member
initializer (the initializer makes every copy initialise all 512 events).  16-thread untimed
baselines, us per iteration per thread: 16 buckets 390.1 now / 346.7 without the initializer;
8 buckets 293.6 / 224.4.  Replayed deals keep caches warm, so absolute times are lower than in a
real run; the ranking of the operations is the result.

### Prediction before the node-table work (2026-09-24 19:00, hypothesis)

Planned bit-identical changes: per-iteration bucket memo, no default initializer on
`Event::stack_after`, card-draw flags on the stack, our hand evaluated once on a complete board
(first agent task); then a flat open-addressing node table with a numeric key built from the
node fields and an incremental history hash, so no key string is built per node (second task).
Per iteration, same benchmark (16 buckets, one thread), in us:

| part | measured now | predicted | why |
|---|---:|---:|---|
| node lookup | 195.2 | about 20 | 1-2 cache misses instead of a chain of pointers and a string compare |
| history string + key assembly | 29.5 | about 2 | incremental hash, no strings on the hot path |
| canonical form + bucket lookup | 22.9 | about 2.5 | memo: 6 lookups per iteration instead of 71 |
| state copy | 17.2 | 1.9 | measured without the initializer |
| everything else | 23 | 23 | unchanged |
| total | 269 (untimed) | about 50 | |

Whole runs, 16 threads, prediction: E[HS] 16 buckets, 40M iterations: 47.5 min measured today,
about 11 min expected; potential-aware 16 buckets, 40M: about 96 min at the old speed, about
22 min expected.  Check: the same benchmark on the new build, then the training log.

### Found 2026-09-24 19:10: the node lookup is slow because of a sharding bug

`NodeTable` picks the shard with the low 8 bits of `std::hash<std::string>` and each shard's
`std::unordered_map` picks its bucket with the low bits of the same hash, so all keys of a shard
share their low 8 bits and fall into bucket_count/256 buckets: lookups scan chains of strings.
Prototype of the current design (random lookups, one thread, idle machine), ns per lookup:

| nodes | current | shard from the top 8 hash bits | flat table, 128-bit numeric key |
|---:|---:|---:|---:|
| 55,000 | 1227.2 | 234.6 | 10.6 |
| 90,000 | 2100.9 | 358.1 | 12.3 |
| 125,000 | 3106.8 | 434.1 | 19.2 |

The current cost grows linearly with the table, as a chain scan does (the traversal benchmark:
998 ns at 55k, 2293 ns at 90k).  Taking the shard from the top bits is a one-line, bit-identical
fix (the key-to-node mapping is unchanged).  Computing a numeric key from the node fields plus an
incremental history hash costs 3.6 ns per node in the same prototype.  Refined prediction per
iteration (16 buckets, one thread, sum of the timed parts 287 us): about 86 us after the shard
fix, memo and copy fix; about 27 us with the flat table and no key strings.

### Step 1 measured against the prediction (2026-09-24 20:30)

Same traversal benchmark (16 E[HS] buckets, 8,000 replayed deals), now with the power-throttling
opt-out and the single thread pinned to a P-core; the old code (headers of 129704d) re-measured
back to back under the same conditions, so the old numbers are lower than in the table above.

| part, us per iteration (1 thread) | old | step 1 | factor |
|---|---:|---:|---:|
| node lookup | 134.6 (1581 ns x 85) | 13.2 (154 ns x 85) | 10.2 |
| canonical form + bucket lookup | 20.9 (71 calls) | 3.2 (5.7 calls) | 6.5 |
| state copy | 15.0 (164 ns) | 1.5 (16.5 ns) | 10.0 |
| history string + key assembly | 27.0 | 24.7 | unchanged (step 2) |
| everything else | 19.9 | 16.4 | |
| untimed iteration | 214.9 | 73.0 | 2.94 |
| untimed iteration, 16 threads (per thread) | 343.5 | 140.4 | 2.45 |

Predicted for step 1: 3.3x on one thread.  Measured: 2.94x on one thread, 2.45x on 16.
Left for step 2: history string + key assembly 24.7 us and node lookup 13.2 us of the 59 us
timed sum.

## Speed-ups 2026-09-24 (bit-identical)

Six changes; no number changes.  Every regret, strategy sum, visit count, bucket and equity
value is bit-identical to the build before and to the Python reference (which did not change);
key, checkpoint and blueprint formats are unchanged (a checkpoint lists its nodes in a
different order after #6; its content is the same).

| # (commit) | change | where | what it removes |
|---|---|---|---|
| 1 (a01f20c) | `PyRandom::sample` without heap allocation up to 64 items: the pool branch copies into a stack array, the "selected" set is a 64-bit mask (heap fallback above 64).  Same branch choice, same `randbelow` calls in the same order as CPython; `k` outside `0..n` now raises `ValueError` like CPython (the port looped forever) | `csrc/pyrandom.h` | a malloc/free per Monte-Carlo sample |
| 2 (71e13ad) | `equity_won_vs_random`: deck, draws and hands on the stack; on a complete board our hand is evaluated once per call instead of once per sample (same draws, same tie split `1/(ties+1)` for any number of opponents, same summation order of `won`) | `csrc/equity.h` | 1 of 2 evaluations per sample of an E[HS] river bucket and of each of the 46 x 100 runouts of a potential-aware turn bucket |
| 3 (f53f182) | per-iteration bucket memo: all traversals of one iteration replay the same deal, so the bucket of (seat, board size) is computed once per iteration and thread (`ThreadCtx::bucket_memo`, reset after the shuffle; `infoset_key_for_bucket` builds the key).  `rnr.h` has the same structure and shares it | `csrc/mccfr.h`, `csrc/rnr.h`, `csrc/abstraction.h` | a `canonical_form()` (0.35-0.5 us) + cache probe per decision node: 53 bucketer calls per iteration of the 2-player 100bb river game -> 5 (table below) |
| 4 (c8caa6d) | `Event` has no default member initializer (found and measured by the main session).  `stack_after = -1` had made `Event` non-trivial, so every `HandState` construction - the copy `HandState child(st)` of every traversal step included - stored -1 into all 512 `events[]` (14 KB) before `copy_from()` overwrote the used prefix.  `HandState::apply`, the only place that creates events, sets every field; entries at or after `n_events` are never read; a `static_assert` keeps `Event` trivial | `csrc/engine.h` | 14 KB of stores per state copy: 132.5 -> 13.8 ns per `HandState` copy, 139.4 -> 26.2 ns for copy + `apply` (standalone loop, pinned P-core, idle machine; the main session measured 286 -> 31 ns inside a copy of `Trainer::traverse`) |
| 5 (6a5613b) | default caps 4M / 32M / 4M entries (flop / turn / river): all 13,960,050 turn forms fit | `negpluribus/fast/__init__.py` | recomputing turn buckets in long runs (73-86% of the CPU time of the 50M potential-aware run with 4M turn slots, see above) |
| 6 (af720d5) | `NodeTable` takes the shard from the top 8 bits of the key hash instead of `h % 256` (found and measured by the main session).  Each shard's `unordered_map` picks its bucket from the low bits of the same hash (MSVC: `hash & (buckets - 1)`), so every key of a shard had the same low 8 bits and only 1/256 of the buckets were used.  The key -> node mapping does not depend on the shard; only the export order changes (tests compare contents) | `csrc/mccfr.h` | chain walks with string compares: a random lookup 1,082 -> 155 ns at 61k nodes and 2,797 -> 278 ns at 183k (standalone, real keys); the 20k-iteration E[HS] run 216 -> 174 us per iteration |

**Bit-identity evidence.**

* Tests: the whole suite passes on the final build (129 tests, 5 of them new, none skipped with
  `data/class_equity.json` and `data/buckets_3p_15bb_flop.json` present; the existing
  bit-for-bit tests compare table contents, so the new export order of #6 does not matter to
  them).  New: `test_cpp_sample_matches_cpython_random_sample`
  (`random.Random.sample` against the port: 20 population sizes x up to 17 k, both CPython
  branches with stack and heap storage, 4 consecutive calls per seed, the generator state
  afterwards; plus 300 seeds x 20 calls of the shapes the bucketers draw: 47/4, 46/3, 45/2 and
  50/15), `test_cpp_equity_on_complete_boards_matches_python` (103 spots x 1/2/3/5 opponents x
  1/10/100 samples incl. boards that play for everybody: value and generator state),
  `test_cpp_turn_histogram_with_production_samples_matches_pure_python` (46 x 100 runouts
  against the fully pure-Python chain), `test_cpp_rnr_bit_identical_on_the_4_street_game`
  (RNR cpp x1 == Python on all four streets, both bucketer kinds).  Changed:
  `test_bounded_bucket_cache_is_bit_identical` now bounds the no-cache computes by 2 seats x
  iterations (the memo; before it: 997 / 1,152 / 1,054 flop / turn / river computes in 80
  iterations of that test's E[HS] run with the fitted fallback bucketer, now 144 / 140 / 122,
  which equals the number of distinct forms), `test_cache_caps_from_kwarg_and_env` checks the
  new default slots.
* Mutation check: without the per-iteration reset of the memo 9 tests fail (MCCFR and RNR cpp
  x1 == Python, the bounded-cache test, resume).
* Same training, old build against new: 2-player 100bb river game (grid 1.0 / 0.5, 1.0, 2
  raises, 8 buckets), `MCCFRTrainer(spec, load_bucketer(f), seed=0, backend="cpp",
  threads=1).train(20000)`, the whole `export_tables()` with every float as `float.hex`,
  compared value by value between HEAD 129704d (built in this worktree), the main checkout's
  build and the builds after the changes ("-Event" = #1-3 and #5, "-Shard" = #1-5, "new" = all
  six); every run (each build's own default caps) also fingerprinted the table with SHA-256:

  | bucket file | infosets | values compared | nodes touched | bucket computes flop / turn / river | runs (old / -Event / -Shard / new / main) | result |
  |---|---:|---:|---:|---:|---:|---|
  | `buckets_hunl100v2_ehs_s0.json` (E[HS], 150 samples) | 61,067 | 392,137 | 1,491,875 | 35,793 / 35,766 / 35,375 | 41 (12 / 12 / 12 / 4 / 1) | identical |
  | `buckets_hunl100v2_pot_s1.json` (potential-aware, 100 x 10 bins) | 60,236 | 387,156 | 1,453,430 | 36,120 / 36,208 / 35,870 | 20 (6 / 5 / 5 / 3 / 1) | identical |
  | the E[HS] file with 1 sample per bucket (the traversal-bound variant below) | 23,056 | 148,556 | 1,466,749 | 36,297 / 36,304 / 35,666 | 27 (8 / 8 / 8 / 3 / -) | identical |

  The per-street compute counts are the same in every build too (every compute is a first
  visit at 20k iterations), so the same canonical forms reach the cache with and without the
  memo.  The 300-iteration runs without any cache (below) are identical as well.  With 16
  threads (not bit-identical by design) old, -Shard and new reach the same infoset counts (to
  within 5 of ~63.6k) and the same compute counts per street to within 0.3% (table below).

**Measured** (i5-14400F, MSVC 14.44 /O2 /fp:precise, Python 3.14.6).  Idle machine: the
main session paused its trainings for these runs (a music player and the desktop app were the
only other load).  Builds interleaved (A B C D A B C D ...), every run a fresh process that opts
out of Windows power throttling (see the note below the tables); single-thread runs pinned to
one P-core logical CPU (CPU 4) at above-normal priority so that every build ran on the same core
type.  Builds: "old" = HEAD 129704d built in this worktree, "-Event" = #1-3 and #5, "-Shard" =
#1-5, "new" = all six.

*One bucket computation* (cache off, so every call computes; per call through the Python
binding, which costs 0.34 us of it; minimum of 7 rounds, median over 3 processes per build):

| | old | new | old / new |
|---|---:|---:|---:|
| E[HS] turn (150 samples) | 29.2 us | 24.3 us | 1.20 |
| E[HS] river | 25.3 us | 16.6 us | 1.53 |
| potential-aware turn (46 x 100 runouts, 10 bins) | 576 us | 309 us | 1.87 |
| potential-aware river (exact, 990 combos; code not changed: the control) | 34.1 us | 34.4 us | 0.99 |

The values of all 32,300 calls (E[HS] as `float.hex`, buckets) are identical between the
builds.

*Training, one thread, 20k iterations* of the 2-player 100bb river game, each build's default
caps (every compute is a first visit at this length, so the caps do not matter), wall-time us
per iteration, median of 3 runs (potential-aware: 2 for old and new, 1 for the intermediate
builds); the runs of one build differ by less than 2%:

| bucketer | old | -Event | -Shard | new | old / new |
|---|---:|---:|---:|---:|---:|
| E[HS] file (150 samples) | 280.5 | 231.9 | 216.3 | 174.4 | 1.61 |
| potential-aware file | 2,541 | 1,775 | 1,727 | 1,677 | 1.52 |
| E[HS] cut points, 1 sample per bucket (traversal-bound) | 127.0 | 111.4 | 97.7 | 79.2 | 1.60 |

Per step (E[HS] / potential-aware / traversal-bound): #1-3 (old -> -Event; #5 does not change
the work at this length, the builds after it only zero a 256 MB instead of a 32 MB turn table
once per run, not measured separately) 1.21 / 1.43 / 1.14;
the `Event` fix #4 (-Event -> -Shard) 1.07 / 1.03 / 1.14, i.e. 15.6 / 48 / 13.7 us per
iteration (the potential-aware figure is one run each; the main session measured 25.9 us,
215.3 -> 189.4, in its copy of the traversal); the shard fix #6 (-Shard -> new) 1.24 / 1.03 /
1.23, i.e. 41.9 / 50 / 18.5 us per iteration (more infosets, longer chains before the fix:
61k / 60k / 23k).  In the traversal-bound variant the bucket work is negligible, so its steps
isolate the memo (127.0 -> 111.4 us), the `Event` fix and the shard fix.

*Standalone loops* (pinned, minimum of 5 x 5M, resp. 5 x 2M):

| | old | new |
|---|---:|---:|
| `HandState` copy (turn node, 4 events) | 132.5 ns | 13.8 ns |
| `HandState` copy + `apply` | 139.4 ns | 26.2 ns |
| `NodeTable::get_or_create`, random existing key, 61,067 nodes (the real keys of the 20k E[HS] run) | 1,082 ns | 155 ns |
| same, 122,134 nodes (the keys twice, one copy with a suffix) | 2,553 ns | 236 ns |
| same, 183,201 nodes | 2,797 ns | 278 ns |

*Bucketer calls per iteration* (no cache at all, 300 iterations, so every call computes):

| bucketer | old flop / turn / river | new flop / turn / river | us per iteration old -> new |
|---|---:|---:|---:|
| E[HS] | 11.9 / 17.5 / 23.9 (53.3) | 1.84 / 1.63 / 1.55 (5.0) | 1,465 -> 163 (9.0x) |
| potential-aware | 11.6 / 17.4 / 22.6 (51.6) | 1.80 / 1.58 / 1.47 (4.9) | 19,075 -> 1,586 (12.0x) |

*Training, 16 threads, 1M iterations*, caps 4M / 32M / 4M for both builds (not pinned; wall
time; not bit-identical by design, so the infoset counts and cache statistics are the sanity
check):

| bucketer | build | wall (runs) | iterations / s | busy cores | infosets | computes flop / turn / river | evictions turn / river |
|---|---|---:|---:|---:|---:|---:|---:|
| E[HS] file | old | 40.5 s, 44.5 s | 23,500 | 15.1 | 63,627 | 979k / 1.84M / 1.96M | 0 / 11.8k |
| | -Shard | 29.8 s, 29.6 s | 33,600 | 15.5 | 63,626-63,627 | 980k / 1.84M / 1.96M | 0 / 11.7k |
| | new | 22.8 s, 22.7 s | 44,000 | 15.7 | 63,628 | 980k / 1.84M / 1.96M | 0 / 11.7k |
| potential-aware file | old | 259.3 s | 3,857 | 15.6 | 63,623 | 981k / 1.84M / 1.97M | 0 / 11.7k |
| | -Shard | 156.5 s | 6,389 | 15.5 | 63,628 | 980k / 1.84M / 1.96M | 0 / 11.7k |
| | new | 148.6 s | 6,729 | 15.6 | 63,626 | 980k / 1.84M / 1.96M | 0 / 11.5k |

old / new at 16 threads: 1.87 (E[HS]) and 1.74 (potential-aware); the shard fix alone
(-Shard / new): 1.31 and 1.05.  The gains are larger than on one thread; a likely reason (not
isolated): the old code's per-sample `malloc`/`free` and its long shard chains cost more when
16 threads share the heap and the node table.  All builds reach the same number of infosets (this game saturates at
~63.6k) and compute the same number of buckets per street to within 0.3% (which forms a thread
meets first depends on the interleaving); with 32M turn slots no turn bucket was evicted.

*Windows power throttling (measured; the cause is a hypothesis).*  In an earlier attempt the
new build's 16-thread potential-aware run took 323 s against 273 s for the old build although
it needed 2,318 instead of 4,095 CPU-seconds: its threads were busy on 7 of 16 logical CPUs on
average, and during the next run 13 of its 16 threads were "Ready" (runnable, not running)
while 75% of the CPU was idle; priority was normal, affinity all 16 CPUs.  Opting the process
out of power throttling (`SetProcessInformation(ProcessPowerThrottling)` with the execution-
speed bit cleared, i.e. no EcoQoS) gave 15+ busy cores again; the pinned single-thread copy
loop took 180 ns in two earlier runs without the opt-out and 132 ns with it.  Hypothesis:
Windows 11 treats processes started from an app that is not in the foreground as background
work and puts them on the E-cores; unattended trainings started from the desktop app may be
throttled the same way.  Check: a 16-thread run should show CPU time / wall time near 15; near
4 means throttled.  All numbers in this section come from opted-out processes.

**Next bottleneck (measured by the main session).**  Before #6 the node lookup
`NodeTable::get_or_create` was the largest part of the traversal: on an idle machine, one
thread, 998 ns per call with 8 buckets and a 55k-infoset table and 2,293 ns per call with 16
buckets and a 90k-infoset table, i.e. 43% and 68% of the iteration time, growing with the
table.  #6 removes the chain walks (1,082 -> 155 ns per lookup at 61k nodes, 2,797 -> 278 ns
at 183k, measured here); what remains per decision node is building the key string, hashing it
twice (shard, then map), a shared lock, the probe with a string compare and the
`unique_ptr<Node>` indirection.  Replacing the string-keyed table (a flat numeric-key table
with an incremental history hash) is a larger, separate redesign and was not started here
(done since: "Numeric node keys and a flat node table" below).

**Memory.**  The only change is the turn cache: 96 -> 320 MB per trainer of a 4-street game.
The 256 MB turn table is zeroed when it is allocated, so it is committed at the first turn
insert, not gradually (measured private bytes after 20k iterations of the 100bb river game:
131 MB with the old caps, 355 MB with the new).  Flop games keep 32 MB, preflop-only games 0.
The memo is 216 bytes per thread; `HandState` keeps its size (14,808 bytes); the other changes
only move per-call temporaries (< 1 KB) from the heap to the stack.

**Memory for 6-max (note; estimates marked as such).**

* The bucket caches belong to one trainer (under `train_parallel` every worker process has its
  own), have a fixed size and do not depend on the number of players: a canonical form is one
  seat's hole cards plus the board.  Defaults 4M / 32M / 4M entries = 320 MB.  The river table
  can be raised with `--bucket-cache 4M,32M,128M` when RAM allows (128M entries -> 134,217,728
  slots = 1 GiB).  Estimate (Poisson occupancy of the 8-way sets): at that 92% load about 89% of
  the 123,156,254 river forms can stay resident, with 256M entries (2 GiB) about 99.5%.  Whether
  it pays is open: the 2-player 50M-iteration runs computed 96.6M river buckets for 123M
  existing forms, i.e. mostly first visits, which no cache avoids.
* In 6-max the infoset table will dominate memory, not the caches: per infoset a `Node` (160
  bytes by its layout: regrets, strategy sums, visits, lock), the key string and the hash-map
  entry.  Measured here: about 360 bytes per infoset (2-player 100bb river game, 8 buckets,
  keys of 37 characters on average; private bytes minus the cache tables, divided by the
  infoset count: 21 MiB for 61,067 infosets after 20k iterations on one thread) and 375-395
  bytes after 1M iterations on 16 threads (thread stacks included).  This game saturates at ~63.6k infosets;
  6-max keys are longer and the infoset count grows by orders of magnitude (estimate, not
  measured), so the table has to be measured at the 3-max step before runs are sized.
* Long-term option: a precomputed dense table per bucketer, 1 byte per canonical form (flop
  1.3 MB, turn 14 MB, river 123 MB), read-only and shareable between processes by memory
  mapping, no cache misses at all.  It needs a perfect index of the canonical forms (a
  bijection form -> 0..N-1; the 44-bit packed key used by `FormCache` is not dense) and a
  one-off precompute.  Estimate from the per-bucket costs measured above (new build, one
  thread, idle): E[HS] river 123M x 16.6 us = 34 CPU-min, turn 14M x 24.3 us = 6 CPU-min, flop
  (not measured) about 1 CPU-min; potential-aware turn 14M x 0.31 ms = 72 CPU-min, river 123M
  x 34 us = 71 CPU-min, flop (not measured) roughly 15 CPU-min.  So roughly 5 minutes (E[HS])
  and 14 minutes (potential-aware) per bucketer on 16 threads, if they scale like the training
  here (7.7x and 11.3x from 1 to 16 threads on this 6P+4E CPU); all of this is an estimate.

## Numeric node keys and a flat node table (2026-09-24, bit-identical)

Step 2 of the node-table work (prediction above).  Results unchanged: every regret, strategy sum,
visit count, bucket and equity value is bit-identical to the build before (dbd883a) and to the
Python reference, which did not change.  Key, checkpoint and blueprint formats are unchanged;
a checkpoint lists its nodes in table order (contents the same).

| commit | change |
|---|---|
| 9cbb451 | `equity_vs_random` bindings reject more than 2 hole or 5 board cards (they wrote past fixed arrays; `ValueError` now) |
| 4bcdb0f | `BetGrid::from_concrete` sorts the grid on the stack instead of two vector copies per raise; it now runs once per new event, so it is on the hot path |
| bba0cc6 | numeric infoset keys, a flat lock-free node table, the key string kept per node, a test mode that checks the numeric keys against the strings (`csrc/nodetable.h`) |

**Numeric key.**  `NodeKey` = the fields street, position index (seat relative to the button),
`n_active` and bucket, plus a 128-bit hash (two independent 64-bit lanes) of the history, the
text after the 4th `|` of the key string.  The traversal never builds that text: it carries a
`HistHash` and, entering a state with a new event, hashes exactly the bytes `history_string`
would write for it ('/' when the street changes except before the first event, ' ' between two
events of one street, then the token `from_concrete` appends).  So a node's key costs one
translation of the new event and a few multiplications instead of the whole history and a
string.  `KeyCodec::of_string` parses a key string into the same key (checkpoint import, lookups
from Python); a string not in the canonical form our code writes (unknown position name,
leading zeros, missing field...) gets a "foreign" key, a hash of the whole string with a flag
bit set, which no traversal key can equal.  Two keys are therefore equal iff their strings are,
up to a collision of the 128-bit hash (none among the 361,809 distinct key strings of the 87
checkpoints and blueprints in `data/`, all canonical; a collision would raise on import).

**Table.**  Power-of-two open addressing, linear probing, 32-byte slots {k1, k2, node, key
string}.  Lookups are lock-free (acquire load of the node pointer, then the keys).  An insert
prepares the node and its key string in the calling thread's arena, reserves room, claims the
empty slot with one CAS and publishes the node with a release store; a thread meeting a claimed
slot waits for that publication (a few stores).  Slots never become empty again, so all
inserters of a key walk the same probe sequence and meet the first claim: no node is ever
duplicated or dropped.  The table doubles when it is half full, at a stop-the-world point: the
table raises a flag, every worker parks at its next lookup, the last one to park grows the
table(s) of its trainer (MCCFR one, RNR two) and wakes the others; a reservation never takes a
table past 3/4, so workers that have not seen the flag yet cannot fill it (the stress test
found that case: 16 threads on a 16-slot table).  Nodes and key strings come from per-thread
arena chunks; the `Node` layout is unchanged.  Growth from the initial 4,096 slots to the
262,144 of a 125k-infoset table took 6 doublings.

**Key strings.**  Built once per node, at creation, with today's `infoset_key_for_bucket`, and
kept for export and import; a lookup by string (Python `get_node`, imports) must match the
stored string too.  **Test mode:** `NEGPLURIBUS_VERIFY_KEYS=1` (set by `tests/conftest.py` for
the whole suite, off in production) rebuilds the string at every lookup and compares it with the
node's, and the RNR trainer also checks its numeric opponent-model index against the string
table; `train()` raises on a mismatch.  The traversals are then as slow as before.  The
Python-facing table accessors wait for a running `train()` (a mutex) instead of racing a
table growth.

**Bit-identity evidence.**

* The suite passes with the test mode on: 135 tests, none skipped.  New:
  `test_numeric_keys_equal_the_parsed_key_strings` (the key computed from the fields and the
  incremental hash equals the key parsed from the string at every state of random trajectories
  in four games, over 2,000 distinct keys; nine non-canonical variants get foreign keys),
  `test_verify_keys_is_on_in_tests_and_catches_a_wrong_stored_key` (a relabelled node makes
  `train()` raise, for MCCFR and for RNR; without the test mode nothing is checked),
  `test_flat_node_table_concurrent_inserts_never_drop_or_duplicate` (16 threads look up the same
  100k keys in different orders, 3 rounds, on a table growing from 16 slots: every key created
  once, every lookup returns that node with its own string; 500 more runs with 2-32 threads,
  5k-24k keys: no anomaly), `test_cpp_threads_grow_the_node_table_without_losing_or_duplicating_nodes`,
  `test_export_import_keep_contents_including_non_canonical_keys` (any order; imported tables
  in two insertion orders keep training identically), `test_cpp_equity_rejects_more_than_two_hole_or_five_board_cards`.
* Old build against new, 2-player 100bb river game, one thread, the whole `export_tables()`
  compared value by value (floats as `float.hex`) and by SHA-256; "main" = the main checkout's
  dbd883a build, "base" = the same sources built here, "new+verify" = the new build in test mode:

  | run | infosets | values compared | main / base / new / new+verify |
  |---|---:|---:|---|
  | E[HS] 8 buckets, 20k iterations from scratch | 61,067 | 392,137 | identical |
  | potential-aware 8 buckets, 20k from scratch | 60,236 | 387,156 | identical |
  | E[HS] 16 buckets (`buckets_hunl100v2_ehs16_s0.json`), 20k from scratch | 111,495 | 720,625 | identical |
  | resume `checkpoint_hunl100v2_ehs16_s0.json` (40M iterations) + 5k | 125,228 | 800,446 | identical |
  | resume `checkpoint_hunl100v2_ehs_s0.json` (50M) + 5k | 63,626 | 406,804 | identical |
  | resume `checkpoint_hunl100v2_pot_s1.json` (20M) + 5k | 63,628 | 406,814 | identical |

**Measured** (idle machine: the main session's trainings paused; every process opted out of
power throttling; one-thread runs pinned to a P-core at above-normal priority; builds
interleaved; "old" = master dbd883a built here, "new" = bba0cc6).

*The two operations this step replaces*, standalone loops (pinned, best of 5):

| | old | new |
|---|---:|---:|
| node lookup, random existing key, 61,067 nodes (the keys of a 20k-iteration 8-bucket run) | 152-158 ns | 13.5-14.1 ns |
| same, 122,134 / 183,201 nodes (the keys twice / three times) | 239 / 280 ns | 19.7 / 24.5 ns |
| node lookup, 125,228 nodes (the keys of `checkpoint_hunl100v2_ehs16_s0.json`) | 243 ns | 20.2 ns |
| same, 250,456 / 375,684 nodes | 310 / 331 ns | 28.4 / 32.5 ns |
| key per node: old = the key string (`infoset_key_for_bucket` into a fresh string); new = hashing the one new event and `KeyCodec::key`; 3,000 states of the 100bb game, 4.4 events each on average | 199-204 ns | 21.4-21.5 ns |

The old key cost grows with the number of events (the whole history is rebuilt), the new one
does not; the nodes a 16-bucket training visits have 6.4 events on average (visit-weighted, from
the 20k-iteration table), where the main session measured 270 + 92 ns in its copy of the
traversal.  The old lookup includes hashing the ~38-character string, the shard lock and a
string compare; the new one is one or two probes of 32-byte slots (the numeric key comes from the
second row).

*Training, one thread, 20k iterations* of the 2-player 100bb river game, us per iteration
(median of 3 runs, potential-aware 2; the runs of a build differ by less than 2%):

| run | nodes per iteration | old | new | old / new |
|---|---:|---:|---:|---:|
| E[HS] 8 buckets, from scratch | 74.6 | 175.7 | 147.9 | 1.19 |
| E[HS] 16 buckets, from scratch | 72.9 | 178.9 | 149.1 | 1.20 |
| potential-aware 8 buckets, from scratch | 72.7 | 1,684 | 1,685 | 1.00 |
| traversal-bound: E[HS] 8-bucket cut points with 1 sample per bucket, from scratch | 73.3 | 79.6 | 51.4 | 1.55 |
| traversal-bound: 16-bucket cut points with 1 sample, continuing `checkpoint_hunl100v2_ehs16_s0.json` (40M iterations, 125,228 infosets) | 213.3 | 191.3 | 94.5 | 2.02 |

From scratch, 20k iterations are dominated by first-time bucket computations (about 5.4 per
iteration: E[HS] 17-24 us each, potential-aware turn 309 us), which this step does not touch;
the traversal-bound rows keep those to about 1 us.

*Training, 16 threads, 1M iterations*, iterations per second (median of 2 runs, potential-aware
1; 15.1-15.7 busy cores in every run):

| run | old | new | new / old |
|---|---:|---:|---:|
| E[HS] 8 buckets, from scratch | 44,914 | 68,446 | 1.52 |
| E[HS] 16 buckets, from scratch | 43,497 | 66,578 | 1.53 |
| potential-aware 8 buckets, from scratch | 6,672 | 6,954 | 1.04 |
| traversal-bound, continuing the 16-bucket checkpoint (1 sample per bucket) | 44,534 | 105,097 | 2.36 |

*Memory per infoset* (process private bytes before and after loading a checkpoint, divided by
its infosets): 125,228 infosets of the 16-bucket checkpoint 343-345 bytes old, 276 new; 63,626
infosets of the 8-bucket one 363-369 old, 288 new.  New breakdown at 125,228 infosets: 262,144
slots x 32 bytes = 67 bytes per infoset (48% load here; 64-128 bytes as the load moves between
1/4 and 1/2), nodes 161 (160 + chunk slack), key strings 39 (38 characters on average + NUL).
So about 225-290 bytes plus the key length per infoset whatever the game (estimate for 6-max,
where keys are longer and the tree much larger; not measured).

*Against the prediction* (recorded before the work, "Prediction before the node-table work";
their benchmark: a copy of the traversal, 16 buckets, one thread, 8,000 replayed deals, 80.6
nodes per iteration).  After step 1 they measured 73.0 us per untimed iteration, i.e. 0.91 us
per node; the old build on my traversal-bound run costs 0.90 us per node (191.3 us / 213.3
nodes), so the two runs are comparable per node.  The new build: 0.44 us per node, 2.02x;
scaled to their 80.6 nodes per iteration, 72.3 -> 35.7 us (derived, not their harness).  The
prediction was "about 27 us" for the sum of the timed parts, 59.0 us after step 1: replacing its
history string + key assembly (24.7 us) and node lookup (13.2 us) with the per-call costs
measured above (85 x 21.5 ns and 85 x 20 ns) gives about 24.6 us (an estimate from their table
and my per-call numbers, not a measurement of their harness).  16 threads: they measured 140.4
us per iteration and thread after step 1 (1.74 us per node); the traversal-bound 16-thread run
above costs 1.82 us per node and thread old, 0.75 new (2.4x).

**What is left** (not changed here): per decision node the state copy, `observe`,
`abstract_actions`, the node spinlock with regret matching, the one `from_concrete` of the new
event and the RNG; per iteration the bucket computations of first-time canonical forms, which
dominate every run from scratch (E[HS] 17-24 us, potential-aware turn 309 us each, 5.4 per
iteration here) and the river forms of long runs (123M forms, 4M cache slots by default).

## Binary checkpoints and blueprints (2026-09-25, bit-identical)

Step 3: checkpoints, blueprint exports and blueprint lookups no longer go through Python
dictionaries.  The C++ core streams the node table into a binary file and back, writes the
average strategy as a binary blueprint, and the agents look probabilities up in a lean C++ table.
JSON stays readable everywhere and is still written on request, byte for byte as before.

| commit | change |
|---|---|
| e8a0408 | `csrc/binio.h`, `csrc/persist.h`: binary checkpoint and blueprint, `BlueprintTable` lookup, JSON written with json.dump's bytes and read by a streaming parser, the L1 diagnostic in C++; imports give nodes the action ids of their street; a checkpoint's linear flag reaches the core; Python: `fast/blueprint.py`, `fast/binfmt.py`, trainer `save_checkpoint` / `load_checkpoint` by format |
| 9cac814 | `train_blueprint.py` writes binary by default (`--json`, `--format json`, `--no-l1`, `--data-dir`), the eval tools read both formats, `scripts/export_json.py`, `tests/test_persist.py` |
| 8b16065 | rounded probabilities stored as `u32 k` (probability = k / 100000.0), lossless |
| dcb226a | the lookup keeps no vector growth slack after a JSON load (35 instead of 45 bytes per infoset) |
| 2977eca | `exploitability.py`, `search_demo.py`, `search_exploit_demo.py` and `--resume` take the newest of `<name>_<tag>.bin` / `.json`; dict tools read binary blueprints |

**Use.**

```text
python scripts/train_blueprint.py ... --backend cpp        # data/checkpoint_<tag>.bin, data/blueprint_<tag>.bin (+ .it<N>.bin)
python scripts/train_blueprint.py ... --backend cpp --json # the old JSON files as well (same bytes as before)
python scripts/train_blueprint.py ... --resume             # checkpoint_<tag>.bin or .json, whichever was written last
python scripts/export_json.py data/checkpoint_X.bin        # -> data/checkpoint_X.json (both trainers load it)
python scripts/export_json.py data/blueprint_X.bin         # -> data/blueprint_X.json (the JSON of BlueprintStrategy.save)
python scripts/export_json.py data/blueprint_X.json data/blueprint_X.bin   # an old JSON blueprint -> binary
```

`eval_archetypes.py`, `compare_checkpoints.py`, `play_slumbot.py` and `match_mozg.py` take either
format (`fast.blueprint.load_blueprint` tells them apart by the first bytes) and, when the core is
built, look probabilities up in C++ (`NEGPLURIBUS_BLUEPRINT=python` gives the old dict).  The
tools that need the dict itself and open `blueprint_<tag>` by name (`exploitability.py`,
`search_demo.py`, `search_exploit_demo.py`) take the `.bin` or the `.json`, whichever was written
last (`fast.blueprint.tagged_path`; with `--json` the binary files are written after their JSON
twins), and get a `BlueprintStrategy` from either.  In
Python: `trainer.save_checkpoint("x.bin")` / `("x.json")`, `trainer.load_checkpoint(path)` (either
format), `trainer.save_blueprint(path)`, `trainer.blueprint()` (the average strategy as a C++
lookup), `trainer.strategy_change(prev)`.  The Python reference trainer is unchanged and keeps
writing JSON; `fast/binfmt.py` reads both binary formats without the core.

**Binary checkpoint** (`NPCKPT01`, little-endian; `csrc/persist.h`):

```text
"NPCKPT01"  u32 version (1)  u32 flags (0)
identity    u32 key scheme, u8 has_game, then i32 n_players, stack_bb, sb, bb, ante, max_street,
            max_raises_per_street, n_buckets; u8 forbid_open_limp, allow_all_in; u16 n + n f64
            preflop fracs; u16 n + n f64 postflop fracs; u16 n + n str16 grid action names;
            str16 bucketer kind, i32 n_buckets, samples, bins, u64 fingerprint (hash of the bit
            patterns of every cut point and centroid)
u8 trainer kind (0 MCCFR, 1 RNR)  i64 iteration  u8 linear
u32 n_rng, n_rng x (624 u32 MT words, i32 index)          every thread's generator
u16 n_names, n_names x str16                              the names the action ids index
u32 n_tables, per table: str16 name ("nodes", "opp_nodes"), u64 n_nodes,
    n_nodes x (u64 k1, u64 k2, str16 key, u8 n, n u8 ids, n f64 regret, n f64 strategy_sum, i64 visits)
u64 checksum of every byte before it
(str16 = u16 length + UTF-8 bytes)
```

Records come in increasing numeric-key order, so a table gives the same bytes whatever its
insertion history (a resumed run and an uninterrupted one write identical files).  Loading
checks the identity first: another player count, stack, blinds, ante, street, grid, raise cap,
bucket count, bucketer kind / parameters or fit (the fingerprint) is refused with the list of
differences, before the table is touched; for preflop-only games the bucketer is not compared
(their keys use the 169 classes).  Then the table is cleared and reserved for the node count,
every node is inserted under its stored numeric key (recomputed from the string when the file's
key scheme differs from the build's, and compared with it in test mode), its action ids mapped
through the names table, and the checksum checked at the end; on any error (short file, bad
checksum, a key twice, a numeric collision) the table is left empty and the error raised, and
the iteration, linear flag and RNG states are applied only after a complete load.  A JSON
checkpoint records no identity and is never refused, as before.

**Binary blueprint** (`NPBLUE01`):

```text
"NPBLUE01"  u32 version (1)  u32 flags (0)  identity (as above; has_game 0 for a converted JSON)
i64 iteration (-1 unknown)  u8 rounded  u8 probability format (1: u32 k, p = k / 100000.0; 0: f64)
u32 player count of the numeric keys  u16 n_names, n_names x str16
u64 n (infosets)  u64 m (actions)
n x (u64 k1, u64 k2)          numeric keys, strictly increasing
(n + 1) x u32 offsets         infoset i has the actions [off[i], off[i + 1])
m x u8 action index           into the names
m x u32 k (or m x f64)        probabilities
u64 checksum of everything above
n x str16 key strings         same order (exports; the lookup skips them)
u64 checksum of the whole file
```

`save_blueprint` writes `round(p, 5)` of the average strategy by default, exactly the numbers of
`BlueprintStrategy.save`'s JSON: Python's `round(x, 5)` is the decimal with five digits nearest
to the exact binary value (ties to even), read back as the nearest double; `std::to_chars(fixed,
5)` + `std::from_chars` compute the same (with an exact fast path for probabilities; checked
against CPython on 1,143,057 values including every tie of the form odd/64).  Such a value is
the double nearest to k * 10^-5, which `k / 100000.0` computes exactly (the division is
correctly rounded and k / 10^5 has no ties), so the file stores `k` (4 bytes instead of 8)
whenever every value reproduces bit for bit (`pack5`); -0.0, unrounded or odd values keep f64.
Hence a binary blueprint and the JSON blueprint of the same training state give the same
decisions.  `rounded=False` keeps the full doubles.

**Lookup** (`BlueprintTable`, Python `CppBlueprint`): the arrays of the file plus a directory
over the top bits of k1 (2^b + 1 u32, 2^b >= n/2: one or two keys per bucket), so a lookup is a
shift, two loads and a short scan.  `policy(key, legal)` is `BlueprintStrategy.policy`: the
numeric key of the string (`KeyCodec::of_string`), None if absent, the probability of each
legal name (the last one if a name appears twice in the entry, 0.0 if absent), their sum with
CPython's `sum()` (`py_sum`), None if it is <= 0, else each value divided by the sum, returned as
Python floats.  The C++ method is bound directly as the agent's `strategy.policy`.  A key is
identified by its 128-bit numeric key; with the key strings loaded (`keys=True`) the string must
match too, and loading a JSON blueprint refuses two strings with one numeric key.  JSON
blueprints load into the same structure (numbers via `std::from_chars`, the same doubles as
Python's `float()`; a key given twice keeps the last entry, as `json.load`; zip() semantics for
entries with fewer probabilities than names).

**JSON.**  `save_checkpoint("x.json")` and `save_blueprint("x.json")` stream the old layouts from
C++ with the bytes `json.dump` writes: floats spelled like `float.__repr__` (shortest digits
from `std::to_chars`, fixed notation for 1e-4 <= |x| < 1e16, else `d.ddde-XX`; NaN / Infinity as
json.dump spells them; checked against CPython on 722,102 values), strings with ensure_ascii
escapes, keys in the table's slot order (the order `export_nodes()` / `strategy()` hand to
json.dump).  `load_checkpoint` of a JSON file (either trainer's) streams it into the table
without building the dict; required members `iteration`, `linear`, `nodes`, optional
`opp_nodes`, `rng_states`; a malformed file raises and leaves the table empty.

**Also changed.**  (1) An imported node gets the action ids the traversal gives a node of its
street.  The narrow and 2bb grids name some raises the same preflop and postflop ("r0.5", "r1")
with different ids; the old import took the first id, so every postflop node of a resumed run
failed the id comparison in the traversal and took the (equivalent) slow path that rebuilds the
action list from the names.  The results were and are identical; resumed runs are faster.
(2) The checkpoint's linear flag now reaches the C++ core, as it does in the Python trainer
(before, only the Python attribute changed).  (3) Files are written to `<path>.tmp` and renamed
over `<path>` when complete: a crash never leaves a truncated checkpoint.  (4) Node rows given
to `import_nodes` / `add_nodes` must have as many regrets and strategy sums as actions
(`ValueError`; before they were read past the end).

**The L1 diagnostic** of `train_blueprint.py` (mean L1 change of the average strategy between
checkpoints, over the keys both have with the same actions) is computed in C++ against a
full-precision `BlueprintTable` snapshot of the previous checkpoint: the same additions in the
same order (the table's slot order, which is the order of `trainer.strategy()`), so the same
double as the Python code on the two dicts.  `--no-l1` skips it and frees the snapshot.

**Bit-identity evidence** (`tests/test_persist.py`, 16 tests, all with the key test mode on; the
whole suite: 213 passed, the 9 failures of `tests/test_mozg_seat.py` are those of the committed
tree, whose `friends/mozg` lacks the engine changes that test expects):

* Resume: 300 iterations, checkpoint as binary and as JSON, resume each in a fresh trainer with
  another seed, 200 more iterations: `export_tables()` equal to a 500-iteration uninterrupted
  run and to each other, RNG states equal after loading, and the binary checkpoint and blueprint
  files written by the three trainers have the same SHA-256 (2 and 3 players, narrow grid).
  4 threads: every thread's stream restored from both formats, tables equal.  The old JSON load
  path (json.load + import_nodes) resumes to the same state.  RNR (two tables) the same.
* JSON bytes: `save_checkpoint("x.json")` equals `json.dump` of the old export dict byte for byte
  (MCCFR and RNR), `save_blueprint("x.json")` equals `strategy().save()`; a hand-made checkpoint
  with foreign keys, escapes, NaN / Infinity / -0.0 / 5e-324 / 1.5e16, extra members and
  indentation loads into the same table through C++ as through json.load.
* Lookup: `policy` equal (float by float) to `BlueprintStrategy.policy` for every key of a
  3-player table and 9 legal lists (missing names, duplicates, unknown names, empty), for the
  in-memory lookup (full precision), the JSON file, the binary file, the pure-Python reader;
  odd JSON entries (duplicate names, all zero, ints, names longer than probabilities, a key
  twice, -0.0 with the sign of the result checked).
* Same decisions: `BlueprintAgent` in a 300-deal duplicate match (3 players, 900 hero hands)
  with the dict, the C++ lookup of the JSON and of the binary file: the same key, legal list and
  probabilities at each of the >800 decisions, the same actions, the same per-deal results;
  the same for the in-memory dict vs the in-memory lookup.
* L1: the C++ number equals `strategy_change(prev_dict, cur_dict)` exactly.
* End to end: the old `train_blueprint.py` (master, dicts) and the new one (C++), 3 players 15bb
  flop narrow grid, 30k iterations, checkpoint every 10k, one thread, seed 0: the printed L1
  values are the same to the last digit (`0.2202393494902491`, `0.14537732315905066`), the
  evaluation lines are the same, the old JSON checkpoint and the new binary one hold the same
  nodes, iteration, linear flag and RNG states, and the old JSON blueprints (final and `.it<N>`)
  and the new binary ones give the same policy on all 328,525 (key, legal) probes.
  `eval_archetypes.py` and `compare_checkpoints.py` print the same lines with the binary file,
  the JSON file and the dict; `play_slumbot.py --mock tag` logs the same 60 hands (keys,
  probabilities, actions, results; only the per-decision wall time differs) with the binary and
  the JSON blueprint.
* Real tables: the 3-max night run's checkpoint (`checkpoint_hunl3m100_pot16_s0.json`, 100M
  iterations, 5,581,446 infosets) resumed from the JSON and from its binary conversion, one
  thread, 3,000 more iterations: the binary checkpoints and blueprints written afterwards have the
  same SHA-256.  Every entry of `blueprint_hunl3m100_pot16_s0.json` (5,581,446) and of
  `blueprint_hunl200w3_pot16_s0.json` (3,042,766), written by the training, equals the binary
  blueprint written from the corresponding checkpoint (names, probabilities, and `policy` on the
  entry's own action list).  A 20,000-deal duplicate match of the HU 200bb blueprint against
  gridrandom: the same key, probabilities and action at all 83,292 decisions with the dict, the
  C++ lookup of the binary file and of the JSON file, the same bb/100.

**Measured** (2026-09-25 04:12-04:40, 16 logical cores; the machine idle except the live Slumbot
match, one Python process using 1.6% of one core; every process opted out of power throttling;
the files in the OS cache; one process per measurement; memory = private bytes of the process,
sampled every 10 ms by a thread; "table" = the trainer's node table loaded, bucket caches empty:
a training process holds its caches on top, 1.3 GB in the night run's configuration).  Games:
**3-max** = the night run (3 players, 100bb, preflop 0.5/1, postflop 0.5/1, 2 raises, potential
16; 100M iterations, 5,581,446 infosets); **HU** = the Slumbot blueprint (2 players, 200bb,
preflop 0.5/1/3, postflop 0.5/1/2/4, 3 raises, potential 16; 3,042,766 infosets).

*One checkpoint of `train_blueprint.py`*: old = `strategy()` dict, the L1 against the previous
dict, the JSON checkpoint (export dict + json.dump), two JSON blueprints (the file and its
`.it<N>` copy), the previous dict held in between; new = the L1 in C++ against the snapshot,
the binary checkpoint, the binary blueprint + a file copy, a new snapshot:

| | 3-max old | 3-max new | HU old | HU new |
|---|---:|---:|---:|---:|
| time of the checkpoint | 134.7 s | 7.2 s | 76.0 s | 4.0 s |
| of which strategy dict / L1 | 12.0 / 5.3 s | - / 0.8 s | 6.7 / 2.4 s | - / 0.4 s |
| of which checkpoint file | 55.0 s | 1.8 s | 31.9 s | 1.0 s |
| of which blueprint files | 62.0 s | 3.7 s | 34.8 s | 2.0 s |
| of which next snapshot | - | 1.0 s | - | 0.5 s |
| memory between checkpoints: table + previous strategy | 1,632 + 3,725 MB | 1,632 + 240 MB | 864 + 2,134 MB | 864 + 135 MB |
| peak memory during the checkpoint | 11,840 MB | 1,958 MB | 6,818 MB | 1,046 MB |

The main session measured the old path in the live night run: 12.72 GB private at its
checkpoints (sampled every 10 s, bucket caches included), consistent with 11.84 GB + 1.3 GB of
caches here.  The new path adds at most 16 bytes per infoset (the sorted slot order) to the table,
plus the snapshot for the L1, 45-47 bytes per infoset (`--no-l1` drops it).  Writing the same state
as JSON from C++ (`--json`) takes 3.8 s (3-max) and 2.2 s (HU) instead of 55.0 and 31.9 s.

*Loading a checkpoint*:

| | 3-max time | 3-max peak | HU time | HU peak |
|---|---:|---:|---:|---:|
| old: json.load + import_nodes | 14.6 s | 5,242 MB | 8.6 s | 2,965 MB |
| new: the same JSON, streamed by C++ | 6.3 s | 1,632 MB | 3.8 s | 864 MB |
| new: binary | 1.6 s | 1,632 MB | 0.9 s | 864 MB |

(peak = the loaded table itself for the new paths)

*Bytes per infoset*:

| | 3-max | HU |
|---|---:|---:|
| node table in memory (slots at 33-36% load + 160-byte nodes + key strings) | 302.2 (96.2 + 160.0 + 46.0) | 290.9 (88.2 + 160.0 + 42.6) |
| checkpoint file, JSON / binary | 148.5 / 113.4 | 162.3 / 113.5 |
| blueprint file, JSON / binary (of which key strings: 47.0 / 43.6) | 85.3 / 79.2 | 85.7 / 76.8 |
| agent's lookup: `BlueprintStrategy` dict (private bytes after loading) | 559.3 | 590.2 |
| agent's lookup: C++ from the binary file | 35.3 | 36.2 |
| agent's lookup: C++ from the JSON file | 35.2 | 36.1 |

The C++ lookup of HU in detail: numeric keys 16.0, offsets 4.0, action indices 2.64, packed
probabilities 10.57 (2.64 actions x 4 bytes), directory 2.76 bytes per infoset.  Loading the
blueprint: dict 10.9 s (3-max) / 6.1 s (HU); C++ from binary 0.15 / 0.08 s; C++ from JSON 4.6 /
2.5 s (transient peak about 1.2 GB / 0.6 GB while parsing).

*Duplicate-match speed* (HU blueprint vs gridrandom, 20,000 deals = 40,000 hands, same seeds,
bucket cache warm, best of 3 alternating runs): dict 5.32 s, C++ lookup 5.20 s; the match is
dominated by the engine, the agents and the key; `policy()` alone on the 83,292 recorded (key,
legal) pairs costs 1,269 ns with the dict and 527 ns with the C++ lookup.

**What is left.**  The node table is now the memory of a run: 290-302 bytes per infoset, of which
160 are the fixed 8-action node (regrets and strategy sums for 8 actions whatever the node has;
2.4-2.6 actions on average here) and 88-96 the slots at 33-36% load.  Nodes sized by their
action count would take the table to roughly 190-200 bytes per infoset (an estimate from the
averages above, not measured), with the same arithmetic.  The RNR trainer still receives its
opponent model and warm start as Python dicts (`fast/rnr.py::tabulate_model`), and the search /
exploit layers (`cfr/search.py`, `cfr/exploit.py`, `exploit/model.py`) read `BlueprintStrategy.table`;
they are unchanged.  A JSON checkpoint converts to binary by resuming from it (the next
checkpoint is binary); there is no standalone converter for that direction, because the binary
file records the game, which only a trainer knows.

## Benchmarks (2026-09-23, Windows 11, 16 logical cores, Python 3.14.6, MSVC 14.44)

### Throughput: `GameSpec(n_players=3, stack_bb=15, max_street=FLOP)`, 30 000 iterations

`python scripts/bench_backends.py --workers 8 --threads 1 8 16`

| backend | wall time | nodes/s | speedup vs pure Python |
|---|---:|---:|---:|
| pure Python (reference, `NEGPLURIBUS_FAST_EVAL=0`) | 154.8 s | 7 840 | 1x |
| Python trainer + compiled evaluator/equity/canonical hooks | 44.5 s | 27 300 | 3.5x |
| multiprocess, 8 workers (Python traversal + hooks) | 8.8 s | 138 000 | 17.6x |
| cpp, 1 thread (bit-identical to Python) | 3.34 s | 363 000 | 46x |
| cpp, 8 threads | 0.54 s | 2 240 000 | 287x |
| cpp, 16 threads | 0.37 s | 3 280 000 | 418x |

Notes: all runs end with ~23.6k infosets and ~1.21M nodes touched.  On one thread about 3 s of
the 3.34 s are E[HS] cache fills (73k canonical forms x 150 samples); the traversal itself runs
at ~2M nodes/s (push/fold game, no buckets: 2.1M nodes/s on one thread).  Multiprocess numbers
include ~1 s of process start-up and the per-round table broadcast.

### Push/fold 10bb: exact exploitability (bb/100) vs iterations

`python scripts/bench_backends.py --only pushfold`

| iterations | python | multiprocess x8, sync 1000 | cpp x1 (== python) | cpp x16 |
|---:|---:|---:|---:|---:|
| 1 000 | 34.29 | 41.05 | 34.29 | 32.64 |
| 5 000 | 24.96 | 33.32 | 24.96 | 23.62 |
| 10 000 | 17.90 | 26.05 | 17.90 | 18.47 |
| 30 000 | 9.57 | 14.87 | 9.57 | 10.61 |
| 100 000 | 3.82 | 5.17 | 3.82 | 4.14 |
| train time | 12.1 s | 3.3 s | 0.1 s | 0.05 s |

Seed-to-seed noise of the reference at 100k iterations: python seed 1 gives 3.46.  The
multiprocess curve with `sync_every=1000` and 8 workers lags: each round is 8 000 iterations on
one frozen strategy, i.e. the strategy is updated 12 times in 100k iterations.  Smaller rounds
fix it at the price of more merges:

| workers, sync_every | 1 000 | 5 000 | 10 000 | 30 000 | 100 000 | time (100k) |
|---|---:|---:|---:|---:|---:|---:|
| 8, 1000 | 41.05 | 33.32 | 26.05 | 14.87 | 5.17 | 3.3 s |
| 8, 250 | 41.05 | 29.86 | 22.48 | 12.75 | 4.21 | 3.7 s |
| 8, 100 | 40.68 | 29.70 | 22.38 | 11.56 | 4.25 | 4.0 s |
| 8, 50 | 40.17 | 28.23 | 23.31 | 12.51 | 4.53 | 4.3 s |
| 4, 250 | 41.14 | 32.31 | 24.46 | 14.36 | 4.92 | 4.7 s |
| 16, 100 | 44.67 | 29.40 | 21.92 | 11.32 | 4.71 | 3.6 s |

(times include the exact-exploitability evaluations, ~0.4 s each.)  Rule of thumb: pick
`sync_every` so that a round is a few percent of the total budget; the default 1000 suits
long runs on the flop games, 100-250 suits toy games.  The C++ threads share the table node by
node and need no such knob.

With more iterations the C++ trainer gets where the Python one cannot go in reasonable time:
1M iterations 0.74 (x1, == python) / 0.49 (x16), 3M iterations 0.18 bb/100 (x16, 1.1 s).

### RNR (exploit layer): `RNRTrainer`, warm start from the blueprint, p = 0.7

| game | python (with hooks) | cpp x1 (== python) | cpp x16 |
|---|---:|---:|---:|
| preflop-r1 10bb, 30k iterations | 9.03 s (26k nodes/s) | 0.12 s (1.96M nodes/s) | 0.02 s (12.8M nodes/s) |
| 2p 20bb flop, 3k / 30k iterations | 2.52 s for 3k (22k nodes/s) | - | 0.24 s for 30k (2.26M nodes/s) |

`scripts/train_blueprint.py` and `scripts/exploitability.py pushfold` were run with
`NEGPLURIBUS_BACKEND=cpp` unchanged: the push/fold curve reaches 3.90 bb/100 at 100k
iterations (python: 3.82), the whole curve including ten exact best responses takes 4 s.

### 169-hand push/call tables, cpp vs python at 100k iterations

Average absolute difference over the 169 classes (test 4 of the acceptance list):

| pair | push | call |
|---|---:|---:|
| python seed 0 vs python seed 1 (reference's own noise) | 0.172 | 0.147 |
| python seed 0 vs cpp x16 seed 0 | 0.160 | 0.144 |
| cpp x16 seed 0, two runs | 0.073 | 0.067 |
| cpp x1 1M vs cpp x16 1M | 0.102 | 0.054 |
| cpp x16 1M vs cpp x16 3M | 0.069 | 0.030 |

The backend difference is the algorithm's own seed-to-seed noise at that budget; the requested
0.1 tolerance is below that noise at 100k iterations, so the test uses 0.2 at 100k and adds a
1M-iteration check (x1 vs x16: 0.15 / 0.10 tolerance, measured 0.102 / 0.054).

## Acceptance tests (`tests/test_backends.py`)

1. evaluator identical on 100k hands (C++ tier and phevaluator tier); equity bit-identical with
   a seeded `random.Random`, within Monte-Carlo noise unseeded; E[HS] and buckets identical;
   `PyRandom::sample` identical to `random.Random.sample` (both branches, stack and heap
   storage, generator state) and equity on complete boards with 1-5 opponents (since 2026-09-24).
2. engine: 2001 random hands x {2, 3, 6} players with random legal actions incl. short stacks
   and antes: identical observations, events, net, showdown seats, winners.
3. keys: >= 500 random states (2461 in the run) over four specs: identical keys and legal lists;
   the numeric key computed without the string equals the key parsed from the string (over
   2,000 distinct keys, four specs), and every C++ trainer test runs in the key test mode
   (`tests/conftest.py`, since 2026-09-24).
4. push/fold 10bb, cpp backend, 100k iterations: epsilon <= 6 bb/100 (measured 4.0-4.7);
   tables vs python within 0.2 (measured 0.16 / 0.14; python seed noise 0.17 / 0.15); 1M
   iterations x1 vs x16 within 0.15 / 0.10 (measured 0.10 / 0.05), epsilon < 1.5 (0.49).
5. flag: `backend=` and `NEGPLURIBUS_BACKEND` switch `MCCFRTrainer` and `RNRTrainer`; default
   stays Python; bad values raise; single-thread cpp == python bit for bit (MCCFR and RNR, both
   also on the 4-street game with either bucketer); checkpoints round-trip in both directions; all 79 reference tests pass with the default
   backend (and run in 29 s instead of 51 s thanks to the hooks).
6. the benchmark table above.

## Notes for the next steps

* `RNRTrainer(backend="cpp")` is what `exploit/pipeline.py::solve_exploit` gets with
  `NEGPLURIBUS_BACKEND=cpp` (its model is an `OpponentModel`, its warm start a
  `BlueprintStrategy`).  `fit_model`'s simulation loop and `duplicate_match` still run the Python
  table/agents; they benefit from the evaluator/equity hooks only.
* Threads share one node table in one process: a flat open-addressing table under numeric keys
  with lock-free lookups, grown by a short stop-the-world pause (since 2026-09-24; before, a
  string-keyed map sharded 256 ways behind `shared_mutex`).  Memory per infoset measured at
  276-288 bytes including the key string (was 343-369); "Numeric node keys".
* `EquityBucketer.fit` and `PotentialAwareBucketer.fit` are Python loops (with the compiled
  equity / histogram helpers when the core is built), fine for the current sizes.
* Potential-aware buckets (docs/buckets.md): `csrc/abstraction.h::PotentialBucketer` is the
  C++ twin of `abstraction/potential.py`, bit-identical (same PyRandom draws from the
  canonical-key seed, same histogram / CDF / EMD summation order; `tests/test_potential_buckets.py`
  checks 5000 situations per street, the single-thread trainer and the RNR trainer).  The
  trainers hold a `shared_ptr<Bucketer>` (virtual `bucket`), `fast.trainer.core_bucketer()`
  picks the twin by type, and the module exposes `potential_histogram` / `river_equity_exact`
  as hooks for the Python reference (`NEGPLURIBUS_FAST_EVAL=0` switches them off).


## Optimizer session commits, measured on this PC (25.09.2026)

Five commits of the cloud optimizer session (repository Xoosiudiaudusy/Zootop, branch
opt/bucket-table, written against a snapshot of this repo taken after search part 1) were
cherry-picked with their authorship: c4db6c5 precomputed bucket tables (Waugh hand index, one byte
per class: flop 1,286,792, turn 13,960,050, river 123,156,254 classes, about 138 MB per bucketer,
`scripts/build_bucket_table.py`, enabled by `NEGPLURIBUS_BUCKET_TABLES=<dir>`), 49488e6 dynamic
hand-out of iterations to threads, a2b94ea MCCFR over a lazily built betting-history tree,
2ecc230 `scripts/bench_opt.py`, d861f7e RNR over the history tree.  They applied without conflicts.

Checks here: the full suite passes except one pre-existing flaky search test (it fails 2 of 6 runs
on d4bfb53 too, before these commits); the table of `buckets_hunl100v2_ehs16_s0` built in 241 s
for the river on 12 threads and matched `bucket()` on 20,000 random hands per street.

Throughput, HU 100bb narrow grid, 16 E[HS] buckets, 12 threads, 1M warm-up then a 2M window,
runs interleaved while other work shared the machine (two rounds, same numbers within 2%):

| build | iterations/s | nodes/s | ratio |
|---|---:|---:|---:|
| master before the commits | 71,128-72,218 | 11.7-12.0 M | 1 |
| with the commits, no tables | 90,350-90,668 | 14.8-15.0 M | 1.26 |
| with the commits and the bucket table | 411,727-412,942 | 67.6-67.7 M | 5.75 |

Nodes per iteration are equal (about 164), so the work is the same.  A short window favours the
tables (the old caches are cold); over a 40M run the ratio will be smaller, to be measured.

### Three more optimizer commits (25.09.2026, afternoon)

525c078 nodes sized to their action count and 16-byte table slots, 0749d81 node key strings
spelled from the history tree / compact tree / trainer tables grown at load 3/4, 0486413 yielding
spinlock and cache-line aligned thread contexts.  Cherry-picked with their authorship; one conflict
in `csrc/search.h` (search part 2 was not in the optimizer's snapshot): the part-2 calls now pass
the action count to `get_or_create` (`na.n`, `N_CONTINUATIONS`) and use `regret()` /
`strategy_sum()`.

Checks here:
- search, one thread, fixed iterations, seed 5, the flop subgame of the search tests: every
  real-path node for every hole is bit-identical before and after (depth `end`: 1,174 nodes;
  `hu_flop_limit` and `next_street`: 1,227 nodes, 1,386 leaves, 16,632 rollouts);
- full suite 255 of 256; the failing test (`test_our_taken_actions_are_fixed_for_our_actual_hole_only`)
  depends on a 0.6 s time budget and failed on master before these commits too (2 of 2 module runs)
  while a 14-thread search duel loaded the machine: about 46 iterations fit in the budget.  It is
  being moved to a fixed iteration count, like `test_our_average_is_accumulated_every_iteration`;
- memory, HU 100bb wide grid, 16 E[HS] buckets, 1 thread, seed 0, 1M iterations, the same 544,117
  infosets in both builds:

| build | slots | nodes | key strings | tree | reported total | private memory grown by training |
|---|---:|---:|---:|---:|---:|---:|
| before (d244cfb) | 123.3 | 160.2 | 38.7 | not reported | 322.2 B/infoset | 160.9 MB |
| after (0486413) | 30.8 | 82.4 | 0 | 19.3 | 132.5 B/infoset | 55.7 MB |

(slots per infoset also fall because the trainer table now grows at load 3/4: 1,048,576 slots
instead of 2,097,152 for the same infosets.)  Throughput of these three commits was not measured
here: the machine was shared with the search duel.
