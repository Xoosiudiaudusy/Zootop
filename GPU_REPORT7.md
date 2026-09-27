# GPU run report 7: GPU vs CPU at equal iterations

Commit: `dc4a200` (origin/opt/gpu). Machine: NVIDIA GeForce RTX 5070 (sm_120), CPU backend cpp x16 threads.
Game: 3-max, 100bb, wide grid, 64 potential-aware postflop buckets (same as round 6).

## R7-1. Build and tests
- `build7.log`: `negpluribus: GPU trainer ON (CUDA v13.4 nvcc, archs 89-real;120)`, `built ...\_fastcore.cp314-win_amd64.pyd`. No errors.
- `tests_gpu7.log`: `14 passed in 2.15s` (test_flatcfr.py, test_batched.py).

## R7-2. Tiny rehearsal
All 8 trainings reached `saved` (in `data\eq3\tiny`), and all 6 duels printed `vs` lines. Total time 13:23:27 to 13:24:59. The numbers carry no meaning, since the trainings were tiny.

## R7-3. Main run: step timings (from `round7_progress.log`)
Start 13:25:13, end 18:58:16 (about 5 h 33 min).

| Step | Start | Duration |
|---|---|---|
| train gpu4k_79M (79.2M it, batch 4096, seed 0) | 13:25:13 | 5 min |
| train gpu16k_79M (79.2M it, batch 16384, seed 0) | 13:30:11 | 3.9 min |
| train gpu16k_79M_s1 (79.2M it, batch 16384, seed 1) | 13:34:06 | 3.9 min |
| train gpu4k_767M (767.3M it, batch 4096, seed 0) | 13:38:00 | 31.4 min |
| train cpu_767M (767.3M it, seed 0) | 14:09:23 | 202.6 min |
| duels gpu4k_79M_vs_cpu0 + gpu16k_79M_vs_cpu0 (in parallel) | 17:32:01 | 28.6 min |
| duels gpu16k_79M_s1_vs_cpu1 + gpu16k_767M_vs_cpu_767M | 18:00:39 | 28.7 min |
| duels gpu4k_767M_vs_cpu_767M + cpu_767M_vs_cpu0 | 18:29:23 | 28.9 min |

## Trainings (from `r7_summary.txt`)

| Run | Iterations | Training time | it/s | Infosets |
|---|---|---|---|---|
| gpu4k_79M | 79,200,000 | 181s (total 198s) | 437,970 | 45,082,858 |
| gpu16k_79M | 79,200,000 | 119s (total 136s) | 663,422 | 45,034,329 |
| gpu16k_79M_s1 | 79,200,000 | 120s (total 136s) | 662,237 | 45,066,242 |
| gpu4k_767M | 767,295,488 | 1737s (total 1757s) | 441,833 | 54,207,553 |
| cpu_767M | 767,295,488 | 11896s | 64,502 | 53,807,591 |

For comparison, from round 6: cpu0 has 44,443,144 infosets, cpu1 has 44,519,012, and gpu16k (767M) has 54,015,459.
At 767M iterations, the GPU with batch 4096 was **6.8x** faster than the CPU (441.8k vs 64.5k it/s).

## Duels: 1M deals each, 3M hands (verbatim)
Each line means one hero plays against two copies of the opponent on identical deals. In 3-max the two lines do not have to be opposite in sign.

```
== r7_duel_gpu4k_79M_vs_cpu0.log
  gpu4k_79M vs cpu0:   +0.86 bb/100  (95% CI +/-1.09, 3000000 hands, off-map 0.0%, 848s)
  cpu0 vs gpu4k_79M:   +0.54 bb/100  (95% CI +/-1.09, 3000000 hands, off-map 0.0%, 863s)
== r7_duel_gpu16k_79M_vs_cpu0.log
  gpu16k_79M vs cpu0:   +0.92 bb/100  (95% CI +/-1.09, 3000000 hands, off-map 0.0%, 847s)
  cpu0 vs gpu16k_79M:   +0.43 bb/100  (95% CI +/-1.09, 3000000 hands, off-map 0.0%, 861s)
== r7_duel_gpu16k_79M_s1_vs_cpu1.log
  gpu16k_79M_s1 vs cpu1:   +0.39 bb/100  (95% CI +/-1.09, 3000000 hands, off-map 0.0%, 829s)
  cpu1 vs gpu16k_79M_s1:   +0.63 bb/100  (95% CI +/-1.10, 3000000 hands, off-map 0.0%, 831s)
== r7_duel_gpu16k_767M_vs_cpu_767M.log
  gpu16k_767M vs cpu_767M:   +0.79 bb/100  (95% CI +/-0.93, 3000000 hands, off-map 0.0%, 860s)
  cpu_767M vs gpu16k_767M:   +1.09 bb/100  (95% CI +/-0.93, 3000000 hands, off-map 0.0%, 858s)
== r7_duel_gpu4k_767M_vs_cpu_767M.log
  gpu4k_767M vs cpu_767M:   +0.93 bb/100  (95% CI +/-0.93, 3000000 hands, off-map 0.0%, 862s)
  cpu_767M vs gpu4k_767M:   +0.95 bb/100  (95% CI +/-0.93, 3000000 hands, off-map 0.0%, 863s)
== r7_duel_cpu_767M_vs_cpu0.log
  cpu_767M vs cpu0:   +4.72 bb/100  (95% CI +/-1.03, 3000000 hands, off-map 0.0%, 827s)
  cpu0 vs cpu_767M:   -3.90 bb/100  (95% CI +/-1.02, 3000000 hands, off-map 0.0%, 840s)
```

## Conclusions
- **The sensitivity check passed:** deep CPU vs shallow CPU gave +4.72 / -3.90 bb/100, well outside the CI. The duel can detect a real difference in training depth.
- **At equal iterations, GPU and CPU are indistinguishable.** In all five GPU-vs-CPU duels, both lines are positive, within about 1 bb/100 of each other, and within or at the edge of the CI (±0.93–1.10). There is no sign that batched updates (4096 or 16384 iterations per snapshot) make training worse per iteration. At this depth there is also no measurable difference between batch 4096 and batch 16384.
- So the round-6 win for GPU came from the number of iterations, not from the "quality" of each one. Per unit of time, GPU is 6.8–10x more efficient.

## Remarks
- There are no `FAILED` entries in `round7_progress.log`, all `.err` files are empty, and `round7_console_err.log` is empty.
- No out-of-memory errors or memory warnings in the logs. Peak memory was not measured.
- Large files (`data\eq3\*.bin`) are not committed.
