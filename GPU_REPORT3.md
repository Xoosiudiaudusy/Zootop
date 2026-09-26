# Отчёт о прогоне GPU-тренера на ПК — Раунд 3 (суббота, 26 сентября 2026, ~14:02–14:12 местного времени)

Исполнитель: opencode (CLI-ассистент), модель GLM (Z.ai). Коммит opt/gpu: `3ce5f0d` («GPU: record slots from a scan instead of single-address atomics, radix sort on the key bits in use (fitted key layout), per-phase device timing; round-3 run instructions»). Папка та же — `C:\Project Manchatten\zootop-gpu` (git worktree, detached HEAD на 3ce5f0d). Код не менялся, настройки окружения — только `NEGPLURIBUS_BUCKET_TABLES` на время замеров.

Машина и интерпретатор те же, что в раундах 1–2 (RTX 5070 sm_120, драйвер 610.62, nvcc 13.4, Python 3.14.6, pybind11 3.1.0, pytest 9.1.1). Таблица корзин — от раунда 2, не пересобиралась (`data\bucket_tables\buckets_ehs_8_aff0bbad1064e725.npbt`, 138 МБ, проверка `0 mismatches in 20000 random hands` была в раунде 2).

## Сборка (R3-1)

Обновление и сборка без замечаний, переменная `CL` не нужна:

```
HEAD is now at 3ce5f0d GPU: record slots from a scan instead of single-address atomics, radix sort on the key bits in use (fitted key layout), per-phase device timing; round-3 run instructions
```

Строки из `build3.log`:

```
-- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
built C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
```

## Тесты бит в бит (R3-2)

`python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu3.log` — хвост лога дословно:

```
..............                                                           [100%]
14 passed in 2.44s
```

Все варианты `[cpu]`/`[emu]`/`[gpu]` и `test_batched.py` прошли, падений нет.

## Замер полный (R3-3): bench4.log

`$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path data\bucket_tables).Path`; `python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 *> bench4.log`. Полный `bench4.log` дословно:

```
bucket tables: C:\Project Manchatten\zootop-gpu\data\bucket_tables
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
identity (GPU vs the batched CPU reference):
  push/fold 10bb: 3000 it, batch 256: infosets 1987, differing 0, extra 0 -> IDENTICAL  [0.0s]
  HU 100bb: 100000 it, batch 4096: infosets 63455, differing 0, extra 0 -> IDENTICAL  [5.0s]
speed on HU 100bb, 5000000 iterations:
  CPU trainer, 16 threads: 9.9s = 506,497 it/s
  GPU batch 4096: 5001216 it in 6.8s = 736,985 it/s (x1.46 vs CPU); host deals+buckets 2.4s (overlapped; waited for it 0.0s), run_batch 6.7s; last batch: 1,188,905 items, 2,938,340 records, device 2.3 fwd + 1.4 back + 0.8 sort + 0.3 add ms = 845,521 it/s device-only
  GPU batch 16384: 5013504 it in 5.1s = 980,710 it/s (x1.94 vs CPU); host deals+buckets 1.5s (overlapped; waited for it 0.0s), run_batch 5.1s; last batch: 4,589,174 items, 11,223,231 records, device 5.5 fwd + 5.3 back + 3.6 sort + 1.7 add ms = 1,020,111 it/s device-only
  GPU batch 32768: 5013504 it in 4.6s = 1,082,048 it/s (x2.14 vs CPU); host deals+buckets 1.4s (overlapped; waited for it 0.0s), run_batch 4.6s; last batch: 9,197,457 items, 22,248,463 records, device 9.3 fwd + 9.9 back + 7.4 sort + 3.6 add ms = 1,089,454 it/s device-only
```

Загрузка во время фаз `GPU batch ...` (опробование раз в секунду; метка `[after: ...]` — последняя на момент замера строка лога, т.е. шла следующая по счёту фаза; формат: `GPU util, память GPU, мощность, загрузка CPU`):

```
73 %, 1934 MiB, 48.50 W, CPU 65%  [after: CPU trainer, ...]      <- шла фаза GPU batch 4096
74 %, 1939 MiB, 73.29 W, CPU 56%  [after: CPU trainer, ...]      <- шла фаза GPU batch 4096
76 %, 1923 MiB, 74.12 W, CPU 58%  [after: CPU trainer, ...]      <- шла фаза GPU batch 4096
87 %, 2269 MiB, 78.21 W, CPU 63%  [after: GPU batch 4096: ...]   <- шла фаза GPU batch 16384
82 %, 2458 MiB, 85.66 W, CPU 46%  [after: GPU batch 4096: ...]   <- шла фаза GPU batch 16384
39 %, 2885 MiB, 82.24 W, CPU 52%  [after: GPU batch 16384: ...]  <- шла фаза GPU batch 32768 (момент между пачками)
94 %, 2877 MiB, 87.19 W, CPU 34%  [after: GPU batch 16384: ...]  <- шла фаза GPU batch 32768
92 %, 2871 MiB, 89.75 W, CPU 3%   [after: GPU batch 32768: ...]  <- завершение/финал
```

## Замер с крупными пачками (R3-3): bench5.log

`python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 --skip-check --batches 65536,131072 *> bench5.log`. Полный `bench5.log` дословно:

```
bucket tables: C:\Project Manchatten\zootop-gpu\data\bucket_tables
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
speed on HU 100bb, 5000000 iterations:
  CPU trainer, 16 threads: 9.2s = 543,626 it/s
  GPU batch 65536: 5046272 it in 4.3s = 1,184,533 it/s (x2.18 vs CPU); host deals+buckets 1.3s (overlapped; waited for it 0.0s), run_batch 4.2s; last batch: 17,774,367 items, 42,075,653 records, device 18.0 fwd + 18.9 back + 13.6 sort + 7.3 add ms = 1,134,145 it/s device-only
  GPU batch 131072: 5111808 it in 3.9s = 1,303,451 it/s (x2.40 vs CPU); host deals+buckets 1.2s (overlapped; waited for it 0.0s), run_batch 3.9s; last batch: 31,436,586 items, 72,734,251 records, device 30.6 fwd + 33.1 back + 23.5 sort + 14.1 add ms = 1,294,533 it/s device-only
```

Загрузка (те же обозначения):

```
9 %, 3583 MiB, 29.56 W, CPU 59%   [after: CPU trainer, ...]      <- старт/прогрев фазы GPU batch 65536
89 %, 3829 MiB, 88.39 W, CPU 62%  [after: CPU trainer, ...]      <- шла фаза GPU batch 65536
62 %, 5779 MiB, 85.48 W, CPU 59%  [after: GPU batch 65536: ...]  <- шла фаза GPU batch 131072
92 %, 5777 MiB, 90.66 W, CPU 60%  [after: GPU batch 65536: ...]  <- шла фаза GPU batch 131072
```

## Замечания исполнителя

1. Соответствие бит в бит подтвердилось после переработки ядер: 14/14 тестов (R3-2) и обе проверки `IDENTICAL` в bench4.
2. Сравнение с раундом 2 на тех же пачках (5 млн итераций, таблица та же):
   - по полной скорости: 4096 — 723,786 → 736,985 it/s; 16384 — 938,652 → 980,710; 32768 — 1,029,856 → 1,082,048; 65536 — 1,154,797 → 1,184,533; 131072 — 1,279,589 → 1,303,451 it/s;
   - по device-only: 4096 — 757,589 → 845,521; 16384 — 963,319 → 1,020,111; 32768 — 1,000,837 → 1,089,454; 65536 — 1,053,377 → 1,134,145; 131072 — 1,224,389 → 1,294,533 it/s (+6…+12%);
   - отношение к CPU: x1.21/x1.57/x1.73/x1.93/x2.14 → x1.46/x1.94/x2.14/x2.18/x2.40. Частично это из-за более медленного CPU-замера в этом раунде (506–544 тыс. it/s против 596–599 тыс. в раунде 2; вариативность CPU-тренера на этой машине уже отмечалась в раунде 1).
3. Мощность GPU во время фаз — до ~91 Вт из 250, утилизация 82–94% (замер секундами, фазы по 4–7 с; это среднее за секунду, кратковременные пики внутри пачки не видны).
4. Появилась разбивка времени устройства: на пачке 131072 — 30.6 fwd + 33.1 back + 23.5 sort + 14.1 add мс; сумма ядер ~101 мс на пачку при run_batch 3.9 с.
5. Во всех фазах `waited for it 0.0s` — подготовка пачек по-прежнему полностью перекрыта.
6. Код не менялся; логи замеров загрузки (gpu_load4/gpu_load5) удалены, содержимое — выше.
