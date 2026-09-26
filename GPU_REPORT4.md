# Отчёт о прогоне GPU-тренера на ПК — Раунд 4 (суббота, 26 сентября 2026, ~14:14–14:24 местного времени)

Исполнитель: opencode (CLI-ассистент), модель GLM (Z.ai). Коммит opt/gpu: `98dd6f0` («GPU: strategies computed once per batch (sigma table), 32-bit keys with a stable sort (a cell's records are already in job order), emulation checks that order; round-4 instructions»). Папка та же — `C:\Project Manchatten\zootop-gpu` (git worktree, detached HEAD на 98dd6f0). Код не менялся; настройка окружения — только `NEGPLURIBUS_BUCKET_TABLES` на время замеров.

Машина и интерпретатор те же, что в раундах 1–3. Таблица корзин — от раунда 2, не пересобиралась (`data\bucket_tables\buckets_ehs_8_aff0bbad1064e725.npbt`, 138 МБ).

## Сборка и тесты (R4: build4.log, tests_gpu4.log)

Обновление и сборка без замечаний:

```
HEAD is now at 98dd6f0 GPU: strategies computed once per batch (sigma table), 32-bit keys with a stable sort (a cell's records are already in job order), emulation checks that order; round-4 instructions
```

Строки из `build4.log`:

```
-- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
built C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
```

`python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu4.log` — хвост дословно:

```
..............                                                           [100%]
14 passed in 2.31s
```

## Замер полный (R4): bench6.log

`$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path data\bucket_tables).Path`; `python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 *> bench6.log`. Полный `bench6.log` дословно:

```
bucket tables: C:\Project Manchatten\zootop-gpu\data\bucket_tables
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
identity (GPU vs the batched CPU reference):
  push/fold 10bb: 3000 it, batch 256: infosets 1987, differing 0, extra 0 -> IDENTICAL  [0.0s]
  HU 100bb: 100000 it, batch 4096: infosets 63455, differing 0, extra 0 -> IDENTICAL  [4.6s]
speed on HU 100bb, 5000000 iterations:
  CPU trainer, 16 threads: 8.8s = 566,311 it/s
  GPU batch 4096: 5001216 it in 3.8s = 1,326,437 it/s (x2.34 vs CPU); host deals+buckets 2.4s (overlapped; waited for it 0.0s), run_batch 3.7s; last batch: 1,188,905 items, 2,938,340 records, device 1.4 fwd + 0.4 back + 0.3 sort + 0.3 add ms = 1,651,549 it/s device-only
  GPU batch 16384: 5013504 it in 2.1s = 2,332,650 it/s (x4.12 vs CPU); host deals+buckets 1.3s (overlapped; waited for it 0.0s), run_batch 2.1s; last batch: 4,589,174 items, 11,223,231 records, device 1.8 fwd + 1.4 back + 1.5 sort + 1.5 add ms = 2,634,219 it/s device-only
  GPU batch 32768: 5013504 it in 1.9s = 2,649,791 it/s (x4.68 vs CPU); host deals+buckets 1.2s (overlapped; waited for it 0.0s), run_batch 1.9s; last batch: 9,197,457 items, 22,248,463 records, device 2.7 fwd + 2.6 back + 3.0 sort + 3.2 add ms = 2,842,628 it/s device-only
```

Загрузка во время bench6 (метка `[after: ...]` — последняя на момент замера строка лога; формат: `GPU util, память GPU, мощность, загрузка CPU`):

```
9 %, 1583 MiB, 29.77 W, CPU 77%  [after: CPU trainer, ...]      <- старт/прогрев фазы GPU batch 4096
59 %, 1626 MiB, 71.66 W, CPU 74% [after: GPU batch 4096: ...]   <- шла фаза GPU batch 16384
81 %, 2072 MiB, 90.93 W, CPU 58% [after: GPU batch 16384: ...]  <- шла фаза GPU batch 32768
86 %, 2430 MiB, 91.84 W, CPU 42% [after: GPU batch 32768: ...]  <- завершение/финал
```

## Замер с крупными пачками (R4): bench7.log

`python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 --skip-check --batches 65536,131072 *> bench7.log`. Полный `bench7.log` дословно:

```
bucket tables: C:\Project Manchatten\zootop-gpu\data\bucket_tables
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
speed on HU 100bb, 5000000 iterations:
  CPU trainer, 16 threads: 8.2s = 606,159 it/s
  GPU batch 65536: 5046272 it in 1.8s = 2,739,388 it/s (x4.52 vs CPU); host deals+buckets 1.1s (overlapped; waited for it 0.0s), run_batch 1.8s; last batch: 17,774,367 items, 42,075,653 records, device 4.3 fwd + 4.7 back + 5.7 sort + 6.6 add ms = 3,071,883 it/s device-only
  GPU batch 131072: 5111808 it in 1.7s = 3,012,157 it/s (x4.97 vs CPU); host deals+buckets 1.1s (overlapped; waited for it 0.0s), run_batch 1.6s; last batch: 31,436,586 items, 72,734,251 records, device 6.7 fwd + 8.4 back + 10.0 sort + 11.8 add ms = 3,563,099 it/s device-only
```

Фазы теперь длятся 1.7–1.8 с, поэтому при опросе раз в секунду в сам прогон bench7 успела попасть только одна строка (фаза 65536):

```
80 %, 3256 MiB, 98.26 W, CPU 62%  [во время GPU batch 65536]
```

Для картины загрузки сделан дополнительный прогон только для замера — те же пачки, 20 млн итераций (`--iters 20000000 --skip-check --batches 65536,131072`); его скоростные строки в раздел замеров не входят, приводятся ниже как контекст загрузки. Хвост его вывода:

```
  CPU trainer, 16 threads: 34.8s = 574,235 it/s
  GPU batch 65536: 20054016 it in 8.1s = 2,479,245 it/s (x4.32 vs CPU); host deals+buckets 4.7s (overlapped; waited for it 0.0s), run_batch 8.0s; last batch: 18,756,673 items, 45,717,627 records, device 4.4 fwd + 4.8 back + 6.7 sort + 7.4 add ms = 2,814,590 it/s device-only
  GPU batch 131072: 20054016 it in 7.1s = 2,813,712 it/s (x4.90 vs CPU); host deals+buckets 4.5s (overlapped; waited for it 0.0s), run_batch 7.0s; last batch: 37,382,758 items, 90,164,823 records, device 7.6 fwd + 9.8 back + 12.6 sort + 14.6 add ms = 2,939,730 it/s device-only
```

Загрузка в том прогоне (метка — строка лога, появившаяся последней к моменту замера, т.е. шла следующая фаза):

```
78 %, 3241 MiB,  73.81 W, CPU 60%  [after: CPU trainer, ...]     <- шла фаза GPU batch 65536
91 %, 3775 MiB, 101.35 W, CPU 68%  [after: CPU trainer, ...]     <- шла фаза GPU batch 65536
75 %, 3773 MiB, 103.63 W, CPU 62%  [after: CPU trainer, ...]     <- шла фаза GPU batch 65536
87 %, 3725 MiB, 101.32 W, CPU 63%  [after: GPU batch 65536: ...] <- шла фаза GPU batch 131072
90 %, 4857 MiB,  96.22 W, CPU 69%  [after: GPU batch 65536: ...] <- шла фаза GPU batch 131072
83 %, 5786 MiB, 101.17 W, CPU 73%  [after: GPU batch 65536: ...] <- шла фаза GPU batch 131072
92 %, 5768 MiB, 106.62 W, CPU 62%  [after: GPU batch 65536: ...] <- шла фаза GPU batch 131072
```

## Замечания исполнителя

1. Соответствие бит в бит подтвердилось и после четвёртой переработки: 14/14 тестов (включая `[gpu]`-варианты и новую проверку порядка записей при эмуляции) и обе проверки `IDENTICAL` в bench6.
2. Сравнение с раундом 3 (5 млн итераций, та же таблица):
   - полная скорость: 4096 — 736,985 → 1,326,437 it/s; 16384 — 980,710 → 2,332,650; 32768 — 1,082,048 → 2,649,791; 65536 — 1,184,533 → 2,739,388; 131072 — 1,303,451 → 3,012,157 it/s (в 2.1–2.4 раза);
   - device-only: 845,521 → 1,651,549; 1,020,111 → 2,634,219; 1,089,454 → 2,842,628; 1,134,145 → 3,071,883; 1,294,533 → 3,563,099 it/s (в 2.0–2.8 раза);
   - отношение к CPU: x1.46/x1.94/x2.14/x2.18/x2.40 → x2.34/x4.12/x4.68/x4.52/x4.97 (CPU в этом раунде 566–606 тыс. it/s — на уровне раунда 2).
3. Мощность GPU во время фаз выросла: до ~107 Вт из 250 (в раунде 3 — до ~91), утилизация 75–92%; память GPU в фазе 131072 — до 5.8 ГБ в 20-миллионном прогоне.
4. Разбивка времени устройства на пачке 131072 (5 млн): 6.7 fwd + 8.4 back + 10.0 sort + 11.8 add мс — все четыре фазы теперь одного порядка, прежнего доминирования проходов нет.
5. Во всех фазах `waited for it 0.0s` — подготовка пачек по-прежнему полностью перекрыта.
6. Код не менялся; логи замеров загрузки удалены, содержимое — выше.
