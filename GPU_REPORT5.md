# Отчёт о прогоне GPU-тренера на ПК — Раунд 5 (суббота, 26 сентября 2026, ~15:20–16:22 местного времени)

Исполнитель: opencode (CLI-ассистент), модель GLM (Z.ai). Коммит opt/gpu: `fed5e93` («compare_checkpoints: buckets from the C++ bucketer and the bucket table (checked against the Python bucketer on random hands); duels with exact potential-aware buckets ~150x faster»). Папка та же — `C:\Project Manchatten\zootop-gpu` (git worktree, detached HEAD на fed5e93). Код не менялся; настройка окружения — `NEGPLURIBUS_BUCKET_TABLES` на время обучений и дуэлей. Машина/интерпретатор те же, что в раундах 1–4. Во время всех обучений ПК посторонней нагрузкой не нагружался (замер «за равное время»).

## Сборка и тесты (R5-1)

```
HEAD is now at fed5e93 compare_checkpoints: buckets from the C++ bucketer and the bucket table (checked against the Python bucketer on random hands); duels with exact potential-aware buckets ~150x faster
```

Строки из `build5.log`:

```
-- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
built C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
```

`python -m pytest -q tests/test_flatcfr.py tests/test_batched.py tests/test_exact_features.py tests/test_exact_bucketer.py *> tests_gpu5.log` — хвост дословно:

```
......................                                                   [100%]
22 passed in 4.91s
```

## Корзины как в проде (R5-2): подбор, точные признаки, таблица

Времена (по `Measure-Command`):

| шаг | время |
|---|---|
| подбор `make_bucketer('potential', 64, None, exact=True).fit(n_situations=4800, seed=0)` (`fit.log`) | 63.0 с |
| точные признаки `build_exact_features(3, ...)` + `build_exact_features(4, ...)` (`features.log`) | 62.7 с |
| таблица `build_bucket_table.py --features-dir data\eq` (`table5.log`) | 71.0 с |

`fit.log` (хвост):

```
         b62: mean eq 0.92  share 0.02  [0000000028]
         b63: mean eq 0.96  share 0.04  [0000000019]
  river: exact equity, cuts at 0.02 0.02 0.04 0.05 0.06 0.08 0.10 0.10 0.11 0.14 0.15 0.16 0.17 0.20 0.22 0.23 0.24 0.26 0.27 0.30 0.32 0.33 0.34 0.35 0.37 0.39 0.41 0.42 0.44 0.45 0.47 0.49 0.51 0.53 0.54 0.56 0.58 0.59 0.61 0.62 0.64 0.65 0.67 0.69 0.71 0.73 0.74 0.75 0.77 0.79 0.81 0.83 0.84 0.85 0.87 0.89 0.90 0.91 0.93 0.95 0.96 0.97 0.99
```

`features.log` (целиком):

```
{'classes': 1286792}
{'classes': 13960050}
```

`table5.log` (хвост) — расхождений нет, все три улицы сверены:

```
-> data\eq\bucket_tables\buckets_potential_64_c9f2f5b5fb1a0b80.npbt, 16 threads
street 1: 1,286,792 classes in 0s (from data\eq\exact_features_flop_b10.bin)
  check: 0 mismatches in 500 random hands
street 2: 13,960,050 classes in 1s (from data\eq\exact_features_turn_b10.bin)
  check: 0 mismatches in 20000 random hands
street 3: 123,156,254 classes in 5s
  check: 0 mismatches in 20000 random hands
written data\eq\bucket_tables\buckets_potential_64_c9f2f5b5fb1a0b80.npbt
```

## Пять обучений по 600 секунд (R5-3)

Все пять прошли штатно: `buckets: loaded ...`, строка скорости, `saved ...blueprint_<tag>.bin`. Общее время каждого прогона по стеню — 602.1–602.3 с. Строки из логов дословно:

| tag | строка из лога | it/s | итераций |
|---|---|---|---|
| eqcpu | `CPU: 274,800,000 iterations in 600s = 457,841 it/s` | 457,841 | 274.8 млн |
| eqgpu4k | `GPU: 819,724,288 iterations in 600s = 1,366,012 it/s` | 1,366,012 | 819.7 млн |
| eqgpu8k | `GPU: 1,118,306,304 iterations in 600s = 1,863,571 it/s` | 1,863,571 | 1118.3 млн |
| eqgpu16k | `GPU: 1,348,730,880 iterations in 600s = 2,247,592 it/s` | 2,247,592 | 1348.7 млн |
| eqgpu32k | `GPU: 1,371,013,120 iterations in 600s = 2,285,002 it/s` | 2,285,002 | 1371.0 млн |

Каждый лог также содержит `done in 600s/601s, 494,828 infosets` и `saved data\eq\blueprint_<tag>.bin`. За равные 600 с GPU выполнил в 3.0 (пачка 4096) … 5.0 (пачка 32768) раза больше итераций, чем CPU.

## Четыре дуэли (R5-4)

Дуэли запускались по две одновременно (как разрешено инструкцией) после завершения всех обучений: пара gpu4k+gpu8k — 208 с, пара gpu16k+gpu32k — 198.3 с. `NEGPLURIBUS_BUCKET_TABLES` была задана. Итоговые строки всех четырёх логов дословно:

```
== duel_gpu4k.log
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 2 players, 100bb stacks, betting through river, grid preflop (1.0,) / postflop (0.5, 1.0) + all-in, 64 postflop buckets (potential)
gpu4k: 494,828 infosets (data\eq\blueprint_eqgpu4k.bin)
cpu: 494,828 infosets (data\eq\blueprint_eqcpu.bin)
  gpu4k vs cpu:   -0.68 bb/100  (95% CI +/-2.58, 400000 hands, off-map 0.0%, 101s)
  cpu vs gpu4k:   +1.86 bb/100  (95% CI +/-2.56, 400000 hands, off-map 0.0%, 103s)

== duel_gpu8k.log
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 2 players, 100bb stacks, betting through river, grid preflop (1.0,) / postflop (0.5, 1.0) + all-in, 64 postflop buckets (potential)
gpu8k: 494,828 infosets (data\eq\blueprint_eqgpu8k.bin)
cpu: 494,828 infosets (data\eq\blueprint_eqcpu.bin)
  gpu8k vs cpu:   -0.41 bb/100  (95% CI +/-2.57, 400000 hands, off-map 0.0%, 101s)
  cpu vs gpu8k:   +1.55 bb/100  (95% CI +/-2.56, 400000 hands, off-map 0.0%, 103s)

== duel_gpu16k.log
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 2 players, 100bb stacks, betting through river, grid preflop (1.0,) / postflop (0.5, 1.0) + all-in, 64 postflop buckets (potential)
gpu16k: 494,828 infosets (data\eq\blueprint_eqgpu16k.bin)
cpu: 494,828 infosets (data\eq\blueprint_eqcpu.bin)
  gpu16k vs cpu:   -1.46 bb/100  (95% CI +/-2.57, 400000 hands, off-map 0.0%, 98s)
  cpu vs gpu16k:   -0.84 bb/100  (95% CI +/-2.55, 400000 hands, off-map 0.0%, 96s)

== duel_gpu32k.log
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 2 players, 100bb stacks, betting through river, grid preflop (1.0,) / postflop (0.5, 1.0) + all-in, 64 postflop buckets (potential)
gpu32k: 494,828 infosets (data\eq\blueprint_eqgpu32k.bin)
cpu: 494,828 infosets (data\eq\blueprint_eqcpu.bin)
  gpu32k vs cpu:   -0.87 bb/100  (95% CI +/-2.58, 400000 hands, off-map 0.0%, 98s)
  cpu vs gpu32k:   +0.21 bb/100  (95% CI +/-2.56, 400000 hands, off-map 0.0%, 97s)
```

Сводка по дуэлям (первое число — герой слева):

| дуэль | результат, bb/100 | 95% CI | значимо? |
|---|---|---|---|
| gpu4k vs cpu | -0.68 | ±2.58 | нет (в пределах CI) |
| gpu8k vs cpu | -0.41 | ±2.57 | нет |
| gpu16k vs cpu | -1.46 | ±2.57 | нет |
| gpu32k vs cpu | -0.87 | ±2.58 | нет |

Все восемь строк (в обе стороны) лежат в пределах доверительных интервалов: статистически значимого отличия стратегий, обученных на GPU за то же время, от CPU-стратегии не выявлено.

## Замечания исполнителя

1. Настройки окружения: только `NEGPLURIBUS_BUCKET_TABLES` (на обучениях и дуэлях) и создание папки `data\eq` (по инструкции). Код не менялся.
2. Подготовка корзин заняла суммарно ~3.3 мин (63 + 62.7 + 71 с) — быстрее оценки «5–10 мин» из инструкции; таблица сверена, `0 mismatches` на всех улицах (улица 1 — 500 рук, улицы 2–3 — по 20 000).
3. Скорость на прод-корзинах (potential-aware 64, exact): CPU 457,841 it/s; GPU 1.37/1.86/2.25/2.29 млн it/s при пачках 4096/8192/16384/32768 — соотношение 3.0–5.0х по числу итераций за равные 600 с. Прирост от пачки 16384 к 32768 почти нулевой (2.248 → 2.285 млн it/s).
4. Во время всех пяти обучений (последовательно, ~50 мин) никаких других задач на ПК не запускалось.
5. Дуэли шли по две одновременно: пара 1 (4k, 8k) — 208 с, пара 2 (16k, 32k) — 198.3 с; каждая дуэль внутри — ~196–206 с на два направления. Все логи содержат `buckets: C++ + table ...` и `off-map 0.0%`.
6. Большие файлы (`data\eq\*.bin`, `*.npbt`, признаки, `blueprint_*.bin`) в git не кладутся. Пустые stderr-файлы дуэлей (`duel_gpu*.err`) удалены.
