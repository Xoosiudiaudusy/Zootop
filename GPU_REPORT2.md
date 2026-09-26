# Отчёт о прогоне GPU-тренера на ПК — Раунд 2 (суббота, 26 сентября 2026, ~13:38–13:58 местного времени)

Исполнитель: opencode (CLI-ассистент), модель GLM (Z.ai). Коммит opt/gpu: `d9c808c` («GPU: MSVC /Zc:preprocessor for CUDA 13 CCCL, host prepares batch i+1 during batch i, bench timing breakdown and bucket table options; round-2 run instructions»). Работа — в той же папке `C:\Project Manchatten\zootop-gpu` (git worktree, detached HEAD на d9c808c). Код не менялся; настройки окружения перечислены ниже и в «Замечаниях».

## Окружение

Та же машина и тот же системный Python, что в раунде 1 (около получаса назад): RTX 5070 (sm_120), драйвер 610.62 / CUDA UMD 13.3, nvcc 13.4 (V13.4.59), cmake 4.4.3, Python 3.14.6, pybind11 3.1.0 и pytest 9.1.1 (установлены в раунде 1, переустановка не потребовалась). `nvidia-smi`/`nvcc --version` заново не снимал — окружение с раунда 1 не менялось.

## Сборка (R2-1)

Перешёл на d9c808c (лог ниже), переменная `CL` не задавалась (в начале команды — `Remove-Item Env:CL`, её и так не было в новом процессе). Сборка прошла **без** каких-либо настроек окружения, C1189 не вернулся:

```
HEAD is now at d9c808c GPU: MSVC /Zc:preprocessor for CUDA 13 CCCL, host prepares batch i+1 during batch i, bench timing breakdown and bucket table options; round-2 run instructions
```

Строки из `build2.log`:

```
-- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
built C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
```

## Тесты бит в бит (R2-2)

`python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu2.log` — хвост лога дословно:

```
..............                                                           [100%]
14 passed in 2.33s
```

Все варианты `[cpu]`/`[emu]`/`[gpu]` (push_fold, three_player_flop, river_potential) и все 5 тестов `test_batched.py` прошли, падений нет.

## Таблица корзин (R2-3)

Первая попытка `python scripts/gpu_bench.py --write-buckets data\buckets_gpubench.json` упала — папки `data` не было и скрипт её сам не создаёт (настройка окружения: создал папку `New-Item -ItemType Directory data`, код не трогал). Дословный вывод первой попытки (хвост):

```
  File "C:\Project Manchatten\zootop-gpu\scripts\..\negpluribus\abstraction\buckets.py", line 93, in save
    with open(path, "w", encoding="utf-8") as f:
         ~~~~^^
FileNotFoundError: [Errno 2] No such file or directory: 'data\\buckets_gpubench.json'
```

После создания папки: сохранение бакетера — 0.2 с (`bucketer saved to data\buckets_gpubench.json`), построение таблицы — **517.2 с** (~8.6 мин), 3 улицы; проверка скрипта — расхождений нет. Хвост `table.log` дословно:

```
  street 3:  93.4%      458s
street 3: 123,156,254 classes in 471s
  check: 0 mismatches in 20000 random hands
written data\bucket_tables\buckets_ehs_8_aff0bbad1064e725.npbt
```

Файл таблицы: `data\bucket_tables\buckets_ehs_8_aff0bbad1064e725.npbt`, 138 403 191 байт (в git не кладу, как предписано).

## Замер с таблицей (R2-4): bench2.log

`$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path data\bucket_tables).Path`, затем `python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 *> bench2.log`. Первая строка — путь таблицы (не `none`), обе проверки IDENTICAL. Полный `bench2.log` дословно:

```
bucket tables: C:\Project Manchatten\zootop-gpu\data\bucket_tables
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
identity (GPU vs the batched CPU reference):
  push/fold 10bb: 3000 it, batch 256: infosets 1987, differing 0, extra 0 -> IDENTICAL  [0.0s]
  HU 100bb: 100000 it, batch 4096: infosets 63455, differing 0, extra 0 -> IDENTICAL  [4.3s]
speed on HU 100bb, 5000000 iterations:
  CPU trainer, 16 threads: 8.4s = 596,279 it/s
  GPU batch 4096: 5001216 it in 6.9s = 723,786 it/s (x1.21 vs CPU); host deals+buckets 2.2s (overlapped; waited for it 0.0s), run_batch 6.9s; last batch: 1,188,905 items, 2,938,340 records, device 3.8 ms traverse + 1.6 ms apply = 757,589 it/s device-only
  GPU batch 16384: 5013504 it in 5.3s = 938,652 it/s (x1.57 vs CPU); host deals+buckets 1.4s (overlapped; waited for it 0.0s), run_batch 5.3s; last batch: 4,589,174 items, 11,223,231 records, device 10.2 ms traverse + 6.8 ms apply = 963,319 it/s device-only
  GPU batch 32768: 5013504 it in 4.9s = 1,029,856 it/s (x1.73 vs CPU); host deals+buckets 1.3s (overlapped; waited for it 0.0s), run_batch 4.8s; last batch: 9,197,457 items, 22,248,463 records, device 18.6 ms traverse + 14.1 ms apply = 1,000,837 it/s device-only
```

Загрузка во время фаз `GPU batch ...` (опрос раз в секунду; метка `[after: ...]` — последняя на момент замера строка лога, т.е. следующая по счёту фаза как раз шла; формат: `GPU util, память GPU, мощность, загрузка CPU`):

```
79 %, 1698 MiB, 59.28 W, CPU 46%  [after: CPU trainer, 16 threads: ...]        <- шла фаза GPU batch 4096
79 %, 1698 MiB, 77.08 W, CPU 55%  [after: CPU trainer, 16 threads: ...]        <- шла фаза GPU batch 4096
80 %, 1698 MiB, 77.24 W, CPU 58%  [after: GPU batch 4096: ...]                 <- шла фаза GPU batch 16384
90 %, 2022 MiB, 82.22 W, CPU 37%  [after: GPU batch 4096: ...]                 <- шла фаза GPU batch 16384
91 %, 2190 MiB, 87.11 W, CPU 39%  [after: GPU batch 4096: ...]                 <- шла фаза GPU batch 16384
86 %, 1548 MiB, 88.46 W, CPU 42%  [after: GPU batch 16384: ...]                <- шла фаза GPU batch 32768
93 %, 2544 MiB, 90.44 W, CPU 37%  [after: GPU batch 16384: ...]                <- шла фаза GPU batch 32768
92 %, 2544 MiB, 91.18 W, CPU 23%  [after: GPU batch 32768: ...]                <- завершение/финал
```

## Доп. прогон с крупными пачками (R2-4): bench3.log

`python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 --skip-check --batches 65536,131072 *> bench3.log`. Полный `bench3.log` дословно:

```
bucket tables: C:\Project Manchatten\zootop-gpu\data\bucket_tables
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
speed on HU 100bb, 5000000 iterations:
  CPU trainer, 16 threads: 8.3s = 598,975 it/s
  GPU batch 65536: 5046272 it in 4.4s = 1,154,797 it/s (x1.93 vs CPU); host deals+buckets 1.2s (overlapped; waited for it 0.0s), run_batch 4.3s; last batch: 17,774,367 items, 42,075,653 records, device 35.2 ms traverse + 27.0 ms apply = 1,053,377 it/s device-only
  GPU batch 131072: 5111808 it in 4.0s = 1,279,589 it/s (x2.14 vs CPU); host deals+buckets 1.1s (overlapped; waited for it 0.0s), run_batch 3.9s; last batch: 31,436,586 items, 72,734,251 records, device 58.7 ms traverse + 48.4 ms apply = 1,224,389 it/s device-only
```

Загрузка (те же обозначения):

```
95 %, 3227 MiB, 84.73 W, CPU 37%  [after: CPU trainer, 16 threads: ...]        <- шла фаза GPU batch 65536
93 %, 3393 MiB, 93.53 W, CPU 31%  [after: GPU batch 65536: ...]                <- шла фаза GPU batch 131072
96 %, 5099 MiB, 87.08 W, CPU 36%  [after: GPU batch 65536: ...]                <- шла фаза GPU batch 131072
96 %, 5099 MiB, 94.75 W, CPU 11%  [after: GPU batch 131072: ...]               <- завершение/финал
```

## Замечания исполнителя

1. Настройки окружения в этом раунде: создание папки `data\` (скрипт `--write-buckets` не создаёт её сам — ошибка записана выше) и переменная `NEGPLURIBUS_BUCKET_TABLES` на время обоих замеров. `$env:CL` не понадобился: флаг `/Zc:preprocessor` теперь в CMake, сборка прошла чисто.
2. Соответствие бит в бит подтвердилось и после изменений кода: все 14 тестов шага R2-2 и обе проверки `IDENTICAL` в bench2 (push/fold 10bb и HU 100bb, 100 тыс. итераций, пачка 4096).
3. Таблицы корзин ускорили CPU-тренер с ~76 тыс. it/s (раунд 1, лучший замер) до ~597–599 тыс. it/s. GPU при этом вырос с x0.92–x1.03 (раунд 1) до **x1.21 / x1.57 / x1.73** (пачки 4096/16384/32768) и **x1.93 / x2.14** (65536/131072); абсолютный максимум — 1 279 589 it/s при пачке 131072.
4. Картина загрузки изменилась радикально: в раунде 1 GPU простаивал (0–15%, ~25–30 Вт), теперь во время GPU-фаз утилизация 79–96%, мощность до ~95 Вт из 250; CPU при этом 11–58%. Во всех фазах `waited for it 0.0s` — подготовка пачек на хосте полностью перекрывается счётом на устройстве.
5. Фазы теперь короткие (4–7 с на 5 млн итераций), поэтому строк замера загрузки немного — по 2–3 на фазу; строки помечены, какая фаза шла в момент замера.
6. При пачке 131072 пиковая память GPU в замере — 5099 MiB; размер последней пачки — 31.4 млн items / 72.7 млн records.
7. Файл таблицы `data\bucket_tables\buckets_ehs_8_aff0bbad1064e725.npbt` (138 МБ) в git не кладётся; временные логи замеров загрузки удалены, их содержимое — в этом отчёте.
