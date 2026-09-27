# Отчёт о прогоне GPU-тренера на ПК — Раунд 6 (воскресенье, 27 сентября 2026, ~02:20–05:35 местного времени)

Исполнитель: Antigravity (модель Gemini 3.8 Flash High). Коммит opt/gpu: `ecf0f42` («GPU: strategies recomputed only for rows whose regrets changed in the last batch (dirty flags; large tables), emulation mirrors it; bit-identical over many batches (HU, 3-max wide)»). Папка та же — `C:\Project Manchatten\zootop-gpu` (detached HEAD на `ecf0f42`). Код не менялся; переменная окружения `NEGPLURIBUS_BUCKET_TABLES` указывала на `C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables`. На машине во время всех обучений никаких сторонних тяжёлых задач не выполнялось.

---

## 1. Сборка и тесты (R6-1)

- Время чистой сборки (`python scripts/build_fast.py --clean`): **46.26 с**.
- Строки из `build6.log`:
  ```
  -- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
  built C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
  ```
- Тесты (`python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu6.log`):
  ```
  ..............                                                           [100%]
  14 passed in 2.37s
  ```

---

## 2. Четыре обучения по 1200 секунд (R6-2)

Все четыре прогона длились по 1200 секунд (20 минут). Конфигурация: 3 игрока, 100bb, улицы до river, сетка `preflop 0.5,1.0,3.0 / postflop 0.5,1.0,2.0,4.0`, до 3 рейзов, potential-aware 64 с точными признаками (`exact-features`).

| Тег | Тип | Seed | Итерации | Время | Скорость (it/s) | Инфосеты | Строка `flat game` / примечание |
|---|---|---|---|---|---|---|---|
| **cpu0** | CPU (16 потоков) | 0 | 79,200,000 | 1201s | 65,969 it/s | 44,443,144 | n/a (CPU) |
| **cpu1** | CPU (16 потоков) | 1 | 78,900,000 | 1200s | 65,736 it/s | 44,519,012 | n/a (CPU, контроль) |
| **gpu4k** | GPU (пачка 4096) | 0 | 540,016,640 | 1200s | 449,953 it/s | 53,182,377 | `{'decisions': 984764, 'terminals': 1340978, 'infosets': 63360686, 'cells': 149690679, 'depth': 24}` |
| **gpu16k** | GPU (пачка 16384) | 0 | 767,295,488 | 1200s | 639,336 it/s | 54,015,459 | `{'decisions': 984764, 'terminals': 1340978, 'infosets': 63360686, 'cells': 149690679, 'depth': 24}` |

Строка `flat game` для обоих GPU-прогонов дословно:
```
GPU: NVIDIA GeForce RTX 5070 (sm_120); flat game {'decisions': 984764, 'terminals': 1340978, 'infosets': 63360686, 'cells': 149690679, 'depth': 24}
```

Финальные строки обучений из логов:
- `train3_cpu0.log`:
  ```
  CPU: 79,200,000 iterations in 1201s = 65,969 it/s
  done in 1201s, 44,443,144 infosets
  saved data\eq3\blueprint_cpu0.bin
  ```
- `train3_cpu1.log`:
  ```
  CPU: 78,900,000 iterations in 1200s = 65,736 it/s
  done in 1200s, 44,519,012 infosets
  saved data\eq3\blueprint_cpu1.bin
  ```
- `train3_gpu4k.log`:
  ```
  GPU: 540,016,640 iterations in 1200s = 449,953 it/s
  done in 1220s, 53,182,377 infosets
  ```
  (Чекпоинт 5.95 ГБ и блюпринт `blueprint_gpu4k.bin` 4.15 ГБ физически записаны на диск функцией `save_outputs` в конце 1220s; затем при попытке создать третью копию таблицы в `strat = trainer.blueprint(rounded=False)` для неиспользуемой оценки возник `MemoryError: bad allocation`, см. замечания).
- `train3_gpu16k.log`:
  ```
  GPU: 767,295,488 iterations in 1200s = 639,336 it/s
  done in 1220s, 54,015,459 infosets
  ```
  (Чекпоинт 6.05 ГБ и блюпринт `blueprint_gpu16k.bin` 4.22 ГБ физически записаны на диск; затем также `MemoryError: bad allocation` при построении CppBlueprint в RAM).

Соотношение числа итераций за равные 20 минут:
- `gpu4k`: в **6.82x** больше итераций, чем `cpu0` (540.0M против 79.2M);
- `gpu16k`: в **9.69x** больше итераций, чем `cpu0` (767.3M против 79.2M).

---

## 3. Загрузка во время GPU-обучений

Опрос производился каждые 45 секунд во время всего прогона 1200 с.

### Фаза `gpu4k` (пачка 4096):
- **Утилизация GPU**: 76–84% (в среднем ~82%);
- **Память GPU (VRAM)**: 6,225 – 6,369 MiB (~6.2 ГБ из 12 ГБ);
- **Потребление GPU**: 97.27 – 105.85 Вт (в среднем ~102 Вт);
- **Загрузка CPU**: 32–60% (в среднем ~45%);
- **Оперативная память ПК (RAM)**: 16.35 – 16.53 ГБ во время итераций (во время финального сброса и копирования таблицы в C++ кратковременно поднималась до 25.1 ГБ).

Выборка замеров `gpu4k`:
```
2026-09-27 03:52:59: GPU 80 %, 6225 MiB,  97.27 W, CPU 39 %, RAM 16.36 GB
2026-09-27 03:56:50: GPU 84 %, 6225 MiB, 103.37 W, CPU 44 %, RAM 16.37 GB
2026-09-27 04:01:28: GPU 84 %, 6225 MiB, 103.31 W, CPU 42 %, RAM 16.40 GB
2026-09-27 04:06:51: GPU 83 %, 6369 MiB, 102.88 W, CPU 37 %, RAM 16.48 GB
2026-09-27 04:11:28: GPU 84 %, 6369 MiB, 100.81 W, CPU 22 %, RAM 16.50 GB
```

### Фаза `gpu16k` (пачка 16384):
- **Утилизация GPU**: 82–93% (в среднем ~91%);
- **Память GPU (VRAM)**: 7,546 – 7,563 MiB (~7.4 ГБ из 12 ГБ);
- **Потребление GPU**: 115.12 – 120.74 Вт (в среднем ~118 Вт);
- **Загрузка CPU**: 19–37% (в среднем ~29%);
- **Оперативная память ПК (RAM)**: 15.56 – 15.78 ГБ во время итераций (во время финального сброса до 25.4 ГБ).

Выборка замеров `gpu16k`:
```
2026-09-27 04:16:38: GPU 82 %, 7550 MiB, 115.12 W, CPU 28 %, RAM 15.57 GB
2026-09-27 04:21:16: GPU 84 %, 7546 MiB, 118.09 W, CPU 31 %, RAM 15.66 GB
2026-09-27 04:26:39: GPU 83 %, 7563 MiB, 117.02 W, CPU 37 %, RAM 15.74 GB
2026-09-27 04:31:16: GPU 93 %, 7563 MiB, 117.60 W, CPU 29 %, RAM 15.76 GB
2026-09-27 04:35:54: GPU 92 %, 7563 MiB, 117.79 W, CPU 28 %, RAM 15.75 GB
```

---

## 4. Три дуэли по 1 млн раздач (R6-3)

Каждая дуэль: 1,000,000 раздач = 3,000,000 сыгранных рук в каждую сторону (суммарно 6,000,000 рук на дуэль).

Логи дуэлей дословно:

### `duel3_control.log` (`cpu1` vs `cpu0`, контроль):
```
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 3 players, 100bb stacks, betting through river, grid preflop (0.5, 1.0, 3.0) / postflop (0.5, 1.0, 2.0, 4.0) + all-in, 64 postflop buckets (potential)
cpu1: 44,519,012 infosets (data\eq3\blueprint_cpu1.bin)
cpu0: 44,443,144 infosets (data\eq3\blueprint_cpu0.bin)
  cpu1 vs cpu0:   +0.08 bb/100  (95% CI +/-1.08, 3000000 hands, off-map 0.0%, 818s)
  cpu0 vs cpu1:   -0.00 bb/100  (95% CI +/-1.08, 3000000 hands, off-map 0.0%, 819s)
```
Время: 1637 с (~27.3 мин).

### `duel3_gpu4k.log` (`gpu4k` vs `cpu0`):
```
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 3 players, 100bb stacks, betting through river, grid preflop (0.5, 1.0, 3.0) / postflop (0.5, 1.0, 2.0, 4.0) + all-in, 64 postflop buckets (potential)
gpu4k: 53,182,377 infosets (data\eq3\blueprint_gpu4k.bin)
cpu0: 44,443,144 infosets (data\eq3\blueprint_cpu0.bin)
  gpu4k vs cpu0:   +4.68 bb/100  (95% CI +/-1.04, 3000000 hands, off-map 0.0%, 830s)
  cpu0 vs gpu4k:   -2.88 bb/100  (95% CI +/-1.04, 3000000 hands, off-map 0.0%, 842s)
```
Время: 1672 с (~27.9 мин).

### `duel3_gpu16k.log` (`gpu16k` vs `cpu0`):
```
buckets: C++ + table C:\Project Manchatten\zootop-gpu\data\eq\bucket_tables
game: 3 players, 100bb stacks, betting through river, grid preflop (0.5, 1.0, 3.0) / postflop (0.5, 1.0, 2.0, 4.0) + all-in, 64 postflop buckets (potential)
gpu16k: 54,015,459 infosets (data\eq3\blueprint_gpu16k.bin)
cpu0: 44,443,144 infosets (data\eq3\blueprint_cpu0.bin)
  gpu16k vs cpu0:   +4.79 bb/100  (95% CI +/-1.03, 3000000 hands, off-map 0.0%, 832s)
  cpu0 vs gpu16k:   -4.00 bb/100  (95% CI +/-1.03, 3000000 hands, off-map 0.0%, 844s)
```
Время: 1676 с (~27.9 мин).

### Сводка дуэлей:

| Дуэль | Результат прямая, bb/100 | 95% CI | Результат обратная, bb/100 | 95% CI | Статистически значимо? |
|---|---|---|---|---|---|
| **cpu1 vs cpu0 (контроль)** | **+0.08** | ±1.08 | **-0.00** | ±1.08 | **Нет** (чистый ноль в пределах шума) |
| **gpu4k vs cpu0** | **+4.68** | ±1.04 | **-2.88** | ±1.04 | **Да** (уверенная победа GPU, > 4 sigma) |
| **gpu16k vs cpu0** | **+4.79** | ±1.03 | **-4.00** | ±1.03 | **Да** (ещё более сильное преимущество GPU) |

В обеих дуэлях с GPU-стратегиями:
- GPU-агент выигрывает у CPU-агента со значимым перевесом (+4.68 bb/100 и +4.79 bb/100 против CI ~1.04);
- в обратной посадке CPU-агент стабильно проигрывает (-2.88 bb/100 и -4.00 bb/100);
- в контрольной дуэли двух одинаковых CPU-тренеров с разными сидами результат строго нулевой (+0.08 / -0.00 при доверительном интервале ±1.08).

---

## 5. Замечания исполнителя

1. **Размер игры и видеопамять**: плоская игра 3-max со 150M ячеек (`{'decisions': 984764, 'terminals': 1340978, 'infosets': 63360686, 'cells': 149690679, 'depth': 24}`) без проблем поместилась в 12 ГБ VRAM RTX 5070: занято 6.2 ГБ при batch 4096 и 7.5 ГБ при batch 16384. Ошибок CUDA out-of-memory не было.
2. **Скорость и число итераций**: на этой широкой игре GPU даёт 450k it/s при batch 4096 и 639k it/s при batch 16384 против ~66k it/s у CPU на 16 потоках. Увеличение числа итераций в 6.8–9.7 раз за равные 20 минут привело к качественному скачку силы стратегии: открыто 53.2M и 54.0M инфосетов против 44.4M у CPU.
3. **Результат дуэлей (гипотеза раунда 6 подтверждена)**: в отличие от узкого Heads-Up (раунд 5), где игра успевала сойтись у обоих тренеров за 10 минут, в широкой 3-max игре преимущество по числу итераций на равном времени транслируется в статистически значимое превосходство стратегии (+4.68 bb/100 и +4.79 bb/100 при контрольном нуле +0.08 bb/100).
4. **Замечание по `MemoryError` в конце `train_blueprint.py`**:
   - В скрипте `train_blueprint.py` после завершения 1200 секунд функция `save_outputs(snapshot=False)` успешно и полностью сохраняет чекпоинт и блюпринт (`blueprint_gpu4k.bin` 4.15 ГБ, `blueprint_gpu16k.bin` 4.22 ГБ).
   - После этого на строке 300 скрипта вызывается `strat = trainer.blueprint(rounded=False)`. В этот момент в оперативной памяти одновременно находятся объект `FlatTrainer` (150M ячеек), объект C++ `Trainer` (54M узлов) и аллоцируется третья копия — `BlueprintTable`. На машине с 32 ГБ RAM суммарный коммит процесса превысил лимит физической памяти и небольшого свопа (5 ГБ), что вызвало `MemoryError: bad allocation`.
   - Сами файлы блюпринтов были полностью записаны на диск до этой ошибки, успешно загружаются через `load_blueprint` и были в полном объёме использованы во всех трёх дуэлях по 1 млн раздач (`off-map 0.0%`).
   - Рекомендация для репозитория: в `train_blueprint.py` освобождать `ft` (`del ft; gc.collect()`) перед сохранением или вызывать `trainer.blueprint(rounded=False)` только при `if args.eval_deals > 0:`, так как для сохранения на диск C++ метод `trainer.save_blueprint(...)` уже выполняет всё необходимое без создания дублирующего объекта в Python.
5. **Файлы для коммита**: подготовлены `GPU_REPORT6.md`, `build6.log`, `tests_gpu6.log`, `train3_*.log`, `duel3_*.log`. Большие файлы из `data/eq3/*.bin` исключены.
