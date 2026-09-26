# Задание: собрать и проверить GPU-тренер на ПК (для помощника-нейросети)

Тебя попросили собрать проект, запустить проверки GPU-тренера, **зафиксировать результат в отчёте** и отдать его.
Разрабатывать, чинить алгоритм или «улучшать» код не нужно. Твоя задача — точно выполнить шаги и честно записать, что получилось.

## Контекст в двух словах
- Проект: покерный бот NegativePluribus (6-max NLHE, как Pluribus). Ядро на C++17 (pybind11-модуль `negpluribus/_fastcore*.pyd`) плюс Python.
- Обучение идёт через MCCFR (Linear CFR). Сделан тренер на GPU (CUDA). Его написали в облаке **без видеокарты**: код собирается, но на настоящем GPU ещё ни разу не запускался.
- Главный критерий: GPU обязан давать **ровно те же числа, бит в бит**, что CPU-эталон (пакетный режим `Trainer` и `FlatTrainer`). Любое расхождение — это баг, его нужно зафиксировать, а не обходить.
- Код GPU-тренера (ветка `opt/gpu`):
  - `csrc/gpucfr.cu` — CUDA;
  - `csrc/gpukernels.h` — работа ядер;
  - `csrc/cfrmath.h` — общая арифметика;
  - `csrc/flatcfr.h`, `csrc/flatgame.h` — плоский формат и CPU-версия;
  - `scripts/gpu_bench.py` — проверка и замер;
  - `tests/test_flatcfr.py` — тесты.
- Железо пользователя: i5-14400F (16 потоков), 32 ГБ, **RTX 5070** (sm_120), Windows + MSVC. CUDA Toolkit, CMake и Visual Studio установлены, проект на этом ПК уже собирали.

## Правила
1. **Не меняй код** в `csrc/`, `negpluribus/`, `scripts/`, `tests/`. Если сборка падает, запиши ошибку в отчёт. Допустимы только настройки окружения: переменные среды, другой генератор CMake, флаги в командной строке. Каждое такое действие опиши в отчёте.
2. Не трогай основную рабочую копию пользователя: работай в **отдельной папке** (git worktree, шаг 1). Не делай commit или push в `main`, `master`, `core`, `opt/gpu` и другие существующие ветки.
3. Отчёт положи в файл, а push делай **только** в новую ветку `opt/gpu-report` (шаг 9). Если push не получается, просто оставь файл: пользователь перешлёт его сам.
4. Выполняй шаги по порядку. Если шаг упал, запиши вывод и переходи к следующему, если он от упавшего не зависит.
5. В отчёт вставляй **дословный вывод** команд (ошибки целиком, длинные логи — последние 60 строк и все строки с `error`). Ничего не пересказывай своими словами вместо вывода.

## Шаги (PowerShell, из папки `zootop` пользователя)

### 0. Окружение
```powershell
nvidia-smi
nvcc --version
cmake --version
python --version
python -c "import pybind11, sys; print(pybind11.__version__, sys.executable)"
git status --short | Select-Object -First 5
```
Все выводы идут в отчёт. Если `nvcc` не найден в PATH, найди его (обычно `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\nvcc.exe`) и запиши путь.
CUDA должна быть **12.8 или новее**: более ранние версии не умеют sm_120 (RTX 50xx).

### 1. Отдельная копия ветки `opt/gpu`
```powershell
git fetch origin opt/gpu
git worktree add ..\zootop-gpu origin/opt/gpu
cd ..\zootop-gpu
git log --oneline -3
```
Если папка `..\zootop-gpu` уже есть, выполни `cd ..\zootop-gpu; git fetch origin opt/gpu; git checkout --detach origin/opt/gpu`.
В `git log` первой строкой должен идти коммит с `GPU run instructions` или новее. Хеш верхнего коммита запиши в отчёт.

### 2. Сборка
```powershell
python scripts/build_fast.py --clean *> build.log
Select-String -Path build.log -Pattern "negpluribus:|error|built" | Select-Object -First 40
```
Нужны две строки: `negpluribus: GPU trainer ON (...)` и `built ...`.
- Если видно `GPU trainer OFF`, CMake не нашёл CUDA. Попробуй по очереди, фиксируя, что помогло:
  1. `$env:CUDACXX = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X\bin\nvcc.exe"` (подставь свою версию), затем снова `python scripts/build_fast.py --clean`.
  2. Из «x64 Native Tools Command Prompt for VS 2022»: `python scripts/build_fast.py --clean --generator Ninja` (если Ninja нет: `pip install ninja`).
- Если ошибка компиляции `gpucfr.cu`, в отчёт идут все строки `error` из `build.log` и 30 строк вокруг первой ошибки. Дальше GPU-шаги не выполнить, но шаг 3 всё равно сделай: он покажет, работает ли CPU-часть.

### 3. Видит ли модуль GPU
```powershell
python -c "import negpluribus._fastcore as f; print(f.__file__); print(f.cuda_available())"
```
Ожидается `(True, '')`. Если `False`, запиши причину, она напечатана вторым элементом.

### 4. Тесты flat/GPU (основная проверка бит в бит)
```powershell
python -m pytest -v tests/test_flatcfr.py tests/test_batched.py *> tests_gpu.log
Get-Content tests_gpu.log -Tail 40
```
Когда GPU доступен, у каждого теста есть три варианта: `[cpu]`, `[emu]`, `[gpu]`. `[gpu]` сравнивает GPU с CPU-эталоном бит в бит.
Отдельно перечисли в отчёте, какие варианты прошли и какие упали. Если упал `[gpu]`, приложи полный текст ошибки или assert.

### 5. Проверка и замер скорости
Таблицы корзин сильно ускоряют CPU-часть. Они не в git, поэтому ищи их в **исходной** папке пользователя (`..\zootop`), а не в worktree: переменная `NEGPLURIBUS_BUCKET_TABLES`, папки вроде `..\zootop\data\bucket_tables`, файлы таблиц корзин. Если нашёл, задай переменную, иначе пропусти (запиши, что таблиц не было):
```powershell
echo $env:NEGPLURIBUS_BUCKET_TABLES
# если пусто и таблицы есть:  $env:NEGPLURIBUS_BUCKET_TABLES = "<путь к папке с таблицами>"
```
Запуск (занимает несколько минут):
```powershell
python scripts/gpu_bench.py *> bench.log
Get-Content bench.log
```
Скрипт печатает:
- устройство;
- проверку `IDENTICAL` / `MISMATCH` для push/fold и HU 100bb;
- скорость обычного CPU-тренера на всех потоках;
- скорость GPU при пачках 4096 / 16384 / 32768.

Весь вывод идёт в отчёт.
- Если напечатано `MISMATCH`, скрипт остановится. Тогда запусти эмуляцию: те же ядра на CPU. Она показывает, в чём баг: в коде ядер или в вызовах CUDA.
  ```powershell
  python scripts/gpu_bench.py --emulate --check-iters 20000 *> bench_emu.log
  Get-Content bench_emu.log
  ```
- Если скрипт упал с ошибкой CUDA, повтори с `$env:CUDA_LAUNCH_BLOCKING = "1"`: ошибка укажет точное ядро. Приложи оба вывода.
- Если всё `IDENTICAL`, но замер идёт очень долго (больше 20 минут), останови его (Ctrl+C) и запусти меньший: `python scripts/gpu_bench.py --iters 200000 --skip-check`.

### 6. Загрузка во время замера (по желанию, если легко)
Пока идёт замер GPU (строки `GPU batch ...`), в другом окне выполни `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 1`. Запиши в отчёт несколько типичных строк, а также загрузку CPU из диспетчера задач. Это покажет, во что упирается скорость: в видеокарту или в процессор.

### 7. Весь набор тестов (контроль, что ничего не сломано)
```powershell
python -m pytest -q *> tests_all.log
Get-Content tests_all.log -Tail 15
```
Занимает около 5 минут. Известное исключение: `tests/test_search_core.py::test_our_average_is_accumulated_every_iteration` иногда падает под нагрузкой (гонка потоков в поиске), к GPU он отношения не имеет. Остальные падения перечисли в отчёте.

### 8. Отчёт
Создай файл `GPU_REPORT.md` в папке `..\zootop-gpu` по шаблону:

```markdown
# Отчёт о прогоне GPU-тренера на ПК (дата, время)
Исполнитель: <какая нейросеть>. Коммит opt/gpu: <хеш из шага 1>.

## Окружение (шаг 0)
<дословные выводы: nvidia-smi (шапка с драйвером и GPU), nvcc --version, cmake, python, pybind11>

## Сборка (шаг 2)
Итог: GPU trainer ON / OFF / ошибка. Что пришлось сделать для сборки: <ничего / переменные / генератор>.
<строки из build.log>

## cuda_available (шаг 3)
<вывод>

## Тесты flat/GPU (шаг 4)
| тест | cpu | emu | gpu |
|---|---|---|---|
| push_fold | ✅/❌ | … | … |
| three_player_flop | … | … | … |
| river_potential | … | … | … |
test_batched.py: <итог>. Ошибки: <дословно>

## Проверка и скорость (шаг 5)
Таблицы корзин: использовались / нет (путь).
<полный bench.log>

## Загрузка (шаг 6)
<строки nvidia-smi, загрузка CPU> или «не делалось»

## Весь набор тестов (шаг 7)
<итоговая строка pytest и список упавших>

## Замечания исполнителя
<всё необычное: предупреждения, долгие шаги, что пробовал; без догадок о причинах, если не проверено>
```

### 9. Отдать отчёт
```powershell
git checkout -b opt/gpu-report
git add -f GPU_REPORT.md build.log tests_gpu.log bench.log tests_all.log bench_emu.log
git commit -m "GPU run report from the user's PC"
git push -u origin opt/gpu-report
```
Если push не прошёл (нет прав или сети), ничего не форси. Скажи пользователю, где лежит `GPU_REPORT.md`, чтобы он переслал файл.
В конце покажи пользователю краткий итог в 3–5 строк:
- собралось ли;
- `IDENTICAL` или нет;
- скорость GPU против CPU;
- где отчёт.


---

# Раунд 2: замер скорости с таблицами корзин

Раунд 1 (отчёт в ветке `opt/gpu-report`) показал две вещи:
- GPU совпадает с CPU **бит в бит**;
- скорость упёрлась в CPU: без таблиц корзины E[HS] считаются Монте-Карло, и GPU простаивал (загрузка 0–15 %).

Что изменилось в коде:
- флаг `/Zc:preprocessor` теперь стоит в CMake, переменная `CL` больше не нужна;
- CPU готовит раздачи следующей пачки, пока GPU считает текущую;
- в замере есть разбивка времени.

Правила те же, что в раунде 1. Работай в той же папке `..\zootop-gpu`.

### R2-1. Обновить копию и собрать
```powershell
cd ..\zootop-gpu
git fetch origin opt/gpu
git checkout --detach origin/opt/gpu
git log --oneline -1
Remove-Item Env:CL -ErrorAction SilentlyContinue
python scripts/build_fast.py --clean *> build2.log
Select-String -Path build2.log -Pattern "negpluribus:|error|built" | Select-Object -First 20
```
Должно быть `GPU trainer ON` и `built`, **без** `$env:CL`. Если снова вылезла ошибка C1189 про препроцессор, запиши это и собери с `$env:CL = "/Zc:preprocessor"`, как в раунде 1.

### R2-2. Контроль бит в бит после изменений
```powershell
python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu2.log
Get-Content tests_gpu2.log -Tail 5
```

### R2-3. Таблица корзин для бенча (около 3–10 минут)
```powershell
python scripts/gpu_bench.py --write-buckets data\buckets_gpubench.json
python scripts/build_bucket_table.py --buckets data\buckets_gpubench.json --out data\bucket_tables *> table.log
Get-Content table.log -Tail 15
```
Время построения и итоговые строки идут в отчёт. Скрипт сам сверяет таблицу со старым способом на 20 тыс. рук на каждой улице. Если он написал о расхождении, это важно: запиши и остановись.

### R2-4. Замер с таблицей
```powershell
$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path data\bucket_tables).Path
python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 *> bench2.log
Get-Content bench2.log
```
Первой строкой должно быть `bucket tables: <путь>`, а не `none`. Параллельно, как в шаге 6 раунда 1, сними загрузку GPU и CPU во время строк `GPU batch ...` (несколько строк на каждую пачку, если успеешь).

Если хватит времени, сделай ещё один прогон с пачками побольше:
```powershell
python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 --skip-check --batches 65536,131072 *> bench3.log
```

### R2-5. Отчёт
Файл `GPU_REPORT2.md`: те же разделы, что в раунде 1, только для шагов R2-1…R2-4. Весь `bench2.log` и `bench3.log` — дословно, плюс загрузка.
```powershell
git checkout -b opt/gpu-report2
git add -f GPU_REPORT2.md build2.log tests_gpu2.log table.log bench2.log bench3.log
git commit -m "GPU run report 2 (bucket tables)"
git push -u origin opt/gpu-report2
```
Файлы `data\bucket_tables\*.npbt` **не коммить**, они большие.


---

# Раунд 3: замер после ускорения ядер

По раунду 2: с таблицами GPU быстрее CPU в 1,2–2,1 раза, GPU загружен на 80–96 %, но берёт всего ~90 Вт из 250. Похоже, он стоит в очереди, а не считает.

Что изменилось в коде:
- убраны атомарные счётчики в одну ячейку, через них шёл каждый элемент;
- сортировка читает только нужные биты ключа;
- в замере есть разбивка времени устройства: `fwd` / `back` / `sort` / `add`.

Правила те же. Папка та же, `..\zootop-gpu`. Таблица корзин от раунда 2 уже лежит в `data\bucket_tables`, её не пересобирать.

### R3-1. Обновить и собрать
```powershell
cd ..\zootop-gpu
git fetch origin opt/gpu
git checkout --detach origin/opt/gpu
git log --oneline -1
python scripts/build_fast.py --clean *> build3.log
Select-String -Path build3.log -Pattern "negpluribus:|error|built" | Select-Object -First 20
```

### R3-2. Бит в бит
```powershell
python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu3.log
Get-Content tests_gpu3.log -Tail 5
```

### R3-3. Замер (как в раунде 2)
```powershell
$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path data\bucket_tables).Path
python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 *> bench4.log
Get-Content bench4.log
python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 --skip-check --batches 65536,131072 *> bench5.log
Get-Content bench5.log
```
Во время строк `GPU batch ...` сними загрузку так же, как в раунде 2: GPU %, память, ватты, CPU %.

### R3-4. Отчёт
Файл `GPU_REPORT3.md` в том же стиле: оба лога дословно, загрузка, замечания.
```powershell
git checkout -b opt/gpu-report3
git add -f GPU_REPORT3.md build3.log tests_gpu3.log bench4.log bench5.log
git commit -m "GPU run report 3"
git push -u origin opt/gpu-report3
```


---

# Раунд 4: стратегии один раз на пачку, 32-битные ключи

По раунду 3: +6–12 %. По разбивке видно, что время съедают прямой и обратный проходы и сортировка.

Что изменилось в коде:
- стратегия каждой строки таблицы считается один раз в начале пачки: раньше каждый элемент заново делал регрет-матчинг с делениями в double;
- ключи сортировки теперь 32-битные, около 20 бит вместо 36, сортировка устойчивая: порядок внутри ячейки задаёт сам буфер.

Всё то же, что в раунде 3, только новые имена файлов:
```powershell
cd ..\zootop-gpu
git fetch origin opt/gpu
git checkout --detach origin/opt/gpu
git log --oneline -1
python scripts/build_fast.py --clean *> build4.log
Select-String -Path build4.log -Pattern "negpluribus:|error|built" | Select-Object -First 20
python -m pytest -q tests/test_flatcfr.py tests/test_batched.py *> tests_gpu4.log
Get-Content tests_gpu4.log -Tail 5
$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path data\bucket_tables).Path
python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 *> bench6.log
Get-Content bench6.log
python scripts/gpu_bench.py --buckets data\buckets_gpubench.json --iters 5000000 --skip-check --batches 65536,131072 *> bench7.log
Get-Content bench7.log
```
Загрузку во время `GPU batch ...` снимай как раньше. Отчёт — `GPU_REPORT4.md`:
```powershell
git checkout -b opt/gpu-report4
git add -f GPU_REPORT4.md build4.log tests_gpu4.log bench6.log bench7.log
git commit -m "GPU run report 4"
git push -u origin opt/gpu-report4
```


---

# Раунд 5: качество на равном времени (дуэль GPU против CPU), корзины как в проде

Раунд 4 дал GPU в 2,3–5 раз быстрее CPU при совпадении бит в бит. Теперь главный вопрос: **играет ли стратегия, обученная на GPU за то же время, не хуже**. Большая пачка реже обновляет стратегию. Основная пачка — 4096, более крупные проверяем с шагом ×2 до 32768 включительно. Ответ даёт только дуэль.

Игра: HU 100bb, узкая сетка (экспериментальная). Корзины как в проде: **potential-aware, 64 корзины, точные признаки** (`exact=True`). Всё в той же папке `..\zootop-gpu`.

Долгий раунд: подготовка корзин (~5–10 мин), 5 обучений по 10 минут, 4 дуэли. **Во время обучения ничем не нагружай ПК**: там замер «за равное время».

### R5-1. Обновить, собрать, тесты
```powershell
cd ..\zootop-gpu
git fetch origin opt/gpu
git checkout --detach origin/opt/gpu
git log --oneline -1
python scripts/build_fast.py --clean *> build5.log
Select-String -Path build5.log -Pattern "negpluribus:|error|built" | Select-Object -First 20
python -m pytest -q tests/test_flatcfr.py tests/test_batched.py tests/test_exact_features.py tests/test_exact_bucketer.py *> tests_gpu5.log
Get-Content tests_gpu5.log -Tail 3
```

### R5-2. Корзины: подбор, точные признаки, таблица
Время каждой команды запиши (`Measure-Command` или по часам).
```powershell
$D = "data\eq"
New-Item -ItemType Directory -Force $D | Out-Null
python -c "from negpluribus.abstraction import make_bucketer; make_bucketer('potential', 64, None, exact=True).fit(n_situations=4800, seed=0, verbose=True).save(r'$D\buckets_pa64.json')" *> fit.log
python -c "from negpluribus.fast import core; c=core(); print(c.build_exact_features(3, 10, 16, r'$D\exact_features_flop_b10.bin')); print(c.build_exact_features(4, 10, 16, r'$D\exact_features_turn_b10.bin'))" *> features.log
python scripts/build_bucket_table.py --buckets $D\buckets_pa64.json --features-dir $D --out $D\bucket_tables *> table5.log
Get-Content fit.log -Tail 3; Get-Content features.log; Get-Content table5.log -Tail 8
```
В `table5.log` должно быть `0 mismatches` на каждой улице и `written ...npbt`. Если есть расхождения, запиши и остановись.

### R5-3. Пять обучений по 600 секунд
```powershell
$env:NEGPLURIBUS_BUCKET_TABLES = (Resolve-Path $D\bucket_tables).Path
foreach ($t in "eqcpu","eqgpu4k","eqgpu8k","eqgpu16k","eqgpu32k") { Copy-Item $D\buckets_pa64.json "$D\buckets_$t.json" }
$G = "--players 2 --stack 100 --street river --preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2 --buckets 64 --buckets-kind potential --exact-features --backend cpp --eval-deals 0 --no-l1 --seed 0 --seconds 600 --data-dir $D".Split(" ")
python scripts/train_blueprint.py @G --tag eqcpu *> train_eqcpu.log
python scripts/train_blueprint.py @G --tag eqgpu4k  --gpu 0 --batch 4096  *> train_eqgpu4k.log
python scripts/train_blueprint.py @G --tag eqgpu8k  --gpu 0 --batch 8192  *> train_eqgpu8k.log
python scripts/train_blueprint.py @G --tag eqgpu16k --gpu 0 --batch 16384 *> train_eqgpu16k.log
python scripts/train_blueprint.py @G --tag eqgpu32k --gpu 0 --batch 32768 *> train_eqgpu32k.log
Select-String -Path train_eq*.log -Pattern "buckets:|iterations in|done in|saved|Error|error"
```
В каждом логе должно быть:
- `buckets: loaded ...buckets_<tag>.json`;
- строка вида `CPU: N iterations in 600s = X it/s` или `GPU: ...`;
- в конце `saved ...blueprint_<tag>.bin`.

### R5-4. Четыре дуэли по 200 тыс. раздач, каждый GPU против CPU
Можно по две одновременно в разных окнах, после того как все обучения закончены. `NEGPLURIBUS_BUCKET_TABLES` оставь заданной, иначе корзины в дуэли будут считаться очень медленно.
```powershell
$C = "--players 2 --stack 100 --street river --preflop-fracs 1.0 --postflop-fracs 0.5,1.0 --max-raises 2 --buckets $D\buckets_pa64.json --deals 200000".Split(" ")
foreach ($b in "4k","8k","16k","32k") {
  python scripts/compare_checkpoints.py @C --a "$D\blueprint_eqgpu$b.bin" --b "$D\blueprint_eqcpu.bin" --label-a "gpu$b" --label-b cpu *> "duel_gpu$b.log"
}
Get-ChildItem duel_gpu*.log | ForEach-Object { "== $_"; Get-Content $_ -Tail 15 }
```
Если дуэль идёт дольше часа, запиши, сколько она успела пройти, и не прерывай.

### R5-5. Отчёт
`GPU_REPORT5.md`:
- время подбора корзин, признаков и таблицы, итоги сверки таблицы;
- итерации и it/s каждого из 5 обучений;
- итог всех 4 дуэлей дословно (строки с `vs`, bb/100 и доверительный интервал) и время каждой;
- замечания.
```powershell
git checkout -b opt/gpu-report5
git add -f GPU_REPORT5.md build5.log tests_gpu5.log fit.log features.log table5.log train_eq*.log duel_gpu*.log
git commit -m "GPU run report 5 (equal-time duel, potential-aware exact 64)"
git push -u origin opt/gpu-report5
```
Большие файлы (`data\eq\*.bin`, `*.npbt`, признаки) **не коммить**.
