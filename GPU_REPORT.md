# Отчёт о прогоне GPU-тренера на ПК (суббота, 26 сентября 2026, 13:10–13:33 местного времени)

Исполнитель: opencode (CLI-ассистент), модель GLM (Z.ai). Коммит opt/gpu: `d4b9411` («GPU run instructions for a helper on the user's PC; gpu_bench --emulate», вершина origin/opt/gpu).

Работа шла в отдельной копии `C:\Project Manchatten\zootop-gpu` (git worktree, detached HEAD на d4b9411), исходная рабочая копия `..\zootop` не тронута (команды шагов 0 выполнялись в ней, изменений файлов нет). Код (`csrc/`, `negpluribus/`, `scripts/`, `tests/`) не менялся; всё, что делалось сверх шагов, — настройка окружения, перечислена ниже по тексту и в «Замечаниях».

## Окружение (шаг 0)

`nvidia-smi` (шапка):

```
Sat Sep 26 13:10:36 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.62                 KMD Version: 610.62        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage |      GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================+
|   0  NVIDIA GeForce RTX 5070      WDDM  |  00000000:01:00.0  On |                  N/A |
|  0%   42C    P5             19W /  250W |    1427MiB /  12227MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------------------------------------------------------+
```

`nvcc --version`:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Sun_Aug_03_13:24:32_Pacific_Standard_Time_2026
Cuda compilation tools, release 13.4, V13.4.59
Build cuda_13.4.r13.4/compiler.38657139_0
```

(CUDA 13.4 ≥ 12.8 — sm_120 поддерживается. Путь nvcc в PATH: найден CMake'ом как `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe`.)

`cmake --version`:

```
cmake version 4.4.3

CMake suite maintained and supported by Kitware (kitware.com).
```

`python --version`:

```
Python 3.14.6
```

`python -c "import pybind11, ..."` — до установки пакета:

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import pybind11, sys; print(pybind11.__version__, sys.executable)
    ^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'pybind11'
```

pybind11 отсутствовал и в системном Python 3.14, и во втором интерпретаторе (`C:\Users\<пользователь>\AppData\Local\Programs\Python\Python311\python.exe`). Настройка окружения: `python -m pip install pybind11` (установлен pybind11 3.1.0). После установки:

```
3.1.0 C:\Python314\python.exe
```

`git status --short` (первые 5 строк): пусто, рабочая копия чистая.

## Сборка (шаг 2)

Итог: **GPU trainer ON**, сборка успешная. Что пришлось сделать: две настройки окружения — `pip install pybind11` (см. выше) и переменная среды `CL=/Zc:preprocessor` для второй попытки сборки (см. ниже). Генератор не менялся (стандартный Visual Studio, `-A x64`), `CUDACXX` не понадобился.

Первая попытка (`python scripts/build_fast.py --clean *> build.log`) упала при компиляции `gpucfr.cu` — CCCL из CUDA 13.4 требует стандарт-конформный препроцессор MSVC. Строки `error` из вывода (цитирую из консоли: build.log первой попытки был перезаписан успешной второй; ошибки давал `Select-String -Path build.log -Pattern "negpluribus:|error|built"`):

```
-- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/include/cccl\cuda/std/__cccl/preprocessor.h(23): fatal error C1189: #error:  MSVC/cl.exe with traditional preprocessor is used. This may lead to unexpected compilation errors. Please switch to the standard conforming preprocessor by passing `/Zc:preprocessor` to cl.exe. You can define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING to suppress this warning. [C:\Project Manchatten\zootop-gpu\build\fast\negp_gpu.vcxproj]
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 13.4.targets(807,9): error MSB3721: выход из команды ""C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"  --use-local-env -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" -x cu   -I"C:\Project Manchatten\zootop-gpu\csrc" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include\cccl" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include"     --keep-dir negp_gpu\x64\Release  -maxrregcount=0     --machine 64 --compile -forward-unknown-to-host-compiler -std=c++17 --generate-code=arch=compute_89,code=[sm_89] --generate-code=arch=compute_120,code=[compute_120,sm_120] --fmad=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler="/EHsc -Ob2"   -D_WINDOWS -DNDEBUG -DNEGP_WITH_CUDA=1 -D"CMAKE_INTDIR=\"Release\"" -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Xcompiler "/EHsc /W1 /nologo /O2 /FS   /MD /GR" -Xcompiler "/Fd\"C:\Project Manchatten\zootop-gpu\build\fast\Release\negp_gpu.pdb\"" -o negp_gpu.dir\Release\gpucfr.obj "C:\Project Manchatten\zootop-gpu\csrc\gpucfr.cu"" с кодом "2". [C:\Project Manchatten\zootop-gpu\build\fast\negp_gpu.vcxproj]
    + FullyQualifiedErrorId : NativeCommandError
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['C:\\Program Files\\CMake\\bin\\cmake.EXE', '--build', 'C:\\Project Manchatten
```

Исправление настройкой окружения (код не менялся): `$env:CL = "/Zc:preprocessor"` — эту переменную читает сам cl.exe при каждом вызове, в том числе когда его вызывает nvcc; после этого `python scripts/build_fast.py --clean *> build.log` прошла целиком. Строки из итогового `build.log`:

```
-- negpluribus: GPU trainer ON (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe, archs 89-real;120)
built C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
```

## cuda_available (шаг 3)

```
C:\Project Manchatten\zootop-gpu\negpluribus\_fastcore.cp314-win_amd64.pyd
(True, '')
```

## Тесты flat/GPU (шаг 4)

Команда: `python -m pytest -v tests/test_flatcfr.py tests/test_batched.py *> tests_gpu.log` (для её запуска понадобилось `python -m pip install pytest`, установлен pytest 9.1.1). Итог: **14 passed in 2.87s**, падений нет.

| тест | cpu | emu | gpu |
|---|---|---|---|
| push_fold | ✅ | ✅ | ✅ |
| three_player_flop | ✅ | ✅ | ✅ |
| river_potential | ✅ | ✅ | ✅ |

`test_batched.py`: 5 passed (`test_philox_known_answers`, `test_deal_is_a_permutation_and_a_function_of_seed_and_iteration`, `test_same_result_for_any_thread_count_and_aligned_splits`, `test_off_is_the_sequential_trainer`). Ошибок нет. Полный вывод (хвост лога):

```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0 -- C:\Python314\python.exe
cachedir: .pytest_cache
rootdir: C:\Project Manchatten\zootop-gpu
configfile: pyproject.toml
plugins: anyio-4.14.2
collecting ... collected 14 items

tests/test_flatcfr.py::test_push_fold_is_the_reference[cpu] PASSED       [  7%]
tests/test_flatcfr.py::test_push_fold_is_the_reference[emu] PASSED       [ 14%]
tests/test_flatcfr.py::test_push_fold_is_the_reference[gpu] PASSED       [ 21%]
tests/test_flatcfr.py::test_three_player_flop_is_the_reference[cpu] PASSED [ 28%]
tests/test_flatcfr.py::test_three_player_flop_is_the_reference[emu] PASSED [ 35%]
tests/test_flatcfr.py::test_three_player_flop_is_the_reference[gpu] PASSED [ 42%]
tests/test_flatcfr.py::test_river_potential_is_the_reference[cpu] PASSED [ 50%]
tests/test_flatcfr.py::test_river_potential_is_the_reference[emu] PASSED [ 57%]
tests/test_flatcfr.py::test_river_potential_is_the_reference[gpu] PASSED [ 64%]
tests/test_flatcfr.py::test_cuda_available_reports_a_reason PASSED       [ 71%]
tests/test_batched.py::test_philox_known_answers PASSED                  [ 78%]
tests/test_batched.py::test_deal_is_a_permutation_and_a_function_of_seed_and_iteration PASSED [ 85%]
tests/test_batched.py::test_same_result_for_any_thread_count_and_aligned_splits PASSED [ 92%]
tests/test_batched.py::test_off_is_the_sequential_trainer PASSED         [100%]

============================= 14 passed in 2.87s ==============================
```

## Проверка и скорость (шаг 5)

Таблицы корзин: **не использовались** — в исходной папке `C:\Project Manchatten\Zootop` их нет (папки `data/` нет, файлов `*.npbt` нет; переменная `NEGPLURIBUS_BUCKET_TABLES` не задана, искал и по коду: `negpluribus/fast/tables.py` ждёт папку с файлами `buckets_*.npbt`).

Запуск `python scripts/gpu_bench.py *> bench.log` завершился сам примерно за 80 секунд (меньший вариант не понадобился). Обе проверки **IDENTICAL** — GPU совпадает с CPU-эталоном бит в бит. Полный `bench.log`:

```
cuda_available: True 
device: NVIDIA GeForce RTX 5070 (sm_120); HU100 flat game {'decisions': 7712, 'terminals': 13074, 'infosets': 63628, 'cells': 171593, 'depth': 15}
identity (GPU vs the batched CPU reference):
  push/fold 10bb: 3000 it, batch 256: infosets 1987, differing 0, extra 0 -> IDENTICAL  [0.1s]
  HU 100bb: 100000 it, batch 4096: infosets 63455, differing 0, extra 0 -> IDENTICAL  [7.8s]
speed on HU 100bb, 1000000 iterations:
  CPU trainer, 16 threads: 13.0s = 76,667 it/s
  GPU batch 4096: 14.2s = 70,581 it/s (x0.92 vs CPU); last batch: 153,603 items, 370,036 records, device 2.1 ms traverse + 0.2 ms apply = 1,817,485 it/s device-only
  GPU batch 16384: 13.1s = 76,110 it/s (x0.99 vs CPU); last batch: 148,286 items, 350,221 records, device 2.2 ms traverse + 0.2 ms apply = 6,979,185 it/s device-only
  GPU batch 32768: 12.7s = 78,691 it/s (x1.03 vs CPU); last batch: 4,261,170 items, 9,961,393 records, device 10.0 ms traverse + 6.5 ms apply = 1,989,632 it/s device-only
```

`MISMATCH` не было, поэтому эмуляция (`--emulate`) не запускалась и файл `bench_emu.log` не создавался. Ошибок CUDA не было, `CUDA_LAUNCH_BLOCKING` не понадобился.

## Загрузка (шаг 6)

Основной прогон шага 5 закончился быстрее, чем успевало заработать внешнее наблюдение, поэтому загрузка снята во время отдельного короткого прогона `python scripts/gpu_bench.py --iters 1000000 --skip-check --batches 32768`, фаза `GPU batch 32768` (строки опроса раз в секунду; CPU — `Win32_Processor.LoadPercentage`, это тот же показатель, что «загрузка CPU» в диспетчере задач; формат: `GPU util, память GPU, мощность, загрузка CPU`):

```
0 %, 2012 MiB, 25.17 W, CPU 11%
7 %, 2493 MiB, 28.43 W, CPU 23%
5 %, 2728 MiB, 28.64 W, CPU 24%
3 %, 2711 MiB, 30.04 W, CPU 37%
2 %, 2693 MiB, 27.21 W, CPU 9%
2 %, 2688 MiB, 26.83 W, CPU 12%
2 %, 2677 MiB, 27.06 W, CPU 10%
1 %, 2677 MiB, 26.92 W, CPU 5%
15 %, 2715 MiB, 28.13 W, CPU 10%
6 %, 2748 MiB, 25.53 W, CPU 9%
```

Тот прогон напечатал: `GPU batch 32768: 38.4s = 26,009 it/s (x1.13 vs CPU); last batch: 4,261,170 items, 9,961,393 records, device 9.5 ms traverse + 10.6 ms apply = 1,632,255 it/s device-only`. Наблюдаемый факт: во время фазы GPU-замера утилизация GPU 0–15%, мощность ~25–30 Вт из 250 Вт, CPU 5–37% — не нагружено ни то, ни другое; device-only скорость по тем же строкам лога — 1.6–2.0 млн it/s при batch 32768.

## Весь набор тестов (шаг 7)

`python -m pytest -q *> tests_all.log` — итоговая строка:

```
229 passed, 3 skipped in 166.65s (0:02:46)
```

Упавших нет, в том числе прошло известное исключение `tests/test_search_core.py::test_our_average_is_accumulated_every_iteration`. 3 пропущенных — штатные skip'ы набора (не связанные с GPU).

## Замечания исполнителя

1. Настройки окружения, которые пришлось сделать (код не менялся, всё остальное — по инструкции):
   - `python -m pip install pybind11` (3.1.0) — пакета не было ни в одном из двух найденных Python (3.14 системный, 3.11 пользовательский); сборка на этом ПК ранее, судя по всему, шла из другого окружения, которого сейчас на машине не видно (в папке проекта venv нет).
   - `python -m pip install pytest` (9.1.1) — для шагов 4 и 7.
   - `$env:CL = "/Zc:preprocessor"` перед повторной сборкой (только на время сборки) — иначе CUDA 13.4/CCCL отказывается собираться с традиционным препроцессором MSVC (fatal error C1189, см. раздел «Сборка»).
2. nvcc 13.4 и драйвер (CUDA UMD 13.3) поддерживают sm_120; собрано с archs `89-real;120`, модуль увидел карту как `NVIDIA GeForce RTX 5070 (sm_120)`.
3. Основной бенч (шаг 5) отработал быстро (~80 с на всё, включая обе проверки IDENTICAL и четыре замера по 1 млн итераций) — рекомендованного запасного варианта с меньшим числом итераций не потребовалось.
4. Замечена заметная вариативность замера скорости между прогонами на одной машине за один час: CPU-тренер 76,667 it/s в основном прогоне против 20,573–23,039 it/s в трёх последующих коротких прогонах (те же 1 млн итераций, 16 потоков); соотношение GPU/CPU при этом было x0.92–x1.03 в основном прогоне и x1.00–x1.15 в последующих. Причину не выяснял — фиксирую как наблюдение.
5. Строки `GPU batch ...` печатаются в конце каждой фазы, поэтому для шага 6 пришлось запускать отдельный прогон и следить за логом; первые две попытки наблюдения попали не в ту фазу (в лог попали нули — их в отчёт не включал, финальные строки выше сняты именно во время фазы GPU batch 32768).
6. Этот nvidia-smi (драйвер 610.62) не принимает `-l`/`-c` вместе с `--query-gpu` («Option ... is not recognized»), поэтому сэмплировал циклом одиночных вызовов; `Get-Counter` в этой системе падает (русская локаль), CPU снимал через `Win32_Processor.LoadPercentage`.
7. Все проверки соответствия (тесты `[gpu]` из шага 4 и обе проверки `IDENTICAL` из шага 5) прошли: расхождений GPU с CPU-эталоном не обнаружено.
8. Логи для шага 9: `build.log`, `tests_gpu.log`, `bench.log`, `tests_all.log`; `bench_emu.log` отсутствует за ненадобностью (см. шаг 5). Временные логи замеров загрузки удалены, их содержимое перенесено в этот отчёт.
