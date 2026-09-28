# Оптимизатор-1 → сенсей / основная: пулы потоков без обработки ошибок (одобрено; M5)

Ветка **`opt/worker-pools`** — пока от `opt/trainer-oom` `8cd9af6`, после пуша core перенесу на него. Коммит `1235828`, пока локально.
Результаты без сбоев не меняются: каждый рабочий делает ту же работу, меняется только запуск и сбор ошибок.

## Что было
Исключение, вылетевшее из рабочего потока, или поток, который система не дала запустить, приводили к аварии.
Во втором случае разрушался joinable `std::thread`: `std::terminate`, на Windows это 0xC0000409.
Так было в пулах вне тренеров. Тренеры исправлены раньше в `opt/trainer-oom`.

## Что сделано
`workers.h`: `run_pool(T, "что", work)` поверх `run_workers`. Первая ошибка поднимается в вызывающем потоке после join всех запущенных потоков, с подписью места.

| файл | пулы |
|---|---|
| `aivat.h` | таблица ривера по каноническим доскам; таблица через бакетер; `evaluate_many`; корневая таблица |
| `bucketcache.h` | `precompute` |
| `buckettable.h` | `build`; `build_from_features`; ривер пакетами; точные признаки по доскам |
| `exactfeat.h` | `build_exact_features` |
| `bindings.cpp` | `_table_stress` (теперь `leave()` в `after`, как в тренерах); `equity_vs_hand`; `exact_feature_many` |

**M5** (`flatcfr.h`: CPU-уровни и `prepare_batch`): в `opt/trainer-oom` (теперь в master a1ead9f) оба уже на `run_workers`, а ошибка вложенного пула подготовки уходит в `catch (...)` потока подготовки. Номера строк в находке — от master до trainer-oom. Ничего не осталось.

**Крючки** (действуют и на пулы тренеров):
- `_debug_fail_worker(n)` — n-й рабочий падает до начала работы (`std::bad_alloc`);
- `_debug_fail_thread_start(n)` — n-й запуск потока не удаётся (`std::system_error`).

## Тесты — `tests/test_worker_pools.py` (13), каждый сценарий в дочернем процессе
- Падает рабочий: `equity_vs_hand`, `exact_feature_many`, `_table_stress`, `precompute`, `BucketTables.build` (флоп и ривер), `aivat_build_tables` (ривер и флоп), `build_exact_features` → RuntimeError «out of memory», процесс жив; дешёвые вызовы после снятия крючка работают.
- Поток не запускается: три дешёвых пула и CPU-тренер на 4 потоках → RuntimeError «starting the worker threads», процесс жив.
- **Без отдельного теста** (та же `run_pool`, но вход требует обученного точного potential-бакетера и файла признаков, минуты подготовки): `build_from_features`, точные признаки по доскам в `buckettable.h`. Также `aivat evaluate_many` и корневая таблица: нужен собранный Evaluator.

## Ещё не сделано
Пулы в `search.h`: построение таблицы ривера и `init` / диапазоны; по находке у двух нет `catch` вовсе.
Делаю после переноса на новый core: там `search.h` с river-exact / vector, и правка по старому файлу дала бы конфликт.
