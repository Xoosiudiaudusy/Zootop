# Оптимизатор-1 → сенсей / основная: ветки перенесены на core `b03512a`

| ветка | коммит | проверка на новом core [замер, облако] |
|---|---|---|
| `opt/worker-pools` | `eeb71ae` | + пулы `search.h`; `search_bench --threads 1 --compare` 12/12 = core; `test_worker_pools` 14 passed; полный набор 426 passed, 3 failed (см. ниже) |
| `opt/gpu-host-pool-oom` | `35d01cc` | коммиты пула + сведение с обработкой ошибок (d559b1e) заново поверх core; `test_trainer_oom` + `test_flatcfr` 38 passed |
| `opt/search-speed2` | `4d60396` | один конфликт в `solve()` (векторная ветка core): переменные пачек итераций объявлены до развилки, вектор не тронут; MCCFR 12/12 и вектор 48/48 бит в бит против core |
| `opt/vector-speed` | `18506ab` | один конфликт в `VBoard` (комментарий 1bcfdb8 к `bucket`); вектор 48/48 бит в бит против core (с 1bcfdb8 и H2) |
| `opt/resume-safety` | `aef7d62` | не трогала (у основной на воротах). С моими следующими ветками не пересекается: `train_blueprint.py`, `fast/blueprint.py`, `fast/runinfo.py` больше никто не меняет |

Старые хэши этих веток перезаписаны: мои ветки, `--force-with-lease`.

## Пулы `search.h` (в `opt/worker-pools`)
- эксплуатируемость (ветки по картам ривера) — `run_pool`;
- таблица ривера поиска — `run_pool`;
- диапазоны (`compute_ranges`) — `run_workers`; раньше без `catch`.

Сам пул `solve()` уже был защищён веткой search-oom.

Область перевода H4 (`search.h` 1596–1626, 2003–2016) не задета: ближайшая правка — пул диапазонов, строки ~2060–2070.
Конфликт при слиянии с H4 возможен только текстовый, рядом.

Тест `test_search_pools_raise` проходит три случая:
- создание поиска на флопе (диапазоны + таблица ривера) → RuntimeError;
- эксплуатируемость на тёрне → RuntimeError;
- после снятия крючков всё работает в том же процессе.

## Полный набор на `opt/worker-pools`: 426 passed, 9 skipped, 36 xfailed, 3 failed
Все три падения — `tests/test_checkpoint_safety.py::test_a_save_that_cannot_replace_the_file_raises_and_keeps_the_previous_one` (ck.bin, ck.json, bp.bin).
- Тот же тест **так же падает на чистом core `b03512a`** в облаке (3 failed, 3 passed, 3 xfailed): «DID NOT RAISE RuntimeError».
- Причина — среда, а не код:
  - тест делает `chmod` файла только на чтение и ждёт, что переименование поверх него не удастся (так на Windows);
  - в Linux право на замену решает каталог, к тому же облако работает от root, поэтому замена проходит.
- Для реестра QA: на Linux тест нужно пропускать или ломать запись иначе (каталог без записи; root и это обходит — тогда пропуск при `os.geteuid() == 0`).
