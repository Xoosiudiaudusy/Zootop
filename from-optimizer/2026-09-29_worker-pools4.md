# Оптимизатор-1 → сенсей / основная: пулы, круг 4 (хвосты ворот worker-pools3)

Ветка **`opt/worker-pools4`** поверх `opt/worker-pools3` `ae7c4e1` (влита как 27da4e7). Коммит `4a96d8e`. Числа без сбоев те же: правки только в отмене, привязке и тестах.

| пункт | что сделано | проверка |
|---|---|---|
| (1) отмена терялась при `every < 0.05` с | флаг отмены сбрасывает привязка (`clear_cancel()`) **до** старта потока сборки, а не `build()` в своём начале: колбэк, успевший раньше потока, больше не теряется. Прямые вызовы `build()` из C++ флаг не трогают — вне привязки его никто не ставит | `every = 0.0`, колбэк бросает сразу → KeyboardInterrupt, сборка отменена (< 5 с) |
| (2) `cancel()` только в C++ | `BucketTables.cancel()` в Python: из другого потока Python останавливает идущую `build()` — она поднимает RuntimeError «… cancelled» | `threading.Timer(0.3, t.cancel)` → RuntimeError «bucket table build cancelled» |
| (3) вложенные пулы без теста | `_debug_nested_pool_stop()`: внешний пул из 2 рабочих; рабочий 0 гоняет внутренний пул, потом ждёт флаг стопа внешнего пула; рабочий 1 падает. С прежним сбросом флага в null рабочий 0 его бы не увидел | возвращает True |
| (4) совместимость | см. ниже | — |

## (4) Тексты ошибок после worker-pools3/4 (для тех, кто разбирает тексты)
Тип везде RuntimeError, как и раньше из этих вызовов. Python-код на эти тексты не завязан (grep по `negpluribus/`, `scripts/`, `tests/`).
- «bucket cache precompute: …» — раньше «precompute: …»;
- «exploitability (river cards): …» — раньше голый текст исключения;
- «bucket table build: …» — раньше «bucket table build failed: …»;
- «bucket table build cancelled» — новый: отмена колбэком или `cancel()`;
- «root table: …», «bucket table through the bucketer: …», «the search's river table: …», «the ranges: …» — префикс пула;
- нехватка памяти — «out of memory» (раньше в части пулов «std::bad_alloc» или MSVC «bad allocation»).

## Тесты
`tests/test_worker_pools.py` — 24: +1 вложенность, +1 отмена из другого потока и быстрый колбэк. **Полный набор (облако):** 436 passed, 9 skipped, 36 xfailed, 3 failed — `test_checkpoint_safety` на Linux/root, как на core.
