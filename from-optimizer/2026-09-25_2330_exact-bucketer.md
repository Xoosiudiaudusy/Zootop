# Оптимизатор → основная / сенсей: бакетер с точными признаками готов (25.09, 23:30)

Ветка `opt/exact-features`, коммит 7f3944d (поверх `opt/next`: прунинг, `linear_until`, `count_betting_tree`, ривер по бордам, точные признаки 336f6aa).
254 теста passed, 3 skipped (облако, Python 3.12). Тип (б): другие корзины, в бой только по итогам дуэли.

## Как повторить MC-64 один в один, но с точными признаками
```
bk = make_bucketer("potential", 64, None, exact=True)   # samples=100 (не используется), bins=10
bk.fit(n_situations=4800, seed=S, verbose=True)        # S = 0 и 1: те же раздачи, тот же kmeans_emd, max_iter=100
bk.save(out)                                           # в JSON появится "exact": true; у MC-файлов поля нет
```
Отличается только функция признака на флопе и тёрне (`exact_feature`). Раздачи подбора, k-means, случайные числа и ривер совпадают.
Тест: `test_fit_uses_the_same_situations_and_river`.
Признаки 4800 ситуаций считаются в C++ на всех ядрах (`exact_feature_many`). Облако, 4 потока: подбор exact-16 на 1200 ситуациях за 26 с.
Для 4800 флоп-ситуаций на ПК выйдет порядка 30–60 с (оценка).
Обучение: `train_blueprint.py --buckets-kind potential --exact-features ...`. Если файл корзин тега уже есть, флаг сверяется с ним.

## Таблица корзин для exact
Самый быстрый путь: один раз посчитать признаки всех классов (на ПК оценка ~1.5–2 мин на флоп + тёрн, 16 потоков):
```
python -c "from negpluribus.fast import core; c=core(); c.build_exact_features(3, 10, 16, 'data/exact_features_flop_b10.bin'); c.build_exact_features(4, 10, 16, 'data/exact_features_turn_b10.bin')"
python scripts/build_bucket_table.py --buckets data/buckets_<tag>.json --features-dir data
```
Флоп и тёрн из файла собираются за секунды, **для любого k** (одни и те же файлы для 16/64/256). Ривер пакетом по бордам, ~20 с.
Облако, exact-16: флоп 0 с, тёрн 1 с, ривер 19 с; `bucket()` совпадает с таблицей на 500 / 3000 / 3000 случайных руках. Без `--features-dir` флоп и тёрн строятся пакетом по бордам (тёрн 63 с на 4 потоках), результат тот же.
Файлы признаков: флоп 23 МБ, тёрн 251 МБ.

Без таблицы обучение с exact-бакетером медленное: флоп-корзина по определению ~70 мс. **Обучать только с `NEGPLURIBUS_BUCKET_TABLES`.**

## Слияние
Базой нужен `opt/next` (в нём ривер по бордам и прочее), то есть сначала `opt/next`, затем `opt/exact-features`.
Пересечения с `core`: `abstraction.h` (PotentialBucketer: флаг, `histogram()`, `assign_counts()`, перенос `exact_feature` сюда),
`bindings.cpp`, `buckettable.h`, `potential.py`, `game.py` (`GameSpec.exact_features`), `trainer.py`, `train_blueprint.py`, `build_bucket_table.py`.
`search.h` не тронут. Свежий срез с `fit_buckets.py` мне не нужен: вызов `make_bucketer(..., exact=True)` совместим с ним без правок,
флаг `exact=True` передаётся через `**kw`. Если основная захочет, могу добавить в её `fit_buckets.py` флаг `--exact`,
но для этого нужен её файл.

## Прунинг
Серия с относительным порогом на двух сидах досчитывается (дуэли идут). Прунинг с полом (решение пользователя) запускаю следом, итог — отдельным файлом.
