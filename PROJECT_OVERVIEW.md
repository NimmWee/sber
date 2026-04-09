# PROJECT OVERVIEW

## Ключевая идея

Мы решаем задачу детекции фактологических галлюцинаций не через внешнюю проверку знаний, а через внутренние сигналы самой модели на паре `prompt + response`.

Это lightweight production-like detector:
- один проход по `prompt + response`
- быстрые uncertainty и internal features
- несколько узких specialist scorer-ов
- простой weighted blend

## Почему это не RAG / не judge / не external verification

Проект:
- не ходит во внешние API
- не делает retrieval
- не подключает Wikipedia / search
- не строит judge pipeline в runtime
- не генерирует новые ответы для проверки

Мы оцениваем уже готовый ответ по его внутренней стабильности и uncertainty-сигналам.

## Почему это быстро

В scoring path нет тяжёлых внешних компонентов.

Основные затраты:
1. один provider forward через локальный checkpoint
2. feature extraction
3. несколько очень лёгких tabular scorer-ов
4. фиксированный blend

Для контроля есть отдельный latency benchmark:
- `scripts/benchmark_latency.py`

## Почему это production-like

- deterministic feature extraction
- локальный runtime без внешних зависимостей
- конфигурируемые пути
- отдельный frozen submission path
- воспроизводимый training pipeline из текстовых данных
- metadata рядом с артефактами

## Почему здесь нет leakage

Train/eval policy жёстко разделена:
- train data: text-based train/dev dataset из committed seed inputs
- internal validation: `dev` split того же training dataset
- public benchmark: только evaluation / overlap-check

Submission path не должен использовать public benchmark как training data или labeled scoring input.

Подробности:
- [docs/DATA_SPLIT_AND_LEAKAGE_POLICY.md](C:\sber\docs\DATA_SPLIT_AND_LEAKAGE_POLICY.md)

## Какие модули дают прирост

Финальный historical best path основан на:
- baseline scorer
- numeric specialist
- entity specialist
- long-response specialist

Именно specialists дали исторический прирост по сравнению с базовой моделью.

## Какие ablations есть

Для внутреннего validation split есть отдельный script:
- `scripts/run_ablation.py`

Он сравнивает:
- baseline only
- baseline + numeric
- baseline + entity
- baseline + long
- full blend

Отчёт сохраняется в `reports/ablation_report.json` и `reports/ablation_report.md`.

## Почему итог — weighted blend

Weighted blend здесь лучше защищается, чем сложный meta-model:
- проще объясняется
- меньше риск переобучения
- легче контролировать latency
- легче воспроизводить

Фиксированные веса финального frozen path задокументированы в:
- [configs/frozen_submission.json](C:\sber\configs\frozen_submission.json)
- [src/submission/frozen_best.py](C:\sber\src\submission\frozen_best.py)

## Что является слабым местом

- модель лучше ранжирует риск, чем делает жёсткую бинарную классификацию
- recall остаётся ограниченной
- latency зависит от локального checkpoint и устройства
- текущий blend — это frozen production-style compromise, а не попытка выжать последний процент через сложную мета-модель

## Что бы масштабировали в production

Если идти дальше после хакатона:
- упаковали бы provider и scorer в единый сервис
- кэшировали бы токенизацию и часть feature extraction
- добавили бы мониторинг распределения score
- аккуратно развивали бы internal validation dataset, не смешивая его с benchmark
