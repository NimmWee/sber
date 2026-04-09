# Детектор фактологических галлюцинаций

Этот репозиторий содержит финальный минимальный pipeline для задачи детекции фактологических галлюцинаций по паре:

- `prompt`
- `response`

Модель возвращает:

- основной выход: `hallucination_probability` в диапазоне `[0, 1]`
- при необходимости сервисный бинарный режим через порог `0.3`

Финальный замороженный вариант:

- historical best commit: `d3fa946`
- variant: `baseline_plus_all_specialists`
- historical PR-AUC: `0.6881`

## Как устроено решение

Runtime path остаётся лёгким:

1. локальный GigaChat-like checkpoint
2. один проход по `prompt + response`
3. извлечение uncertainty / structural / internal features
4. лёгкие scorer-ы:
   - baseline
   - numeric
   - entity
   - long-response
5. фиксированный weighted blend

В runtime path:

- нет внешних API
- нет retrieval / RAG
- нет judge pipeline
- не генерируются новые ответы

## Что нужно для работы

### 1. Конфиг модели

Нужно указать путь к локальному checkpoint:

- локально: `configs/token_stat_provider.local.json`
- в Kaggle: `configs/token_stat_provider.kaggle.json`

### 2. Данные для обучения

Для воспроизводимого train path в репозитории уже есть:

- `data/public_seed_facts.jsonl` — текстовые seed-данные
- `data/bench/knowledge_bench_public.csv` — только для overlap-check, не для обучения

### 3. Данные для scoring

Для private scoring нужен CSV:

- `data/bench/knowledge_bench_private.csv`

Ожидаемые колонки:

- `prompt`
- `response`

## Структура репозитория

- `configs/` — конфиги модели
- `data/` — текстовые seed-данные и benchmark-файлы
- `model/` — артефакты обученной модели
- `src/` — основной код
- `scripts/` — команды запуска
- `notebooks/` — пустая директория под ноутбуки

## Основные команды

Все команды ниже запускаются из корня репозитория.

### Установка

```bash
bash scripts/install.sh
```

### Обучение

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

Что делает `train.sh`:

1. собирает текстовый train dataset
2. обучает frozen final path
3. сохраняет артефакты в `model/frozen_best/`

### Скоринг private benchmark

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

По умолчанию:

- вход: `data/bench/knowledge_bench_private.csv`
- выход: `data/bench/knowledge_bench_private_scores.csv`

### Явный скоринг через Python

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --input-path data/bench/knowledge_bench_private.csv \
  --artifact-dir model/frozen_best \
  --output-path data/bench/knowledge_bench_private_scores.csv
```

## Формат результата

Выходной CSV содержит:

- `prompt`
- `response`
- `hallucination_probability`

Если нужен бинарный сервисный режим, используй порог `0.3` уже после получения вероятностей.

Пример:

```python
hallucination = hallucination_probability >= 0.3
```

Где:

- `true` — галлюцинация есть
- `false` — галлюцинации нет

Важно:

- основной output проекта — именно probability
- boolean режим вторичен и зависит от выбранного operational threshold

## Запуск в Kaggle

Если репозиторий расположен в:

- `/kaggle/working/sber`

и checkpoint доступен локально, то рабочий сценарий такой:

```bash
cd /kaggle/working/sber

bash scripts/install.sh

bash scripts/train.sh --config configs/token_stat_provider.kaggle.json

bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/YOUR_DATASET_NAME/knowledge_bench_private.csv \
  --output-path /kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv
```

## Активные файлы запуска

В рабочем минимальном варианте используются:

- `scripts/install.sh`
- `scripts/train.sh`
- `scripts/score_private.sh`
- `scripts/build_text_training_dataset.py`
- `scripts/train_frozen_submission.py`
- `scripts/score_frozen_submission.py`

## Коротко

Чтобы воспроизвести решение:

1. настрой путь к checkpoint в config
2. положи private CSV в `data/bench/knowledge_bench_private.csv`
3. выполни:

```bash
bash scripts/install.sh
bash scripts/train.sh --config configs/token_stat_provider.local.json
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

Итоговый результат появится в:

- `data/bench/knowledge_bench_private_scores.csv`
