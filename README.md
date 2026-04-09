# Детектор фактических галлюцинаций

Этот репозиторий содержит финальную submission-версию решения для хакатона Sber.

Модель получает на вход:
- `prompt`
- `response`

На выходе возвращается:
- `hallucination_probability` в диапазоне `[0, 1]`

Финальный замороженный вариант:
- historical best commit: `d3fa946`
- variant: `baseline_plus_all_specialists`
- historical PR-AUC: `0.6881`

Сейчас именно этот путь является активным submission path.

## Что лежит в репозитории

- `configs/` — конфиги модели и frozen submission
- `data/bench/` — benchmark CSV-файлы
- `data/public_seed_facts.jsonl` — публичные текстовые seed-данные
- `model/frozen_best/` — артефакты обученной финальной модели
- `src/` — код извлечения признаков, инференса, подготовки данных и frozen scoring path
- `scripts/` — основные команды запуска
- `notebooks/` — пустая директория под ноутбуки, если понадобится

## Важное ограничение

В runtime scoring path:
- не используются внешние API
- не используется retrieval / RAG
- не генерируются новые ответы
- скоринг работает только по готовым парам `prompt + response`

## Откуда берутся данные для обучения

Репозиторий не зависит от анонимных матриц признаков.

Для воспроизводимости сохранены:
- текстовые seed-данные: `data/public_seed_facts.jsonl`
- код сборки текстового датасета: `src/data/textual_dataset.py`
- код препроцессинга текста в признаки: `src/data/textual_preprocessing.py`
- скрипт сборки датасета: `scripts/build_text_training_dataset.py`

Публичный preview benchmark:
- `data/bench/knowledge_bench_public.csv`
- используется только для evaluation-only overlap checks
- не используется как train data

## Что нужно положить вручную

Перед обучением и скорингом нужно:

1. Указать локальный путь к GigaChat checkpoint в:
   - `configs/token_stat_provider.local.json`

2. Для финального скоринга положить private benchmark в:
   - `data/bench/knowledge_bench_private.csv`

Если этих файлов нет, train/score path должен падать с понятным сообщением.

## Ожидаемый формат private benchmark

Входной файл:
- `data/bench/knowledge_bench_private.csv`

Ожидаемые колонки:
- `prompt`
- `response`

Выходной файл:
- `data/bench/knowledge_bench_private_scores.csv`

Колонки в выходе:
- `prompt`
- `response`
- `hallucination_probability`

## Быстрый запуск из корня репозитория

Все команды ниже предполагают, что текущая директория — корень репозитория.

### 1. Установка

```bash
bash scripts/install.sh
```

Если в shell нет команды `python`, но есть другой интерпретатор, можно явно передать его:

```bash
export PYTHON_BIN=/path/to/python
bash scripts/install.sh
```

### 2. Обучение финального frozen path

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

Эта команда:
- использует committed text-based inputs
- собирает текстовый train dataset
- обучает frozen final path
- сохраняет артефакты в `model/frozen_best/`

### 3. Скоринг private benchmark

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

По умолчанию скрипт:
- читает `data/bench/knowledge_bench_private.csv`
- пишет `data/bench/knowledge_bench_private_scores.csv`

Если нужно передать пути явно:

```bash
bash scripts/score_private.sh \
  --config configs/token_stat_provider.local.json \
  --input-path data/bench/knowledge_bench_private.csv \
  --artifact-dir model/frozen_best \
  --output-path data/bench/knowledge_bench_private_scores.csv
```

## Быстрый запуск в Kaggle

В репозитории уже есть готовый committed config для Kaggle:
- `configs/token_stat_provider.kaggle.json`

Если private файл называется `knowledge_bench_private_no_labels.csv`, то из корня репозитория в Kaggle можно запускать так:

```bash
bash scripts/install.sh
bash scripts/train.sh --config configs/token_stat_provider.kaggle.json
bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<YOUR_DATASET_NAME>/knowledge_bench_private_no_labels.csv \
  --output-path data/bench/knowledge_bench_private_scores.csv
```

Если в kaggle shell нет `python`, но есть `python3`, можно явно указать интерпретатор:

```bash
export PYTHON_BIN=python3
bash scripts/install.sh
bash scripts/train.sh --config configs/token_stat_provider.kaggle.json
bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<YOUR_DATASET_NAME>/knowledge_bench_private_no_labels.csv \
  --output-path data/bench/knowledge_bench_private_scores.csv
```

## Прямые Python entrypoints

Если удобнее запускать без shell-обёрток:

### Сборка текстового датасета

```bash
python scripts/build_text_training_dataset.py
```

### Обучение

```bash
python scripts/train_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --dataset-path data/processed/textual_training_dataset.jsonl \
  --artifact-dir model/frozen_best
```

### Скоринг

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --input-path data/bench/knowledge_bench_private.csv \
  --artifact-dir model/frozen_best \
  --output-path data/bench/knowledge_bench_private_scores.csv
```

## Активные скрипты

В финальной структуре используются только:
- `scripts/install.sh`
- `scripts/train.sh`
- `scripts/score_private.sh`
- `scripts/build_text_training_dataset.py`
- `scripts/preprocess_text_training_dataset.py`
- `scripts/train_frozen_submission.py`
- `scripts/score_frozen_submission.py`

## Итог

Если коротко, для воспроизведения решения нужно:

1. Настроить `configs/token_stat_provider.local.json`
2. Положить `data/bench/knowledge_bench_private.csv`
3. Выполнить:

```bash
bash scripts/install.sh
bash scripts/train.sh --config configs/token_stat_provider.local.json
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

После этого появится:
- `data/bench/knowledge_bench_private_scores.csv`
