# Детектор фактологических галлюцинаций

Репозиторий содержит компактное решение для задачи детекции фактологических галлюцинаций по паре:

- `prompt`
- `response`

На выходе модель возвращает:

- основной режим: `hallucination_probability` в диапазоне `[0, 1]`
- сервисный режим: `hallucination = true/false` при пороге `0.3`

Финальный замороженный вариант, который используется в проекте:

- historical best commit: `d3fa946`
- variant: `baseline_plus_all_specialists`
- historical PR-AUC: `0.6881`

## Как мы решаем задачу

Здесь не используется внешний fact-checking и не строится judge pipeline.

Подход intentionally production-like:

1. локальный GigaChat-like checkpoint
2. один проход по `prompt + response`
3. извлечение быстрых признаков:
   - token uncertainty
   - entropy
   - top1-top2 margin
   - structural features
   - compact internal features
4. несколько лёгких scorer-ов:
   - baseline
   - numeric specialist
   - entity specialist
   - long-response specialist
5. фиксированный weighted blend

В runtime path:

- нет внешних API
- нет RAG
- нет multi-pass generation
- не генерируются новые ответы

## Что реализовано

### 1. Построение текстового train dataset

Обучение не опирается на анонимные готовые числовые признаки.

В репозитории есть текстовые входы и код сборки:

- `data/public_seed_facts.jsonl`
- `scripts/build_text_training_dataset.py`
- `src/data/textual_dataset.py`

Именно из этих текстовых данных строится тренировочный JSONL-датасет.

### 2. Frozen training path

Скрипт `train.sh`:

1. собирает текстовый датасет
2. обучает frozen final path
3. сохраняет артефакты в `model/frozen_best/`

### 3. Private scoring

Скрипт `score_private.sh` принимает готовый CSV с `prompt + response` и создаёт результат без изменения самих ответов.

### 4. Kaggle-ready запуск

В репозитории есть отдельный конфиг:

- `configs/token_stat_provider.kaggle.json`

Он рассчитан на checkpoint по пути:

- `/kaggle/temp/GigaChat3`

Если модель лежит в другом месте, достаточно поправить `checkpoint_path`.

## Структура репозитория

- `configs/` — конфиги загрузки локального checkpoint
- `data/` — seed-данные и benchmark-файлы
- `model/` — артефакты обученной модели
- `src/` — основной код инференса, признаков и scoring head
- `scripts/` — основные entrypoint-скрипты
- `notebooks/` — директория под ноутбуки

## Какие файлы важны

### Конфиги

- `configs/token_stat_provider.json` — базовый шаблон
- `configs/token_stat_provider.kaggle.json` — запуск в Kaggle
- `configs/token_stat_provider.local.json` — локальный запуск

### Данные

- `data/public_seed_facts.jsonl` — публичные seed-факты для train pipeline
- `data/bench/knowledge_bench_public.csv` — только для overlap-check / sanity-check
- `data/bench/knowledge_bench_private_no_labels.csv` — пример private-style файла без разметки

### Основные скрипты

- `scripts/install.sh`
- `scripts/train.sh`
- `scripts/score_private.sh`
- `scripts/build_text_training_dataset.py`
- `scripts/train_frozen_submission.py`
- `scripts/score_frozen_submission.py`

## Поддерживаемые форматы

### Вход для scoring

CSV с колонками:

- `prompt`
- `response`

Дополнительно поддерживается колонка:

- `model_answer` вместо `response`

### Основной выход

CSV с колонками:

- `prompt`
- `response`
- `hallucination_probability`

### Сервисный boolean-режим

CSV с колонками:

- `prompt`
- `response`
- `hallucination`

Где:

- `true` — галлюцинация есть
- `false` — галлюцинации нет

Важно:

- для соревнования и анализа основной режим — именно `hallucination_probability`
- boolean-режим нужен только как operational layer поверх вероятности

## Запуск локально

Все команды ниже выполняются из корня репозитория.

### 1. Установка

```bash
bash scripts/install.sh
```

### 2. Обучение

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

### 3. Скоринг private CSV

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

По умолчанию используются пути:

- вход: `data/bench/knowledge_bench_private.csv`
- выход: `data/bench/knowledge_bench_private_scores.csv`

### 4. Явный вызов scoring через Python

Probability mode:

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --artifact-dir model/frozen_best \
  --input-path data/bench/knowledge_bench_private.csv \
  --output-path data/bench/knowledge_bench_private_scores.csv \
  --output-mode probability
```

Boolean mode:

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --artifact-dir model/frozen_best \
  --input-path data/bench/knowledge_bench_private.csv \
  --output-path data/bench/knowledge_bench_private_scores_boolean.csv \
  --output-mode boolean \
  --label-threshold 0.3
```

## Запуск в Kaggle

Если репозиторий лежит в:

- `/kaggle/working/sber`

и checkpoint модели находится по пути:

- `/kaggle/temp/GigaChat3`

то рабочий сценарий такой:

```bash
cd /kaggle/working/sber

bash scripts/install.sh

bash scripts/train.sh --config configs/token_stat_provider.kaggle.json

python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.kaggle.json \
  --artifact-dir model/frozen_best \
  --input-path data/bench/knowledge_bench_private_no_labels.csv \
  --output-path data/bench/knowledge_bench_private_scores.csv \
  --output-mode probability
```

Для boolean-режима в Kaggle:

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.kaggle.json \
  --artifact-dir model/frozen_best \
  --input-path data/bench/knowledge_bench_private_no_labels.csv \
  --output-path data/bench/knowledge_bench_private_scores_boolean.csv \
  --output-mode boolean \
  --label-threshold 0.3
```

## Почему выбран именно такой подход

Мы не делали тяжёлый verifier или judge pipeline.

Причины простые:

- такой runtime труднее удержать в latency-ограничении
- сложнее объяснять архитектуру на защите
- выше риск внешней зависимости
- сложнее воспроизводимость

Текущий подход проще защищать:

- один проход
- детерминированные признаки
- дешёвые scorer-ы
- прозрачный blend

## Практические ограничения

- нужно иметь локальный checkpoint модели
- scoring path не работает без предварительного `train.sh`
- boolean-порог `0.3` — это сервисный threshold, а не соревновательная метрика
- historical PR-AUC `0.6881` относится к frozen best variant, а не к любому последующему cleanup-коммиту

## Коротко

Если нужно воспроизвести решение с нуля:

1. настрой `checkpoint_path` в конфиге
2. запусти установку:

```bash
bash scripts/install.sh
```

3. обучи frozen path:

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

4. проскорь private CSV:

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

Итоговый результат появится в:

- `data/bench/knowledge_bench_private_scores.csv`
