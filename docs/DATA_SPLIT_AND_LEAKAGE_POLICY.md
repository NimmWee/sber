# DATA SPLIT AND LEAKAGE POLICY

## Что является train data

Train data в этом проекте строится из:
- `data/public_seed_facts.jsonl`
- `scripts/build_text_training_dataset.py`
- `src/data/textual_dataset.py`

После сборки получается text-based train/dev dataset:
- `train`
- `dev`

Этот датасет используется для:
- обучения scorer-ов
- внутренней валидации

## Что является internal validation

Internal validation — это `dev` split text-based training dataset.

Он используется для:
- внутренней проверки качества
- ablation report
- контроля, что specialists дают вклад на internal split

## Что является public benchmark

Public benchmark:
- `data/bench/knowledge_bench_public.csv`

Он используется только для:
- evaluation
- overlap-check
- sanity-check

## Что public benchmark НЕ делает

Public benchmark не должен использоваться для:
- fit
- обучения head-ов
- настройки threshold
- настройки blend weights
- feature selection для final submission path
- specialist tuning для финального решения

## Какие проверки это гарантируют

В проект добавлены защитные ограничения:

1. Training path падает, если ему передали `knowledge_bench_public.csv` как train dataset.
2. Preprocess path падает, если ему передали `knowledge_bench_public.csv` как training dataset.
3. Submission scoring path требует unlabeled input и падает, если в scoring input есть:
   - `label`
   - `is_hallucination`
   - `target`
4. Build dataset script использует public benchmark только для overlap checks, а не для fit.

## Почему это важно

На защите главный вопрос будет звучать примерно так:

> А вы точно не подгоняли решение под public benchmark?

Текущая политика проекта отвечает на этот вопрос так:
- public benchmark отделён от train pipeline
- training и internal validation идут только на text-based train/dev dataset
- submission scoring path не должен принимать labeled public benchmark как вход

Это делает решение заметно более прозрачным и защищаемым.
