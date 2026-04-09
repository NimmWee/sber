# KAGGLE VALIDATION

Этот документ описывает, как проверить проект в Kaggle без локальных Windows-путей и без ручного гадания по структуре каталогов.

## Базовые пути в Kaggle

- рабочая копия репозитория: `/kaggle/working/sber`
- входные датасеты Kaggle: `/kaggle/input/<dataset-name>/...`
- локальный checkpoint модели: `/kaggle/temp/GigaChat3`
- отчёты проекта: `/kaggle/working/sber/reports`

Для запуска из Kaggle используйте:
- `configs/token_stat_provider.kaggle.json`

## Smoke-check

Минимальный сценарий нужен, чтобы быстро проверить, что репозиторий поднимается и scoring path работает.

```bash
cd /kaggle/working/sber
bash scripts/install.sh
python -m unittest discover tests -v
bash scripts/train.sh --config configs/token_stat_provider.kaggle.json
bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --output-path /kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv
```

Проверяемые артефакты:
- `/kaggle/working/sber/model/frozen_best/`
- `/kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv`

## Full validation

Полный сценарий для демонстрации включает scoring, latency benchmark и внутренний ablation.

```bash
cd /kaggle/working/sber
bash scripts/install.sh
python -m unittest discover tests -v
bash scripts/train.sh --config configs/token_stat_provider.kaggle.json

bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --output-path /kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv

python scripts/benchmark_latency.py \
  --config configs/token_stat_provider.kaggle.json \
  --dataset-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --artifact-dir /kaggle/working/sber/model/frozen_best \
  --report-dir /kaggle/working/sber/reports \
  --max-samples 32

python scripts/run_ablation.py \
  --config configs/token_stat_provider.kaggle.json \
  --dataset-path /kaggle/working/sber/data/processed/textual_training_dataset.jsonl \
  --artifact-dir /kaggle/working/sber/model/frozen_best \
  --report-dir /kaggle/working/sber/reports
```

Ожидаемые артефакты:
- `/kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv`
- `/kaggle/working/sber/reports/latency_report.json`
- `/kaggle/working/sber/reports/latency_report.md`
- `/kaggle/working/sber/reports/ablation_report.json`
- `/kaggle/working/sber/reports/ablation_report.md`

## Как запускать tests

```bash
cd /kaggle/working/sber
python -m unittest discover tests -v
```

Если нужен только узкий sanity check:

```bash
cd /kaggle/working/sber
python -m unittest tests.unit.test_submission_policy tests.unit.test_score_frozen_submission_cli tests.unit.test_reporting_scripts tests.unit.test_kaggle_paths_and_docs -v
```

## Как запускать scoring

Основной режим для соревнования — probability output:

```bash
cd /kaggle/working/sber
bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --output-path /kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv
```

Если нужен boolean serving mode:

```bash
cd /kaggle/working/sber
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --output-path /kaggle/working/sber/data/bench/knowledge_bench_private_boolean.csv \
  --output-mode boolean \
  --label-threshold 0.3
```

## Как запускать benchmark_latency.py

```bash
cd /kaggle/working/sber
python scripts/benchmark_latency.py \
  --config configs/token_stat_provider.kaggle.json \
  --dataset-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --artifact-dir /kaggle/working/sber/model/frozen_best \
  --report-dir /kaggle/working/sber/reports \
  --max-samples 32
```

## Как запускать run_ablation.py

```bash
cd /kaggle/working/sber
python scripts/run_ablation.py \
  --config configs/token_stat_provider.kaggle.json \
  --dataset-path /kaggle/working/sber/data/processed/textual_training_dataset.jsonl \
  --artifact-dir /kaggle/working/sber/model/frozen_best \
  --report-dir /kaggle/working/sber/reports
```

## Понятные причины падения в Kaggle

Если в Kaggle отсутствует config:
- передайте `--config configs/token_stat_provider.kaggle.json`

Если отсутствует checkpoint:
- смонтируйте или заранее положите модель в `/kaggle/temp/GigaChat3`

Если отсутствует private dataset:
- проверьте путь в `/kaggle/input/<dataset-name>/...`

Если отсутствует processed dataset для ablation:
- сначала выполните `bash scripts/train.sh --config configs/token_stat_provider.kaggle.json`

## Что не делает Kaggle validation

- не использует external API
- не генерирует новые ответы в scoring path
- не использует public benchmark для fit/tuning
- не модифицирует prompt/response во входном private CSV
