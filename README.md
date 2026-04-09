# Детектор фактологических галлюцинаций

## 1. Что решает проект

Проект оценивает пару:
- `prompt`
- `response`

и выдаёт:
- основной выход для соревнования: `hallucination_probability` в `[0, 1]`
- дополнительный сервисный режим: `hallucination = true/false`

Главная метрика проекта:
- `PR-AUC`

## 2. Почему архитектура соответствует условиям задачи

Финальный submission path:
- использует локальный GigaChat-like checkpoint
- делает один проход по `prompt + response`
- не использует внешние API
- не использует retrieval / RAG
- не использует judge pipeline в runtime
- не генерирует новый ответ на этапе scoring

Основные сигналы:
- token uncertainty
- entropy
- top1-top2 margin
- structural features
- compact internal features
- lightweight specialists:
  - baseline
  - numeric
  - entity
  - long-response

Финальный score строится как фиксированный `weighted blend`.

## 3. Как устроен pipeline

### Training
1. Из committed text-based seed данных строится train/dev датасет
2. Локальная модель извлекает сигналы из текста
3. Обучаются лёгкие LightGBM-головы
4. Артефакты сохраняются в `model/frozen_best/`

### Scoring
1. На вход подаётся готовый CSV с `prompt + response`
2. Модель извлекает сигналы
3. Считаются scorer-ы
4. Формируется `hallucination_probability`
5. При необходимости probability преобразуется в `true/false` через serving threshold

## 4. Что используется на train

Используется только текстовый train/dev pipeline:
- `data/public_seed_facts.jsonl`
- `scripts/build_text_training_dataset.py`
- `src/data/textual_dataset.py`
- `src/data/textual_preprocessing.py`

Public benchmark не является train data.

## 5. Что используется только на eval

Файл:
- `data/bench/knowledge_bench_public.csv`

используется только для:
- overlap checks
- sanity-check
- evaluation/reporting

Он не используется для:
- `fit`
- threshold tuning
- blend tuning
- feature selection в submission path

Подробности:
- [docs/DATA_SPLIT_AND_LEAKAGE_POLICY.md](C:\sber\docs\DATA_SPLIT_AND_LEAKAGE_POLICY.md)

## 6. Как запустить training

### Установка

```bash
bash scripts/install.sh
```

### Обучение frozen финального решения

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

### Запуск в Kaggle

Рекомендуемая рабочая папка репозитория в Kaggle:
- `/kaggle/working/sber`

Рекомендуемые входные данные:
- private benchmark: `/kaggle/input/<dataset-name>/knowledge_bench_private.csv`
- локальный checkpoint: `/kaggle/temp/GigaChat3`

Пример полного train/scoring запуска в Kaggle:

```bash
cd /kaggle/working/sber
bash scripts/install.sh
bash scripts/train.sh --config configs/token_stat_provider.kaggle.json
bash scripts/score_private.sh \
  --config configs/token_stat_provider.kaggle.json \
  --input-path /kaggle/input/<dataset-name>/knowledge_bench_private.csv \
  --output-path /kaggle/working/sber/data/bench/knowledge_bench_private_scores.csv
```

Аналог через Python:

```bash
python scripts/train_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --dataset-path data/processed/textual_training_dataset.jsonl \
  --artifact-dir model/frozen_best
```

## 7. Как запустить scoring

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

По умолчанию скрипт читает:
- `data/bench/knowledge_bench_private.csv`

и пишет:
- `data/bench/knowledge_bench_private_scores.csv`

## 8. Как получить probability output

Probability mode — основной submission mode и режим по умолчанию:

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --output-mode probability
```

Выходные колонки:
- `prompt`
- `response`
- `hallucination_probability`

Почему именно probability — основной режим:
- PR-AUC зависит от ранжирования score
- threshold не влияет на PR-AUC
- probability output лучше отражает качество модели для соревнования

## 9. Как получить boolean output

Boolean mode — это только дополнительный operational режим:

```bash
python scripts/score_frozen_submission.py \
  --config configs/token_stat_provider.local.json \
  --output-mode boolean \
  --label-threshold 0.3
```

Выходные колонки:
- `prompt`
- `response`
- `hallucination`

Где:
- `true` — галлюцинация есть
- `false` — галлюцинации нет

Важно:
- threshold используется только для serving mode
- threshold не является основной целью оптимизации
- threshold не влияет на PR-AUC

## 10. Latency benchmark

```bash
python scripts/benchmark_latency.py \
  --config configs/token_stat_provider.local.json \
  --dataset-path data/bench/knowledge_bench_private.csv \
  --artifact-dir model/frozen_best \
  --report-dir reports \
  --max-samples 32
```

Скрипт сохраняет:
- `reports/latency_report.json`
- `reports/latency_report.md`

В Kaggle при запуске из `/kaggle/working/sber` отчёты по умолчанию будут лежать в:
- `/kaggle/working/sber/reports/latency_report.json`
- `/kaggle/working/sber/reports/latency_report.md`

## 11. Ablation report

```bash
python scripts/run_ablation.py \
  --config configs/token_stat_provider.local.json \
  --dataset-path data/processed/textual_training_dataset.jsonl \
  --artifact-dir model/frozen_best \
  --report-dir reports
```

Скрипт сохраняет:
- `reports/ablation_report.json`
- `reports/ablation_report.md`

В Kaggle при запуске из `/kaggle/working/sber` отчёты по умолчанию будут лежать в:
- `/kaggle/working/sber/reports/ablation_report.json`
- `/kaggle/working/sber/reports/ablation_report.md`

## 12. Активный frozen best variant

Финальный submission candidate:
- historical best commit: `d3fa946`
- variant: `baseline_plus_all_specialists`
- historical PR-AUC: `0.6881`

Submission path оформлен в:
- [src/submission/frozen_best.py](C:\sber\src\submission\frozen_best.py)

## 13. Ограничения

- scoring path не должен использовать внешние API
- scoring path не должен генерировать новый ответ
- public benchmark не должен использоваться как training data
- boolean mode не должен подменять probability mode как соревновательный режим
- latency нужно проверять отдельно benchmark-скриптом

## 14. Полезные документы

- [PROJECT_OVERVIEW.md](C:\sber\PROJECT_OVERVIEW.md)
- [docs/DATA_SPLIT_AND_LEAKAGE_POLICY.md](C:\sber\docs\DATA_SPLIT_AND_LEAKAGE_POLICY.md)
- [docs/DEFENSE_TALK_TRACK.md](C:\sber\docs\DEFENSE_TALK_TRACK.md)
- [docs/KAGGLE_VALIDATION.md](C:\sber\docs\KAGGLE_VALIDATION.md)
- [docs/REPRODUCIBILITY.md](C:\sber\docs\REPRODUCIBILITY.md)
- [docs/SOLUTION_CARD.md](C:\sber\docs\SOLUTION_CARD.md)
