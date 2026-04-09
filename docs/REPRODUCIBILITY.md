# REPRODUCIBILITY

## Что нужно для воспроизведения

1. Репозиторий
2. Локальный GigaChat-like checkpoint
3. Конфиг провайдера
4. Private benchmark CSV

## Какие данные уже committed

- `data/public_seed_facts.jsonl`
- `data/bench/knowledge_bench_public.csv`

Этого достаточно, чтобы воспроизвести text-based training pipeline и overlap checks.

## Что нужно положить вручную

### Локальный checkpoint

Укажи путь к локальному checkpoint в:
- `configs/token_stat_provider.local.json`

### Private benchmark

Положи private benchmark в:
- `data/bench/knowledge_bench_private.csv`

## Какие команды запускать

### Install

```bash
bash scripts/install.sh
```

### Train

```bash
bash scripts/train.sh --config configs/token_stat_provider.local.json
```

### Score

```bash
bash scripts/score_private.sh --config configs/token_stat_provider.local.json
```

## Что логируется

Training metadata сохраняет:
- timestamp
- training_run_id
- git commit hash, если доступен
- версии библиотек
- blend metadata
- random seed

## Как проект ведёт себя при ошибках

Если отсутствует:
- training dataset
- private benchmark
- checkpoint
- frozen artifacts

пользователь должен получить понятную actionable ошибку, а не немой traceback без контекста.
