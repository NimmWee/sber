# SOLUTION CARD

## Задача

Детекция фактологических галлюцинаций по паре:
- `prompt`
- `response`

## Вход

- `prompt`
- `response`

## Выход

Основной:
- `hallucination_probability`

Дополнительный:
- `hallucination = true/false`

## Используемые сигналы

- token uncertainty
- entropy
- top1-top2 margin
- structural features
- compact internal features
- specialist features для чисел, сущностей и длинных ответов

## Модель

- baseline lightweight head
- numeric specialist head
- entity specialist head
- long-response specialist head
- фиксированный weighted blend

## Ограничения

- без внешних API
- без retrieval / RAG
- без runtime judge pipeline
- без обучения на public benchmark

## Почему решение соответствует правилам

- scoring идёт только по готовому `prompt + response`
- нет повторной генерации
- нет внешней фактологической проверки
- public benchmark отделён от training pipeline

## Что не используется

- external verification
- search
- Wikipedia lookup
- multi-pass generation
- heavy meta-model

## Latency mindset

Решение строится как lightweight detector:
- один provider forward
- дешёвые признаки
- лёгкие scorer-ы
- простой blend

Отдельный latency benchmark:
- `scripts/benchmark_latency.py`

## Риски

- recall ограничена
- итоговое качество зависит от локального checkpoint
- latency нужно подтверждать benchmark-отчётом на целевом железе

## Future work

- улучшение recall без отказа от lightweight path
- дополнительный production instrumentation
- расширение internal train/dev dataset без leakage
