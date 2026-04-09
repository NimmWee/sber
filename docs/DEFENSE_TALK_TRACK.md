# DEFENSE TALK TRACK

## 1. Проблема

Мы решаем задачу детекции фактологических галлюцинаций в ответах LLM по паре `prompt + response`.

На входе уже есть готовый ответ модели, а на выходе нужен score риска галлюцинации.

## 2. Ограничения

Условия задачи не позволяют строить решение на:
- внешних API
- retrieval / RAG
- heavy judge pipeline
- обучении на public benchmark

Поэтому мы сознательно выбрали lightweight detector.

## 3. Идея решения

Мы не проверяем факт внешним источником.
Мы оцениваем, насколько сам ответ выглядит фактологически нестабильным с точки зрения внутренних сигналов модели.

## 4. Почему решение подходит под latency

В scoring path:
- один проход локальной модели
- feature extraction
- несколько лёгких scorer-ов
- простой weighted blend

Нет повторной генерации, нет retrieval, нет тяжёлых проверок.

## 5. Почему решение не нарушает правила

- нет внешних API
- нет runtime judge pipeline
- нет обучения на public benchmark
- public benchmark используется только для evaluation/overlap-check

## 6. Как устроены признаки

Мы используем:
- token uncertainty
- entropy
- margin
- structural features
- compact internal features

Это дешёвые deterministic сигналы, которые можно стабильно воспроизводить.

## 7. Зачем нужны specialists

Базовый scorer ловит общую неуверенность.

Specialists добавляют более целевой сигнал:
- numeric specialist — ошибки в числах и датах
- entity specialist — ошибки в именах и сущностях
- long-response specialist — ошибки в длинных ответах

## 8. Почему итог — weighted blend

Weighted blend:
- проще объяснить
- проще воспроизвести
- проще контролировать по latency
- меньше риск, чем у сложной meta-model

## 9. Какие слабые места

- recall ограничена
- модель сильнее как ranking detector, чем как жёсткий binary filter
- latency зависит от конкретного checkpoint и железа

## 10. Как бы масштабировали в production

- сервис с единым cached provider
- мониторинг распределения score
- стабильный internal validation dataset
- постепенное расширение train data без benchmark leakage
