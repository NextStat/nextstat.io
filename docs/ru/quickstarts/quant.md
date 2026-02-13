# Быстрый старт (количественный аналитик): фильтр Калмана + EM-подгонка + прогноз

Что вы сделаете:

- смоделируете зашумлённый временной ряд AR(1)
- подгоните шумы процесса/измерений (Q/R) методом EM
- построите краткосрочный прогноз

## Установка

```bash
python -m pip install /path/to/nextstat-*.whl
```

Режим разработки из репозитория (без установки):

```bash
cd /path/to/nextstat.io
export PYTHONPATH=bindings/ns-py/python
```

## Запуск

```bash
python docs/quickstarts/code/quant_kalman_ar1.py
```

Вывод:

- печатает подогнанные Q/R и сводку сходимости EM
- записывает JSON-артефакт: `docs/quickstarts/out/quant_kalman_ar1_result.json`

Следующие шаги:

- используйте `nextstat.timeseries.local_level_model(...)` для трекинга локального уровня
- поддержка пропущенных наблюдений — вставьте `None` в ряд `ys`
