# Быстрый старт (специалист по данным): логистическая регрессия за 5 минут

Что вы сделаете:

- сгенерируете небольшой синтетический датасет для классификации
- обучите логистическую регрессию с помощью NextStat
- выведете коэффициенты и базовые метрики (accuracy, log-loss)

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
python docs/quickstarts/code/ds_logistic.py
```

Вывод:

- печатает оценённые коэффициенты и стандартные ошибки
- печатает accuracy и log-loss

Следующие шаги:

- используйте формульный интерфейс: `nextstat.glm.logistic.from_formula(...)`
- добавьте ridge/MAP для стабильности при сепарации: `l2=...`
