# Быстрый старт (биолог): подгонка однокамерной пероральной PK-модели

Что вы сделаете:

- сгенерируете небольшой синтетический PK-датасет (пероральное дозирование, абсорбция первого порядка)
- подгоните параметры методом максимального правдоподобия (MLE)
- сравните предсказания модели с наблюдениями

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
python docs/quickstarts/code/bio_pk_1c_oral.py
```

Вывод:

- печатает оценённые параметры и неопределённости
- записывает JSON-артефакт: `docs/quickstarts/out/bio_pk_1c_oral_result.json`

Следующие шаги:

- попробуйте популяционную PK (NLME): `OneCompartmentOralPkNlmeModel(...)`
