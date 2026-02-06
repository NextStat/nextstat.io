---
title: "ROOT/TRExFitter Parity (HistFactory) — NextStat Validation"
status: draft
---

# ROOT/TRExFitter Parity (HistFactory) — NextStat Validation

Цель: прогнать одни и те же HistFactory модели через **ROOT/HistFactory** (эталон в HEP‑экосистеме) и через **NextStat**, сравнить расхождения и померить скорость.

На практике TRExFitter обычно является *генератором* HistFactory XML + ROOT histograms и/или RooWorkspace, а математика “движка” живёт в ROOT/RooFit/RooStats. Поэтому минимальный “эталонный” контур — это ROOT `hist2workspace` + RooFit профилирование.

## Prerequisites

1) ROOT (с HistFactory/RooStats) доступен в PATH:

```bash
command -v root
command -v hist2workspace
```

2) Python bindings NextStat собраны/установлены:

```bash
cd bindings/ns-py
maturin develop --release
```

3) Для конвертации HistFactory XML ↔ pyhf JSON нужен `uproot`:

```bash
pip install -e "bindings/ns-py[validation]"
```

## Apex2 workflow (Planning → Exploration → Execution → Verification)

Ниже самый воспроизводимый путь, который удобно запускать на кластере (где есть ROOT и TRExFitter).

### Planning (окружение и зависимости)

Минимально нужно:
- `root` + `hist2workspace` в `PATH`
- Python 3 + зависимости для валидации (`pyhf`, `uproot`, и python bindings NextStat)

Рекомендуемая проверка prereqs (быстро, без прогонов):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py --root-prereq-only
```

Если в кластере нет `.venv`, используй любой эквивалентный Python (conda/venv/модуль), но важно:
- `PYTHONPATH=bindings/ns-py/python`
- `pip install -e "bindings/ns-py[validation]"` выполнен в этом env

### Exploration (найти тестовые модели)

Тестовые модели для ROOT/TRExFitter в этом контуре это HistFactory экспорты с `combination.xml`.

Если у тебя есть директория с экспортами TRExFitter (или любыми HistFactory export-ами), можно:

1) Сгенерировать cases JSON (наиболее контролируемо, удобно для CI/архива):

```bash
./.venv/bin/python tests/generate_apex2_root_cases.py \
  --search-dir /abs/path/to/trex/output \
  --out tmp/apex2_root_cases.json \
  --include-fixtures \
  --absolute-paths
```

`name` каждого кейса генерится как относительный путь папки экспорта (от `--search-dir`), чтобы избежать коллизий (в больших наборах часто повторяются одинаковые имена подпапок).

2) Либо не генерировать вручную, а дать директорию прямо мастер-раннеру (см. Execution).

### Execution (прогоны)

#### Вариант A: один master-report (pyhf + ROOT-suite)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --root-search-dir /abs/path/to/trex/output \
  --root-include-fixtures \
  --root-cases-absolute-paths
```

Артефакт:
- `tmp/apex2_master_report.json`

Внутри будет:
- `pyhf.status` (`ok`/`fail`)
- `root.status` (`ok`/`fail`/`skipped`)
- ссылки на `tmp/apex2_pyhf_report.json` и `tmp/apex2_root_suite_report.json`

#### Вариант B: отдельно ROOT-suite (если нужно фокусно)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \
  --cases tmp/apex2_root_cases.json \
  --keep-going \
  --out tmp/apex2_root_suite_report.json
```

#### Вариант C: только pyhf parity + speed (без ROOT)

Этот прогон не требует ROOT/TRExFitter и полезен как быстрый “эталон” на любом окружении:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_pyhf_validation_report.py \
  --out tmp/apex2_pyhf_report.json \
  --sizes 2,16,64,256 \
  --n-random 8 \
  --seed 0
```

Если нужно дополнительно прогнать fit (может быть медленнее):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_pyhf_validation_report.py \
  --out tmp/apex2_pyhf_report.json \
  --fit
```

### Verification (интерпретация и “почему”)

1) Первичный “зеленый/красный” сигнал:
- `pyhf.status == ok` значит NLL/expected_data совпадают с эталоном `pyhf`
- `root.status == ok` значит q(mu) профиль совпал с ROOT в заданных допусках
- `root.status == skipped` значит не было prereqs (например, нет `hist2workspace` или `uproot`)

2) Если ROOT-suite дал `fail`, в `tmp/apex2_root_suite_report.json` для каждого кейса есть:
- `run_dir` (папка с артефактами одного прогона)
- `summary_path`
- `diff.max_abs_dq_mu` и `diff.d_mu_hat`

3) Для разбора расхождений по конкретному `run_dir` (без ROOT) используй:

```bash
./.venv/bin/python tests/explain_root_vs_nextstat_profile_diff.py \
  --run-dir /abs/path/to/tmp/root_parity_suite/<case>/run_<timestamp>
```

4) Для профилировки:
- pyhf speedup смотри в `tmp/apex2_pyhf_report.json` (и в `tmp/apex2_master_report.json` в `pyhf.stdout_tail`/`pyhf.report`)
- ROOT-suite speedup смотри в `tmp/apex2_root_suite_report.json` по ключу `cases[*].perf.speedup_nextstat_vs_root_scan` (это `root_profile_scan_wall / nextstat_profile_scan`)

### Cookbook: примеры для всех вариантов

Ниже набор “копипаст” команд для кластера/локально.

1) Быстрый prereq-check только для ROOT-suite (без прогонов):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py --prereq-only
```

2) Master-report, но ROOT часть только prereq-check (pyhf прогонится полностью):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py --root-prereq-only
```

3) Master-report с автопоиском `combination.xml` (TRExFitter export dir):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --root-search-dir /abs/path/to/trex/output \
  --root-include-fixtures \
  --root-cases-absolute-paths
```

4) Master-report с кастомным glob (если `combination.xml` лежит иначе):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --root-search-dir /abs/path/to/trex/output \
  --root-glob "**/*/combination.xml" \
  --root-cases-absolute-paths
```

5) Master-report с кастомной сеткой mu (влияет на auto-generated cases):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --root-search-dir /abs/path/to/trex/output \
  --root-mu-start 0.0 \
  --root-mu-stop 10.0 \
  --root-mu-points 101
```

6) Master-report с заранее сгенерированным cases JSON:

```bash
./.venv/bin/python tests/generate_apex2_root_cases.py \
  --search-dir /abs/path/to/trex/output \
  --out tmp/apex2_root_cases.json \
  --include-fixtures \
  --absolute-paths
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --root-cases tmp/apex2_root_cases.json \
  --root-out tmp/apex2_root_suite_report.json \
  --out tmp/apex2_master_report.json
```

7) Только ROOT-suite с кастомными thresholds и workdir:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \
  --cases tmp/apex2_root_cases.json \
  --dq-atol 1e-3 \
  --mu-hat-atol 1e-3 \
  --workdir tmp/root_parity_suite \
  --keep-going \
  --out tmp/apex2_root_suite_report.json
```

8) Один ROOT кейс “fail-fast” через Apex2 wrapper (удобно для дебага):

Старт от `pyhf` JSON:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_profile_report.py \
  --pyhf-json tests/fixtures/simple_workspace.json \
  --measurement GaussExample \
  --out tmp/apex2_root_profile_report.json
```

Старт от HistFactory XML (экспорт TRExFitter):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_profile_report.py \
  --histfactory-xml /abs/path/to/combination.xml \
  --rootdir /abs/path/to \
  --out tmp/apex2_root_profile_report.json
```

9) Сгенерировать cases JSON с кастомным glob и сеткой mu (и оставить пути относительными):

```bash
./.venv/bin/python tests/generate_apex2_root_cases.py \
  --search-dir /abs/path/to/trex/output \
  --glob "**/combination.xml" \
  --out tmp/apex2_root_cases.json \
  --start 0.0 --stop 5.0 --points 51
```

10) ROOT-suite без `--cases` (только встроенный smoke fixture):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \
  --out tmp/apex2_root_suite_report.json
```

11) Низкоуровневый прогон одного кейса (пишет полный `summary.json` + артефакты в workdir):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \
  --histfactory-xml /abs/path/to/combination.xml \
  --rootdir /abs/path/to \
  --start 0.0 --stop 5.0 --points 51 \
  --workdir /abs/path/to/scratch/root_parity \
  --keep
```

12) Master-report с кастомными путями для артефактов (удобно на кластере в `$SCRATCH`):

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py \
  --out /abs/path/to/scratch/apex2_master_report.json \
  --pyhf-out /abs/path/to/scratch/apex2_pyhf_report.json \
  --root-out /abs/path/to/scratch/apex2_root_suite_report.json \
  --root-search-dir /abs/path/to/trex/output \
  --root-cases-out /abs/path/to/scratch/apex2_root_cases.json
```

### Cluster job templates (SLURM / PBS / HTCondor)

Ниже шаблоны job-скриптов. Они предполагают shared filesystem (репозиторий доступен на compute nodes) и что Python окружение уже содержит зависимости (`pyhf`, `uproot`) и установленный/собранный NextStat python binding.

Если в кластере ты работаешь через `module`, заменяй строки `module load ...` на ваши реальные имена модулей/версии.

#### SLURM (`sbatch`)

Файл `apex2_root_parity.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=apex2-root-parity
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# Optional: load env
# module purge
# module load python
# module load root

REPO="${SLURM_SUBMIT_DIR}"
TREX_EXPORT_DIR="/abs/path/to/trex/output"

OUTDIR="${SCRATCH:-/tmp}/apex2_nextstat/${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

cd "${REPO}"
export PYTHONPATH="${REPO}/bindings/ns-py/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -V
command -v root || true
command -v hist2workspace || true

# Fast prereq check (records skipped if missing)
python3 tests/apex2_master_report.py \
  --out "${OUTDIR}/apex2_master_report.json" \
  --pyhf-out "${OUTDIR}/apex2_pyhf_report.json" \
  --root-out "${OUTDIR}/apex2_root_suite_report.json" \
  --root-cases-out "${OUTDIR}/apex2_root_cases.json" \
  --root-search-dir "${TREX_EXPORT_DIR}" \
  --root-include-fixtures \
  --root-cases-absolute-paths

echo "Artifacts written to: ${OUTDIR}"
```

Запуск:

```bash
sbatch apex2_root_parity.slurm
```

#### PBS/Torque (`qsub`)

Файл `apex2_root_parity.pbs`:

```bash
#!/usr/bin/env bash
#PBS -N apex2-root-parity
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -j oe

set -euo pipefail

# Optional: load env
# module purge
# module load python
# module load root

REPO="${PBS_O_WORKDIR}"
TREX_EXPORT_DIR="/abs/path/to/trex/output"

OUTDIR="${SCRATCH:-/tmp}/apex2_nextstat/${PBS_JOBID}"
mkdir -p "${OUTDIR}"

cd "${REPO}"
export PYTHONPATH="${REPO}/bindings/ns-py/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -V
command -v root || true
command -v hist2workspace || true

python3 tests/apex2_master_report.py \
  --out "${OUTDIR}/apex2_master_report.json" \
  --pyhf-out "${OUTDIR}/apex2_pyhf_report.json" \
  --root-out "${OUTDIR}/apex2_root_suite_report.json" \
  --root-cases-out "${OUTDIR}/apex2_root_cases.json" \
  --root-search-dir "${TREX_EXPORT_DIR}" \
  --root-include-fixtures \
  --root-cases-absolute-paths

echo "Artifacts written to: ${OUTDIR}"
```

Запуск:

```bash
qsub apex2_root_parity.pbs
```

#### HTCondor

HTCondor обычно требует `.sub` файл и исполняемый скрипт. Ниже shared-filesystem вариант (репо уже доступно на worker node).

Файл `apex2_root_parity.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="/abs/path/to/nextstat.io"
TREX_EXPORT_DIR="/abs/path/to/trex/output"

OUTDIR="${SCRATCH:-/tmp}/apex2_nextstat/${CLUSTER:-condor}.${PROCESS:-0}"
mkdir -p "${OUTDIR}"

cd "${REPO}"
export PYTHONPATH="${REPO}/bindings/ns-py/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 tests/apex2_master_report.py \
  --out "${OUTDIR}/apex2_master_report.json" \
  --pyhf-out "${OUTDIR}/apex2_pyhf_report.json" \
  --root-out "${OUTDIR}/apex2_root_suite_report.json" \
  --root-cases-out "${OUTDIR}/apex2_root_cases.json" \
  --root-search-dir "${TREX_EXPORT_DIR}" \
  --root-include-fixtures \
  --root-cases-absolute-paths

echo "Artifacts written to: ${OUTDIR}"
```

Файл `apex2_root_parity.sub`:

```ini
universe = vanilla
executable = apex2_root_parity.sh
output = condor.$(Cluster).$(Process).out
error = condor.$(Cluster).$(Process).err
log = condor.$(Cluster).log
request_cpus = 4
request_memory = 8GB

queue 1
```

Запуск:

```bash
chmod +x apex2_root_parity.sh
condor_submit apex2_root_parity.sub
```

## 1) Проверка профилирования q(mu) vs ROOT

### Вариант A: стартуем от pyhf JSON (fixtures)

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \
  --pyhf-json tests/fixtures/simple_workspace.json \
  --measurement GaussExample \
  --start 0.0 --stop 5.0 --points 51
```

Скрипт:
- экспортирует workspace в HistFactory XML + `data.root` через `pyhf.writexml`
- строит RooWorkspace через `hist2workspace`
- делает free fit и fixed‑POI fits в ROOT → q(mu)
- делает `nextstat.infer.profile_scan` на той же сетке mu
- печатает JSON summary и пишет артефакты в `tmp/root_parity/...`

### Вариант B: стартуем от HistFactory Combination XML (например, экспорт TRExFitter)

Если у тебя есть `combination.xml`, который ссылается на XML каналов и ROOT histograms (часто `data.root`), можно прогнать так:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \
  --histfactory-xml /abs/path/to/combination.xml \
  --start 0.0 --stop 5.0 --points 51
```

Опция `--rootdir` нужна только если в XML относительные пути должны резолвиться не от папки с `combination.xml`.

## 2) Что считать “совпадением”

Ожидаемые источники отличий ROOT vs pyhf/NextStat:
- разные минимизаторы/стратегии и критерии остановки
- разные дефолтные ограничения/параметризации (особенно на границах)
- нюансы включения константных членов в NLL (offsets/normalization)

Рекомендуемая метрика на первом проходе:
- `mu_hat` (best fit POI)
- `max_abs_dq_mu` по сетке mu (q(mu) разница)
- стабильность статуса минимизации (ROOT status codes)

## 3) Performance / profiling

`tests/validate_root_profile_scan.py` печатает wall‑time для:
- `hist2workspace` (построение RooWorkspace)
- ROOT profile scan
- NextStat profile scan

Для честной профилировки “движка” обычно отдельно сравнивают:
- время *построения модели* (парсинг/инициализация)
- время *одного NLL eval*
- время *одного fit* и *скана из N фиксированных fit’ов*

Следующий шаг — добавить отдельный бенч “NLL eval в ROOT vs NextStat” и “fit time”, но сначала важно зафиксировать паритет математики на q(mu).
