# NextStat-NLP: Automated Extraction of Statistical Inputs from Clinical Text

## Abstract (outline)

- Problem: manual transcription from literature to statistical models is error-prone
- Solution: GLiNER2-based NER pipeline that extracts survival data, informative priors, and dosing regimens from clinical text
- Key result: end-to-end pipeline text -> validated schema -> NextStat inference
- Contribution: reproducible audit trail, heuristic fallback for CI, benchmarked accuracy

## 1. Introduction

- Gap: no standardised tooling for text-to-stats in pharma/clinical workflows
- Motivation: informative priors from literature, survival ETL from case reports, regimen extraction from protocols
- Scope: NextStat-NLP as optional plugin for NextStat statistical engine

## 2. Methods

### 2.1 Architecture

- Backend protocol: GLiNER2 (torch), GLiNER2 (ONNX Runtime), GLiNER2 (MLX/Metal on Apple Silicon), Heuristic
- Schema-driven extraction: label set defines what to extract
- Deterministic parsing layer: regex-based time/event/numeric normalisation
- Validation layer: type checks, range checks, consistency

### 2.2 Extraction pipelines

| Pipeline | Input | Output | Schema labels |
|----------|-------|--------|---------------|
| Survival | Clinical case text | `SurvivalDataset` | subject_id, time, event, age, dose, sex, stage, ecog_ps |
| Priors | Literature abstract | `PriorBundle` | parameter_name, distribution, mean, sd, units, constraint |
| Regimens | Protocol text | `RegimenTable` | dose_amount, dose_unit, route, frequency, start_time, duration |

### 2.3 Provenance and reproducibility

- `ProvenanceBundle`: model version, runtime versions, input hashes, timestamp
- All outputs frozen dataclasses with `text_hash` for audit

## 3. Results

### Table 1: Extraction quality (heuristic baseline)

| Pipeline | Golden set | Precision | Recall | F1 |
|----------|-----------|-----------|--------|----|
| Survival | 5 oncology records | TBD | TBD | TBD |
| Priors | 5 PK parameters | TBD | TBD | TBD |
| Regimens | 3 dosing records | TBD | TBD | TBD |

### Table 2: End-to-end timing

| Pipeline | Backend | Cold start (ms) | Warm (ms) |
|----------|---------|-----------------|-----------|
| Survival | heuristic | TBD | TBD |
| Survival | gliner2 | TBD | TBD |
| Survival | onnx | TBD | TBD |
| Survival | mlx | TBD | TBD |

### Table 3: End-to-end: text -> NextStat inference

| Model | Text source | Extraction time | Inference time | Total |
|-------|-------------|-----------------|----------------|-------|
| Cox PH | 5 oncology records | TBD | TBD | TBD |
| NLME FOCE | PK priors + regimens | TBD | TBD | TBD |

## 4. Discussion

- Heuristic backend: useful for CI, limited for real text
- GLiNER2: general-purpose NER, no fine-tuning needed for structured clinical text
- Limitations: ambiguous event/censoring, multi-line records, non-English text
- Future: fine-tuned models, active learning, multi-language support

## 5. Conclusion

- NextStat-NLP bridges the gap between clinical literature and statistical inference
- Reproducible, auditable, and extensible via backend protocol

## References

- GLiNER2: [citation TBD]
- NextStat: [citation TBD]
