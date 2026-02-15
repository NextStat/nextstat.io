# GLM Fit Speed Benchmark

Benchmarks NextStat GLM MLE fitting against statsmodels, scikit-learn, and glum
across four families (Linear, Logistic, Poisson, Negative Binomial) and three
dataset sizes (1k, 10k, 100k observations).

## Cases

| Case | Family | N | p | Competitors |
|------|--------|---|---|-------------|
| linear_{1k,10k,100k} | Gaussian | varies | 10 | statsmodels, sklearn |
| logistic_{1k,10k,100k} | Binomial | varies | 10 | statsmodels, sklearn, glum |
| poisson_{1k,10k,100k} | Poisson | varies | 10 | statsmodels, glum |
| negbin_{1k,10k,100k} | NegBin | varies | 10 | statsmodels |

## Parity

Coefficient max abs/rel diff and log-likelihood comparison.

## Running

```bash
python suite.py --out-dir /tmp/bench_blitz_glm --seed 42
```
