# NextStat Model Specification: HistFactory Probability Densities and Modifiers

**Version:** 0.1.0 (Draft)
**Authors:** NextStat Contributors
**Date:** February 2026

---

## Abstract

This document provides the mathematical specification of the probability model implemented in NextStat. The model follows the HistFactory framework [1] as realized in the pyhf JSON workspace format [2, 3]. We define the full likelihood function, all supported modifier types with their interpolation formulas, constraint terms, and the construction of expected event rates. This specification serves as the reference for the implementation in `ns-translate` and the validation against pyhf.

---

## 1. Introduction

NextStat implements HistFactory-style binned likelihood models as specified in [1] and realized in the pyhf JSON format [2]. The probability model describes the expected number of events in bins of histograms across one or more analysis channels, parameterized by a signal strength parameter $\mu$ and nuisance parameters $\boldsymbol{\theta}$ encoding systematic uncertainties.

This document defines:

- The structure of the full likelihood (Section 2)
- The construction of expected event rates from nominal templates and modifiers (Section 3)
- Each modifier type with its mathematical formula (Section 4)
- Interpolation schemes (Section 5)
- Constraint terms (Section 6)
- The complete likelihood formula (Section 7)
- Asimov dataset construction (Section 8)

All formulas are consistent with the pyhf implementation (NumPy backend) and validated by the deterministic parity test suite.

## 2. Probability Model Structure

### 2.1 Channels and Bins

A HistFactory model consists of a set of *channels* $\{c_1, c_2, \ldots\}$, each containing a histogram with $B_c$ bins. The bins across all channels are concatenated into a single vector of length $N = \sum_c B_c$.

For each bin $i$, we observe $n_i$ events and predict $\lambda_i(\mu, \boldsymbol{\theta})$ expected events.

### 2.2 Samples

Each channel contains one or more *samples* (signal, background components). The expected count in bin $i$ is the sum over all samples $s$ in the channel containing bin $i$:

$$\lambda_i(\mu, \boldsymbol{\theta}) = \sum_{s \in \text{channel}(i)} \lambda_{s,i}(\mu, \boldsymbol{\theta})$$

### 2.3 Parameters

The model parameters $\boldsymbol{\phi} = (\mu, \boldsymbol{\theta})$ comprise:

- **Parameters of interest (POI):** typically the signal strength $\mu$.
- **Nuisance parameters:** systematic uncertainty parameters $\boldsymbol{\theta}$, which may be constrained (with associated auxiliary measurements) or unconstrained.

Each parameter has:

| Attribute | Description |
|-----------|-------------|
| Name | Unique identifier |
| Initial value | Starting point for optimization |
| Bounds | $(a, b)$ box constraints |
| Constrained | Whether an auxiliary constraint term exists |
| Constraint center $\theta_0$ | Central value of the constraint |
| Constraint width $\sigma$ | Width of the constraint |

### 2.4 Likelihood Factorization

The full likelihood factorizes into three components:

$$\mathcal{L}(\mu, \boldsymbol{\theta}) = \underbrace{\mathcal{L}_\text{main}(\mu, \boldsymbol{\theta})}_{\text{Poisson bins}} \times \underbrace{\mathcal{L}_\text{aux}^\text{Poisson}(\boldsymbol{\theta})}_{\text{ShapeSys}} \times \underbrace{\mathcal{L}_\text{aux}^\text{Gauss}(\boldsymbol{\theta})}_{\text{Gaussian constraints}}$$

## 3. Expected Event Rates

### 3.1 Rate Construction

For each sample $s$ in a channel, the expected count in bin $i$ is computed from the nominal template $\nu_{s,i}$ by applying modifiers:

$$\lambda_{s,i}(\boldsymbol{\phi}) = \left(\nu_{s,i} + \Delta_{s,i}^\text{add}(\boldsymbol{\phi})\right) \times \prod_{m \in \text{mult}} \kappa_{m,s,i}(\boldsymbol{\phi})$$

where:

- $\nu_{s,i}$ is the nominal (unmodified) expected count from the sample template.
- $\Delta_{s,i}^\text{add}$ is the sum of additive modifier contributions (HistoSys).
- $\kappa_{m,s,i}$ is the multiplicative factor from modifier $m$.

### 3.2 Modifier Application Order

Modifiers are applied in the order they appear in the pyhf workspace specification. The implementation processes all modifiers for each sample in a single pass:

1. Start with nominal rate $\nu_{s,i}$.
2. Apply multiplicative modifiers (NormFactor, NormSys, ShapeSys, StatError, ShapeFactor, Lumi) as factors.
3. Apply additive modifiers (HistoSys) as increments.

### 3.3 Rate Clamping

To prevent numerical issues (division by zero, logarithm of non-positive values), expected rates are clamped:

$$\lambda_i \gets \max(\lambda_i, \epsilon), \quad \epsilon = 10^{-10}$$

## 4. Modifier Types

### 4.1 NormFactor

**Purpose:** Free-floating overall normalization of a sample. Typically used for the POI.

**Parameters:** One parameter $\mu$.

**Effect:** Multiplicative on all bins of the sample:

$$\kappa_{s,i} = \mu$$

**Properties:**

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(0, 10)$ |
| Constrained | No |

### 4.2 NormSys

**Purpose:** Log-normal systematic uncertainty on sample normalization. Encodes the effect of a systematic source that scales all bins of a sample up or down.

**Parameters:** One parameter $\alpha$ (standard deviations from nominal).

**Input data:** Per-modifier up/down scale factors $(h, l)$ (e.g., $h = 1.05$, $l = 0.95$ for a 5% uncertainty).

**Effect:** Multiplicative on all bins:

$$\kappa_{s,i} = f_\text{code4}(\alpha; h, l)$$

where $f_\text{code4}$ is the pyhf "code 4" interpolation (see Section 5.1).

**Properties:**

| Attribute | Value |
|-----------|-------|
| Initial | $0.0$ |
| Bounds | $(-5, 5)$ |
| Constrained | Yes (Gaussian) |
| $\theta_0$ | $0.0$ |
| $\sigma$ | $1.0$ |

### 4.3 HistoSys

**Purpose:** Shape (bin-by-bin) systematic uncertainty. Encodes the effect of a systematic source that deforms the histogram shape.

**Parameters:** One parameter $\alpha$.

**Input data:** Per-bin up and down templates $h_i^\text{data}$, $l_i^\text{data}$.

**Effect:** Additive per bin:

$$\Delta_{s,i} = f_\text{code4p}(\alpha; l_i^\text{data}, \nu_{s,i}, h_i^\text{data})$$

where $f_\text{code4p}$ is the pyhf "code 4p" piecewise interpolation (see Section 5.2).

**Properties:**

| Attribute | Value |
|-----------|-------|
| Initial | $0.0$ |
| Bounds | $(-5, 5)$ |
| Constrained | Yes (Gaussian) |
| $\theta_0$ | $0.0$ |
| $\sigma$ | $1.0$ |

### 4.4 ShapeSys (Barlow-Beeston)

**Purpose:** Per-bin multiplicative modifier with Poisson auxiliary constraint. Models statistical uncertainty in background estimates from control regions or simulation.

**Parameters:** One parameter $\gamma_i$ per bin (vector-valued modifier).

**Input data:** Per-bin uncertainties $\sigma_i$.

**Effect:** Multiplicative per bin:

$$\kappa_{s,i} = \gamma_i$$

**Auxiliary construction:** For each bin, a pseudo-count is computed:

$$\tau_i = \left(\frac{\nu_{s,i}}{\sigma_i}\right)^2$$

The auxiliary constraint is Poisson (see Section 6.2).

**Properties (per bin):**

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(10^{-10}, 10)$ |
| Constrained | No (Poisson auxiliary instead) |

### 4.5 StatError

**Purpose:** Per-bin multiplicative modifier with Gaussian constraint. Models the MC statistical uncertainty, aggregated across all samples that declare StatError in a channel.

**Parameters:** One parameter $\gamma_i$ per bin, shared across all samples in the channel that have a StatError modifier.

**Input data:** Per-sample, per-bin absolute uncertainties $\sigma_{s,i}$.

**Effect:** Multiplicative per bin:

$$\kappa_{s,i} = \gamma_i$$

**Aggregation:** When multiple samples in a channel declare StatError, the relative uncertainty is computed as:

$$\sigma_i^\text{rel} = \frac{\sqrt{\sum_s \sigma_{s,i}^2}}{\sum_s \nu_{s,i}}$$

where the sums are over all samples in the channel with StatError.

**Properties (per bin):**

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(10^{-10}, 10)$ |
| Constrained | Yes (Gaussian) |
| $\theta_0$ | $1.0$ |
| $\sigma$ | $\sigma_i^\text{rel}$ (computed) |

### 4.6 ShapeFactor

**Purpose:** Free-floating per-bin normalization. Used when the shape of a background component is determined entirely by data (no MC template).

**Parameters:** One parameter $\gamma_i$ per bin.

**Effect:** Multiplicative per bin:

$$\kappa_{s,i} = \gamma_i$$

**Properties (per bin):**

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(0, 10)$ |
| Constrained | No |

### 4.7 Lumi

**Purpose:** Luminosity uncertainty. Scales all bins of all samples uniformly.

**Parameters:** One shared parameter $\theta_\text{lumi}$.

**Input data:** Relative luminosity uncertainty $\delta_L$ (e.g., $0.02$ for 2%).

**Effect:** Multiplicative on all bins:

$$\kappa_{s,i} = \theta_\text{lumi}$$

**Properties:**

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(0, 10)$ |
| Constrained | Yes (Gaussian) |
| $\theta_0$ | $1.0$ |
| $\sigma$ | $\delta_L$ |

### 4.8 Modifier Summary

| Modifier | Scope | Type | Parameters | Constrained |
|----------|-------|------|------------|-------------|
| NormFactor | Sample-wide | Multiplicative | 1 | No |
| NormSys | Sample-wide | Multiplicative | 1 | Yes (Gaussian) |
| HistoSys | Per-bin | Additive | 1 | Yes (Gaussian) |
| ShapeSys | Per-bin | Multiplicative | $B$ | Poisson auxiliary |
| StatError | Per-bin | Multiplicative | $B$ | Yes (Gaussian) |
| ShapeFactor | Per-bin | Multiplicative | $B$ | No |
| Lumi | Sample-wide | Multiplicative | 1 | Yes (Gaussian) |

## 5. Interpolation Codes

### 5.1 Code 4: Exponential Interpolation (NormSys)

The pyhf "code 4" interpolation smoothly connects logarithmic (exponential) behavior at $|\alpha| \geq 1$ with a degree-6 polynomial for $|\alpha| < 1$.

**Given:** up-factor $h$ and down-factor $l$ (e.g., $h = 1.05$, $l = 0.95$).

**For $|\alpha| \geq 1$** (exponential regime):

$$f_\text{code4}(\alpha) = \begin{cases} h^{|\alpha|} = \exp(|\alpha| \ln h) & \text{if } \alpha \geq 1 \\ l^{|\alpha|} = \exp(|\alpha| \ln l) & \text{if } \alpha \leq -1 \end{cases}$$

**For $|\alpha| < 1$** (polynomial regime):

$$f_\text{code4}(\alpha) = 1 + \sum_{j=1}^{6} a_j \alpha^j$$

The coefficients $\mathbf{a} = (a_1, \ldots, a_6)$ are determined by matching the function value and first two derivatives at $\alpha = \pm 1$. They are computed as:

$$\mathbf{a} = A^{-1} \mathbf{b}$$

where $\mathbf{b} = (h - 1,\; l - 1,\; h\ln h,\; -l\ln l,\; h(\ln h)^2,\; l(\ln l)^2)^T$ and $A^{-1}$ is the fixed $6 \times 6$ matrix:

$$A^{-1} = \begin{pmatrix}
 \frac{1}{2} &  -\frac{1}{2} &  \frac{1}{2} &  \frac{1}{2} & -\frac{1}{8} & -\frac{1}{8} \\
 \frac{3}{2} &  \frac{3}{2}  & -\frac{5}{4} & -\frac{5}{4} &  \frac{3}{8} &  \frac{3}{8} \\
 0            &  0             &  \frac{1}{4} & -\frac{1}{4} &  0           &  0 \\
-\frac{5}{2} & -\frac{5}{2}  &  \frac{7}{4} &  \frac{7}{4} & -\frac{1}{2} & -\frac{1}{2} \\
 0            &  0             & -\frac{1}{4} &  \frac{1}{4} &  0           &  0 \\
 1            &  1             & -\frac{3}{4} & -\frac{3}{4} &  \frac{1}{8} &  \frac{1}{8}
\end{pmatrix}$$

**Properties:** $f_\text{code4}(0) = 1$, $f_\text{code4}(1) = h$, $f_\text{code4}(-1) = l$, and the function and its first two derivatives are continuous everywhere.

### 5.2 Code 4p: Piecewise Polynomial Interpolation (HistoSys)

The pyhf "code 4p" interpolation for histogram shape variations uses a piecewise function that is cubic for $|\alpha| \leq 1$ and linear for $|\alpha| > 1$.

**Given:** per-bin values from the down template $l_i$, nominal $\nu_i$, and up template $h_i$.

**Derived quantities:**

$$S_i = \frac{1}{2}(h_i - l_i) \quad \text{(symmetric slope)}$$

$$A_i = \frac{1}{16}(h_i - 2\nu_i + l_i) \quad \text{(asymmetry)}$$

**For $|\alpha| \leq 1$** (polynomial regime):

$$\Delta_i(\alpha) = S_i \cdot \alpha + A_i \cdot (3\alpha^2 - 10) \cdot \alpha^2 \cdot \frac{1}{(\text{normalization})}$$

More precisely, following the pyhf implementation:

$$\Delta_i(\alpha) = S_i \cdot \alpha + A_i \cdot p(\alpha)$$

where $p(\alpha) = 24\alpha^4 - 40\alpha^2 + 15$ (ensuring $p(\pm 1) = \pm 1$, $p(0) = 0$, and smooth transitions).

**For $\alpha > 1$** (linear extrapolation):

$$\Delta_i(\alpha) = (h_i - \nu_i) \cdot \alpha$$

**For $\alpha < -1$** (linear extrapolation):

$$\Delta_i(\alpha) = (\nu_i - l_i) \cdot \alpha$$

**Properties:** $\Delta_i(0) = 0$, $\Delta_i(1) = h_i - \nu_i$, $\Delta_i(-1) = -(\ \nu_i - l_i)$, continuous with continuous first derivative.

## 6. Constraint Terms

### 6.1 Gaussian Constraints

For a constrained parameter $\theta$ with constraint center $\theta_0$ and width $\sigma$:

$$-\ln\mathcal{L}_\text{Gauss}(\theta) = \frac{1}{2}\left(\frac{\theta - \theta_0}{\sigma}\right)^2 + \ln\sigma + \frac{1}{2}\ln 2\pi$$

This corresponds to the auxiliary measurement $\theta_0 \sim \mathcal{N}(\theta, \sigma)$.

**Applies to:** NormSys, HistoSys, StatError, Lumi.

**Constant terms.** The terms $\ln\sigma + \frac{1}{2}\ln 2\pi$ are independent of $\theta$ and do not affect the fit. However, they are included for absolute NLL parity with pyhf and are precomputed in the `constraint_const` cache.

### 6.2 Poisson Auxiliary Constraints (ShapeSys / Barlow-Beeston)

For each ShapeSys bin with parameter $\gamma_i$ and pseudo-count $\tau_i$:

**Auxiliary expected count:**

$$\lambda_i^\text{aux} = \gamma_i \cdot \tau_i$$

**Auxiliary observed count:**

- Data mode: $n_i^\text{aux} = \tau_i$
- Asimov mode: $n_i^\text{aux} = \hat{\gamma}_i \cdot \tau_i$ (where $\hat{\gamma}_i$ is the fitted value)

**Constraint contribution:**

$$-\ln\mathcal{L}_\text{Pois,aux}(\gamma_i) = \gamma_i \tau_i - n_i^\text{aux} \ln(\gamma_i \tau_i) + \ln\Gamma(n_i^\text{aux} + 1)$$

### 6.3 Constraint Summary

| Modifier | Constraint Type | $\theta_0$ | $\sigma$ |
|----------|----------------|------------|----------|
| NormSys | Gaussian | $0.0$ | $1.0$ |
| HistoSys | Gaussian | $0.0$ | $1.0$ |
| StatError | Gaussian | $1.0$ | $\sigma_i^\text{rel}$ (computed) |
| Lumi | Gaussian | $1.0$ | $\delta_L$ (specified) |
| ShapeSys | Poisson | — | — |
| NormFactor | None | — | — |
| ShapeFactor | None | — | — |

## 7. Full Likelihood

### 7.1 Negative Log-Likelihood

Combining all terms, the full NLL is:

$$\text{nll}(\boldsymbol{\phi}) = \underbrace{\sum_{i=1}^{N}\left[\lambda_i(\boldsymbol{\phi}) - n_i \ln\lambda_i(\boldsymbol{\phi}) + \ln\Gamma(n_i + 1)\right]}_{\text{main Poisson terms}} + \underbrace{\sum_{k \in \text{ShapeSys}} \left[\gamma_k \tau_k - n_k^\text{aux} \ln(\gamma_k \tau_k) + \ln\Gamma(n_k^\text{aux} + 1)\right]}_{\text{Poisson auxiliary (Barlow-Beeston)}} + \underbrace{\sum_{j \in \text{Gauss}} \left[\frac{1}{2}\left(\frac{\theta_j - \theta_{0,j}}{\sigma_j}\right)^2 + \ln\sigma_j + \frac{1}{2}\ln 2\pi\right]}_{\text{Gaussian constraints}}$$

### 7.2 Canonical Quantities

For compatibility with pyhf:

$$\text{logpdf}(\boldsymbol{\phi}) = -\text{nll}(\boldsymbol{\phi})$$

$$\text{twice\_nll}(\boldsymbol{\phi}) = 2 \cdot \text{nll}(\boldsymbol{\phi}) = -2 \cdot \text{logpdf}(\boldsymbol{\phi})$$

### 7.3 Sparse Optimization

When $n_i = 0$, the term $-n_i \ln\lambda_i = 0$ regardless of $\lambda_i$. The implementation uses an observation mask to skip these terms, reducing computation for histograms with many empty bins.

### 7.4 SIMD Evaluation

The main Poisson NLL sum is evaluated using `f64x4` SIMD lanes (4 bins per vector operation). Operations: multiply, subtract, add, reduce. The natural logarithm is extracted to scalar `f64::ln()` for bit-exact accuracy ($\texttt{wide::f64x4::ln()}$ has $\sim$1000 ULP error).

## 8. Asimov Dataset Construction

The Asimov dataset [4] provides the expected data under a given hypothesis, used for computing expected test statistics and the Brazil band.

### 8.1 Background-Only Asimov ($\mu = 0$)

1. **Conditional fit:** Minimize NLL with $\mu$ fixed to $0$, obtaining fitted nuisance parameters $\hat{\boldsymbol{\theta}}_0$.

2. **Expected main data:** Evaluate the model at $(\mu = 0, \hat{\boldsymbol{\theta}}_0)$:

$$n_i^\text{Asimov} = \lambda_i(0, \hat{\boldsymbol{\theta}}_0)$$

3. **Expected auxiliary data (Gaussian):** Replace constraint centers with fitted values:

$$\theta_{0,j}^\text{Asimov} = \hat{\theta}_{0,j}$$

4. **Expected auxiliary data (Poisson/ShapeSys):** Replace auxiliary observations:

$$n_k^{\text{aux, Asimov}} = \hat{\gamma}_k \cdot \tau_k$$

### 8.2 Properties

The Asimov dataset has the property that the MLE on it returns the parameters used to generate it (to within numerical precision). This makes it suitable for computing the median expected test statistic without Monte Carlo sampling.

## 9. pyhf JSON Workspace Format

NextStat reads the pyhf JSON workspace format. A workspace contains:

```json
{
  "channels": [
    {
      "name": "SR",
      "samples": [
        {
          "name": "signal",
          "data": [10.0, 5.0],
          "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null}
          ]
        },
        {
          "name": "background",
          "data": [50.0, 40.0],
          "modifiers": [
            {"name": "bkg_norm", "type": "normsys",
             "data": {"hi": 1.1, "lo": 0.9}}
          ]
        }
      ]
    }
  ],
  "observations": [
    {"name": "SR", "data": [55.0, 48.0]}
  ],
  "measurements": [
    {
      "name": "meas",
      "config": {
        "poi": "mu",
        "parameters": []
      }
    }
  ],
  "version": "1.0.0"
}
```

The translation from JSON to the internal `HistFactoryModel` is handled by `ns-translate`, which:

1. Parses the JSON via `serde`.
2. Resolves channel-sample-modifier relationships.
3. Precomputes interpolation coefficients, $\tau$ values, and constraint constants.
4. Constructs the `PreparedModel` with cached data for efficient evaluation.

## References

[1] K. Cranmer, G. Lewis, L. Moneta, A. Shibata, W. Verkerke, "HistFactory: A tool for creating statistical models for use with RooFit and RooStats," CERN-OPEN-2012-016, 2012.

[2] L. Heinrich, M. Feickert, G. Stark, K. Cranmer, "pyhf: pure-Python implementation of HistFactory statistical models," JOSS 6(58), 2823, 2021.

[3] ATLAS Collaboration, "Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods," ATL-PHYS-PUB-2019-029, 2019.

[4] G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics," Eur. Phys. J. C71, 1554, 2011; erratum ibid. C73, 2501, 2013.
