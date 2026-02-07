# NextStat Model Specification: HistFactory Probability Densities and Modifiers

**Version:** 0.2.0 (Draft)
**Authors:** NextStat Contributors
**Date:** February 2026

---

## Abstract

This document provides the mathematical specification of the probability model implemented in NextStat. The model follows the HistFactory framework [1] as realized in the pyhf JSON workspace format [2, 3]. We define the full likelihood function, all supported modifier types with their interpolation formulas, constraint terms, and the construction of expected event rates. This specification serves as the canonical reference for the implementation in `ns-translate` and the validation against pyhf.

---

## 1. Introduction

NextStat implements HistFactory-style binned likelihood models as specified in [1] and realized in the pyhf JSON format [2]. The probability model describes the expected number of events in bins of histograms across one or more analysis channels, parameterized by a signal strength parameter $\mu$ and nuisance parameters $\boldsymbol{\alpha}$, $\boldsymbol{\gamma}$ encoding systematic uncertainties.

This document defines:

- The likelihood template and index conventions (Section 2)
- The construction of expected event rates from nominal templates and modifiers (Section 3)
- Each modifier type with its mathematical formula (Section 4)
- All interpolation schemes with their properties (Section 5)
- Constraint terms with consistent Bayesian and frequentist interpretations (Section 6)
- The complete likelihood formula (Section 7)
- Asimov dataset construction (Section 8)
- The pyhf JSON workspace format (Section 9)

All formulas are consistent with the pyhf implementation (NumPy backend, InterpCode=4 for normalization, piecewise linear for shape) and validated by the deterministic parity test suite.

## 2. The Likelihood Template

### 2.1 Index Conventions

Following [1], we use the following mnemonic index conventions:

- $b \in \text{bins}$
- $c \in \text{channels}$
- $s \in \text{samples}$
- $p \in \text{parameters}$

A **channel** is a region of the data defined by a corresponding event selection; channels have disjoint event selections. A **sample** is a set of scattering processes that can be added together incoherently.

### 2.2 Parameter Subsets

We define the following subsets of parameters:

- $\mathbb{N} = \{\phi_p\}$: unconstrained normalization factors (NormFactor)
- $\mathbb{S} = \{\alpha_p\}$: parameters associated with systematic uncertainties that have external constraints (OverallSys and HistoSys)
- $\boldsymbol{\Gamma} = \{\gamma_{cb}\}$: bin-by-bin uncertainties with constraints (StatError, ShapeSys — but *not* those associated with unconstrained ShapeFactor)

### 2.3 The Template

The parametrized probability density function is of the form:

$$\mathcal{P}(n_{cb}, a_p \mid \phi_p, \alpha_p, \gamma_b) = \prod_{c \in \text{channels}} \prod_{b \in \text{bins}} \text{Pois}(n_{cb} \mid \nu_{cb}) \cdot G(L_0 \mid \lambda, \Delta_L) \cdot \prod_{p \in \mathbb{S}+\boldsymbol{\Gamma}} f_p(a_p \mid \alpha_p)$$

where $f_p(a_p \mid \alpha_p)$ is a constraint term describing an auxiliary measurement $a_p$ that constrains the nuisance parameter $\alpha_p$.

### 2.4 Expected Event Rate

The expected (mean) number of events in a given bin is:

$$\nu_{cb}(\phi_p, \alpha_p, \gamma_b) = \lambda_{cs} \; \gamma_{cb} \; \phi_{cs}(\boldsymbol{\alpha}) \; \eta_{cs}(\boldsymbol{\alpha}) \; \sigma_{csb}(\boldsymbol{\alpha})$$

where:

- $\lambda_{cs}$: luminosity parameter for a given channel and sample. For samples with luminosity uncertainty (`NormalizeByTheory=True`), this is a common luminosity parameter. For data-driven samples, $\lambda_{cs} = L_0$ (fixed).
- $\gamma_{cb}$: bin-by-bin scale factor for statistical uncertainties (StatError, ShapeSys) and data-driven shape extrapolations (ShapeFactor). For statistical errors, $\gamma_{cb}$ is shared across all samples in the channel.
- $\phi_{cs}$: product of unconstrained normalization factors for a given sample within a given channel, typically including the POI:

$$\phi_{cs} = \prod_{p \in \mathbb{N}_c} \phi_p$$

- $\eta_{cs}(\boldsymbol{\alpha})$: parametrized normalization uncertainties (OverallSys) for a given sample within a given channel (a factor around 1).
- $\sigma_{csb}(\boldsymbol{\alpha})$: parametrized histogram (the nominal histogram modified by HistoSys) for a given sample within a given channel.

### 2.5 Building Blocks

The conceptual building blocks of the HistFactory model are organized as follows (cf. Table 1 of [1]):

|  | Constrained | Unconstrained |
|--|------------|---------------|
| **Normalization Variation** | OverallSys ($\eta_{cs}$) | NormFactor ($\phi_p$) |
| **Coherent Shape Variation** | HistoSys ($\sigma_{csb}$) | — |
| **Bin-by-bin Variation** | ShapeSys & StatError ($\gamma_{cb}$) | ShapeFactor ($\gamma_{csb}$) |

## 3. Expected Event Rates

### 3.1 Rate Construction

For each sample $s$ in a channel $c$, the expected count in bin $b$ is computed from the nominal template $\nu^0_{csb}$ by applying modifiers:

$$\nu_{cb} = \sum_{s \in \text{samples}_c} \lambda_{cs} \; \gamma_{cb} \; \phi_{cs} \; \eta_{cs}(\boldsymbol{\alpha}) \; \sigma_{csb}(\boldsymbol{\alpha})$$

In the pyhf JSON realization, the implementation applies modifiers sequentially:

1. Start with the nominal rate $\nu^0_{csb}$.
2. Apply multiplicative modifiers: NormFactor, NormSys, ShapeSys, StatError, ShapeFactor, Lumi.
3. Apply additive modifiers: HistoSys (as a delta to the parametrized histogram $\sigma_{csb}$).
4. Sum over all samples in the channel.

### 3.2 Rate Clamping

To prevent numerical issues (division by zero, logarithm of non-positive values), expected rates are clamped:

$$\nu_{cb} \gets \max(\nu_{cb}, \epsilon), \quad \epsilon = 10^{-10}$$

## 4. Modifier Types

### 4.1 NormFactor (Unconstrained Normalization)

**Purpose:** Free-floating overall normalization of a sample. Typically used for the POI.

**Parameters:** One parameter $\phi_p$ contributing to $\phi_{cs} = \prod_p \phi_p$.

**Effect:** Multiplicative on all bins of the sample.

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(0, 10)$ |
| Constrained | No |

### 4.2 OverallSys / NormSys (Constrained Normalization)

**Purpose:** Systematic uncertainty on sample normalization. The effect of a systematic source (e.g., jet energy scale) on the overall normalization of a given sample.

**Parameters:** One parameter $\alpha_p$ per systematic source.

**Input data:** Per-modifier up/down scale factors $(\eta^+_{ps}, \eta^-_{ps})$ representing the effect when $\alpha_p = +1$ and $\alpha_p = -1$ respectively. The nominal value is $\eta^0_{ps} = 1$.

**Note:** The distinction between the sign of the source $\alpha$ and the effect $\eta$ allows for anti-correlated systematics: $\eta^+ > 1$ does not necessarily follow from $\alpha = +1$.

**Effect:** The net normalization factor for sample $s$ is:

$$\eta_s(\boldsymbol{\alpha}) = \prod_{p \in \text{Syst}} I(\alpha_p; 1, \eta^+_{ps}, \eta^-_{ps})$$

where $I$ is the interpolation function (see Section 5). NextStat uses InterpCode=4 (polynomial interpolation, exponential extrapolation), matching the pyhf default.

| Attribute | Value |
|-----------|-------|
| Initial | $0.0$ |
| Bounds | $(-5, 5)$ |
| Constrained | Yes (Gaussian) |
| $\alpha^0_p$ | $0.0$ |
| $\sigma_p$ | $1.0$ |

### 4.3 HistoSys (Constrained Shape Variation)

**Purpose:** Bin-by-bin systematic uncertainty. Encodes the effect of a systematic source on the shape and normalization of a histogram.

**Parameters:** One parameter $\alpha_p$ (shared with other HistoSys/OverallSys for the same source).

**Input data:** Per-bin variational histograms $\sigma^+_{psb}$ and $\sigma^-_{psb}$ representing the effect when $\alpha_p = +1$ and $\alpha_p = -1$ respectively. The nominal histogram is $\sigma^0_{sb}$.

**Effect:** The parametrized histogram is:

$$\sigma_{sb}(\boldsymbol{\alpha}) = \sigma^0_{sb} + \sum_{p \in \text{Syst}} I_\text{shape}(\alpha_p; \sigma^0_{sb}, \sigma^+_{psb}, \sigma^-_{psb})$$

where $I_\text{shape}$ is the shape interpolation function (piecewise linear by default; see Section 5).

| Attribute | Value |
|-----------|-------|
| Initial | $0.0$ |
| Bounds | $(-5, 5)$ |
| Constrained | Yes (Gaussian) |
| $\alpha^0_p$ | $0.0$ |
| $\sigma_p$ | $1.0$ |

### 4.4 ShapeSys (Barlow-Beeston)

**Purpose:** Per-bin multiplicative modifier with Poisson auxiliary constraint. Models statistical uncertainty from Monte Carlo simulation when the histograms are sparsely populated [4].

**Parameters:** One parameter $\gamma_b$ per bin.

**Input data:** Per-bin uncertainties $\delta_b$.

**Effect:** Multiplicative per bin:

$$\nu_b \to \nu_b(\boldsymbol{\alpha}) + \gamma_b \; \nu^\text{MC}_b(\boldsymbol{\alpha})$$

The factor $\gamma_b$ reflects that the true rate may differ from the MC estimate $\nu^\text{MC}_b$ by some amount.

**Auxiliary construction.** If the total statistical uncertainty is $\delta_b$, then the pseudo-count is:

$$\tau_b = \left(\frac{\nu^\text{MC}_b}{\delta_b}\right)^2$$

Treating the MC estimate as an auxiliary measurement, the constraint is:

$$\text{Pois}(n^\text{aux}_b \mid \gamma_b \tau_b)$$

where $n^\text{aux}_b = \tau_b$ in nominal data mode and $n^\text{aux}_b = \hat\gamma_b \tau_b$ in Asimov mode.

**Analytic conditional MLE.** Following [1] (Eq. 10–13), the conditional maximum likelihood estimate $\hat{\hat{\gamma}}_b(\boldsymbol{\alpha})$ can be solved analytically:

$$\hat{\hat{\gamma}}_b(\boldsymbol{\alpha}) = \frac{-B + \sqrt{B^2 - 4AC}}{2A}$$

with:

$$A = (\nu^\text{MC}_b)^2 + \tau_b \nu^\text{MC}_b$$

$$B = \nu_b \tau_b + \nu_b \nu^\text{MC}_b - n_b \nu^\text{MC}_b - n^\text{aux}_b \nu^\text{MC}_b$$

$$C = -n^\text{aux}_b \nu_b$$

This analytic solution can be used to speed up fits by eliminating the $\gamma_b$ parameters from the numerical minimization.

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(10^{-10}, 10)$ |
| Constrained | Poisson auxiliary |

### 4.5 StatError (Gaussian-Constrained MC Statistical Uncertainty)

**Purpose:** Per-bin multiplicative modifier with Gaussian constraint. A lighter-weight alternative to the full Barlow-Beeston treatment, where one nuisance parameter per bin is associated with the total MC estimate across all samples.

**Parameters:** One parameter $\gamma_b$ per bin, shared across all samples in the channel that declare StatError.

**Input data:** Per-sample, per-bin absolute uncertainties $\delta_{s,b}$.

**Effect:** Multiplicative per bin: $\nu_{b} \to \gamma_b \cdot \nu_{b}$

**Aggregation.** When multiple samples declare StatError, the relative uncertainty is:

$$\sigma_b^\text{rel} = \frac{\sqrt{\sum_s \delta_{s,b}^2}}{\sum_s \nu^0_{s,b}}$$

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(10^{-10}, 10)$ |
| Constrained | Yes (Gaussian) |
| $\gamma^0_b$ | $1.0$ |
| $\sigma_b$ | $\sigma_b^\text{rel}$ (computed) |

### 4.6 ShapeFactor (Unconstrained Shape)

**Purpose:** Free-floating per-bin normalization, unconstrained. Used when each bin's content is parametrized individually (e.g., data-driven shape extrapolation).

**Parameters:** One parameter $\gamma_{csb}$ per bin per sample.

**Effect:** Multiplicative per bin.

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(0, 10)$ |
| Constrained | No |

### 4.7 Lumi (Luminosity Uncertainty)

**Purpose:** Common normalization uncertainty from luminosity measurement, applied to all theory-normalized samples.

**Parameters:** One shared parameter $\lambda$.

**Constraint:** $G(L_0 \mid \lambda, \Delta_L)$ where $L_0$ is the measured luminosity and $\Delta_L$ is the relative uncertainty.

| Attribute | Value |
|-----------|-------|
| Initial | $1.0$ |
| Bounds | $(0, 10)$ |
| Constrained | Yes (Gaussian) |
| $\lambda^0$ | $1.0$ |
| $\sigma$ | $\Delta_L$ (relative luminosity uncertainty) |

### 4.8 Modifier Summary

| Modifier | HistFactory Name | Scope | Type | Parameters | Constrained |
|----------|-----------------|-------|------|------------|-------------|
| NormFactor | `NormFactor` | Sample-wide | Multiplicative | 1 | No |
| NormSys | `OverallSys` | Sample-wide | Multiplicative | 1 | Yes (Gaussian) |
| HistoSys | `HistoSys` | Per-bin | Additive | 1 | Yes (Gaussian) |
| ShapeSys | `ShapeSys` | Per-bin | Multiplicative | $B$ | Poisson auxiliary |
| StatError | `StatError` | Per-bin | Multiplicative | $B$ | Yes (Gaussian) |
| ShapeFactor | `ShapeFactor` | Per-bin | Multiplicative | $B$ | No |
| Lumi | (measurement) | All samples | Multiplicative | 1 | Yes (Gaussian) |

## 5. Interpolation & Extrapolation

The treatment of systematic uncertainties requires defining interpolation algorithms to produce continuous functions $\eta_s(\boldsymbol{\alpha})$ and $\sigma_{sb}(\boldsymbol{\alpha})$ from the discrete $\pm 1\sigma$ variations. Following [1], we parametrize $\alpha_p$ such that $\alpha_p = 0$ is the nominal value and $\alpha_p = \pm 1$ are the "$\pm 1\sigma$ variations."

Four interpolation strategies are available. NextStat implements Code 0 (piecewise linear) and Code 4/4p (polynomial+exponential), with defaults matching pyhf. Codes 1 and 2 are documented for reference but not yet implemented.

### 5.1 Piecewise Linear (InterpCode=0)

**Normalization:**

$$\eta_s(\boldsymbol{\alpha}) = 1 + \sum_{p \in \text{Syst}} I_\text{lin.}(\alpha_p; 1, \eta^+_{sp}, \eta^-_{sp})$$

**Shape:**

$$\sigma_{sb}(\boldsymbol{\alpha}) = \sigma^0_{sb} + \sum_{p \in \text{Syst}} I_\text{lin.}(\alpha_p; \sigma^0_{sb}, \sigma^+_{psb}, \sigma^-_{psb})$$

with:

$$I_\text{lin.}(\alpha; I^0, I^+, I^-) = \begin{cases} \alpha(I^+ - I^0) & \alpha \geq 0 \\ \alpha(I^0 - I^-) & \alpha < 0 \end{cases}$$

**Pros:** Most straightforward interpolation.

**Cons:** (1) Discontinuous first derivative (kink) at $\alpha = 0$, causing difficulties for gradient-based minimizers like Minuit/L-BFGS. (2) Can extrapolate to negative values (e.g., if $\eta^- = 0.5$, then $\eta(\alpha) < 0$ when $\alpha < -2$).

**Default for:** $\sigma_{sb}(\boldsymbol{\alpha})$ (HistoSys) in pyhf.

### 5.2 Piecewise Exponential (InterpCode=1)

**Normalization:**

$$\eta_s(\boldsymbol{\alpha}) = \prod_{p \in \text{Syst}} I_\text{exp.}(\alpha_p; 1, \eta^+_{sp}, \eta^-_{sp})$$

with:

$$I_\text{exp.}(\alpha; I^0, I^+, I^-) = \begin{cases} (I^+/I^0)^\alpha & \alpha \geq 0 \\ (I^-/I^0)^{-\alpha} & \alpha < 0 \end{cases}$$

**Pros:** (1) Ensures $\eta(\alpha) \geq 0$. (2) For small uncertainties, agrees with linear interpolation near $\alpha \sim 0$.

**Cons:** (1) Discontinuous first derivative at $\alpha = 0$. (2) For large uncertainties, develops asymmetric behavior even when $\eta^+ - 1 = 1 - \eta^-$.

**Note:** When paired with a Gaussian constraint on $\alpha$, this is equivalent to linear interpolation with a log-normal constraint in $\ln(\alpha)$.

**Default for:** $\eta_s(\boldsymbol{\alpha})$ (OverallSys) in ROOT HistFactory.

### 5.3 Quadratic Interpolation + Linear Extrapolation (InterpCode=2)

$$I_\text{quad.|lin.}(\alpha; I^0, I^+, I^-) = \begin{cases} (b + 2a)(\alpha - 1) & \alpha > 1 \\ a\alpha^2 + b\alpha & |\alpha| \leq 1 \\ (b - 2a)(\alpha + 1) & \alpha < -1 \end{cases}$$

with:

$$a = \tfrac{1}{2}(I^+ + I^-) - I^0, \qquad b = \tfrac{1}{2}(I^+ - I^-)$$

**Pros:** Avoids the kink at $\alpha = 0$ (continuous first derivative).

**Cons:** (1) When both up and down variations have the same sign of effect relative to nominal, can produce intermediate values with the opposite sign. (2) Can extrapolate to negative values.

### 5.4 Polynomial Interpolation + Exponential Extrapolation (InterpCode=4)

This is the strategy used by pyhf and NextStat as the default for normalization uncertainties (OverallSys).

$$I_\text{poly|exp.}(\alpha; I^0, I^+, I^-, \alpha_0) = \begin{cases} (I^+/I^0)^\alpha & \alpha \geq \alpha_0 \\ 1 + \sum_{i=1}^{6} a_i \alpha^i & |\alpha| < \alpha_0 \\ (I^-/I^0)^{-\alpha} & \alpha \leq -\alpha_0 \end{cases}$$

where $\alpha_0 = 1$ (default) and the coefficients $a_i$ are fixed by matching the function value, first derivative, and second derivative at $\alpha = \pm \alpha_0$:

$$\mathbf{a} = A^{-1} \mathbf{b}$$

with $\mathbf{b} = (h - 1,\; l - 1,\; h\ln h,\; -l\ln l,\; h(\ln h)^2,\; l(\ln l)^2)^T$ where $h = I^+/I^0$ and $l = I^-/I^0$, and $A^{-1}$ is the fixed $6 \times 6$ matrix:

$$A^{-1} = \begin{pmatrix}
 \frac{1}{2} &  -\frac{1}{2} &  \frac{1}{2} &  \frac{1}{2} & -\frac{1}{8} & -\frac{1}{8} \\
 \frac{3}{2} &  \frac{3}{2}  & -\frac{5}{4} & -\frac{5}{4} &  \frac{3}{8} &  \frac{3}{8} \\
 0            &  0             &  \frac{1}{4} & -\frac{1}{4} &  0           &  0 \\
-\frac{5}{2} & -\frac{5}{2}  &  \frac{7}{4} &  \frac{7}{4} & -\frac{1}{2} & -\frac{1}{2} \\
 0            &  0             & -\frac{1}{4} &  \frac{1}{4} &  0           &  0 \\
 1            &  1             & -\frac{3}{4} & -\frac{3}{4} &  \frac{1}{8} &  \frac{1}{8}
\end{pmatrix}$$

**Pros:** (1) Avoids the kink at $\alpha = 0$ (continuous first and second derivatives). (2) Ensures $\eta(\alpha) \geq 0$.

**Default for:** $\eta_s(\boldsymbol{\alpha})$ (OverallSys) in pyhf and NextStat.

### 5.5 Code 4p: Piecewise Polynomial Shape Interpolation (HistoSys)

For histogram shape variations, pyhf uses a piecewise function that is polynomial for $|\alpha| \leq 1$ and linear for $|\alpha| > 1$.

**Derived quantities per bin:**

$$S_b = \tfrac{1}{2}(\sigma^+_{b} - \sigma^-_{b}) \quad \text{(symmetric slope)}$$

$$A_b = \tfrac{1}{16}(\sigma^+_{b} - 2\sigma^0_{b} + \sigma^-_{b}) \quad \text{(asymmetry)}$$

**Interpolation:**

$$\Delta_b(\alpha) = \begin{cases} (\sigma^+_b - \sigma^0_b) \cdot \alpha & \alpha > 1 \\ S_b \cdot \alpha + A_b \cdot p(\alpha) & |\alpha| \leq 1 \\ (\sigma^0_b - \sigma^-_b) \cdot \alpha & \alpha < -1 \end{cases}$$

where $p(\alpha) = 24\alpha^4 - 40\alpha^2 + 15$ is a polynomial satisfying $p(0) = 15 A_b$, $p(\pm 1) = -1$, with continuous first derivative at $\alpha = \pm 1$.

**Properties:** $\Delta_b(0) = 0$, $\Delta_b(1) = \sigma^+_b - \sigma^0_b$, $\Delta_b(-1) = -(\sigma^0_b - \sigma^-_b)$.

### 5.6 Interpolation Defaults Summary

| Modifier | InterpCode | Accumulation | Default in pyhf |
|----------|-----------|--------------|----------------|
| OverallSys ($\eta$) | 4 (poly+exp) | Multiplicative | Yes |
| HistoSys ($\sigma$) | 0 (piecewise linear) | Additive | Yes |

NextStat matches these defaults exactly for pyhf parity. Both Code 0 and Code 4p are implemented for HistoSys and selectable via `HistoSysInterpCode`.

## 6. Constraint Terms

### 6.1 Consistent Bayesian and Frequentist Modeling

Following [1], we distinguish between the *source* of uncertainty ($\alpha_p$) and its *effect* on rates and shapes ($\eta$, $\sigma$). The "$\pm 1\sigma$ variations" $\eta^\pm_{ps}$ and $\sigma^\pm_{psb}$ describe the effect when the source is at $\alpha_p = \pm 1$.

In the frequentist framework, the auxiliary measurement $a_p$ constrains the nuisance parameter $\alpha_p$ via a constraint term $f_p(a_p \mid \alpha_p)$. In the Bayesian framework, this same term can be paired with a prior $\pi_0(\alpha_p)$ to form a posterior.

The table below summarizes consistent treatments (cf. Table 4 of [1]):

| PDF | Likelihood $\propto$ | Prior $\pi_0$ | Posterior $\pi$ |
|-----|---------------------|---------------|-----------------|
| $G(a_p \mid \alpha_p, \sigma_p)$ | $G(\alpha_p \mid a_p, \sigma_p)$ | $\pi_0(\alpha_p) \propto \text{const}$ | $G(\alpha_p \mid a_p, \sigma_p)$ |
| $\text{Pois}(n_p \mid \tau_p \beta_p)$ | $P_\Gamma(\beta_p \mid A=\tau_p, B=1+n_p)$ | $\pi_0(\beta_p) \propto \text{const}$ | $P_\Gamma(\beta_p \mid A=\tau_p, B=1+n_p)$ |
| $P_\text{LN}(n_p \mid \beta_p, \sigma_p)$ | $\beta_p \cdot P_\text{LN}(\beta_p \mid n_p, \sigma_p)$ | $\pi_0(\beta_p) \propto \text{const}$ | $P_\text{LN}(\beta_p \mid n_p, \sigma_p)$ |
| $P_\text{LN}(n_p \mid \beta_p, \sigma_p)$ | $\beta_p \cdot P_\text{LN}(\beta_p \mid n_p, \sigma_p)$ | $\pi_0(\beta_p) \propto 1/\beta_p$ | $P_\text{LN}(\beta_p \mid n_p, \sigma_p)$ |

### 6.2 Gaussian Constraint

The Gaussian constraint is a good approximation when the maximum likelihood estimate of $\alpha_p$ from the auxiliary measurement has a Gaussian distribution. The global observable $a_p$ can be identified with this MLE estimate.

$$G(a_p \mid \alpha_p, \sigma_p) = \frac{1}{\sqrt{2\pi\sigma_p^2}} \exp\left[-\frac{(a_p - \alpha_p)^2}{2\sigma_p^2}\right]$$

with $\sigma_p = 1$ by default.

**NLL contribution:**

$$-\ln\mathcal{L}_\text{Gauss}(\alpha_p) = \frac{1}{2}\left(\frac{a_p - \alpha_p}{\sigma_p}\right)^2 + \ln\sigma_p + \frac{1}{2}\ln 2\pi$$

**Constant terms.** The terms $\ln\sigma_p + \frac{1}{2}\ln 2\pi$ are independent of $\alpha_p$ and do not affect the fit. However, they are included for absolute NLL parity with pyhf and are precomputed in the `constraint_const` cache.

**Note:** If $\alpha_p$ represents a shifted and rescaled version of a bounded physical parameter, the Gaussian can attribute positive probability to unphysical regions.

**Applies to:** OverallSys, HistoSys, StatError, Lumi.

### 6.3 Poisson ("Gamma") Constraint

When the auxiliary measurement is based on counting events (e.g., a Poisson process), a Poisson distribution more accurately describes the constraint. The truncated Gaussian can lead to undercoverage (overly optimistic) results, making this practically relevant.

For the Poisson constraint, we reparametrize to $\beta_p > 0$ centered around 1. The nominal rate is factored out into a constant $\tau_p$ and the mean of the Poisson is $\tau_p \beta_p$:

$$\text{Pois}(n_p \mid \tau_p \beta_p) = \frac{(\tau_p \beta_p)^{n_p} e^{-\tau_p \beta_p}}{n_p!}$$

The nominal auxiliary measurement is:

$$n^0_p = \tau_p = (1/\sigma_p^\text{rel})^2$$

The relationship between $\alpha$ (used for the response of systematics) and $\beta$ is:

$$\alpha_p(\beta_p) = \sqrt{\tau_p} \; (\beta_p - 1)$$

satisfying $\alpha(\beta = 1) = 0$ and $\alpha(\beta = 1 \pm \tau_p^{-1/2}) = \pm 1$.

**NLL contribution:**

$$-\ln\mathcal{L}_\text{Pois}(\gamma_b) = \gamma_b \tau_b - n^\text{aux}_b \ln(\gamma_b \tau_b) + \ln\Gamma(n^\text{aux}_b + 1)$$

**Applies to:** ShapeSys (Barlow-Beeston bins).

### 6.4 Log-Normal Constraint

The log-normal distribution represents a random variable whose logarithm follows a normal distribution. It is appropriate when the value is a random proportion of the previous observation, or when the parameter must be strictly positive.

$$P_\text{LN}(n_p \mid \beta_p, \kappa_p) = \frac{1}{\sqrt{2\pi}\ln\kappa_p} \frac{1}{n_p} \exp\left[-\frac{\ln(n_p/\beta_p)^2}{2(\ln\kappa_p)^2}\right]$$

where $\kappa_p = 1 + \sigma_p^\text{rel}$.

**Note:** The log-normal constraint is not currently the default in NextStat or pyhf but is available in the HistFactory framework. Support is planned for future releases.

### 6.5 Constraint Summary

| Modifier | Constraint Type | Center | Width | Global Observable |
|----------|----------------|--------|-------|-------------------|
| OverallSys | Gaussian | $\alpha^0_p = 0$ | $\sigma_p = 1$ | `nom_alpha_<name>` |
| HistoSys | Gaussian | $\alpha^0_p = 0$ | $\sigma_p = 1$ | `nom_alpha_<name>` |
| StatError | Gaussian | $\gamma^0_b = 1$ | $\sigma_b^\text{rel}$ (computed) | `nom_gamma_stat_<ch>_bin_<b>` |
| Lumi | Gaussian | $\lambda^0 = 1$ | $\Delta_L$ | `nominalLumi` |
| ShapeSys | Poisson | — | — | `nom_gamma_<name>_bin_<b>` |
| NormFactor | None | — | — | — |
| ShapeFactor | None | — | — | — |

## 7. Full Likelihood

### 7.1 Negative Log-Likelihood

Combining all terms, the full NLL is:

$$\text{nll}(\boldsymbol{\phi}) = \underbrace{\sum_{c} \sum_{b} \left[\nu_{cb} - n_{cb} \ln\nu_{cb} + \ln\Gamma(n_{cb} + 1)\right]}_{\text{main Poisson terms}}$$

$$+ \underbrace{\sum_{k \in \text{ShapeSys}} \left[\gamma_k \tau_k - n_k^\text{aux} \ln(\gamma_k \tau_k) + \ln\Gamma(n_k^\text{aux} + 1)\right]}_{\text{Poisson auxiliary (Barlow-Beeston)}}$$

$$+ \underbrace{\sum_{j \in \text{Gauss}} \left[\frac{1}{2}\left(\frac{\theta_j - \theta_{0,j}}{\sigma_j}\right)^2 + \ln\sigma_j + \frac{1}{2}\ln 2\pi\right]}_{\text{Gaussian constraints}}$$

### 7.2 Canonical Quantities

For compatibility with pyhf:

- $\text{logpdf}(\boldsymbol{\phi}) = -\text{nll}(\boldsymbol{\phi})$
- $\text{twice\_nll}(\boldsymbol{\phi}) = 2 \cdot \text{nll}(\boldsymbol{\phi}) = -2 \cdot \text{logpdf}(\boldsymbol{\phi})$

### 7.3 Implementation Optimizations

**Sparse bins.** When $n_{cb} = 0$, the term $-n_{cb} \ln\nu_{cb} = 0$ regardless of $\nu_{cb}$. The implementation uses an observation mask to skip these terms.

**Precomputed constants.** The Gaussian constraint constant $\sum_j [\ln\sigma_j + \frac{1}{2}\ln 2\pi]$ and the log-factorials $\ln\Gamma(n_{cb} + 1)$ are precomputed once and cached in `PreparedModel`.

**SIMD evaluation.** The main Poisson NLL sum is evaluated using `f64x4` SIMD lanes. The natural logarithm is extracted to scalar `f64::ln()` for bit-exact accuracy (`wide::f64x4::ln()` has $\sim$1000 ULP error).

## 8. Asimov Dataset Construction

The Asimov dataset [5] provides the expected data under a given hypothesis, used for computing expected test statistics and the Brazil band.

### 8.1 Background-Only Asimov ($\mu = 0$)

1. **Conditional fit:** Minimize NLL with $\mu$ fixed to $0$, obtaining fitted nuisance parameters $\hat{\boldsymbol{\theta}}_0$.

2. **Expected main data:** $n_{cb}^\text{Asimov} = \nu_{cb}(0, \hat{\boldsymbol{\theta}}_0)$

3. **Expected auxiliary data (Gaussian):** $a_p^\text{Asimov} = \hat\alpha_p$

4. **Expected auxiliary data (Poisson/ShapeSys):** $n_k^{\text{aux, Asimov}} = \hat\gamma_k \cdot \tau_k$

### 8.2 Properties

The Asimov dataset has the property that the MLE on it returns the parameters used to generate it (to within numerical precision). This makes it suitable for computing the median expected test statistic without Monte Carlo sampling.

## 9. pyhf JSON Workspace Format

### 9.1 Schema Overview

NextStat reads the pyhf JSON workspace format. A workspace contains four top-level keys:

| Key | Content |
|-----|---------|
| `channels` | Array of channels, each with named samples and modifiers |
| `observations` | Observed bin counts per channel |
| `measurements` | Analysis configuration (POI, parameter overrides, luminosity) |
| `version` | Schema version (`"1.0.0"`) |

### 9.2 Example Workspace

```json
{
  "channels": [
    {
      "name": "channel1",
      "samples": [
        {
          "name": "signal",
          "data": [12.0, 11.0],
          "modifiers": [
            {"name": "mu", "type": "normfactor", "data": null},
            {"name": "syst1", "type": "normsys",
             "data": {"hi": 1.05, "lo": 0.95}}
          ]
        },
        {
          "name": "background1",
          "data": [50.0, 52.0],
          "modifiers": [
            {"name": "syst2", "type": "normsys",
             "data": {"hi": 1.07, "lo": 0.93}},
            {"name": "syst3", "type": "normsys",
             "data": {"hi": 1.03, "lo": 0.95}}
          ]
        },
        {
          "name": "background2",
          "data": [33.0, 25.0],
          "modifiers": [
            {"name": "syst3", "type": "normsys",
             "data": {"hi": 0.97, "lo": 1.02}},
            {"name": "bkg2_stat", "type": "staterror",
             "data": [5.0, 4.0]}
          ]
        }
      ]
    }
  ],
  "observations": [
    {"name": "channel1", "data": [55.0, 48.0]}
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

This example corresponds to the standard HistFactory example (cf. Section 5.1 of [1]):

| Syst | Signal | Background1 | Background2 |
|------|--------|-------------|-------------|
| syst1 | $\eta^+ = 1.05$, $\eta^- = 0.95$ | — | — |
| syst2 | — | $\eta^+ = 1.07$, $\eta^- = 0.93$ | — |
| syst3 | — | $\eta^+ = 1.03$, $\eta^- = 0.95$ | $\eta^+ = 0.97$, $\eta^- = 1.02$ |

Note that `syst3` affects both backgrounds, with opposite signs of effect (anti-correlated).

### 9.3 Translation to Internal Model

The translation from JSON to the internal `HistFactoryModel` is handled by `ns-translate`:

1. Parse the JSON via `serde`.
2. Resolve channel-sample-modifier relationships.
3. Identify shared parameters across samples and channels.
4. Precompute Code 4 interpolation coefficients ($a_1, \ldots, a_6$) for each NormSys modifier.
5. Precompute $\tau_b$ values for ShapeSys modifiers.
6. Precompute Gaussian constraint constant $\sum_j [\ln\sigma_j + \frac{1}{2}\ln 2\pi]$.
7. Construct the `PreparedModel` with flattened data, log-factorials, and observation mask.

## References

[1] K. Cranmer, G. Lewis, L. Moneta, A. Shibata, W. Verkerke, "HistFactory: A tool for creating statistical models for use with RooFit and RooStats," CERN-OPEN-2012-016, 2012.

[2] L. Heinrich, M. Feickert, G. Stark, K. Cranmer, "pyhf: pure-Python implementation of HistFactory statistical models," JOSS 6(58), 2823, 2021.

[3] ATLAS Collaboration, "Reproducing searches for new physics with the ATLAS experiment through publication of full statistical likelihoods," ATL-PHYS-PUB-2019-029, 2019.

[4] R. Barlow, C. Beeston, "Fitting using finite Monte Carlo samples," Comp. Phys. Comm. 77, 219–228, 1993.

[5] G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for likelihood-based tests of new physics," Eur. Phys. J. C71, 1554, 2011; erratum ibid. C73, 2501, 2013.
