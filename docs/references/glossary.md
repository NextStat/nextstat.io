# Glossary

A cross-domain glossary of key terms used throughout NextStat documentation.
Terms are grouped by domain; general-purpose terms appear first.

> **Convention:** acronyms in public-facing docs must be expanded on first use
> (see `docs/references/terminology.md`).

---

## General Statistics

| Term | Definition |
|------|-----------|
| **MLE** | Maximum Likelihood Estimation — finds parameter values that maximize the likelihood of observed data. |
| **NLL** | Negative log-likelihood — the objective function minimized during MLE. Lower is better. |
| **Gradient** | Vector of partial derivatives of the NLL with respect to each parameter. Used by L-BFGS-B for optimization. |
| **Hessian** | Matrix of second derivatives of the NLL. Its inverse approximates the covariance matrix of parameter estimates. |
| **Covariance matrix** | Square matrix describing the joint uncertainty of all parameters. Diagonal elements are variances; off-diagonal are correlations. |
| **Confidence interval (CI)** | A frequentist interval `[lo, hi]` such that repeated experiments would contain the true value at the stated coverage rate. |
| **Credible interval** | A Bayesian interval containing the true value with stated posterior probability. |
| **p-value** | Probability of observing data at least as extreme as the current observation, assuming the null hypothesis. |
| **Nuisance parameter (NP)** | A model parameter that is not of direct interest but must be accounted for (e.g., systematic uncertainties). |
| **Parameter of interest (POI)** | The primary quantity being measured or tested (e.g., signal strength μ). |
| **Constraint term** | A penalty added to the NLL to encode prior knowledge about a nuisance parameter (typically Gaussian). |
| **Profile likelihood** | Likelihood evaluated at the best-fit nuisance parameters for each fixed POI value. |
| **L-BFGS-B** | Limited-memory Broyden–Fletcher–Goldfarb–Shanno with box constraints — NextStat's default optimizer. |

## Regression & GLM

| Term | Definition |
|------|-----------|
| **GLM** | Generalized Linear Model — extends linear regression to non-normal response distributions via a link function. |
| **Link function** | Maps the linear predictor to the mean of the response distribution (e.g., logit for binary outcomes). |
| **Logistic regression** | GLM with a logit link for binary classification. |
| **Poisson regression** | GLM with a log link for count data. |
| **Negative binomial regression** | GLM for overdispersed count data (variance > mean). |
| **Robust standard errors (HC)** | Heteroscedasticity-consistent standard errors that remain valid when the variance model is misspecified. |

## Bayesian Inference

| Term | Definition |
|------|-----------|
| **NUTS** | No-U-Turn Sampler — an adaptive HMC algorithm that automatically tunes trajectory length. NextStat's default MCMC sampler. |
| **HMC** | Hamiltonian Monte Carlo — uses gradient information to propose distant samples with high acceptance. |
| **Posterior** | The probability distribution of parameters given observed data; proportional to likelihood × prior. |
| **MAP** | Maximum A Posteriori — the mode of the posterior; equivalent to penalized MLE. |
| **ESS** | Effective Sample Size — estimates how many independent samples a correlated MCMC chain is worth. |
| **R-hat (R̂)** | Convergence diagnostic comparing within-chain and between-chain variance. Values near 1.0 indicate convergence. |
| **Laplace approximation** | Approximates the posterior as a Gaussian centered at the MAP with covariance from the inverse Hessian. |

## Hierarchical / Mixed Models

| Term | Definition |
|------|-----------|
| **LMM** | Linear Mixed Model — a regression model with both fixed effects (population-level) and random effects (group-level). |
| **GLMM** | Generalized Linear Mixed Model — extends LMM to non-Gaussian responses. |
| **Fixed effects** | Coefficients estimated at the population level, shared across all groups. |
| **Random effects** | Group-specific deviations from the population mean, assumed drawn from a distribution (typically Normal). |
| **Marginal likelihood** | The likelihood with random effects integrated out analytically (for Gaussian LMMs). |

## Time Series & State Space

| Term | Definition |
|------|-----------|
| **Kalman filter** | Recursive algorithm for optimal state estimation in linear-Gaussian state space models. |
| **EM** | Expectation-Maximization — iterative algorithm for parameter estimation in latent-variable models. Used for fitting state space models. |
| **GARCH** | Generalized Autoregressive Conditional Heteroscedasticity — a volatility model where variance depends on past squared residuals and past variances. |
| **Stochastic volatility (SV)** | A latent-variable model where log-variance follows an AR(1) process. More flexible than GARCH. |
| **State space model** | A model with hidden states evolving over time, observed through a noisy measurement equation. |

## Survival Analysis

| Term | Definition |
|------|-----------|
| **Hazard function** | Instantaneous rate of event occurrence at time t, given survival to t. |
| **Survival function S(t)** | Probability of surviving beyond time t. |
| **Censoring** | When the event time is only partially observed (e.g., a patient is still alive at study end — right censoring). |
| **Weibull model** | A parametric survival model with shape and scale parameters; includes exponential as a special case. |

## Causal Inference & Econometrics

| Term | Definition |
|------|-----------|
| **IV / 2SLS** | Instrumental Variables / Two-Stage Least Squares — estimates causal effects when confounders are unobserved, using an instrument. |
| **DID** | Difference-in-Differences — estimates treatment effects by comparing pre/post changes in treated vs control groups. |
| **AIPW** | Augmented Inverse Probability Weighting — a doubly-robust estimator combining outcome regression and propensity score. |
| **Propensity score** | Estimated probability of receiving treatment given covariates; used for weighting or matching. |
| **Panel data** | Data with repeated observations over time for multiple entities (cross-sectional time series). |
| **Fixed effects (econometric)** | Entity-specific intercepts that absorb time-invariant unobserved heterogeneity. |

## Pharmacometrics (PK/PD)

| Term | Definition |
|------|-----------|
| **PK** | Pharmacokinetics — the study of how a drug moves through the body (absorption, distribution, metabolism, elimination). |
| **PD** | Pharmacodynamics — the study of a drug's biological effects as a function of concentration. |
| **NLME** | Nonlinear Mixed-Effects model — population PK/PD model with fixed (population) and random (individual) parameters. |
| **Compartment model** | Represents the body as interconnected compartments (e.g., gut, central, peripheral) with first-order transfer. |
| **Clearance (CL)** | Volume of plasma completely cleared of drug per unit time. |
| **Volume of distribution (V)** | Apparent volume into which a drug distributes in the body. |
| **Absorption rate (Ka)** | First-order rate constant for drug absorption from the gut into the central compartment. |
| **Bioavailability (F)** | Fraction of the administered dose that reaches systemic circulation. |
| **LLOQ** | Lower Limit of Quantification — the lowest concentration that can be reliably measured. Below-LLOQ handling: Ignore, ReplaceHalf, or Censored likelihood. |
| **Random effects (PK)** | Individual deviations from population parameters, typically log-normal: `CL_i = CL_pop × exp(η_i)`. |

## HEP (High Energy Physics)

| Term | Definition |
|------|-----------|
| **HistFactory** | A binned-likelihood workspace format widely used in HEP. NextStat reads pyhf-compatible JSON. |
| **pyhf** | Python reference implementation for HistFactory likelihoods. NextStat's JSON format is pyhf-compatible. |
| **CLs** | Modified frequentist confidence level — a method for setting upper limits that avoids excluding signals to which an experiment has no sensitivity. |
| **Asimov dataset** | A synthetic dataset where all observables equal their expected values (no statistical fluctuation). Used for expected limits. |
| **Signal strength (μ)** | The POI in a search: μ = 0 means background-only, μ = 1 means the nominal signal hypothesis. |
| **Systematic uncertainty** | A source of uncertainty other than statistical fluctuation (e.g., detector calibration, theory assumptions). |
| **NormSys** | A multiplicative normalization systematic: the expected yield is scaled by a factor that depends on a nuisance parameter α. |
| **HistoSys** | A shape systematic: the bin contents are interpolated between up/down templates as a function of α. |
| **Ranking plot** | Shows the impact of each nuisance parameter on the POI by comparing fits with that parameter fixed vs free. |
| **TRExFitter** | A HEP analysis framework for HistFactory-based fits. NextStat can import TRExFitter `.config` files. |
| **RooFit** | ROOT's statistical modeling toolkit. NextStat validates unbinned fits against RooFit as a reference. |
| **HS3** | HEP Statistics Serialization Standard — a JSON schema for exchanging statistical models between frameworks. |

## Unbinned Analysis

| Term | Definition |
|------|-----------|
| **Extended likelihood** | Likelihood where the total number of events is Poisson-distributed with expected yield ν, and each event has a probability density p(x). |
| **Mixture model** | A PDF that is a weighted sum of component PDFs: `p(x) = Σ (ν_k / ν_tot) × p_k(x)`. |
| **Normalizing flow** | A neural network that learns an invertible mapping from a simple base distribution to a complex target density. |
| **FlowPdf** | NextStat's ONNX-backed normalizing flow PDF, usable as a drop-in process PDF in unbinned fits. |
| **DcrSurrogate** | A conditional flow that replaces binned template morphing with a smooth, continuous, bin-free neural surrogate. |
| **FAIR-HUC** | The training protocol for DCR surrogates: generate from HistFactory templates at many α-points, train conditional NSF, validate normalization. |
| **EventStore** | NextStat's in-memory columnar event storage for unbinned data. Supports Parquet I/O and per-event weights. |

## GPU & Infrastructure

| Term | Definition |
|------|-----------|
| **CUDA** | NVIDIA's GPU computing platform. NextStat uses f64 precision for CUDA kernels. |
| **Metal** | Apple's GPU API for Apple Silicon. NextStat uses f32 precision for Metal shaders. |
| **GPU parity contract** | Tolerance tiers ensuring GPU-accelerated results match CPU reference within specified bounds (see `docs/gpu-contract.md`). |
| **Batch fitting** | Running many independent fits (e.g., toy experiments) in parallel on GPU with a shared NLL kernel. |
| **Device-resident** | Data that remains on the GPU between operations, avoiding host↔device copies. |
| **TensorRT** | NVIDIA's inference optimizer. NextStat can use TensorRT EP for neural flow evaluation with engine caching. |

## Data Formats

| Term | Definition |
|------|-----------|
| **pyhf JSON** | The HistFactory workspace JSON format (channels, samples, modifiers, observations). |
| **HS3 JSON** | The HEP Statistics Serialization Standard JSON format for general statistical models. |
| **Parquet** | Apache columnar storage format. Used for both binned histogram tables and unbinned event data. |
| **Arrow** | Apache in-memory columnar format. NextStat uses Arrow record batches as the intermediate representation for Parquet I/O. |
| **SoA** | Structure-of-Arrays layout: observables stored as contiguous per-column vectors for cache-friendly GPU access. |
