//! Econometrics & Causal Inference (Phase 12).
//!
//! This module provides:
//! - **Panel linear regression** with fixed effects (entity/time demeaning) and
//!   cluster-robust standard errors (Liang–Zeger HC0 sandwich).
//! - **Difference-in-Differences** (DiD) with two-period canonical estimator and
//!   multi-period event-study (leads/lags) helpers.
//! - **Instrumental Variables** (IV / 2SLS) with first-stage F-statistic and
//!   weak-instrument diagnostics (Stock–Yogo critical values).
//! - **Doubly-robust AIPW** (Augmented Inverse Probability Weighting) with
//!   Rosenbaum sensitivity analysis hooks.

pub mod aipw;
pub mod did;
pub mod hdfe;
pub mod iv;
pub mod panel;

pub use aipw::{AipwResult, RosenbaumResult, aipw_ate, rosenbaum_bounds};
pub use did::{DidResult, EventStudyResult, did_canonical, event_study};
pub use hdfe::FixedEffectsSolver;
pub use iv::{FirstStageResult, IvResult, iv_2sls};
pub use panel::{PanelFEResult, cluster_robust_se, panel_fe_fit};
