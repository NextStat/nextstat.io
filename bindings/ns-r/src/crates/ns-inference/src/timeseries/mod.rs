//! Time series and state space models (Phase 8).

mod internal;

/// Linear-Gaussian Kalman filter / smoother.
pub mod kalman;

/// EM parameter estimation for linear-Gaussian state space models.
pub mod em;

/// Forecasting utilities for linear-Gaussian state space models.
pub mod forecast;

/// Simulation utilities for linear-Gaussian state space models.
pub mod simulate;

/// Parameter transforms and bounds for time series models.
pub mod params;

/// Volatility models (GARCH(1,1), approximate stochastic volatility).
pub mod volatility;
