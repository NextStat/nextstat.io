//! Typed registry for Ads Systematics V1.
//!
//! This is the shared source of truth for the first doctrine-aligned ads
//! systematics surface. It does not yet calibrate values or run sensitivity
//! analysis by itself; it defines the stable registry entries that downstream
//! layers can reference without copying semantics from markdown documents.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsUncertaintyLayer {
    Measurement,
    Causal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsSystematicKind {
    NormSys,
    HistoSys,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsSystematicSourceClass {
    DataEstimated,
    HybridApiPlusBenchmark,
    HybridApiPlusPrior,
    PolicyDefaultOrVendorCalibrated,
    PolicyDefaultOrIncrementalityCalibrated,
    InternalNextstatDiagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsCombinationPolicy {
    ProfileLikelihood,
    SensitivityOnly,
    ScenarioOnly,
    SensitivityFirstThenProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdsSystematic {
    pub code: &'static str,
    pub systematic_id: &'static str,
    pub uncertainty_layer: AdsUncertaintyLayer,
    pub kind: AdsSystematicKind,
    pub source_class: AdsSystematicSourceClass,
    pub default_enabled: bool,
    pub default_nominal: &'static str,
    pub default_sigma: &'static str,
    pub constraint: &'static str,
    pub combination_policy: AdsCombinationPolicy,
}

const DEFAULT_SYSTEMATICS: [AdsSystematic; 6] = [
    AdsSystematic {
        code: "S1",
        systematic_id: "conversion_lag_curve",
        uncertainty_layer: AdsUncertaintyLayer::Measurement,
        kind: AdsSystematicKind::HistoSys,
        source_class: AdsSystematicSourceClass::DataEstimated,
        default_enabled: true,
        default_nominal: "account-specific lag baseline",
        default_sigma: "per-bin historical standard error",
        constraint: "mass-preserving redistribution across approved lag bins",
        combination_policy: AdsCombinationPolicy::ProfileLikelihood,
    },
    AdsSystematic {
        code: "S2",
        systematic_id: "viewability_baseline",
        uncertainty_layer: AdsUncertaintyLayer::Measurement,
        kind: AdsSystematicKind::NormSys,
        source_class: AdsSystematicSourceClass::HybridApiPlusBenchmark,
        default_enabled: true,
        default_nominal: "Active View baseline",
        default_sigma: "max(observed spread, benchmark 0.15)",
        constraint: "[0.30, 1.00]",
        combination_policy: AdsCombinationPolicy::ProfileLikelihood,
    },
    AdsSystematic {
        code: "S3",
        systematic_id: "cross_device_partial_rate",
        uncertainty_layer: AdsUncertaintyLayer::Measurement,
        kind: AdsSystematicKind::NormSys,
        source_class: AdsSystematicSourceClass::HybridApiPlusPrior,
        default_enabled: true,
        default_nominal: "observed cross-device share",
        default_sigma: "benchmark gap for non-logged-in users",
        constraint: "[0.80, 1.40]",
        combination_policy: AdsCombinationPolicy::ProfileLikelihood,
    },
    AdsSystematic {
        code: "S4",
        systematic_id: "residual_fraud_rate",
        uncertainty_layer: AdsUncertaintyLayer::Measurement,
        kind: AdsSystematicKind::NormSys,
        source_class: AdsSystematicSourceClass::PolicyDefaultOrVendorCalibrated,
        default_enabled: true,
        default_nominal: "0.10",
        default_sigma: "0.05",
        constraint: "[0.00, 0.30]",
        combination_policy: AdsCombinationPolicy::SensitivityFirstThenProfile,
    },
    AdsSystematic {
        code: "S5",
        systematic_id: "organic_cannibalization",
        uncertainty_layer: AdsUncertaintyLayer::Causal,
        kind: AdsSystematicKind::NormSys,
        source_class: AdsSystematicSourceClass::PolicyDefaultOrIncrementalityCalibrated,
        default_enabled: true,
        default_nominal: "0.00",
        default_sigma: "0.15",
        constraint: "[0.00, 0.50]",
        combination_policy: AdsCombinationPolicy::SensitivityOnly,
    },
    AdsSystematic {
        code: "S6",
        systematic_id: "experiment_interference",
        uncertainty_layer: AdsUncertaintyLayer::Causal,
        kind: AdsSystematicKind::NormSys,
        source_class: AdsSystematicSourceClass::InternalNextstatDiagnostic,
        default_enabled: true,
        default_nominal: "0.00",
        default_sigma: "diagnostic-derived",
        constraint: "[0.00, 0.30]",
        combination_policy: AdsCombinationPolicy::SensitivityOnly,
    },
];

/// Return the canonical Ads Systematics V1 registry in deterministic S1..S6 order.
pub fn default_systematics() -> Vec<AdsSystematic> {
    DEFAULT_SYSTEMATICS.to_vec()
}

/// Return the canonical Ads Systematics V1 registry as a static slice.
pub fn default_systematics_slice() -> &'static [AdsSystematic] {
    &DEFAULT_SYSTEMATICS
}

/// Look up a default systematic by short code (`S1`..`S6`).
pub fn default_systematic_by_code(code: &str) -> Option<AdsSystematic> {
    DEFAULT_SYSTEMATICS.iter().copied().find(|item| item.code == code)
}

/// Look up a default systematic by stable identifier.
pub fn default_systematic_by_id(systematic_id: &str) -> Option<AdsSystematic> {
    DEFAULT_SYSTEMATICS.iter().copied().find(|item| item.systematic_id == systematic_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_systematics_registry_is_complete_and_ordered() {
        let systematics = default_systematics();
        assert_eq!(systematics.len(), 6);
        assert_eq!(
            systematics.iter().map(|item| item.code).collect::<Vec<_>>(),
            vec!["S1", "S2", "S3", "S4", "S5", "S6"]
        );
        assert_eq!(
            systematics.iter().map(|item| item.systematic_id).collect::<Vec<_>>(),
            vec![
                "conversion_lag_curve",
                "viewability_baseline",
                "cross_device_partial_rate",
                "residual_fraud_rate",
                "organic_cannibalization",
                "experiment_interference",
            ]
        );
    }

    #[test]
    fn measurement_and_causal_entries_match_doctrine_split() {
        let systematics = default_systematics();
        let measurement = systematics
            .iter()
            .filter(|item| item.uncertainty_layer == AdsUncertaintyLayer::Measurement)
            .count();
        let causal = systematics
            .iter()
            .filter(|item| item.uncertainty_layer == AdsUncertaintyLayer::Causal)
            .count();

        assert_eq!(measurement, 4);
        assert_eq!(causal, 2);
    }

    #[test]
    fn conversion_lag_curve_is_histosys_profiled_measurement() {
        let item = default_systematic_by_code("S1").expect("S1 must exist");
        assert_eq!(item.systematic_id, "conversion_lag_curve");
        assert_eq!(item.kind, AdsSystematicKind::HistoSys);
        assert_eq!(item.uncertainty_layer, AdsUncertaintyLayer::Measurement);
        assert_eq!(item.source_class, AdsSystematicSourceClass::DataEstimated);
        assert_eq!(item.combination_policy, AdsCombinationPolicy::ProfileLikelihood);
    }

    #[test]
    fn causal_systematics_are_sensitivity_only() {
        let s5 = default_systematic_by_id("organic_cannibalization").expect("S5");
        let s6 = default_systematic_by_id("experiment_interference").expect("S6");

        assert_eq!(s5.uncertainty_layer, AdsUncertaintyLayer::Causal);
        assert_eq!(s6.uncertainty_layer, AdsUncertaintyLayer::Causal);
        assert_eq!(s5.combination_policy, AdsCombinationPolicy::SensitivityOnly);
        assert_eq!(s6.combination_policy, AdsCombinationPolicy::SensitivityOnly);
    }

    #[test]
    fn registry_lookups_use_stable_code_and_id() {
        assert_eq!(
            default_systematic_by_code("S4").map(|item| item.systematic_id),
            Some("residual_fraud_rate")
        );
        assert_eq!(
            default_systematic_by_id("viewability_baseline").map(|item| item.code),
            Some("S2")
        );
        assert!(default_systematic_by_code("S7").is_none());
        assert!(default_systematic_by_id("unknown").is_none());
    }
}
