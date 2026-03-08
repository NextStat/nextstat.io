use serde::Serialize;

use crate::detect::{WorkspaceFormat, detect_format};

use ns_core::{Error, Result};

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum WorkspaceAuditArtifact {
    Pyhf(crate::pyhf::audit::WorkspaceAudit),
    Simplified(Box<crate::simplified::audit::SimplifiedLikelihoodAudit>),
}

pub fn audit_workspace_json(json_str: &str) -> Result<WorkspaceAuditArtifact> {
    match detect_format(json_str) {
        WorkspaceFormat::Hs3 => Err(Error::NotImplemented(
            "workspace audit supports pyhf or simplified-likelihood JSON; HS3 is not supported"
                .to_string(),
        )),
        WorkspaceFormat::SimplifiedLikelihood => {
            let spec: crate::simplified::schema::SimplifiedLikelihoodWorkspace =
                serde_json::from_str(json_str)?;
            let audit = crate::simplified::audit::audit_simplified_likelihood(&spec)?;
            Ok(WorkspaceAuditArtifact::Simplified(Box::new(audit)))
        }
        WorkspaceFormat::Pyhf | WorkspaceFormat::Unknown => {
            let json: serde_json::Value = serde_json::from_str(json_str)?;
            Ok(WorkspaceAuditArtifact::Pyhf(crate::pyhf::audit::workspace_audit(&json)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkspaceAuditArtifact, audit_workspace_json};

    #[test]
    fn audit_workspace_json_accepts_pyhf_fixture() {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        let audit = audit_workspace_json(json).expect("pyhf fixture should audit");

        match audit {
            WorkspaceAuditArtifact::Pyhf(audit) => {
                assert_eq!(audit.channel_count, 1);
                assert!(audit.total_samples > 0);
            }
            WorkspaceAuditArtifact::Simplified(_) => {
                panic!("expected pyhf audit artifact");
            }
        }
    }

    #[test]
    fn audit_workspace_json_accepts_simplified_fixture() {
        let json = include_str!("../../../tests/fixtures/sl_covariance_three_bin.json");
        let audit = audit_workspace_json(json).expect("simplified fixture should audit");

        match audit {
            WorkspaceAuditArtifact::Pyhf(_) => {
                panic!("expected simplified audit artifact");
            }
            WorkspaceAuditArtifact::Simplified(audit) => {
                assert_eq!(audit.schema_version, "nextstat_simplified_likelihood_audit_v0");
                assert_eq!(audit.input_schema_version, "nextstat_simplified_likelihood_v0");
                assert_eq!(audit.uncertainty_model_kind, "covariance");
            }
        }
    }

    #[test]
    fn audit_workspace_json_rejects_hs3_fixture() {
        let json = include_str!("../../../tests/fixtures/workspace-postFit_PTV.json");
        let err = audit_workspace_json(json).expect_err("HS3 audit should be rejected");
        assert!(err.to_string().contains("HS3"), "error should mention HS3, got: {err}");
    }
}
