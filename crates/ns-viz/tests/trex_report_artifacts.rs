use approx::assert_abs_diff_eq;
use std::collections::HashMap;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repo root")
}

fn load_histfactory_fixture()
-> (ns_translate::pyhf::Workspace, ns_translate::pyhf::HistFactoryModel) {
    let root = repo_root();
    let ws_path = root.join("tests/fixtures/histfactory/workspace.json");
    let bytes = std::fs::read(ws_path).expect("read fixture workspace");
    let ws: ns_translate::pyhf::Workspace =
        serde_json::from_slice(&bytes).expect("parse workspace");
    let model = ns_translate::pyhf::HistFactoryModel::from_workspace(&ws).expect("build model");
    (ws, model)
}

#[test]
fn trex_report_distributions_artifact_contract_smoke() {
    let (ws, model) = load_histfactory_fixture();

    let params_prefit: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let params_postfit = params_prefit.clone();

    let mut data_by_channel: HashMap<String, Vec<f64>> = HashMap::new();
    for obs in &ws.observations {
        data_by_channel.insert(obs.name.clone(), obs.data.clone());
    }

    let root = repo_root();
    let xml = root.join("tests/fixtures/histfactory/combination.xml");
    let edges =
        ns_translate::histfactory::bin_edges_by_channel_from_xml(&xml).expect("bin edges from xml");

    let artifact = ns_viz::distributions::distributions_artifact(
        &model,
        &data_by_channel,
        &edges,
        &params_prefit,
        &params_postfit,
        1,
        None,
    )
    .expect("distributions artifact");

    assert_eq!(artifact.schema_version, "trex_report_distributions_v0");
    assert!(!artifact.channels.is_empty());

    for ch in &artifact.channels {
        // Basic shape checks.
        assert!(ch.bin_edges.len() >= 2);
        let n_bins = ch.bin_edges.len() - 1;
        assert_eq!(ch.data_y.len(), n_bins);
        assert_eq!(ch.total_prefit_y.len(), n_bins);
        assert_eq!(ch.total_postfit_y.len(), n_bins);
        assert_eq!(ch.ratio_y.len(), n_bins);
        assert_eq!(ch.ratio_yerr_lo.len(), n_bins);
        assert_eq!(ch.ratio_yerr_hi.len(), n_bins);

        // Ratio should be data / total_postfit where denom is finite and non-zero.
        for i in 0..n_bins.min(5) {
            let denom = ch.total_postfit_y[i];
            if denom.is_finite() && denom != 0.0 {
                let want = ch.data_y[i] / denom;
                assert_abs_diff_eq!(ch.ratio_y[i], want, epsilon = 1e-9);
            }
        }
    }
}

#[test]
fn trex_report_distributions_supports_blinding() {
    let (ws, model) = load_histfactory_fixture();

    let params_prefit: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let params_postfit = params_prefit.clone();

    let mut data_by_channel: HashMap<String, Vec<f64>> = HashMap::new();
    for obs in &ws.observations {
        data_by_channel.insert(obs.name.clone(), obs.data.clone());
    }

    let root = repo_root();
    let xml = root.join("tests/fixtures/histfactory/combination.xml");
    let edges =
        ns_translate::histfactory::bin_edges_by_channel_from_xml(&xml).expect("bin edges from xml");

    let mut blinded = std::collections::HashSet::new();
    blinded.insert("SR".to_string());

    let artifact = ns_viz::distributions::distributions_artifact(
        &model,
        &data_by_channel,
        &edges,
        &params_prefit,
        &params_postfit,
        1,
        Some(&blinded),
    )
    .expect("distributions artifact");

    let sr = artifact
        .channels
        .iter()
        .find(|c| c.channel_name == "SR")
        .expect("SR channel");
    assert_eq!(sr.data_is_blinded, Some(true));
    assert!(sr.data_y.iter().all(|&v| v == 0.0));
    assert!(sr.ratio_y.iter().all(|&v| v == 0.0));
}

#[test]
fn trex_report_pulls_and_corr_golden_simple_fit() {
    let (_ws, model) = load_histfactory_fixture();

    let n = model.parameters().len();
    let mut params = Vec::with_capacity(n);
    let mut uncs = Vec::with_capacity(n);
    for p in model.parameters() {
        params.push(p.constraint_center.unwrap_or(p.init));
        // Ensure strictly positive widths for correlation normalization.
        uncs.push(p.constraint_width.unwrap_or(1.0).max(1e-12));
    }

    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = uncs[i] * uncs[i];
    }

    let fit = ns_core::FitResult {
        parameters: params,
        uncertainties: uncs,
        covariance: Some(cov),
        nll: 0.0,
        converged: true,
        n_iter: 0,
        n_fev: 0,
        n_gev: 0,
    };

    let pulls = ns_viz::pulls::pulls_artifact(&model, &fit, 1).expect("pulls artifact");
    assert_eq!(pulls.schema_version, "trex_report_pulls_v0");
    assert!(!pulls.entries.is_empty());
    for e in &pulls.entries {
        if e.kind == "nuisance" {
            assert_abs_diff_eq!(e.pull, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(e.constraint, 1.0, epsilon = 1e-12);
        }
    }

    let corr = ns_viz::corr::corr_artifact(&model, &fit, 1, false).expect("corr artifact");
    assert_eq!(corr.schema_version, "trex_report_corr_v0");
    assert_eq!(corr.parameter_names.len(), n);
    assert_eq!(corr.corr.len(), n);
    for i in 0..n {
        assert_eq!(corr.corr[i].len(), n);
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(corr.corr[i][j], want, epsilon = 1e-12);
        }
    }
}

#[test]
fn trex_report_yields_invariants_prefit_equals_postfit() {
    let (_ws, model) = load_histfactory_fixture();

    let params_prefit: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let artifact = ns_viz::yields::yields_artifact(&model, &params_prefit, &params_prefit, 1, None)
        .expect("yields artifact");

    assert_eq!(artifact.schema_version, "trex_report_yields_v0");
    assert!(!artifact.channels.is_empty());

    for ch in &artifact.channels {
        assert_abs_diff_eq!(ch.total_prefit, ch.total_postfit, epsilon = 1e-9);
        let sum_samples: f64 = ch.samples.iter().map(|s| s.prefit).sum();
        assert_abs_diff_eq!(sum_samples, ch.total_prefit, epsilon = 1e-6);
    }
}

#[test]
fn trex_report_uncertainty_breakdown_grouping_prefix_1() {
    let ranking = ns_viz::RankingArtifact {
        names: vec!["lumi_test".to_string(), "gamma_stat_foo".to_string(), "JES_bar".to_string()],
        delta_mu_up: vec![0.2, -0.1, 0.05],
        delta_mu_down: vec![-0.15, 0.08, -0.02],
        pull: vec![0.0, 0.0, 0.0],
        constraint: vec![1.0, 1.0, 1.0],
    };

    let unc = ns_viz::uncertainty::uncertainty_breakdown_from_ranking(&ranking, "prefix_1", 1)
        .expect("uncertainty breakdown");
    assert_eq!(unc.schema_version, "trex_report_uncertainty_v0");

    let mut by_name = std::collections::HashMap::new();
    for g in &unc.groups {
        by_name.insert(g.name.as_str(), g.impact);
    }

    assert!(by_name.contains_key("lumi"));
    assert!(by_name.contains_key("stat"));
    assert!(by_name.contains_key("JES"));

    assert_abs_diff_eq!(*by_name.get("lumi").unwrap(), 0.2, epsilon = 1e-12);
    assert_abs_diff_eq!(*by_name.get("stat").unwrap(), 0.1, epsilon = 1e-12);
    assert_abs_diff_eq!(*by_name.get("JES").unwrap(), 0.05, epsilon = 1e-12);

    let total = (0.2_f64 * 0.2 + 0.1 * 0.1 + 0.05 * 0.05).sqrt();
    assert_abs_diff_eq!(unc.total, total, epsilon = 1e-12);
}

#[test]
fn trex_report_uncertainty_breakdown_grouping_category_v1() {
    let ranking = ns_viz::RankingArtifact {
        names: vec![
            "lumi_2022".to_string(),
            "PDF_setA".to_string(),
            "alphaS".to_string(),
            "muR".to_string(),
            "scale_ttbar".to_string(),
            "gamma_stat_SR".to_string(),
            "JES_barrel".to_string(),
        ],
        delta_mu_up: vec![0.2, 0.12, 0.06, 0.08, 0.04, 0.10, 0.05],
        delta_mu_down: vec![-0.15, -0.10, -0.03, -0.02, -0.01, 0.08, -0.02],
        pull: vec![0.0; 7],
        constraint: vec![1.0; 7],
    };

    let unc = ns_viz::uncertainty::uncertainty_breakdown_from_ranking(&ranking, "category_v1", 1)
        .expect("uncertainty breakdown");

    let mut by_name = std::collections::HashMap::new();
    for g in &unc.groups {
        by_name.insert(g.name.as_str(), (g.impact, g.n_parameters));
    }

    // category buckets exist
    assert!(by_name.contains_key("lumi"));
    assert!(by_name.contains_key("pdf"));
    assert!(by_name.contains_key("scale"));
    assert!(by_name.contains_key("stat"));

    // simple sanity: lumi impact is just |0.2|
    assert_abs_diff_eq!(by_name.get("lumi").unwrap().0, 0.2, epsilon = 1e-12);

    // pdf has two entries (PDF_setA + alphaS)
    assert_eq!(by_name.get("pdf").unwrap().1, 2);
    let pdf_expected = (0.12_f64 * 0.12 + 0.06 * 0.06).sqrt();
    assert_abs_diff_eq!(by_name.get("pdf").unwrap().0, pdf_expected, epsilon = 1e-12);

    // scale has two entries (muR + scale_ttbar)
    assert_eq!(by_name.get("scale").unwrap().1, 2);
    let scale_expected = (0.08_f64 * 0.08 + 0.04 * 0.04).sqrt();
    assert_abs_diff_eq!(by_name.get("scale").unwrap().0, scale_expected, epsilon = 1e-12);

    // stat
    assert_abs_diff_eq!(by_name.get("stat").unwrap().0, 0.10, epsilon = 1e-12);

    // JES falls back to prefix_1 ("JES")
    assert!(by_name.contains_key("JES"));
    assert_abs_diff_eq!(by_name.get("JES").unwrap().0, 0.05, epsilon = 1e-12);
}
