use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_inference::measurement_combine::{
    MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0, MEASUREMENT_COMBINATION_SCHEMA_V0,
    MeasurementCombinationScenarioSpec, MeasurementCombinationScenarioStudySpec,
    MeasurementCombinationSolver, MeasurementCombinationSpec, MeasurementInput,
    ScenarioErrorOnErrorAssignment, SystematicSource, calibrate_measurements_toys_with_solver,
    combine_measurements_with_solver, compare_measurement_combination_calibration_campaign_solvers,
    run_measurement_combination_calibration_campaign_with_solver,
};
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Duration;

fn load_topmass_full_fixture() -> MeasurementCombinationSpec {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/measurement_combine_gvm_topmass_full.json"
    ))
    .expect("top-mass fixture should deserialize")
}

fn ar1_corr(n: usize, rho: f64) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; n];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = rho.powi(i.abs_diff(j) as i32);
        }
    }
    out
}

fn synthetic_measurement_combination_spec(
    poi: &str,
    n_measurements: usize,
    n_systematics: usize,
    error_on_error: f64,
) -> MeasurementCombinationSpec {
    let measurements = (0..n_measurements)
        .map(|i| {
            let smooth = ((i as f64) * 0.31).sin() * 0.04;
            let slow = ((i as f64) * 0.07).cos() * 0.02;
            MeasurementInput {
                name: format!("meas_{i:03}"),
                value: 172.3 + 0.006 * (i as f64) + smooth + slow,
            }
        })
        .collect::<Vec<_>>();

    let stat_sigma =
        (0..n_measurements).map(|i| 0.12 + 0.004 * ((i % 9) as f64)).collect::<Vec<_>>();
    let stat_corr = ar1_corr(n_measurements, 0.18);
    let mut stat_covariance = vec![vec![0.0; n_measurements]; n_measurements];
    for i in 0..n_measurements {
        for j in 0..n_measurements {
            stat_covariance[i][j] = stat_sigma[i] * stat_sigma[j] * stat_corr[i][j];
        }
    }

    let systematics = (0..n_systematics)
        .map(|s| {
            let phase = (s as f64) * 0.37;
            let magnitudes = (0..n_measurements)
                .map(|i| {
                    let wave = (((i + 1) as f64) * 0.11 + phase).sin().abs();
                    0.03 + 0.025 * wave + 0.002 * ((s % 5) as f64)
                })
                .collect::<Vec<_>>();
            let rho_mag = 0.15 + 0.65 * (((s * 7 + 3) % 11) as f64) / 10.0;
            let rho = if s % 4 == 0 { -rho_mag.min(0.6) } else { rho_mag.min(0.82) };
            let aux_mean = ((s as f64) * PI / 19.0).sin() * 0.02;
            SystematicSource {
                name: format!("syst_{s:03}"),
                magnitudes,
                corr: ar1_corr(n_measurements, rho),
                error_on_error,
                aux_mean,
            }
        })
        .collect::<Vec<_>>();

    MeasurementCombinationSpec {
        schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
        poi: poi.to_string(),
        measurements,
        stat_covariance,
        systematics,
    }
}

fn spec_with_error_on_error(
    spec: &MeasurementCombinationSpec,
    systematic_name: &str,
    error_on_error: f64,
) -> MeasurementCombinationSpec {
    let mut cloned = spec.clone();
    if let Some(systematic) =
        cloned.systematics.iter_mut().find(|systematic| systematic.name == systematic_name)
    {
        systematic.error_on_error = error_on_error;
    } else if let Some(systematic) = cloned.systematics.first_mut() {
        systematic.error_on_error = error_on_error;
    }
    cloned
}

fn synthetic_scenario_study_spec(
    spec: &MeasurementCombinationSpec,
) -> MeasurementCombinationScenarioStudySpec {
    let first = spec.systematics.first().expect("synthetic spec should have systematics");
    let second =
        spec.systematics.get(1).expect("synthetic spec should have at least two systematics");
    let third =
        spec.systematics.get(2).expect("synthetic spec should have at least three systematics");
    MeasurementCombinationScenarioStudySpec {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
        scenarios: vec![
            MeasurementCombinationScenarioSpec {
                name: "leading_syst".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: first.name.clone(),
                    value: 0.10,
                }],
            },
            MeasurementCombinationScenarioSpec {
                name: "pairwise_systs".to_string(),
                error_on_error: vec![
                    ScenarioErrorOnErrorAssignment { systematic: first.name.clone(), value: 0.20 },
                    ScenarioErrorOnErrorAssignment { systematic: second.name.clone(), value: 0.20 },
                ],
            },
            MeasurementCombinationScenarioSpec {
                name: "triple_systs".to_string(),
                error_on_error: vec![
                    ScenarioErrorOnErrorAssignment { systematic: first.name.clone(), value: 0.30 },
                    ScenarioErrorOnErrorAssignment { systematic: second.name.clone(), value: 0.30 },
                    ScenarioErrorOnErrorAssignment { systematic: third.name.clone(), value: 0.30 },
                ],
            },
        ],
    }
}

fn topmass_scenario_study_spec(
    spec: &MeasurementCombinationSpec,
) -> MeasurementCombinationScenarioStudySpec {
    let first = spec.systematics.first().expect("top-mass spec should have systematics");
    let second = spec
        .systematics
        .iter()
        .find(|systematic| systematic.name != first.name)
        .expect("top-mass spec should have at least two systematics");
    MeasurementCombinationScenarioStudySpec {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
        scenarios: vec![
            MeasurementCombinationScenarioSpec {
                name: "leading_source".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: first.name.clone(),
                    value: 0.30,
                }],
            },
            MeasurementCombinationScenarioSpec {
                name: "paired_sources".to_string(),
                error_on_error: vec![
                    ScenarioErrorOnErrorAssignment { systematic: first.name.clone(), value: 0.30 },
                    ScenarioErrorOnErrorAssignment { systematic: second.name.clone(), value: 0.20 },
                ],
            },
        ],
    }
}

fn bench_measurement_combine_fit(c: &mut Criterion) {
    let paper_topmass = load_topmass_full_fixture();
    let synthetic_32x24 = synthetic_measurement_combination_spec("mu", 32, 24, 0.05);
    let synthetic_64x48 = synthetic_measurement_combination_spec("mu", 64, 48, 0.05);

    let mut group = c.benchmark_group("measurement_combine_fit");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let bench_cases = [
        ("paper_topmass_full", &paper_topmass, vec![MeasurementCombinationSolver::Auto]),
        (
            "synthetic_gvm_32x24",
            &synthetic_32x24,
            vec![
                MeasurementCombinationSolver::Auto,
                MeasurementCombinationSolver::AnalyticPerturbative,
                MeasurementCombinationSolver::NumericalPaper,
            ],
        ),
        (
            "synthetic_gvm_64x48",
            &synthetic_64x48,
            vec![
                MeasurementCombinationSolver::Auto,
                MeasurementCombinationSolver::AnalyticPerturbative,
            ],
        ),
    ];

    for (label, spec, solvers) in bench_cases {
        for solver in solvers {
            group.bench_with_input(
                BenchmarkId::new(label, format!("{solver:?}")),
                spec,
                |b, spec| {
                    b.iter(|| {
                        let result =
                            combine_measurements_with_solver(black_box(spec), 0.68, solver)
                                .expect("measurement combination benchmark should succeed");
                        black_box(result.mu_hat)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_measurement_combine_calibration(c: &mut Criterion) {
    let paper_topmass_calibration =
        spec_with_error_on_error(&load_topmass_full_fixture(), "b-JES", 0.50);
    let synthetic_32x24_calibration = synthetic_measurement_combination_spec("mu", 32, 24, 0.05);
    let synthetic_64x48_calibration = synthetic_measurement_combination_spec("mu", 64, 48, 0.05);
    let synthetic_32x24_campaign_base = synthetic_measurement_combination_spec("mu", 32, 24, 0.0);
    let topmass_campaign_base = load_topmass_full_fixture();
    let synthetic_campaign = synthetic_scenario_study_spec(&synthetic_32x24_campaign_base);
    let topmass_campaign = topmass_scenario_study_spec(&topmass_campaign_base);

    let mut calibration_group = c.benchmark_group("measurement_combine_calibration");
    calibration_group.sample_size(10);
    calibration_group.warm_up_time(Duration::from_millis(500));
    calibration_group.measurement_time(Duration::from_secs(3));

    calibration_group.bench_with_input(
        BenchmarkId::new("paper_topmass_full", "Auto_n_toys_16"),
        &paper_topmass_calibration,
        |b, spec| {
            b.iter(|| {
                let report = calibrate_measurements_toys_with_solver(
                    black_box(spec),
                    0.68,
                    MeasurementCombinationSolver::Auto,
                    16,
                    42,
                )
                .expect("top-mass calibration benchmark should succeed");
                black_box(report.summary.mean_sigma_star_to_sigma_ratio)
            })
        },
    );

    for solver in [
        MeasurementCombinationSolver::Auto,
        MeasurementCombinationSolver::AnalyticPerturbative,
        MeasurementCombinationSolver::NumericalPaper,
    ] {
        calibration_group.bench_with_input(
            BenchmarkId::new("synthetic_gvm_32x24", format!("{solver:?}_n_toys_8")),
            &synthetic_32x24_calibration,
            |b, spec| {
                b.iter(|| {
                    let report = calibrate_measurements_toys_with_solver(
                        black_box(spec),
                        0.68,
                        solver,
                        8,
                        42,
                    )
                    .expect("synthetic calibration benchmark should succeed");
                    black_box(report.summary.mean_sigma_star_to_sigma_ratio)
                })
            },
        );
    }

    for solver in [MeasurementCombinationSolver::Auto, MeasurementCombinationSolver::NumericalPaper]
    {
        calibration_group.bench_with_input(
            BenchmarkId::new("synthetic_gvm_64x48", format!("{solver:?}_n_toys_4")),
            &synthetic_64x48_calibration,
            |b, spec| {
                b.iter(|| {
                    let report = calibrate_measurements_toys_with_solver(
                        black_box(spec),
                        0.68,
                        solver,
                        4,
                        42,
                    )
                    .expect("large synthetic calibration benchmark should succeed");
                    black_box(report.summary.mean_sigma_star_to_sigma_ratio)
                })
            },
        );
    }

    calibration_group.finish();

    let mut campaign_group = c.benchmark_group("measurement_combine_calibration_campaign");
    campaign_group.sample_size(10);
    campaign_group.warm_up_time(Duration::from_millis(500));
    campaign_group.measurement_time(Duration::from_secs(3));

    campaign_group.bench_with_input(
        BenchmarkId::new("paper_topmass_full", "Auto_s2_seeds2_n_toys_8"),
        &topmass_campaign_base,
        |b, spec| {
            b.iter(|| {
                let report = run_measurement_combination_calibration_campaign_with_solver(
                    black_box(spec),
                    black_box(&topmass_campaign),
                    0.68,
                    MeasurementCombinationSolver::Auto,
                    8,
                    &[42, 43],
                )
                .expect("top-mass campaign benchmark should succeed");
                black_box(report.aggregate.max_calibration_mean_sigma_star_to_sigma_ratio)
            })
        },
    );

    campaign_group.bench_with_input(
        BenchmarkId::new("synthetic_gvm_32x24", "Auto_s3_seeds2_n_toys_4"),
        &synthetic_32x24_campaign_base,
        |b, spec| {
            b.iter(|| {
                let report = run_measurement_combination_calibration_campaign_with_solver(
                    black_box(spec),
                    black_box(&synthetic_campaign),
                    0.68,
                    MeasurementCombinationSolver::Auto,
                    4,
                    &[42, 43],
                )
                .expect("synthetic campaign benchmark should succeed");
                black_box(report.aggregate.max_calibration_mean_sigma_star_to_sigma_ratio)
            })
        },
    );

    campaign_group.finish();

    let parity_scenario = MeasurementCombinationScenarioStudySpec {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
        scenarios: vec![MeasurementCombinationScenarioSpec {
            name: "bjes_0p05".to_string(),
            error_on_error: vec![ScenarioErrorOnErrorAssignment {
                systematic: "b-JES".to_string(),
                value: 0.05,
            }],
        }],
    };

    let mut parity_group = c.benchmark_group("measurement_combine_solver_parity_calibration");
    parity_group.sample_size(10);
    parity_group.warm_up_time(Duration::from_millis(500));
    parity_group.measurement_time(Duration::from_secs(3));

    parity_group.bench_with_input(
        BenchmarkId::new("paper_topmass_full", "NumericalPaper_vs_Analytic_n_toys_8"),
        &topmass_campaign_base,
        |b, spec| {
            b.iter(|| {
                let report = compare_measurement_combination_calibration_campaign_solvers(
                    black_box(spec),
                    black_box(&parity_scenario),
                    0.68,
                    MeasurementCombinationSolver::NumericalPaper,
                    MeasurementCombinationSolver::AnalyticPerturbative,
                    8,
                    &[2026, 2027],
                )
                .expect("top-mass solver parity calibration benchmark should succeed");
                black_box(report.aggregate.max_calibration_ratio_center_abs_diff)
            })
        },
    );

    parity_group.finish();
}

fn bench_measurement_combine_multithread(c: &mut Criterion) {
    let threads = rayon::current_num_threads();
    let paper_topmass = load_topmass_full_fixture();
    let synthetic_32x24 = synthetic_measurement_combination_spec("mu", 32, 24, 0.05);

    // --- Calibration: 256 toys ---
    // Run with: RAYON_NUM_THREADS=N cargo bench -- 'mt_'
    let mut cal_group = c.benchmark_group("mt_calibration_256toys");
    cal_group.sample_size(10);
    cal_group.warm_up_time(Duration::from_millis(300));
    cal_group.measurement_time(Duration::from_secs(5));

    let spec_cal = spec_with_error_on_error(&paper_topmass, "b-JES", 0.50);
    cal_group.bench_function(format!("paper_topmass/{threads}t"), |b| {
        b.iter(|| {
            let report = calibrate_measurements_toys_with_solver(
                black_box(&spec_cal),
                0.68,
                MeasurementCombinationSolver::Auto,
                256,
                42,
            )
            .expect("calibration should succeed");
            black_box(report.summary.mean_sigma_star_to_sigma_ratio)
        })
    });

    cal_group.bench_function(format!("synthetic_32x24/{threads}t"), |b| {
        b.iter(|| {
            let report = calibrate_measurements_toys_with_solver(
                black_box(&synthetic_32x24),
                0.68,
                MeasurementCombinationSolver::Auto,
                256,
                42,
            )
            .expect("calibration should succeed");
            black_box(report.summary.mean_sigma_star_to_sigma_ratio)
        })
    });

    cal_group.finish();

    // --- Campaign: 8 scenarios x 8 seeds x 32 toys ---
    let first = paper_topmass.systematics.first().unwrap();
    let second = paper_topmass.systematics.iter().find(|s| s.name != first.name).unwrap();
    let big_scenarios = MeasurementCombinationScenarioStudySpec {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
        scenarios: (0..8)
            .map(|i| {
                let eoe_val = 0.05 + 0.05 * (i as f64);
                MeasurementCombinationScenarioSpec {
                    name: format!("scenario_{i}"),
                    error_on_error: vec![
                        ScenarioErrorOnErrorAssignment {
                            systematic: first.name.clone(),
                            value: eoe_val,
                        },
                        ScenarioErrorOnErrorAssignment {
                            systematic: second.name.clone(),
                            value: eoe_val * 0.5,
                        },
                    ],
                }
            })
            .collect(),
    };
    let seeds_8: Vec<u64> = (42..50).collect();

    let mut campaign_group = c.benchmark_group("mt_campaign_8s_8seeds_32toys");
    campaign_group.sample_size(10);
    campaign_group.warm_up_time(Duration::from_millis(300));
    campaign_group.measurement_time(Duration::from_secs(8));

    campaign_group.bench_function(format!("paper_topmass/{threads}t"), |b| {
        b.iter(|| {
            let report = run_measurement_combination_calibration_campaign_with_solver(
                black_box(&paper_topmass),
                black_box(&big_scenarios),
                0.68,
                MeasurementCombinationSolver::Auto,
                32,
                &seeds_8,
            )
            .expect("campaign should succeed");
            black_box(report.aggregate.max_calibration_mean_sigma_star_to_sigma_ratio)
        })
    });

    campaign_group.finish();
}

criterion_group!(
    benches,
    bench_measurement_combine_fit,
    bench_measurement_combine_calibration,
    bench_measurement_combine_multithread,
);
criterion_main!(benches);
