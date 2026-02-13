#![cfg(feature = "arrow-io")]

use ns_translate::arrow::{export, ingest};
use ns_translate::pyhf::schema::{
    Channel, HistoSysData, Measurement, MeasurementConfig, Modifier, NormSysData, Observation,
    Sample, Workspace,
};

#[test]
fn arrow_ingest_with_modifiers_roundtrips_basic_modifier_types() {
    let ws = Workspace {
        channels: vec![Channel {
            name: "ch".into(),
            samples: vec![Sample {
                name: "bkg".into(),
                data: vec![1.0, 2.0],
                modifiers: vec![
                    Modifier::NormFactor { name: "mu".into(), data: None },
                    Modifier::NormSys {
                        name: "alpha".into(),
                        data: NormSysData { hi: 1.1, lo: 0.9 },
                    },
                    Modifier::HistoSys {
                        name: "shape".into(),
                        data: HistoSysData { hi_data: vec![1.1, 2.2], lo_data: vec![0.9, 1.8] },
                    },
                    Modifier::StatError { name: "staterr_ch".into(), data: vec![0.1, 0.2] },
                ],
            }],
        }],
        observations: vec![Observation { name: "ch".into(), data: vec![1.0, 2.0] }],
        measurements: vec![Measurement {
            name: "meas".into(),
            config: MeasurementConfig { poi: "mu".into(), parameters: vec![] },
        }],
        version: Some("1.0.0".into()),
    };

    let yields_batch = export::yields_from_workspace(&ws).expect("yields export should succeed");
    let mods_batch =
        export::modifiers_to_record_batch(&ws).expect("modifiers export should succeed");

    let config = ingest::ArrowIngestConfig {
        poi: "mu".into(),
        observations: None,
        measurement_name: "meas".into(),
    };
    let ws2 = ingest::from_record_batches_with_modifiers(&[yields_batch], &[mods_batch], &config)
        .expect("ingest should succeed");

    assert_eq!(ws2.channels.len(), 1);
    assert_eq!(ws2.channels[0].name, "ch");
    assert_eq!(ws2.channels[0].samples.len(), 1);

    let s = &ws2.channels[0].samples[0];
    assert_eq!(s.name, "bkg");
    assert_eq!(s.data, vec![1.0, 2.0]);

    let mut n_normfactor = 0usize;
    let mut n_normsys = 0usize;
    let mut n_histosys = 0usize;
    let mut n_staterror = 0usize;

    for m in &s.modifiers {
        match m {
            Modifier::NormFactor { name, .. } => {
                assert_eq!(name, "mu");
                n_normfactor += 1;
            }
            Modifier::NormSys { name, data } => {
                assert_eq!(name, "alpha");
                assert!((data.hi - 1.1).abs() < 1e-12);
                assert!((data.lo - 0.9).abs() < 1e-12);
                n_normsys += 1;
            }
            Modifier::HistoSys { name, data } => {
                assert_eq!(name, "shape");
                assert_eq!(data.hi_data, vec![1.1, 2.2]);
                assert_eq!(data.lo_data, vec![0.9, 1.8]);
                n_histosys += 1;
            }
            Modifier::StatError { name, data } => {
                assert_eq!(name, "staterr_ch");
                assert_eq!(data.as_slice(), &[0.1, 0.2]);
                n_staterror += 1;
            }
            _ => {}
        }
    }

    assert_eq!(n_normfactor, 1);
    assert_eq!(n_normsys, 1);
    assert_eq!(n_histosys, 1);
    assert_eq!(n_staterror, 1);
}
