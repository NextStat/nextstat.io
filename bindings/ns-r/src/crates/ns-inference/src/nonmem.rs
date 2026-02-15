//! NONMEM-format dataset reader for pharmacometric analysis.
//!
//! Parses CSV files with standard NONMEM column conventions:
//! - **ID**: subject identifier
//! - **TIME**: observation/dosing time
//! - **DV**: dependent variable (observed concentration)
//! - **AMT**: dose amount (> 0 for dosing events)
//! - **EVID**: event ID (0 = observation, 1 = dose)
//! - **MDV**: missing DV flag (0 = DV valid, 1 = DV missing)
//! - **CMT**: compartment (optional; 1 = depot/oral, 2 = central/IV)
//! - **RATE**: infusion rate (optional; > 0 → infusion with duration = AMT/RATE)
//!
//! Converts parsed records into [`DosingRegimen`] and observation vectors
//! suitable for PK model fitting.

use ns_core::{Error, Result};
use std::collections::BTreeMap;

use crate::dosing::{DoseEvent, DoseRoute, DosingRegimen};

/// A single record (row) from a NONMEM-style dataset.
#[derive(Debug, Clone)]
pub struct NonmemRecord {
    /// Subject identifier (string, mapped to integer index).
    pub id: String,
    /// Time of event.
    pub time: f64,
    /// Dependent variable (observed concentration). Meaningful only when `evid == 0 && mdv == 0`.
    pub dv: f64,
    /// Dose amount. > 0 for dosing records.
    pub amt: f64,
    /// Event ID: 0 = observation, 1 = dose event.
    pub evid: u8,
    /// Missing DV flag: 0 = DV is valid, 1 = DV should be ignored.
    pub mdv: u8,
    /// Compartment number (default 1). 1 = depot/oral, 2 = central/IV.
    pub cmt: u8,
    /// Infusion rate. If > 0, the dose is an infusion with duration = AMT / RATE.
    pub rate: f64,
}

/// A parsed NONMEM-format dataset.
///
/// Records are grouped by subject ID and sorted by time within each subject.
#[derive(Debug, Clone)]
pub struct NonmemDataset {
    /// All records, sorted by (subject_index, time).
    records: Vec<NonmemRecord>,
    /// Unique subject IDs in order of first appearance.
    subject_ids: Vec<String>,
    /// Map from subject ID string → index in `subject_ids`.
    subject_map: BTreeMap<String, usize>,
}

impl NonmemDataset {
    /// Parse a NONMEM-format CSV string.
    ///
    /// Column names are case-insensitive. Required: `ID`, `TIME`, `DV`.
    /// Optional (defaulted): `AMT` (0), `EVID` (inferred), `MDV` (inferred),
    /// `CMT` (1), `RATE` (0).
    pub fn from_csv(csv_text: &str) -> Result<Self> {
        let mut lines = csv_text.lines();
        let header_line = lines.next().ok_or_else(|| Error::Validation("empty CSV".to_string()))?;

        let headers: Vec<String> =
            header_line.split(',').map(|h| h.trim().to_uppercase()).collect();

        let col = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

        let id_col = col("ID").ok_or_else(|| Error::Validation("missing ID column".to_string()))?;
        let time_col =
            col("TIME").ok_or_else(|| Error::Validation("missing TIME column".to_string()))?;
        let dv_col = col("DV").ok_or_else(|| Error::Validation("missing DV column".to_string()))?;
        let amt_col = col("AMT");
        let evid_col = col("EVID");
        let mdv_col = col("MDV");
        let cmt_col = col("CMT");
        let rate_col = col("RATE");

        let mut records = Vec::new();
        let mut subject_ids: Vec<String> = Vec::new();
        let mut subject_map: BTreeMap<String, usize> = BTreeMap::new();

        for (line_no, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();

            let parse_f64 = |idx: usize, name: &str| -> Result<f64> {
                fields
                    .get(idx)
                    .and_then(|s| if s.is_empty() || *s == "." { None } else { Some(*s) })
                    .map(|s| {
                        s.parse::<f64>().map_err(|_| {
                            Error::Validation(format!(
                                "line {}: cannot parse {name} '{s}' as f64",
                                line_no + 2
                            ))
                        })
                    })
                    .unwrap_or(Ok(0.0))
            };

            let parse_u8 = |idx: usize, name: &str| -> Result<u8> {
                fields
                    .get(idx)
                    .and_then(|s| if s.is_empty() || *s == "." { None } else { Some(*s) })
                    .map(|s| {
                        s.parse::<u8>().map_err(|_| {
                            Error::Validation(format!(
                                "line {}: cannot parse {name} '{s}' as u8",
                                line_no + 2
                            ))
                        })
                    })
                    .unwrap_or(Ok(0))
            };

            let id_str = fields
                .get(id_col)
                .ok_or_else(|| {
                    Error::Validation(format!("line {}: missing ID field", line_no + 2))
                })?
                .to_string();

            let time = parse_f64(time_col, "TIME")?;
            let dv = parse_f64(dv_col, "DV")?;
            let amt = amt_col.map(|c| parse_f64(c, "AMT")).unwrap_or(Ok(0.0))?;
            let rate = rate_col.map(|c| parse_f64(c, "RATE")).unwrap_or(Ok(0.0))?;
            let cmt = cmt_col.map(|c| parse_u8(c, "CMT")).unwrap_or(Ok(1))?;

            let evid = if let Some(c) = evid_col {
                parse_u8(c, "EVID")?
            } else if amt > 0.0 {
                1
            } else {
                0
            };

            let mdv = if let Some(c) = mdv_col {
                parse_u8(c, "MDV")?
            } else if evid != 0 {
                1
            } else {
                0
            };

            if !subject_map.contains_key(&id_str) {
                let idx = subject_ids.len();
                subject_ids.push(id_str.clone());
                subject_map.insert(id_str.clone(), idx);
            }

            records.push(NonmemRecord { id: id_str, time, dv, amt, evid, mdv, cmt, rate });
        }

        if records.is_empty() {
            return Err(Error::Validation("no data records found".to_string()));
        }

        records.sort_by(|a, b| {
            let ia = subject_map[&a.id];
            let ib = subject_map[&b.id];
            ia.cmp(&ib).then(a.time.total_cmp(&b.time))
        });

        Ok(Self { records, subject_ids, subject_map })
    }

    /// Number of unique subjects.
    pub fn n_subjects(&self) -> usize {
        self.subject_ids.len()
    }

    /// Unique subject IDs in order of first appearance.
    pub fn subject_ids(&self) -> &[String] {
        &self.subject_ids
    }

    /// All parsed records.
    pub fn records(&self) -> &[NonmemRecord] {
        &self.records
    }

    /// Records for a specific subject.
    pub fn subject_records(&self, id: &str) -> Vec<&NonmemRecord> {
        self.records.iter().filter(|r| r.id == id).collect()
    }

    /// Extract observation data: `(times, dv_values, subject_indices)`.
    ///
    /// Only includes records where `evid == 0` and `mdv == 0`.
    pub fn observation_data(&self) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut times = Vec::new();
        let mut dv = Vec::new();
        let mut subj_idx = Vec::new();
        for r in &self.records {
            if r.evid == 0 && r.mdv == 0 {
                times.push(r.time);
                dv.push(r.dv);
                subj_idx.push(self.subject_map[&r.id]);
            }
        }
        (times, dv, subj_idx)
    }

    /// Build a [`DosingRegimen`] for a specific subject from dosing records.
    ///
    /// Converts EVID=1 records to dose events:
    /// - `CMT == 1` → `DoseRoute::Oral` (default bioavailability = 1.0)
    /// - `CMT == 2` and `RATE == 0` → `DoseRoute::IvBolus`
    /// - `CMT == 2` and `RATE > 0` → `DoseRoute::Infusion` (duration = AMT / RATE)
    pub fn dosing_regimen(&self, id: &str) -> Result<DosingRegimen> {
        self.dosing_regimen_with_bioav(id, 1.0)
    }

    /// Build a [`DosingRegimen`] for a specific subject with custom oral bioavailability.
    pub fn dosing_regimen_with_bioav(
        &self,
        id: &str,
        bioavailability: f64,
    ) -> Result<DosingRegimen> {
        let events: Vec<DoseEvent> = self
            .records
            .iter()
            .filter(|r| r.id == id && r.evid == 1 && r.amt > 0.0)
            .map(|r| {
                let route = if r.cmt == 1 {
                    DoseRoute::Oral { bioavailability }
                } else if r.rate > 0.0 {
                    DoseRoute::Infusion { duration: r.amt / r.rate }
                } else {
                    DoseRoute::IvBolus
                };
                DoseEvent { time: r.time, amount: r.amt, route }
            })
            .collect();

        if events.is_empty() {
            return Err(Error::Validation(format!("no dosing records found for subject '{id}'")));
        }
        DosingRegimen::from_events(events)
    }

    /// Build dosing regimens for all subjects.
    pub fn all_dosing_regimens(&self) -> Result<BTreeMap<String, DosingRegimen>> {
        self.all_dosing_regimens_with_bioav(1.0)
    }

    /// Build dosing regimens for all subjects with custom oral bioavailability.
    pub fn all_dosing_regimens_with_bioav(
        &self,
        bioavailability: f64,
    ) -> Result<BTreeMap<String, DosingRegimen>> {
        let mut map = BTreeMap::new();
        for id in &self.subject_ids {
            match self.dosing_regimen_with_bioav(id, bioavailability) {
                Ok(reg) => {
                    map.insert(id.clone(), reg);
                }
                Err(_) => {
                    // Subject has no dosing records (observation-only); skip.
                }
            }
        }
        if map.is_empty() {
            return Err(Error::Validation("no dosing records found in dataset".to_string()));
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CSV: &str = "\
ID,TIME,AMT,DV,EVID,MDV,CMT
1,0,100,0,1,1,1
1,0.5,0,2.3,0,0,1
1,1,0,4.1,0,0,1
1,2,0,3.5,0,0,1
2,0,100,0,1,1,1
2,0.5,0,1.9,0,0,1
2,1,0,3.8,0,0,1
2,2,0,3.2,0,0,1
";

    #[test]
    fn parse_minimal_csv() {
        let ds = NonmemDataset::from_csv(MINIMAL_CSV).unwrap();
        assert_eq!(ds.n_subjects(), 2);
        assert_eq!(ds.subject_ids(), &["1", "2"]);
        assert_eq!(ds.records().len(), 8);
    }

    #[test]
    fn observation_data_extraction() {
        let ds = NonmemDataset::from_csv(MINIMAL_CSV).unwrap();
        let (times, dv, subj_idx) = ds.observation_data();
        assert_eq!(times.len(), 6);
        assert_eq!(dv.len(), 6);
        assert_eq!(subj_idx.len(), 6);
        assert!((dv[0] - 2.3).abs() < 1e-10);
        assert_eq!(subj_idx[0], 0);
        assert_eq!(subj_idx[3], 1);
    }

    #[test]
    fn dosing_regimen_oral() {
        let ds = NonmemDataset::from_csv(MINIMAL_CSV).unwrap();
        let reg = ds.dosing_regimen("1").unwrap();
        assert_eq!(reg.n_doses(), 1);
        assert!((reg.total_amount() - 100.0).abs() < 1e-12);
        match reg.events()[0].route {
            DoseRoute::Oral { bioavailability } => {
                assert!((bioavailability - 1.0).abs() < 1e-12);
            }
            _ => panic!("expected oral route for CMT=1"),
        }
    }

    #[test]
    fn iv_bolus_csv() {
        let csv = "\
ID,TIME,AMT,DV,EVID,MDV,CMT
1,0,200,0,1,1,2
1,0.5,0,15.0,0,0,2
1,1,0,12.0,0,0,2
";
        let ds = NonmemDataset::from_csv(csv).unwrap();
        let reg = ds.dosing_regimen("1").unwrap();
        assert!(matches!(reg.events()[0].route, DoseRoute::IvBolus));
    }

    #[test]
    fn infusion_csv() {
        let csv = "\
ID,TIME,AMT,DV,EVID,MDV,CMT,RATE
1,0,100,0,1,1,2,50
1,1,0,8.0,0,0,2,0
1,2,0,12.0,0,0,2,0
1,4,0,9.0,0,0,2,0
";
        let ds = NonmemDataset::from_csv(csv).unwrap();
        let reg = ds.dosing_regimen("1").unwrap();
        match reg.events()[0].route {
            DoseRoute::Infusion { duration } => {
                assert!((duration - 2.0).abs() < 1e-12, "duration = AMT/RATE = 100/50 = 2");
            }
            _ => panic!("expected infusion route"),
        }
    }

    #[test]
    fn multi_dose_regimen() {
        let csv = "\
ID,TIME,AMT,DV,EVID,MDV,CMT
1,0,100,0,1,1,1
1,0.5,0,2.0,0,0,1
1,12,100,0,1,1,1
1,12.5,0,5.0,0,0,1
1,24,100,0,1,1,1
1,24.5,0,7.0,0,0,1
";
        let ds = NonmemDataset::from_csv(csv).unwrap();
        let reg = ds.dosing_regimen("1").unwrap();
        assert_eq!(reg.n_doses(), 3);
        assert!((reg.total_amount() - 300.0).abs() < 1e-12);
    }

    #[test]
    fn all_dosing_regimens() {
        let ds = NonmemDataset::from_csv(MINIMAL_CSV).unwrap();
        let regs = ds.all_dosing_regimens().unwrap();
        assert_eq!(regs.len(), 2);
        assert!(regs.contains_key("1"));
        assert!(regs.contains_key("2"));
    }

    #[test]
    fn inferred_evid_mdv() {
        let csv = "\
ID,TIME,AMT,DV
1,0,100,0
1,1,0,4.5
1,2,0,3.2
";
        let ds = NonmemDataset::from_csv(csv).unwrap();
        let (times, dv, _) = ds.observation_data();
        assert_eq!(times.len(), 2);
        assert!((dv[0] - 4.5).abs() < 1e-10);
        assert!((dv[1] - 3.2).abs() < 1e-10);

        let reg = ds.dosing_regimen("1").unwrap();
        assert_eq!(reg.n_doses(), 1);
    }

    #[test]
    fn missing_required_columns() {
        assert!(NonmemDataset::from_csv("TIME,DV\n0,1.0").is_err());
        assert!(NonmemDataset::from_csv("ID,DV\n1,1.0").is_err());
        assert!(NonmemDataset::from_csv("ID,TIME\n1,0").is_err());
    }

    #[test]
    fn empty_csv_rejected() {
        assert!(NonmemDataset::from_csv("").is_err());
        assert!(NonmemDataset::from_csv("ID,TIME,DV\n").is_err());
    }

    #[test]
    fn custom_bioavailability() {
        let ds = NonmemDataset::from_csv(MINIMAL_CSV).unwrap();
        let reg = ds.dosing_regimen_with_bioav("1", 0.85).unwrap();
        match reg.events()[0].route {
            DoseRoute::Oral { bioavailability } => {
                assert!((bioavailability - 0.85).abs() < 1e-12);
            }
            _ => panic!("expected oral route"),
        }
    }

    #[test]
    fn subject_records_filter() {
        let ds = NonmemDataset::from_csv(MINIMAL_CSV).unwrap();
        let recs = ds.subject_records("1");
        assert_eq!(recs.len(), 4);
        assert!(recs.iter().all(|r| r.id == "1"));
    }
}
