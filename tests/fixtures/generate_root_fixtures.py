#!/usr/bin/env python3
"""Generate ROOT fixture files for ns-root tests.

Requires: pip install uproot awkward numpy
"""

import json
import os

import numpy as np
import awkward as ak
import uproot

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.join(FIXTURES_DIR, "histfactory")


def create_simple_root():
    """Create a simple ROOT file with known TH1D histograms."""
    path = os.path.join(FIXTURES_DIR, "simple_histos.root")

    with uproot.recreate(path) as f:
        # Simple 3-bin histogram with known content
        f["hist1"] = (
            np.array([10.0, 20.0, 30.0]),  # bin contents
            np.array([0.0, 1.0, 2.0, 3.0]),  # bin edges
        )

        # Histogram with 4 bins
        f["hist_sw2"] = (
            np.array([5.0, 15.0, 25.0, 35.0]),
            np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        )

        # Histogram in a subdirectory
        f["subdir/nested_hist"] = (
            np.array([1.0, 2.0]),
            np.array([0.0, 10.0, 20.0]),
        )

    print(f"Created {path}")

    # Write expected values as JSON for Rust tests
    expected = {
        "hist1": {
            "n_bins": 3,
            "x_min": 0.0,
            "x_max": 3.0,
            "bin_content": [10.0, 20.0, 30.0],
            "bin_edges": [0.0, 1.0, 2.0, 3.0],
        },
        "hist_sw2": {
            "n_bins": 4,
            "x_min": 0.0,
            "x_max": 2.0,
            "bin_content": [5.0, 15.0, 25.0, 35.0],
            "bin_edges": [0.0, 0.5, 1.0, 1.5, 2.0],
        },
        "subdir/nested_hist": {
            "n_bins": 2,
            "x_min": 0.0,
            "x_max": 20.0,
            "bin_content": [1.0, 2.0],
            "bin_edges": [0.0, 10.0, 20.0],
        },
    }
    expected_path = os.path.join(FIXTURES_DIR, "simple_histos_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Created {expected_path}")


def create_histfactory_fixtures():
    """Create a HistFactory workspace in both ROOT+XML and pyhf JSON formats.

    Simple model: 1 channel (SR), 2 samples (signal, background).
    Signal: normfactor "mu" (POI)
    Background: normsys "bkg_norm" + staterror
    """
    # Channel: 3 bins
    obs_data = [15.0, 25.0, 12.0]
    sig_nominal = [5.0, 10.0, 3.0]
    bkg_nominal = [10.0, 18.0, 9.0]

    # NormSys: hi/lo factors for background
    normsys_hi = 1.1
    normsys_lo = 0.9

    # StatError uncertainties (sqrt of sumw2 / nominal for relative)
    staterror_rel = [0.1, 0.08, 0.15]  # relative uncertainties

    bin_edges = [0.0, 1.0, 2.0, 3.0]

    # ── ROOT file ────────────────────────────────────
    root_path = os.path.join(HF_DIR, "data.root")
    with uproot.recreate(root_path) as f:
        # Observed data
        f["SR/data_obs"] = (np.array(obs_data), np.array(bin_edges))

        # Signal nominal
        f["SR/signal_nominal"] = (np.array(sig_nominal), np.array(bin_edges))

        # Background nominal
        f["SR/bkg_nominal"] = (np.array(bkg_nominal), np.array(bin_edges))

        # Background normsys up/down
        bkg_up = [b * normsys_hi for b in bkg_nominal]
        bkg_down = [b * normsys_lo for b in bkg_nominal]
        f["SR/bkg_normsys_up"] = (np.array(bkg_up), np.array(bin_edges))
        f["SR/bkg_normsys_down"] = (np.array(bkg_down), np.array(bin_edges))

    print(f"Created {root_path}")

    # ── combination.xml ──────────────────────────────
    combination_xml = """<!DOCTYPE Combination SYSTEM "HistFactorySchema.dtd">
<Combination OutputFilePrefix="results">
  <Input>channel_SR.xml</Input>
  <Measurement Name="NominalMeasurement" Lumi="1.0" LumiRelErr="0.0" ExportOnly="True">
    <POI>mu</POI>
  </Measurement>
</Combination>
"""
    comb_path = os.path.join(HF_DIR, "combination.xml")
    with open(comb_path, "w") as f:
        f.write(combination_xml)
    print(f"Created {comb_path}")

    # ── channel_SR.xml ───────────────────────────────
    # Compute absolute stat errors
    staterror_abs = [r * n for r, n in zip(staterror_rel, bkg_nominal)]

    channel_xml = f"""<!DOCTYPE Channel SYSTEM "HistFactorySchema.dtd">
<Channel Name="SR" InputFile="{os.path.basename(root_path)}" HistoPath="SR">
  <Data HistoName="data_obs" />
  <Sample Name="signal" HistoName="signal_nominal" NormalizeByTheory="True">
    <NormFactor Name="mu" Val="1.0" Low="0.0" High="10.0" />
  </Sample>
  <Sample Name="background" HistoName="bkg_nominal" NormalizeByTheory="True">
    <OverallSys Name="bkg_norm" Low="{normsys_lo}" High="{normsys_hi}" />
    <StatError Activate="True" />
  </Sample>
</Channel>
"""
    chan_path = os.path.join(HF_DIR, "channel_SR.xml")
    with open(chan_path, "w") as f:
        f.write(channel_xml)
    print(f"Created {chan_path}")

    # ── pyhf JSON workspace ──────────────────────────
    # This must produce the exact same model as the XML+ROOT path.
    # StatError uses sqrt(sumw2) which for unweighted histograms = sqrt(N).
    # uproot writes sumw2 = bin_content for unweighted histograms.
    staterror_data = [np.sqrt(b) for b in bkg_nominal]

    workspace = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "signal",
                        "data": sig_nominal,
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            }
                        ],
                    },
                    {
                        "name": "background",
                        "data": bkg_nominal,
                        "modifiers": [
                            {
                                "name": "bkg_norm",
                                "type": "normsys",
                                "data": {"hi": normsys_hi, "lo": normsys_lo},
                            },
                            {
                                "name": "staterror_SR",
                                "type": "staterror",
                                "data": staterror_data,
                            },
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "SR", "data": obs_data}],
        "measurements": [
            {
                "name": "NominalMeasurement",
                "config": {
                    "poi": "mu",
                    "parameters": [],
                },
            }
        ],
        "version": "1.0.0",
    }

    ws_path = os.path.join(HF_DIR, "workspace.json")
    with open(ws_path, "w") as f:
        json.dump(workspace, f, indent=2)
    print(f"Created {ws_path}")


def create_simple_tree():
    """Create a ROOT file with a TTree for TTree reader tests.

    Tree "events" with 1000 entries and branches:
      pt       (float32)
      eta      (float32)
      njet     (int32)
      mbb      (float64)
      weight_mc     (float64)
      weight_jes_up   (float64)
      weight_jes_down (float64)
    """
    path = os.path.join(FIXTURES_DIR, "simple_tree.root")

    rng = np.random.RandomState(42)
    n = 1000

    pt = rng.exponential(50.0, n).astype(np.float32)
    eta = rng.uniform(-2.5, 2.5, n).astype(np.float32)
    njet = rng.poisson(3, n).astype(np.int32)
    mbb = rng.normal(125.0, 30.0, n).astype(np.float64)
    weight_mc = rng.uniform(0.5, 1.5, n).astype(np.float64)
    weight_jes_up = weight_mc * rng.normal(1.05, 0.02, n)
    weight_jes_down = weight_mc * rng.normal(0.95, 0.02, n)

    with uproot.recreate(path) as f:
        # Use mktree + extend to write a classic TTree (not RNTuple)
        f.mktree("events", {
            "pt": np.float32,
            "eta": np.float32,
            "njet": np.int32,
            "mbb": np.float64,
            "weight_mc": np.float64,
            "weight_jes_up": np.float64,
            "weight_jes_down": np.float64,
        })
        f["events"].extend({
            "pt": pt,
            "eta": eta,
            "njet": njet,
            "mbb": mbb,
            "weight_mc": weight_mc,
            "weight_jes_up": weight_jes_up,
            "weight_jes_down": weight_jes_down,
        })

    print(f"Created {path}")

    # Write expected values for Rust tests
    expected = {
        "n_entries": n,
        "branches": {
            "pt": {
                "type": "float32",
                "first_5": pt[:5].tolist(),
                "sum": float(pt.sum()),
            },
            "eta": {
                "type": "float32",
                "first_5": eta[:5].tolist(),
                "sum": float(eta.sum()),
            },
            "njet": {
                "type": "int32",
                "first_5": njet[:5].tolist(),
                "sum": int(njet.sum()),
            },
            "mbb": {
                "type": "float64",
                "first_5": mbb[:5].tolist(),
                "sum": float(mbb.sum()),
            },
            "weight_mc": {
                "type": "float64",
                "first_5": weight_mc[:5].tolist(),
                "sum": float(weight_mc.sum()),
            },
        },
        # numpy.histogram reference for cross-validation
        "mbb_histogram": {
            "bin_edges": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0],
            "bin_content_unweighted": np.histogram(
                mbb, bins=[0.0, 50.0, 100.0, 150.0, 200.0, 300.0]
            )[0].tolist(),
            "bin_content_weighted": np.histogram(
                mbb,
                bins=[0.0, 50.0, 100.0, 150.0, 200.0, 300.0],
                weights=weight_mc,
            )[0].tolist(),
        },
        # With selection: njet >= 4
        "mbb_histogram_selected": {
            "bin_edges": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0],
            "bin_content_weighted": np.histogram(
                mbb[njet >= 4],
                bins=[0.0, 50.0, 100.0, 150.0, 200.0, 300.0],
                weights=weight_mc[njet >= 4],
            )[0].tolist(),
        },
    }

    expected_path = os.path.join(FIXTURES_DIR, "simple_tree_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Created {expected_path}")

def create_vector_tree():
    """Create a ROOT file with a TTree containing a jagged (variable-length) array branch.

    This fixture is used to test expression-style indexing like `jet_pt[0]`.
    """
    path = os.path.join(FIXTURES_DIR, "vector_tree.root")

    jet_pt = ak.Array([
        [10.0, 11.0],
        [20.0],
        [],
        [30.0, 31.0, 32.0],
        [40.0],
        [],
        [50.0, 51.0],
        [60.0],
    ])
    n = len(jet_pt)

    # Expected materialized scalar columns for indexing.
    jet_pt_0 = [10.0, 20.0, 0.0, 30.0, 40.0, 0.0, 50.0, 60.0]
    jet_pt_1 = [11.0, 0.0, 0.0, 31.0, 0.0, 0.0, 51.0, 0.0]

    with uproot.recreate(path) as f:
        f.mktree("events", {
            "jet_pt": "var * float32",
        })
        f["events"].extend({
            "jet_pt": ak.values_astype(jet_pt, np.float32),
        })

    print(f"Created {path}")

    expected = {
        "n_entries": n,
        "jet_pt_0": jet_pt_0,
        "jet_pt_1": jet_pt_1,
    }
    expected_path = os.path.join(FIXTURES_DIR, "vector_tree_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Created {expected_path}")

def create_fixed_array_tree():
    """Create a ROOT file with a TTree containing a fixed-length array branch.

    This is used to validate expressions like `eig[0]` when the underlying storage
    is fixed-size per entry (no entry-offset table).
    """
    path = os.path.join(FIXTURES_DIR, "fixed_array_tree.root")
    expected_path = os.path.join(FIXTURES_DIR, "fixed_array_tree_expected.json")

    # 8 entries, 4 elements each.
    eig = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [0.0, 0.0, 0.0, 0.0],
            [5.5, 6.5, 7.5, 8.5],
            [9.0, 8.0, 7.0, 6.0],
            [100.0, 200.0, 300.0, 400.0],
            [11.0, 22.0, 33.0, 44.0],
            [3.14, 2.71, 1.61, 0.0],
        ],
        dtype=np.float32,
    )

    with uproot.recreate(path) as f:
        f.mktree(
            "events",
            {
                "eig": "4 * float32",
            },
        )
        f["events"].extend(
            {
                "eig": eig,
            }
        )

    expected = {
        "n_entries": int(eig.shape[0]),
        "eig_0": eig[:, 0].astype(np.float64).tolist(),
        "eig_1": eig[:, 1].astype(np.float64).tolist(),
        "eig_3": eig[:, 3].astype(np.float64).tolist(),
    }
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)

    print(f"Created {path}")
    print(f"Created {expected_path}")

if __name__ == "__main__":
    create_simple_root()
    create_histfactory_fixtures()
    create_simple_tree()
    create_vector_tree()
    create_fixed_array_tree()
    print("\nAll fixtures created successfully.")
