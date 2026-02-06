#!/usr/bin/env python3
"""Generate ROOT fixture files for ns-root tests.

Requires: pip install uproot awkward numpy
"""

import json
import os

import numpy as np
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

        # Histogram with sumw2
        import uproot.writing

        # Use uproot's histogram writing
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


if __name__ == "__main__":
    create_simple_root()
    create_histfactory_fixtures()
    print("\nAll fixtures created successfully.")
