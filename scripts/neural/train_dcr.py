#!/usr/bin/env python3
"""Train a DCR (Direct Classifier Ratio) surrogate for HistFactory template morphing.

Implements the FAIR-HUC protocol: generate synthetic data from a binned
HistFactory model at many nuisance parameter points α, then train a
conditional Neural Spline Flow p_θ(x | α) to approximate the morphed
template distributions.

The resulting ONNX models + manifest can be used as a ``dcr_surrogate``
PDF in NextStat unbinned specs, replacing ``MorphingHistogramPdf`` with
smooth, continuous, bin-free morphing.

Usage
-----
# From a NextStat JSON workspace:
python scripts/neural/train_dcr.py \\
    --workspace workspace.json \\
    --channel SR \\
    --process background \\
    --output-dir models/bkg_dcr \\
    --n-samples-per-point 5000 \\
    --n-alpha-points 200

# From explicit histogram templates:
python scripts/neural/train_dcr.py \\
    --templates templates.json \\
    --output-dir models/bkg_dcr

Requirements
------------
pip install torch zuko onnx onnxruntime numpy scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sample_from_histogram(
    bin_edges: np.ndarray,
    bin_contents: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample events from a histogram PDF via inverse CDF."""
    widths = np.diff(bin_edges)
    densities = bin_contents / (np.sum(bin_contents * widths) + 1e-30)
    cdf = np.concatenate([[0.0], np.cumsum(densities * widths)])
    cdf = cdf / cdf[-1]

    u = rng.uniform(0, 1, size=n_samples)
    bin_idx = np.searchsorted(cdf, u, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_contents) - 1)

    lo = bin_edges[bin_idx]
    hi = bin_edges[bin_idx + 1]
    x = rng.uniform(lo, hi)
    return x


def interpolate_template(
    nominal: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    alpha: float,
    interp_code: str = "code0",
) -> np.ndarray:
    """Interpolate histogram template at nuisance parameter value α.

    code0: piecewise linear (HistFactory InterpCode 0)
    code4p: polynomial + exponential (HistFactory InterpCode 4p)
    """
    if interp_code == "code0":
        if alpha >= 0:
            return nominal + alpha * (up - nominal)
        else:
            return nominal - alpha * (down - nominal)
    elif interp_code == "code4p":
        if abs(alpha) >= 1:
            if alpha >= 0:
                return nominal * np.exp(alpha * np.log(up / (nominal + 1e-30) + 1e-30))
            else:
                return nominal * np.exp(-alpha * np.log(down / (nominal + 1e-30) + 1e-30))
        else:
            # Polynomial interpolation for |α| < 1
            t = alpha
            f_up = up / (nominal + 1e-30)
            f_down = down / (nominal + 1e-30)
            log_up = np.log(np.clip(f_up, 1e-30, None))
            log_down = np.log(np.clip(f_down, 1e-30, None))
            # 6th order polynomial that matches function and first two derivatives at ±1
            S = 0.5 * (log_up + log_down)
            A = 0.5 * (log_up - log_down)
            log_factor = t * (A + t * S * (1 - t**2 / 3))
            return nominal * np.exp(log_factor)
    else:
        raise ValueError(f"Unknown interp_code: {interp_code}")


def load_templates_from_json(path: Path) -> dict:
    """Load template specifications from a JSON file.

    Expected format:
    {
      "observable": "mass",
      "bin_edges": [5000, 5100, ...],
      "nominal": [100.0, 95.0, ...],
      "systematics": [
        {
          "name": "jes_alpha",
          "up": [102.0, 94.0, ...],
          "down": [98.0, 96.0, ...],
          "interp_code": "code0"
        }
      ]
    }
    """
    with open(path) as f:
        return json.load(f)


def load_templates_from_workspace(
    workspace_path: Path,
    channel_name: str,
    process_name: str,
) -> dict:
    """Load template data from a NextStat/pyhf JSON workspace.

    Extracts bin edges, nominal, and systematic up/down templates for the
    specified channel and process.
    """
    with open(workspace_path) as f:
        ws = json.load(f)

    channels = ws.get("channels", [])
    target_ch = None
    for ch in channels:
        if ch["name"] == channel_name:
            target_ch = ch
            break
    if target_ch is None:
        print(f"ERROR: channel '{channel_name}' not found in workspace", file=sys.stderr)
        sys.exit(1)

    target_sample = None
    for sample in target_ch.get("samples", []):
        if sample["name"] == process_name:
            target_sample = sample
            break
    if target_sample is None:
        print(
            f"ERROR: process '{process_name}' not found in channel '{channel_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    nominal = np.array(target_sample["data"], dtype=np.float64)
    n_bins = len(nominal)

    # Extract bin edges from observations or infer
    observations = ws.get("observations", [])
    obs_data = None
    for obs in observations:
        if obs["name"] == channel_name:
            obs_data = obs
            break

    if obs_data and "binning" in obs_data:
        bin_edges = np.array(obs_data["binning"], dtype=np.float64)
    else:
        bin_edges = np.arange(n_bins + 1, dtype=np.float64)

    systematics = []
    for modifier in target_sample.get("modifiers", []):
        if modifier["type"] == "histosys":
            data = modifier["data"]
            systematics.append({
                "name": modifier["name"],
                "up": np.array(data["hi_data"], dtype=np.float64),
                "down": np.array(data["lo_data"], dtype=np.float64),
                "interp_code": "code0",
            })

    observable = channel_name
    return {
        "observable": observable,
        "bin_edges": bin_edges.tolist(),
        "nominal": nominal.tolist(),
        "systematics": systematics,
    }


def generate_training_data(
    templates: dict,
    n_samples_per_point: int,
    n_alpha_points: int,
    alpha_range: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (x, alpha) training pairs from template interpolation.

    Returns:
        x: [N_total] observable values
        alpha: [N_total, n_systematics] nuisance parameter values
    """
    rng = np.random.default_rng(seed)
    bin_edges = np.array(templates["bin_edges"])
    nominal = np.array(templates["nominal"])
    systs = templates["systematics"]
    n_syst = len(systs)

    all_x = []
    all_alpha = []

    for _ in range(n_alpha_points):
        # Sample alpha uniformly
        alphas = rng.uniform(-alpha_range, alpha_range, size=n_syst)

        # Interpolate template
        morphed = nominal.copy()
        for j, s in enumerate(systs):
            up = np.array(s["up"])
            down = np.array(s["down"])
            interp_code = s.get("interp_code", "code0")
            morphed = interpolate_template(morphed, up, down, alphas[j], interp_code)

        # Ensure non-negative
        morphed = np.maximum(morphed, 0.0)

        # Sample from morphed histogram
        if morphed.sum() > 0:
            x = sample_from_histogram(bin_edges, morphed, n_samples_per_point, rng)
            alpha_broadcast = np.tile(alphas, (n_samples_per_point, 1))
            all_x.append(x)
            all_alpha.append(alpha_broadcast)

    x = np.concatenate(all_x).astype(np.float32)
    alpha = np.concatenate(all_alpha).astype(np.float32)
    return x, alpha


def build_flow(features: int, context: int, transforms: int, hidden: int, bins: int):
    """Build conditional NSF flow."""
    import zuko

    return zuko.flows.NSF(
        features=features,
        context=context,
        transforms=transforms,
        hidden_features=[hidden] * 3,
        bins=bins,
    )


def train_dcr(
    flow,
    x: np.ndarray,
    alpha: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> list[float]:
    """Train DCR surrogate by maximizing conditional log-likelihood."""
    flow = flow.to(device)
    flow.train()

    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(-1)
    a_t = torch.tensor(alpha, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = x_t.shape[0]
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_t[idx]
            ab = a_t[idx]

            dist = flow(ab)
            loss = -dist.log_prob(xb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:4d}/{epochs}  NLL = {avg:.4f}")

    return losses


def export_dcr_onnx(flow, context: int, output_dir: Path) -> tuple[str, str]:
    """Export log_prob and sample ONNX models for DCR surrogate."""
    import torch.nn as nn

    flow.eval()
    flow = flow.cpu()

    class LogProbWrap(nn.Module):
        def __init__(self, flow, ctx):
            super().__init__()
            self.flow = flow
            self.ctx = ctx

        def forward(self, x):
            obs = x[:, :1]
            ctx = x[:, 1:]
            return self.flow(ctx).log_prob(obs)

    class SampleWrap(nn.Module):
        def __init__(self, flow, ctx):
            super().__init__()
            self.flow = flow
            self.ctx = ctx

        def forward(self, z):
            noise = z[:, :1]
            ctx = z[:, 1:]
            return self.flow(ctx).transform(noise)

    input_dim = 1 + context
    dummy = torch.randn(2, input_dim)

    lp_file = "log_prob.onnx"
    torch.onnx.export(
        LogProbWrap(flow, context),
        dummy,
        str(output_dir / lp_file),
        input_names=["input"],
        output_names=["log_prob"],
        dynamic_axes={"input": {0: "batch"}, "log_prob": {0: "batch"}},
        opset_version=17,
    )
    print(f"  exported {lp_file}")

    s_file = "sample.onnx"
    torch.onnx.export(
        SampleWrap(flow, context),
        dummy,
        str(output_dir / s_file),
        input_names=["input"],
        output_names=["samples"],
        dynamic_axes={"input": {0: "batch"}, "samples": {0: "batch"}},
        opset_version=17,
    )
    print(f"  exported {s_file}")

    return lp_file, s_file


def validate_dcr(
    output_dir: Path,
    templates: dict,
    context: int,
    n_alpha_test: int = 50,
    n_quad: int = 128,
) -> dict:
    """Validate DCR surrogate against original templates."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_dir / "log_prob.onnx"))
    bin_edges = np.array(templates["bin_edges"])
    nominal = np.array(templates["nominal"])
    systs = templates["systematics"]
    lo, hi = bin_edges[0], bin_edges[-1]

    # Gauss-Legendre nodes for normalization check
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    half = (hi - lo) / 2.0
    mid = (lo + hi) / 2.0
    x_quad = (mid + half * nodes).astype(np.float32)
    w_quad = weights * half

    rng = np.random.default_rng(123)
    nll_diffs = []
    norm_devs = []

    for _ in range(n_alpha_test):
        alphas = rng.uniform(-2.0, 2.0, size=len(systs)).astype(np.float32)

        # Reference: binned template NLL proxy (sum of -log(template_density) at quad points)
        morphed = nominal.copy()
        for j, s in enumerate(systs):
            up = np.array(s["up"])
            down = np.array(s["down"])
            interp_code = s.get("interp_code", "code0")
            morphed = interpolate_template(morphed, up, down, float(alphas[j]), interp_code)
        morphed = np.maximum(morphed, 0.0)
        widths = np.diff(bin_edges)
        total = np.sum(morphed * widths)
        if total <= 0:
            continue
        densities_ref = morphed / total

        # Neural: evaluate log p(x | alpha) at quad points
        ctx = np.tile(alphas.reshape(1, -1), (n_quad, 1))
        inp = np.hstack([x_quad.reshape(-1, 1), ctx]).astype(np.float32)
        log_p = session.run(None, {"input": inp})[0]

        # Normalization check
        integral = float(np.sum(np.exp(log_p) * w_quad))
        norm_devs.append(abs(integral - 1.0))

        # NLL comparison at quad points
        bin_idx = np.searchsorted(bin_edges, x_quad, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, len(densities_ref) - 1)
        ref_log_p = np.log(np.clip(densities_ref[bin_idx] , 1e-30, None))
        nll_diff = float(np.mean(np.abs(log_p.flatten() - ref_log_p)))
        nll_diffs.append(nll_diff)

    results = {
        "mean_nll_diff": float(np.mean(nll_diffs)) if nll_diffs else float("nan"),
        "max_nll_diff": float(np.max(nll_diffs)) if nll_diffs else float("nan"),
        "mean_norm_deviation": float(np.mean(norm_devs)) if norm_devs else float("nan"),
        "max_norm_deviation": float(np.max(norm_devs)) if norm_devs else float("nan"),
        "n_test_points": len(nll_diffs),
    }

    print(f"  Mean |NLL diff|     = {results['mean_nll_diff']:.4f}")
    print(f"  Max  |NLL diff|     = {results['max_nll_diff']:.4f}")
    print(f"  Mean |norm dev|     = {results['mean_norm_deviation']:.6f}")
    print(f"  Max  |norm dev|     = {results['max_norm_deviation']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train a DCR surrogate for HistFactory template morphing"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--templates", type=Path, help="Template JSON file")
    grp.add_argument("--workspace", type=Path, help="NextStat/pyhf JSON workspace")
    parser.add_argument("--channel", type=str, default=None, help="Channel name (for --workspace)")
    parser.add_argument("--process", type=str, default=None, help="Process name (for --workspace)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples-per-point", type=int, default=5000)
    parser.add_argument("--n-alpha-points", type=int, default=200)
    parser.add_argument("--alpha-range", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transforms", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-validate", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load templates ---
    if args.templates:
        print(f"Loading templates from {args.templates}")
        templates = load_templates_from_json(args.templates)
    else:
        if not args.channel or not args.process:
            print("ERROR: --channel and --process required with --workspace", file=sys.stderr)
            sys.exit(1)
        print(f"Loading templates from {args.workspace} (channel={args.channel}, process={args.process})")
        templates = load_templates_from_workspace(args.workspace, args.channel, args.process)

    syst_names = [s["name"] for s in templates["systematics"]]
    n_syst = len(syst_names)
    observable = templates.get("observable", "x")
    bin_edges = templates["bin_edges"]
    lo, hi = bin_edges[0], bin_edges[-1]

    print(f"  Observable: {observable}")
    print(f"  Support: [{lo}, {hi}]")
    print(f"  Systematics ({n_syst}): {syst_names}")
    print(f"  Bins: {len(bin_edges) - 1}")

    # --- Generate training data ---
    print(f"\nGenerating training data: {args.n_alpha_points} α-points × {args.n_samples_per_point} samples")
    x, alpha = generate_training_data(
        templates, args.n_samples_per_point, args.n_alpha_points, args.alpha_range, args.seed
    )
    print(f"  Total training events: {len(x)}")

    # --- Build and train ---
    print(f"\nBuilding conditional NSF: 1D, {n_syst} context, {args.transforms} transforms")
    flow = build_flow(1, n_syst, args.transforms, args.hidden, args.bins)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  {n_params:,} trainable parameters")

    print(f"\nTraining on {args.device} for {args.epochs} epochs...")
    losses = train_dcr(flow, x, alpha, args.epochs, args.batch_size, args.lr, args.device)

    # --- Export ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting ONNX to {args.output_dir}")
    lp_file, s_file = export_dcr_onnx(flow, n_syst, args.output_dir)

    training_info = {
        "protocol": "FAIR-HUC",
        "epochs": args.epochs,
        "n_alpha_points": args.n_alpha_points,
        "n_samples_per_point": args.n_samples_per_point,
        "alpha_range": args.alpha_range,
        "transforms": args.transforms,
        "hidden": args.hidden,
        "bins": args.bins,
        "n_params": n_params,
        "final_nll": float(losses[-1]),
        "seed": args.seed,
    }

    manifest = {
        "schema_version": "nextstat_flow_v0",
        "flow_type": "dcr_nsf",
        "features": 1,
        "context_features": n_syst,
        "observable_names": [observable],
        "context_names": syst_names,
        "support": [[lo, hi]],
        "base_distribution": "standard_normal",
        "models": {"log_prob": lp_file, "sample": s_file},
        "training": training_info,
    }

    manifest_path = args.output_dir / "flow_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {manifest_path}")

    # --- Validate ---
    if not args.skip_validate:
        print("\nValidating DCR surrogate against original templates...")
        val_results = validate_dcr(args.output_dir, templates, n_syst)
        manifest["validation"] = val_results
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
