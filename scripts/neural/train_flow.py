#!/usr/bin/env python3
"""Train a normalizing flow (Neural Spline Flow) and export to ONNX for NextStat.

Produces two ONNX models and a ``flow_manifest.json`` compatible with
``ns-unbinned``'s ``FlowPdf``.

Usage
-----
# Unconditional 1-D flow on [0, 10]:
python scripts/neural/train_flow.py \
    --data samples.npy \
    --observables mass \
    --support 0 10 \
    --output-dir models/signal_flow

# Conditional 2-D flow with 1 context param:
python scripts/neural/train_flow.py \
    --data samples.npy \
    --observables x y \
    --support 0 1 0 1 \
    --context-features 1 \
    --context-names alpha \
    --output-dir models/cond_flow

Requirements
------------
pip install torch zuko onnx onnxruntime numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def build_flow(
    features: int,
    context: int,
    transforms: int,
    hidden: int,
    bins: int,
) -> nn.Module:
    """Build a Neural Spline Flow (NSF) via zuko."""
    import zuko

    return zuko.flows.NSF(
        features=features,
        context=context,
        transforms=transforms,
        hidden_features=[hidden] * 3,
        bins=bins,
    )


class LogProbWrapper(nn.Module):
    """Wraps a zuko flow for ONNX export: input x [, c] → log_prob [batch]."""

    def __init__(self, flow: nn.Module, context: int):
        super().__init__()
        self.flow = flow
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.context > 0:
            obs = x[:, : -self.context]
            ctx = x[:, -self.context :]
        else:
            obs = x
            ctx = None
        dist = self.flow(ctx)
        return dist.log_prob(obs)


class SampleWrapper(nn.Module):
    """Wraps a zuko flow for ONNX export: input z [, c] → x [batch, features]."""

    def __init__(self, flow: nn.Module, context: int, features: int):
        super().__init__()
        self.flow = flow
        self.context = context
        self.features = features

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.context > 0:
            noise = z[:, : self.features]
            ctx = z[:, self.features :]
        else:
            noise = z
            ctx = None
        dist = self.flow(ctx)
        return dist.transform(noise)


def train(
    flow: nn.Module,
    data: np.ndarray,
    context_data: np.ndarray | None,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> list[float]:
    """Train flow by maximizing log-likelihood. Returns per-epoch loss history."""
    flow = flow.to(device)
    flow.train()

    x = torch.tensor(data, dtype=torch.float32, device=device)
    c = (
        torch.tensor(context_data, dtype=torch.float32, device=device)
        if context_data is not None
        else None
    )

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = x.shape[0]
    losses: list[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x[idx]
            cb = c[idx] if c is not None else None

            dist = flow(cb)
            loss = -dist.log_prob(xb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:4d}/{epochs}  NLL = {avg_loss:.4f}")

    return losses


def export_onnx(
    flow: nn.Module,
    features: int,
    context: int,
    output_dir: Path,
) -> tuple[str, str]:
    """Export log_prob and sample ONNX models. Returns filenames."""
    flow.eval()
    flow = flow.cpu()

    # --- log_prob model ---
    logprob_wrapper = LogProbWrapper(flow, context)
    logprob_wrapper.eval()
    input_dim = features + context
    dummy_x = torch.randn(2, input_dim)
    logprob_file = "log_prob.onnx"

    torch.onnx.export(
        logprob_wrapper,
        dummy_x,
        str(output_dir / logprob_file),
        input_names=["input"],
        output_names=["log_prob"],
        dynamic_axes={"input": {0: "batch"}, "log_prob": {0: "batch"}},
        opset_version=17,
    )
    print(f"  exported {logprob_file}")

    # --- sample model ---
    sample_wrapper = SampleWrapper(flow, context, features)
    sample_wrapper.eval()
    sample_input_dim = features + context
    dummy_z = torch.randn(2, sample_input_dim)
    sample_file = "sample.onnx"

    torch.onnx.export(
        sample_wrapper,
        dummy_z,
        str(output_dir / sample_file),
        input_names=["input"],
        output_names=["samples"],
        dynamic_axes={"input": {0: "batch"}, "samples": {0: "batch"}},
        opset_version=17,
    )
    print(f"  exported {sample_file}")

    return logprob_file, sample_file


def write_manifest(
    output_dir: Path,
    features: int,
    context: int,
    observable_names: list[str],
    context_names: list[str],
    support: list[list[float]],
    logprob_file: str,
    sample_file: str,
    training_info: dict | None = None,
) -> None:
    """Write flow_manifest.json."""
    manifest = {
        "schema_version": "nextstat_flow_v0",
        "flow_type": "nsf",
        "features": features,
        "context_features": context,
        "observable_names": observable_names,
        "context_names": context_names,
        "support": support,
        "base_distribution": "standard_normal",
        "models": {
            "log_prob": logprob_file,
            "sample": sample_file,
        },
    }
    if training_info:
        manifest["training"] = training_info

    path = output_dir / "flow_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {path}")


def verify_onnx(output_dir: Path, features: int, context: int) -> None:
    """Quick sanity check: run both models through ONNX Runtime."""
    import onnxruntime as ort

    input_dim = features + context
    x = np.random.randn(8, input_dim).astype(np.float32)

    sess_lp = ort.InferenceSession(str(output_dir / "log_prob.onnx"))
    lp = sess_lp.run(None, {"input": x})[0]
    assert lp.shape == (8,), f"log_prob shape mismatch: {lp.shape}"
    assert np.all(np.isfinite(lp)), "log_prob contains non-finite values"
    print(f"  log_prob check OK: shape={lp.shape}, range=[{lp.min():.2f}, {lp.max():.2f}]")

    sess_s = ort.InferenceSession(str(output_dir / "sample.onnx"))
    z = np.random.randn(8, input_dim).astype(np.float32)
    s = sess_s.run(None, {"input": z})[0]
    assert s.shape == (8, features), f"sample shape mismatch: {s.shape}"
    assert np.all(np.isfinite(s)), "sample contains non-finite values"
    print(f"  sample  check OK: shape={s.shape}, range=[{s.min():.2f}, {s.max():.2f}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Neural Spline Flow and export to ONNX for NextStat"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (.npy, shape [N, features] or [N] for 1-D)",
    )
    parser.add_argument(
        "--context-data",
        type=Path,
        default=None,
        help="Path to context data (.npy, shape [N, context_features]). Optional.",
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        required=True,
        help="Observable names (one per feature dimension)",
    )
    parser.add_argument(
        "--support",
        nargs="+",
        type=float,
        required=True,
        help="Support bounds as lo hi lo hi ... (2 values per observable)",
    )
    parser.add_argument(
        "--context-features",
        type=int,
        default=0,
        help="Number of context features (default: 0, unconditional)",
    )
    parser.add_argument(
        "--context-names",
        nargs="*",
        default=[],
        help="Context parameter names (one per context feature)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (default: 500)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--transforms", type=int, default=5, help="Number of NSF transforms (default: 5)"
    )
    parser.add_argument(
        "--hidden", type=int, default=64, help="Hidden layer width (default: 64)"
    )
    parser.add_argument("--bins", type=int, default=8, help="Spline bins (default: 8)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip ONNX Runtime verification"
    )

    args = parser.parse_args()

    features = len(args.observables)
    context = args.context_features

    if len(args.support) != features * 2:
        print(
            f"ERROR: --support must have {features * 2} values (2 per observable), got {len(args.support)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if context > 0 and len(args.context_names) != context:
        print(
            f"ERROR: --context-names must have {context} entries, got {len(args.context_names)}",
            file=sys.stderr,
        )
        sys.exit(1)

    support = [[args.support[2 * i], args.support[2 * i + 1]] for i in range(features)]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load data ---
    print(f"Loading data from {args.data}")
    data = np.load(args.data).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] != features:
        print(
            f"ERROR: data has {data.shape[1]} columns but {features} observables specified",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  {data.shape[0]} events, {features} features")

    context_data = None
    if args.context_data is not None:
        context_data = np.load(args.context_data).astype(np.float32)
        if context_data.ndim == 1:
            context_data = context_data.reshape(-1, 1)
        if context_data.shape[1] != context:
            print(
                f"ERROR: context data has {context_data.shape[1]} columns but {context} context features specified",
                file=sys.stderr,
            )
            sys.exit(1)
        if context_data.shape[0] != data.shape[0]:
            print(
                f"ERROR: context data has {context_data.shape[0]} rows but data has {data.shape[0]}",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Build and train ---
    print(f"Building NSF: {features}D, {context} context, {args.transforms} transforms, {args.hidden} hidden, {args.bins} bins")
    flow = build_flow(features, context, args.transforms, args.hidden, args.bins)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  {n_params:,} trainable parameters")

    print(f"Training on {args.device} for {args.epochs} epochs...")
    losses = train(flow, data, context_data, args.epochs, args.batch_size, args.lr, args.device)

    # --- Export ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX to {args.output_dir}")
    logprob_file, sample_file = export_onnx(flow, features, context, args.output_dir)

    training_info = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "transforms": args.transforms,
        "hidden": args.hidden,
        "bins": args.bins,
        "n_params": n_params,
        "n_events": int(data.shape[0]),
        "final_nll": float(losses[-1]),
        "seed": args.seed,
    }

    write_manifest(
        args.output_dir,
        features,
        context,
        args.observables,
        args.context_names,
        support,
        logprob_file,
        sample_file,
        training_info,
    )

    if not args.skip_verify:
        print("Verifying ONNX models...")
        verify_onnx(args.output_dir, features, context)

    print("Done.")


if __name__ == "__main__":
    main()
