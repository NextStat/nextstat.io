#!/usr/bin/env python3
"""Generate minimal ONNX test fixtures for DcrSurrogate integration tests.

Creates a conditional 1-D "flow" that computes a Gaussian with systematic shift:
    log p(x | α) = -0.5 * ((x - δ*α)/σ)^2 - 0.5 * ln(2πσ^2)

where δ = 0.5 (shift per unit α), σ = 1.0.  This simulates a simple
HistFactory-like template morphing: at α=0 we get the nominal standard
normal, and at α≠0 the distribution shifts.

The sample model applies x = z + δ*α.

Usage:
    python scripts/neural/generate_dcr_test_fixtures.py

Output:
    tests/fixtures/dcr_test/
        flow_manifest.json
        log_prob.onnx
        sample.onnx
"""

import json
import os
import sys

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("ERROR: onnx package required. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


DELTA = 0.5  # shift per unit alpha
SIGMA = 1.0


def make_conditional_logprob_model() -> onnx.ModelProto:
    """Create ONNX model: log N(x; δ*α, σ).

    Input:  input [batch, 2]  — column 0 = x (observable), column 1 = α (context)
    Output: log_prob [batch]

    log_prob = -0.5 * ((x - δ*α) / σ)^2 - 0.5 * ln(2πσ^2)
    """
    # Constants
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    delta = numpy_helper.from_array(np.array([DELTA], dtype=np.float32), name="delta")
    sigma = numpy_helper.from_array(np.array([SIGMA], dtype=np.float32), name="sigma")
    log2pi_sigma = numpy_helper.from_array(
        np.array([0.5 * np.log(2 * np.pi * SIGMA**2)], dtype=np.float32),
        name="log2pi_sigma",
    )

    # Slice indices
    zero_i = numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_i")
    one_i = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one_i")
    two_i = numpy_helper.from_array(np.array([2], dtype=np.int64), name="two_i")
    axis1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis1")

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2])
    log_prob_out = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, ["batch"])

    nodes = [
        # Slice x = input[:, 0:1]
        helper.make_node("Slice", ["input", "zero_i", "one_i", "axis1"], ["x_col"]),
        # Slice alpha = input[:, 1:2]
        helper.make_node("Slice", ["input", "one_i", "two_i", "axis1"], ["alpha_col"]),
        # mu = delta * alpha
        helper.make_node("Mul", ["delta", "alpha_col"], ["mu"]),
        # diff = x - mu
        helper.make_node("Sub", ["x_col", "mu"], ["diff"]),
        # normalized = diff / sigma
        helper.make_node("Div", ["diff", "sigma"], ["normalized"]),
        # sq = normalized^2
        helper.make_node("Mul", ["normalized", "normalized"], ["sq"]),
        # half_sq = 0.5 * sq
        helper.make_node("Mul", ["half", "sq"], ["half_sq"]),
        # neg_half_sq = -half_sq
        helper.make_node("Neg", ["half_sq"], ["neg_half_sq"]),
        # logp_2d = neg_half_sq - log2pi_sigma  (shape [batch, 1])
        helper.make_node("Sub", ["neg_half_sq", "log2pi_sigma"], ["logp_2d"]),
        # Squeeze to [batch]
        helper.make_node("Squeeze", ["logp_2d", "axis1"], ["log_prob"]),
    ]

    graph = helper.make_graph(
        nodes,
        "dcr_logprob_graph",
        [inp],
        [log_prob_out],
        initializer=[half, delta, sigma, log2pi_sigma, zero_i, one_i, two_i, axis1],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def make_conditional_sample_model() -> onnx.ModelProto:
    """Create ONNX model: x = z + δ*α.

    Input:  input [batch, 2]  — column 0 = z (noise), column 1 = α (context)
    Output: samples [batch, 1]
    """
    delta = numpy_helper.from_array(np.array([DELTA], dtype=np.float32), name="delta")
    zero_i = numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_i")
    one_i = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one_i")
    two_i = numpy_helper.from_array(np.array([2], dtype=np.int64), name="two_i")
    axis1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis1")

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2])
    x_out = helper.make_tensor_value_info("samples", TensorProto.FLOAT, ["batch", 1])

    nodes = [
        helper.make_node("Slice", ["input", "zero_i", "one_i", "axis1"], ["z_col"]),
        helper.make_node("Slice", ["input", "one_i", "two_i", "axis1"], ["alpha_col"]),
        helper.make_node("Mul", ["delta", "alpha_col"], ["shift"]),
        helper.make_node("Add", ["z_col", "shift"], ["samples"]),
    ]

    graph = helper.make_graph(
        nodes,
        "dcr_sample_graph",
        [inp],
        [x_out],
        initializer=[delta, zero_i, one_i, two_i, axis1],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def make_manifest() -> dict:
    return {
        "schema_version": "nextstat_flow_v0",
        "flow_type": "dcr_test_shifted_gaussian",
        "features": 1,
        "context_features": 1,
        "observable_names": ["mass"],
        "context_names": ["alpha_syst"],
        "support": [[-6.0, 6.0]],
        "base_distribution": "standard_normal",
        "models": {
            "log_prob": "log_prob.onnx",
            "sample": "sample.onnx",
        },
        "training": {
            "library": "test_fixture",
            "architecture": "shifted_gaussian",
            "note": "DCR test fixture: Gaussian with systematic shift δ=0.5 per unit α",
            "protocol": "FAIR-HUC",
        },
        "validation": {
            "normalization_check": 1.0,
        },
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(repo_root, "tests", "fixtures", "dcr_test")
    os.makedirs(out_dir, exist_ok=True)

    logprob_model = make_conditional_logprob_model()
    sample_model = make_conditional_sample_model()
    manifest = make_manifest()

    lp_path = os.path.join(out_dir, "log_prob.onnx")
    s_path = os.path.join(out_dir, "sample.onnx")
    m_path = os.path.join(out_dir, "flow_manifest.json")

    onnx.save(logprob_model, lp_path)
    onnx.save(sample_model, s_path)
    with open(m_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated DCR test fixtures in {out_dir}/")
    print(f"  {lp_path} ({os.path.getsize(lp_path)} bytes)")
    print(f"  {s_path} ({os.path.getsize(s_path)} bytes)")
    print(f"  {m_path} ({os.path.getsize(m_path)} bytes)")

    # Verify
    import onnxruntime as ort

    sess = ort.InferenceSession(lp_path)

    # Nominal: α=0 → standard normal
    test_inp = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    result = sess.run(None, {"input": test_inp})
    log_probs = result[0]
    expected = -0.5 * test_inp[:, 0] ** 2 - 0.5 * np.log(2 * np.pi)
    print(f"\nVerification (nominal α=0):")
    for i in range(len(test_inp)):
        x, a = test_inp[i]
        print(f"  x={x:+.1f}, α={a:+.1f}: log_prob={log_probs[i]:.6f}, expected={expected[i]:.6f}")

    # Shifted: α=2 → Gaussian(μ=1, σ=1)
    test_inp2 = np.array([[1.0, 2.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    result2 = sess.run(None, {"input": test_inp2})
    log_probs2 = result2[0]
    mu = DELTA * 2.0
    expected2 = -0.5 * ((test_inp2[:, 0] - mu) / SIGMA) ** 2 - 0.5 * np.log(2 * np.pi * SIGMA**2)
    print(f"\nVerification (shifted α=2, μ={mu}):")
    for i in range(len(test_inp2)):
        x, a = test_inp2[i]
        print(f"  x={x:+.1f}, α={a:+.1f}: log_prob={log_probs2[i]:.6f}, expected={expected2[i]:.6f}")

    # Sample model
    sess_s = ort.InferenceSession(s_path)
    z_inp = np.array([[0.0, 2.0], [1.0, -1.0]], dtype=np.float32)
    result_s = sess_s.run(None, {"input": z_inp})
    print(f"\nVerification (sample):")
    for i in range(len(z_inp)):
        z, a = z_inp[i]
        x_out = result_s[0][i, 0]
        x_expected = z + DELTA * a
        print(f"  z={z:+.1f}, α={a:+.1f}: x={x_out:.4f}, expected={x_expected:.4f}")


if __name__ == "__main__":
    main()
