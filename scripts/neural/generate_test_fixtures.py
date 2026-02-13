#!/usr/bin/env python3
"""Generate minimal ONNX test fixtures for ns-unbinned integration tests.

Creates a simple 1-D "flow" that computes standard normal log-probability:
    log p(x) = -0.5 * x^2 - 0.5 * ln(2π)

And a sample (inverse) model that applies the identity transform (z → x = z).

Usage:
    python scripts/neural/generate_test_fixtures.py

Output:
    tests/fixtures/flow_test/
        flow_manifest.json
        flow_logprob.onnx
        flow_sample.onnx
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


def make_logprob_model() -> onnx.ModelProto:
    """Create an ONNX model that computes standard normal log_prob.

    Input:  x [batch, 1]
    Output: log_prob [batch]

    log_prob = -0.5 * x^2 - 0.5 * ln(2π)
    """
    # Constants
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    log2pi_half = numpy_helper.from_array(
        np.array([0.5 * np.log(2 * np.pi)], dtype=np.float32), name="log2pi_half"
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 1])
    log_prob_out = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, ["batch"])

    # x_sq = x * x
    mul_node = helper.make_node("Mul", ["x", "x"], ["x_sq"])
    # half_x_sq = half * x_sq
    mul_half = helper.make_node("Mul", ["half", "x_sq"], ["half_x_sq"])
    # neg_half_x_sq = -half_x_sq
    neg_node = helper.make_node("Neg", ["half_x_sq"], ["neg_half_x_sq"])
    # logp_2d = neg_half_x_sq - log2pi_half  (shape: [batch, 1])
    sub_node = helper.make_node("Sub", ["neg_half_x_sq", "log2pi_half"], ["logp_2d"])
    # Squeeze to [batch]
    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="squeeze_axes")
    squeeze_node = helper.make_node("Squeeze", ["logp_2d", "squeeze_axes"], ["log_prob"])

    graph = helper.make_graph(
        [mul_node, mul_half, neg_node, sub_node, squeeze_node],
        "logprob_graph",
        [x],
        [log_prob_out],
        initializer=[half, log2pi_half, axes_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def make_sample_model() -> onnx.ModelProto:
    """Create an ONNX model that implements identity transform (z → x = z).

    For a standard normal flow, sampling is just: x = z where z ~ N(0,1).

    Input:  z [batch, 1]
    Output: x [batch, 1]
    """
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", 1])
    x_out = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 1])

    # Identity: x = z
    identity_node = helper.make_node("Identity", ["z"], ["x"])

    graph = helper.make_graph(
        [identity_node],
        "sample_graph",
        [z],
        [x_out],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def make_manifest() -> dict:
    """Create the flow_manifest.json for the test fixture."""
    return {
        "schema_version": "nextstat_flow_v0",
        "flow_type": "test_standard_normal",
        "features": 1,
        "context_features": 0,
        "observable_names": ["x"],
        "context_names": [],
        "support": [[-6.0, 6.0]],
        "base_distribution": "standard_normal",
        "models": {
            "log_prob": "flow_logprob.onnx",
            "sample": "flow_sample.onnx",
        },
        "training": {
            "library": "test_fixture",
            "architecture": "identity",
            "note": "Minimal test fixture: standard normal log-prob + identity sampling",
        },
        "validation": {
            "pit_ks_pvalue": 1.0,
            "closure_bias_percent": 0.0,
            "normalization_check": 1.0,
        },
    }


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(repo_root, "tests", "fixtures", "flow_test")
    os.makedirs(out_dir, exist_ok=True)

    logprob_model = make_logprob_model()
    sample_model = make_sample_model()
    manifest = make_manifest()

    logprob_path = os.path.join(out_dir, "flow_logprob.onnx")
    sample_path = os.path.join(out_dir, "flow_sample.onnx")
    manifest_path = os.path.join(out_dir, "flow_manifest.json")

    onnx.save(logprob_model, logprob_path)
    onnx.save(sample_model, sample_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated test fixtures in {out_dir}/")
    print(f"  {logprob_path} ({os.path.getsize(logprob_path)} bytes)")
    print(f"  {sample_path} ({os.path.getsize(sample_path)} bytes)")
    print(f"  {manifest_path} ({os.path.getsize(manifest_path)} bytes)")

    # Verify by running inference
    import onnxruntime as ort

    sess = ort.InferenceSession(logprob_path)
    test_x = np.array([[0.0], [1.0], [-1.0]], dtype=np.float32)
    result = sess.run(None, {"x": test_x})
    log_probs = result[0]
    expected = -0.5 * test_x.flatten() ** 2 - 0.5 * np.log(2 * np.pi)
    print(f"\nVerification (log_prob):")
    for i, (lp, exp) in enumerate(zip(log_probs, expected)):
        print(f"  x={test_x[i, 0]:+.1f}: log_prob={lp:.6f}, expected={exp:.6f}, diff={abs(lp - exp):.2e}")

    sess_s = ort.InferenceSession(sample_path)
    test_z = np.array([[0.5], [-0.5]], dtype=np.float32)
    result_s = sess_s.run(None, {"z": test_z})
    print(f"\nVerification (sample): z={test_z.flatten().tolist()} → x={result_s[0].flatten().tolist()}")


if __name__ == "__main__":
    main()
