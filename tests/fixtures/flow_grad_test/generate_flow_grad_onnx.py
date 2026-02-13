#!/usr/bin/env python3
"""Generate ONNX flow models for G3 analytical gradient test.

Creates a **conditional Gaussian** flow with 2 context parameters (mu, sigma).

    log_prob(x | mu, sigma) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2π)

Three ONNX models are exported:

1. ``log_prob.onnx``       — input ``(x, c) → log_prob [batch]``
2. ``log_prob_grad.onnx``  — input ``(x, c) → (log_prob [batch], d_log_prob_d_context [batch, 2])``
3. ``sample.onnx``         — input ``(z, c) → x [batch, 1]``

The analytical Jacobian is:

    ∂logp/∂mu    = (x - mu) / sigma^2
    ∂logp/∂sigma = ((x - mu)^2 / sigma^3) - 1/sigma

Usage:
    python generate_flow_grad_onnx.py
"""
import json
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_log_prob_model() -> onnx.ModelProto:
    """log_prob(x, c) → [batch]  where c = [mu, sigma]."""
    # Input: [batch, 3] = [x, mu, sigma]  (features=1, context_features=2)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, [None])

    # Constants
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    log2pi = numpy_helper.from_array(
        np.array([0.5 * np.log(2 * np.pi)], dtype=np.float32), name="log2pi"
    )
    idx_x = numpy_helper.from_array(np.int64(0), name="idx_x")
    idx_mu = numpy_helper.from_array(np.int64(1), name="idx_mu")
    idx_sigma = numpy_helper.from_array(np.int64(2), name="idx_sigma")

    nodes = [
        # Slice columns: x = input[:, 0], mu = input[:, 1], sigma = input[:, 2]
        helper.make_node("Gather", ["input", "idx_x"], ["x"], axis=1),
        helper.make_node("Gather", ["input", "idx_mu"], ["mu"], axis=1),
        helper.make_node("Gather", ["input", "idx_sigma"], ["sigma"], axis=1),
        # z = (x - mu) / sigma
        helper.make_node("Sub", ["x", "mu"], ["x_minus_mu"]),
        helper.make_node("Div", ["x_minus_mu", "sigma"], ["z"]),
        # z^2
        helper.make_node("Mul", ["z", "z"], ["z_sq"]),
        # -0.5 * z^2
        helper.make_node("Mul", ["half", "z_sq"], ["half_z_sq"]),
        helper.make_node("Neg", ["half_z_sq"], ["neg_half_z_sq"]),
        # -log(sigma)
        helper.make_node("Log", ["sigma"], ["log_sigma"]),
        helper.make_node("Neg", ["log_sigma"], ["neg_log_sigma"]),
        # log_prob = -0.5*z^2 - log(sigma) - 0.5*log(2*pi)
        helper.make_node("Add", ["neg_half_z_sq", "neg_log_sigma"], ["t1"]),
        helper.make_node("Sub", ["t1", "log2pi"], ["log_prob"]),
    ]

    graph = helper.make_graph(
        nodes,
        "log_prob_graph",
        [X],
        [Y],
        initializer=[half, one, log2pi, idx_x, idx_mu, idx_sigma],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def make_log_prob_grad_model() -> onnx.ModelProto:
    """log_prob_grad(x, c) → (log_prob [batch], d_log_prob_d_context [batch, 2])."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])
    Y_logp = helper.make_tensor_value_info("log_prob", TensorProto.FLOAT, [None])
    Y_jac = helper.make_tensor_value_info(
        "d_log_prob_d_context", TensorProto.FLOAT, [None, 2]
    )

    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    log2pi = numpy_helper.from_array(
        np.array([0.5 * np.log(2 * np.pi)], dtype=np.float32), name="log2pi"
    )
    neg_one = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="neg_one")
    idx_x = numpy_helper.from_array(np.int64(0), name="idx_x")
    idx_mu = numpy_helper.from_array(np.int64(1), name="idx_mu")
    idx_sigma = numpy_helper.from_array(np.int64(2), name="idx_sigma")

    nodes = [
        # Slice columns
        helper.make_node("Gather", ["input", "idx_x"], ["x"], axis=1),
        helper.make_node("Gather", ["input", "idx_mu"], ["mu"], axis=1),
        helper.make_node("Gather", ["input", "idx_sigma"], ["sigma"], axis=1),
        # z = (x - mu) / sigma
        helper.make_node("Sub", ["x", "mu"], ["x_minus_mu"]),
        helper.make_node("Div", ["x_minus_mu", "sigma"], ["z"]),
        # z^2
        helper.make_node("Mul", ["z", "z"], ["z_sq"]),
        # --- log_prob ---
        helper.make_node("Mul", ["half", "z_sq"], ["half_z_sq"]),
        helper.make_node("Neg", ["half_z_sq"], ["neg_half_z_sq"]),
        helper.make_node("Log", ["sigma"], ["log_sigma"]),
        helper.make_node("Neg", ["log_sigma"], ["neg_log_sigma"]),
        helper.make_node("Add", ["neg_half_z_sq", "neg_log_sigma"], ["t1"]),
        helper.make_node("Sub", ["t1", "log2pi"], ["log_prob"]),
        # --- Jacobian ---
        # sigma^2
        helper.make_node("Mul", ["sigma", "sigma"], ["sigma_sq"]),
        # sigma^3
        helper.make_node("Mul", ["sigma_sq", "sigma"], ["sigma_cu"]),
        # d_logp/d_mu = (x - mu) / sigma^2 = z / sigma
        helper.make_node("Div", ["z", "sigma"], ["dlogp_dmu"]),
        # d_logp/d_sigma = (x-mu)^2 / sigma^3 - 1/sigma
        helper.make_node("Mul", ["x_minus_mu", "x_minus_mu"], ["xmu_sq"]),
        helper.make_node("Div", ["xmu_sq", "sigma_cu"], ["term_a"]),
        helper.make_node("Div", ["one", "sigma"], ["inv_sigma"]),
        helper.make_node("Sub", ["term_a", "inv_sigma"], ["dlogp_dsigma"]),
        # Stack [dlogp_dmu, dlogp_dsigma] → [batch, 2]
        # Unsqueeze each to [batch, 1], then Concat
        helper.make_node("Unsqueeze", ["dlogp_dmu", "neg_one"], ["dmu_2d"]),
        helper.make_node("Unsqueeze", ["dlogp_dsigma", "neg_one"], ["dsigma_2d"]),
        helper.make_node(
            "Concat", ["dmu_2d", "dsigma_2d"], ["d_log_prob_d_context"], axis=1
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "log_prob_grad_graph",
        [X],
        [Y_logp, Y_jac],
        initializer=[half, one, log2pi, neg_one, idx_x, idx_mu, idx_sigma],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def make_sample_model() -> onnx.ModelProto:
    """sample(z, c) → x [batch, 1]  where x = mu + sigma * z."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    idx_z = numpy_helper.from_array(np.int64(0), name="idx_z")
    idx_mu = numpy_helper.from_array(np.int64(1), name="idx_mu")
    idx_sigma = numpy_helper.from_array(np.int64(2), name="idx_sigma")
    neg_one = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="neg_one")

    nodes = [
        helper.make_node("Gather", ["input", "idx_z"], ["z"], axis=1),
        helper.make_node("Gather", ["input", "idx_mu"], ["mu"], axis=1),
        helper.make_node("Gather", ["input", "idx_sigma"], ["sigma"], axis=1),
        helper.make_node("Mul", ["sigma", "z"], ["sigma_z"]),
        helper.make_node("Add", ["mu", "sigma_z"], ["x_flat"]),
        helper.make_node("Unsqueeze", ["x_flat", "neg_one"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "sample_graph",
        [X],
        [Y],
        initializer=[idx_z, idx_mu, idx_sigma, neg_one],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def verify_models():
    """Quick numerical verification of all three models."""
    import onnxruntime as ort

    # Test data: batch=4, x values, mu=1.0, sigma=2.0
    x_vals = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    mu, sigma = 1.0, 2.0

    # Expected log_prob
    z = (x_vals - mu) / sigma
    expected_logp = -0.5 * z**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)

    # Expected Jacobian
    expected_dmu = z / sigma
    expected_dsigma = (x_vals - mu) ** 2 / sigma**3 - 1.0 / sigma

    inp = np.column_stack(
        [x_vals, np.full(4, mu, dtype=np.float32), np.full(4, sigma, dtype=np.float32)]
    )

    # Verify log_prob
    sess_lp = ort.InferenceSession("log_prob.onnx")
    (logp,) = sess_lp.run(None, {"input": inp})
    assert np.allclose(logp, expected_logp, atol=1e-5), f"log_prob mismatch: {logp} vs {expected_logp}"

    # Verify log_prob_grad
    sess_grad = ort.InferenceSession("log_prob_grad.onnx")
    logp2, jac = sess_grad.run(None, {"input": inp})
    assert np.allclose(logp2, expected_logp, atol=1e-5), f"grad log_prob mismatch"
    assert np.allclose(jac[:, 0], expected_dmu, atol=1e-5), f"d/dmu mismatch: {jac[:, 0]} vs {expected_dmu}"
    assert np.allclose(jac[:, 1], expected_dsigma, atol=1e-5), f"d/dsigma mismatch: {jac[:, 1]} vs {expected_dsigma}"

    # Verify sample
    sess_s = ort.InferenceSession("sample.onnx")
    z_in = np.column_stack(
        [z, np.full(4, mu, dtype=np.float32), np.full(4, sigma, dtype=np.float32)]
    )
    (x_out,) = sess_s.run(None, {"input": z_in})
    assert np.allclose(x_out.flatten(), x_vals, atol=1e-5), f"sample roundtrip mismatch"

    print("All models verified ✅")


def main():
    import os
    import pathlib

    out_dir = pathlib.Path(__file__).parent
    os.chdir(out_dir)

    # Export models
    lp = make_log_prob_model()
    onnx.save(lp, "log_prob.onnx")
    print(f"Saved log_prob.onnx ({os.path.getsize('log_prob.onnx')} bytes)")

    lpg = make_log_prob_grad_model()
    onnx.save(lpg, "log_prob_grad.onnx")
    print(f"Saved log_prob_grad.onnx ({os.path.getsize('log_prob_grad.onnx')} bytes)")

    smp = make_sample_model()
    onnx.save(smp, "sample.onnx")
    print(f"Saved sample.onnx ({os.path.getsize('sample.onnx')} bytes)")

    # Manifest
    manifest = {
        "schema_version": "nextstat_flow_v0",
        "flow_type": "conditional_gaussian",
        "features": 1,
        "context_features": 2,
        "observable_names": ["x"],
        "context_names": ["mu", "sigma"],
        "support": [[-10.0, 10.0]],
        "base_distribution": "standard_normal",
        "models": {
            "log_prob": "log_prob.onnx",
            "sample": "sample.onnx",
            "log_prob_grad": "log_prob_grad.onnx",
        },
        "training": {
            "library": "test_fixture",
            "architecture": "conditional_gaussian",
            "note": "Minimal conditional Gaussian for G3 analytical gradient parity test",
        },
        "validation": {
            "pit_ks_pvalue": 1.0,
            "closure_bias_percent": 0.0,
            "normalization_check": 1.0,
        },
    }
    with open("flow_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved flow_manifest.json")

    # Optional verification (requires onnxruntime)
    try:
        verify_models()
    except ImportError:
        print("onnxruntime not available — skipping verification")


if __name__ == "__main__":
    main()
