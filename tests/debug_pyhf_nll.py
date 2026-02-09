#!/usr/bin/env python3
"""
Debug pyhf NLL computation to understand exact formula.

This script breaks down NLL into components to understand
what pyhf includes in the likelihood.
"""

import json
from pathlib import Path
import pyhf
import numpy as np


def load_fixture(name: str) -> dict:
    """Load a test fixture JSON file."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / name) as f:
        return json.load(f)


def debug_nll_components():
    """Debug NLL component by component."""
    print("=" * 70)
    print("Debugging pyhf NLL Components")
    print("=" * 70)

    workspace = load_fixture("simple_workspace.json")
    ws = pyhf.Workspace(workspace)
    model = ws.model("GaussExample")

    print(f"\nModel info:")
    print(f"  Parameters: {model.config.par_names}")
    print(f"  Parameter order: {list(enumerate(model.config.par_names))}")
    print(f"  N parameters: {model.config.npars}")

    # Get observations
    observations = ws.data(model)
    print(f"\nObservations: {observations}")

    # Parameters: [mu, gamma_bin_0, gamma_bin_1]
    params = np.array([1.0, 1.0, 1.0])
    print(f"\nParameters: {params}")

    # Expected data
    expected = model.expected_data(params)
    print(f"\nExpected data: {expected}")

    # Compute NLL
    twice_nll = pyhf.infer.mle.twice_nll(params, observations, model)
    if hasattr(twice_nll, 'item'):
        twice_nll = twice_nll.item()
    nll = twice_nll / 2.0
    print(f"\nTotal NLL: {nll:.10f}")
    print(f"Twice NLL: {twice_nll:.10f}")

    # Now let's manually compute components
    print("\n" + "=" * 70)
    print("Manual NLL Computation")
    print("=" * 70)

    # 1. Poisson main likelihood
    poisson_nll = 0.0
    for i, (obs, exp) in enumerate(zip(observations, expected)):
        # -log(Poisson(obs|exp)) = exp - obs*log(exp) + log(obs!)
        exp = max(exp, 1e-10)

        # Without factorial
        term_no_fact = exp - obs * np.log(exp)

        # With factorial (Stirling)
        if obs > 0:
            ln_factorial = obs * np.log(obs) - obs + 0.5 * np.log(2 * np.pi * obs)
        else:
            ln_factorial = 0.0

        term_with_fact = term_no_fact + ln_factorial

        poisson_nll += term_with_fact

        print(f"\nBin {i}:")
        print(f"  obs={obs:.1f}, exp={exp:.1f}")
        print(f"  Poisson (no factorial): {term_no_fact:.6f}")
        print(f"  ln({obs}!): {ln_factorial:.6f}")
        print(f"  Poisson (with factorial): {term_with_fact:.6f}")

    print(f"\n>>> Total Poisson NLL: {poisson_nll:.10f}")

    # 2. Constraint terms (if any)
    # For shapesys, pyhf uses Barlow-Beeston with auxiliary data

    # Get constraint terms from model
    # pyhf stores these in model.config.auxdata
    print(f"\n" + "=" * 70)
    print("Constraint Terms")
    print("=" * 70)

    auxdata = model.config.auxdata()
    print(f"Auxiliary data: {auxdata}")

    # For shapesys with uncertainties [5.0, 6.0] and nominals [50.0, 60.0]:
    # tau_i = (nominal_i / sigma_i)^2
    # Constraint: Poisson(tau_i | gamma_i * tau_i)

    constraint_nll = 0.0

    # shapesys parameters are indices 1 and 2
    # From workspace: background sample has nominal [50.0, 60.0], uncertainties [5.0, 6.0]
    nominals = np.array([50.0, 60.0])
    uncertainties = np.array([5.0, 6.0])
    gammas = params[1:3]  # [1.0, 1.0]

    print(f"\nShapeSys parameters:")
    print(f"  Nominals: {nominals}")
    print(f"  Uncertainties: {uncertainties}")
    print(f"  Gammas: {gammas}")

    for i, (nom, sigma, gamma) in enumerate(zip(nominals, uncertainties, gammas)):
        tau = (nom / sigma) ** 2
        print(f"\n  Bin {i}:")
        print(f"    tau = (nom/sigma)^2 = ({nom}/{sigma})^2 = {tau:.1f}")

        # Poisson constraint: -log(Poisson(tau | gamma * tau))
        # = gamma * tau - tau * log(gamma * tau) + log(tau!)

        exp_aux = gamma * tau

        # Without factorial
        term_no_fact = exp_aux - tau * np.log(exp_aux)

        # With factorial
        if tau > 0:
            ln_factorial_tau = tau * np.log(tau) - tau + 0.5 * np.log(2 * np.pi * tau)
        else:
            ln_factorial_tau = 0.0

        term_with_fact = term_no_fact + ln_factorial_tau

        constraint_nll += term_with_fact

        print(f"    Constraint (no factorial): {term_no_fact:.6f}")
        print(f"    ln({tau}!): {ln_factorial_tau:.6f}")
        print(f"    Constraint (with factorial): {term_with_fact:.6f}")

    print(f"\n>>> Total Constraint NLL: {constraint_nll:.10f}")

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Poisson NLL:     {poisson_nll:.10f}")
    print(f"Constraint NLL:  {constraint_nll:.10f}")
    print(f"Total (manual):  {(poisson_nll + constraint_nll):.10f}")
    print(f"Total (pyhf):    {nll:.10f}")
    print(f"Difference:      {abs(poisson_nll + constraint_nll - nll):.10f}")


if __name__ == "__main__":
    debug_nll_components()
