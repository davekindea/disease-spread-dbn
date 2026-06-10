"""Phase 4 sensitivity experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelParams, SimConfig
from .learning import em_learn
from .network import make_contact_network
from .simulation import epidemic_counts, simulate_epidemic
from .visualization import plot_sensitivity


def experiment_beta_sensitivity(
    output_dir: Path,
    betas: list[float] | None = None,
    config: SimConfig | None = None,
) -> dict:
    """Figure 5: peak infectious count vs transmission rate beta."""
    config = config or SimConfig()
    betas = betas or [0.10, 0.20, 0.30, 0.40, 0.50]
    peaks = []

    for beta in betas:
        G = make_contact_network(config.n_nodes, config.network_kind, config.seed)
        params = ModelParams(beta=beta, sigma=0.20, gamma=0.10)
        X, _ = simulate_epidemic(G, params, config)
        counts = epidemic_counts(X)
        peaks.append(max(counts["I"]))

    plot_sensitivity(
        betas, peaks,
        xlabel="Transmission rate (beta)",
        ylabel="Peak infectious count",
        title="Figure 5: Sensitivity to beta",
        output_dir=output_dir,
        filename="fig5_sensitivity_beta.png",
    )
    return {"betas": betas, "peak_I": peaks}


def experiment_test_rate_sensitivity(
    output_dir: Path,
    test_rates: list[float] | None = None,
    config: SimConfig | None = None,
    true_params: ModelParams | None = None,
) -> dict:
    """Figure 6: EM parameter error vs test observation rate."""
    config = config or SimConfig()
    test_rates = test_rates or [0.30, 0.50, 0.70, 0.90]
    true_params = true_params or ModelParams(beta=0.30, sigma=0.20, gamma=0.10)
    errors = []

    G = make_contact_network(config.n_nodes, config.network_kind, config.seed)

    for rate in test_rates:
        cfg = SimConfig(
            n_nodes=config.n_nodes,
            n_timesteps=config.n_timesteps,
            network_kind=config.network_kind,
            test_probability=rate,
            seed=config.seed,
            patient_zero=config.patient_zero,
        )
        _, Y = simulate_epidemic(G, true_params, cfg)
        init = ModelParams(beta=0.10, sigma=0.10, gamma=0.10)
        learned, _ = em_learn(G, Y, init, n_iter=25, patient_zero=cfg.patient_zero, verbose=False)
        err = np.linalg.norm(learned.as_array() - true_params.as_array())
        errors.append(float(err))

    plot_sensitivity(
        test_rates, errors,
        xlabel="Test observation probability",
        ylabel="||theta_hat - theta_true||",
        title="Figure 6: EM Error vs Test Rate",
        output_dir=output_dir,
        filename="fig6_sensitivity_test_rate.png",
    )
    return {"test_rates": test_rates, "param_errors": errors}


def experiment_network_topology(
    output_dir: Path,
    config: SimConfig | None = None,
    params: ModelParams | None = None,
) -> dict:
    """Figure 7: final epidemic size across network topologies."""
    config = config or SimConfig()
    params = params or ModelParams()
    kinds = ["er", "ws", "ba"]
    final_I = []

    for kind in kinds:
        G = make_contact_network(config.n_nodes, kind, config.seed)
        X, _ = simulate_epidemic(G, params, config)
        counts = epidemic_counts(X)
        final_I.append(counts["I"][-1])

    plot_sensitivity(
        kinds, final_I,
        xlabel="Network topology",
        ylabel="Final infectious count",
        title="Figure 7: Sensitivity to Network Topology",
        output_dir=output_dir,
        filename="fig7_sensitivity_topology.png",
    )
    return {"topologies": kinds, "final_I": final_I}
