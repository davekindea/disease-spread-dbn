#!/usr/bin/env python3
"""
Disease Spread Modeling Using Dynamic Bayesian Networks
=======================================================

Full PGM project pipeline covering:
  - Representation: SEIR DBN structure and CPTs
  - Inference: forward-backward filtering / smoothing
  - Learning: EM for beta, sigma, gamma
  - Experiments: sensitivity analyses

Usage:
    python run.py                  # run full pipeline
    python run.py --quick          # smaller/faster run
    python run.py --query 0 10     # P(node 0 infectious at t=10)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Non-interactive backend — avoids GUI hangs on Windows when saving figures
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import ModelParams, SimConfig
from src.experiments import (
    experiment_beta_sensitivity,
    experiment_network_topology,
    experiment_test_rate_sensitivity,
)
from src.inference import infer_infectious_probability, query_node_infectious
from src.learning import em_learn
from src.model import build_dbn_structure
from src.network import make_contact_network, network_summary
from src.simulation import epidemic_counts, simulate_epidemic
from src.visualization import (
    plot_em_convergence,
    plot_epidemic_curve,
    plot_network,
    plot_network_posterior,
    plot_posterior_heatmap,
)


def run_pipeline(output_dir: Path, quick: bool = False) -> None:
    """Execute all three PGM pillars and generate report figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimConfig(
        n_nodes=12 if quick else 20,
        n_timesteps=30 if quick else 50,
        network_kind="ws",
        test_probability=0.70,
        seed=42,
        patient_zero=0,
    )
    true_params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)

    print("=" * 60)
    print("PGM Project: Disease Spread DBN")
    print("=" * 60)

    # --- Phase 1: Representation ---
    print("\n[Phase 1] Representation")
    G = make_contact_network(config.n_nodes, config.network_kind, config.seed)
    summary = network_summary(G)
    structure = build_dbn_structure(G)
    print(f"  Network: {summary['n_nodes']} nodes, avg degree {summary['avg_degree']:.1f}")
    print(f"  DBN: {structure['description']}")
    plot_network(G, output_dir, seed=config.seed)

    # --- Phase 1: Simulation ---
    print("\n[Phase 1] Simulation")
    X_true, Y_obs = simulate_epidemic(G, true_params, config)
    counts = epidemic_counts(X_true)
    print(f"  Peak infectious: {max(counts['I'])} at t={counts['I'].index(max(counts['I']))}")
    print(f"  Observations recorded: {(Y_obs >= 0).sum()} / {Y_obs.size}")
    plot_epidemic_curve(counts, output_dir)

    # --- Phase 2: Inference ---
    print("\n[Phase 2] Inference (forward-backward)")
    P_I, beliefs = infer_infectious_probability(
        G, Y_obs, true_params, patient_zero=config.patient_zero, smooth=True,
    )
    plot_posterior_heatmap(P_I, output_dir)
    plot_network_posterior(G, P_I, output_dir, seed=config.seed)

    # Example query from project description
    example_node, example_t = 0, min(10, config.n_timesteps - 1)
    p_inf = query_node_infectious(
        G, Y_obs, true_params, node=example_node, time=example_t,
        patient_zero=config.patient_zero,
    )
    true_state = ["S", "E", "I", "R"][X_true[example_t, example_node]]
    print(f"\n  Query: P(node {example_node} is infectious at t={example_t}) = {p_inf:.3f}")
    print(f"  (True state at t={example_t}: {true_state})")

    # --- Phase 3: Learning (EM) ---
    print("\n[Phase 3] Learning (EM algorithm)")
    print(f"  True parameters: beta={true_params.beta}, sigma={true_params.sigma}, gamma={true_params.gamma}")
    init_params = ModelParams(beta=0.10, sigma=0.10, gamma=0.10)
    learned, history = em_learn(
        G, Y_obs, init_params,
        n_iter=20 if quick else 30,
        patient_zero=config.patient_zero,
    )
    print(f"  Learned:         beta={learned.beta:.3f}, sigma={learned.sigma:.3f}, gamma={learned.gamma:.3f}")
    plot_em_convergence(history, true_params.as_array(), output_dir)

    # --- Phase 4: Sensitivity experiments ---
    if not quick:
        print("\n[Phase 4] Sensitivity experiments")
        experiment_beta_sensitivity(output_dir, config=config)
        experiment_test_rate_sensitivity(output_dir, config=config, true_params=true_params)
        experiment_network_topology(output_dir, config=config, params=true_params)
        print("  Figures 5-7 saved.")

    print("\n" + "=" * 60)
    print(f"Done. All figures saved to: {output_dir.resolve()}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEIR epidemic DBN — PGM course project",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"),
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Faster run with fewer nodes/timesteps (skip some experiments)",
    )
    parser.add_argument(
        "--query", nargs=2, type=int, metavar=("NODE", "TIME"),
        help="Run inference query: P(node infectious at time)",
    )
    args = parser.parse_args()

    if args.query is not None:
        node, time = args.query
        config = SimConfig()
        true_params = ModelParams()
        G = make_contact_network(config.n_nodes, config.network_kind, config.seed)
        _, Y_obs = simulate_epidemic(G, true_params, config)
        p = query_node_infectious(
            G, Y_obs, true_params, node=node, time=time,
            patient_zero=config.patient_zero,
        )
        print(f"P(node {node} is infectious at t={time} | observations) = {p:.4f}")
        return

    run_pipeline(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
