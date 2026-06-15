#!/usr/bin/env python3
"""
Disease Spread Modeling Using Dynamic Bayesian Networks
=======================================================

Models temporal disease spread through a population using a DBN with:
  - Latent SEIR states (Susceptible, Exposed, Infectious, Recovered)
  - Observations: reported symptoms / test results
  - Representation: graph structure and CPTs
  - Inference: forward-backward belief propagation
  - Learning: EM on simulated or real-world epidemic data

Usage:
    python run.py                      # real-world Geneva contact-tracing data
    python run.py --data synthetic     # simulated epidemic data
    python run.py --quick              # smaller/faster run
    python run.py --query 0 10         # P(node 0 infectious at t=10 | observations)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
from src.real_data import load_real_data
from src.simulation import epidemic_counts, simulate_epidemic
from src.visualization import (
    plot_em_convergence,
    plot_epidemic_curve,
    plot_network,
    plot_network_posterior,
    plot_posterior_heatmap,
)


def _load_dataset(config: SimConfig, data_source: str):
    """Load graph, observations, and optional ground-truth states."""
    if data_source == "real":
        print("\n[Data] Loading real-world Geneva COVID-19 contact tracing")
        bundle = load_real_data(config)
        meta = bundle.metadata
        print(f"  Dataset: {bundle.dataset_name}")
        print(f"  Source: {meta['source']}")
        print(f"  Subgraph: {meta['n_nodes']} people, {meta['n_timesteps']} days")
        print(f"  Period: {meta['date_start']} to {meta['date_end']}")
        print(f"  Positive tests observed: {meta['n_positive_tests']}")
        return bundle.graph, bundle.Y_obs, bundle.X_true, bundle.patient_zero, bundle

    G = make_contact_network(config.n_nodes, config.network_kind, config.seed)
    true_params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)
    X_true, Y_obs = simulate_epidemic(G, true_params, config)
    return G, Y_obs, X_true, config.patient_zero, None


def run_pipeline(
    output_dir: Path,
    quick: bool = False,
    data_source: str = "real",
) -> None:
    """Execute all three PGM pillars and generate report figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimConfig(
        n_nodes=20 if quick else 30,
        n_timesteps=30 if quick else 50,
        network_kind="ws",
        test_probability=0.70,
        seed=42,
        patient_zero=0,
        data_source=data_source,
    )
    model_params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)

    print("=" * 60)
    print("PGM Project: Disease Spread DBN")
    print(f"Data mode: {data_source}")
    print("=" * 60)

    G, Y_obs, X_true, patient_zero, bundle = _load_dataset(config, data_source)

    # --- Phase 1: Representation ---
    print("\n[Phase 1] Representation")
    summary = network_summary(G)
    structure = build_dbn_structure(G)
    print(f"  Network: {summary['n_nodes']} nodes, avg degree {summary['avg_degree']:.1f}")
    print(f"  DBN: {structure['description']}")
    plot_network(G, output_dir, seed=config.seed)

    # --- Phase 1: Data summary ---
    print("\n[Phase 1] Observations")
    if X_true is not None:
        counts = epidemic_counts(X_true)
        peak_I = max(counts["I"])
        print(f"  Peak infectious (approx.): {peak_I} at t={counts['I'].index(peak_I)}")
    else:
        daily_pos = (Y_obs == 1).sum(axis=1)
        counts = {
            "I": daily_pos.tolist(),
            "E": [0] * len(daily_pos),
            "R": [0] * len(daily_pos),
        }
        print(f"  Peak daily positive tests: {max(counts['I'])}")
    print(f"  Observations recorded: {(Y_obs >= 0).sum()} / {Y_obs.size}")
    plot_epidemic_curve(counts, output_dir)

    # --- Phase 2: Inference ---
    print("\n[Phase 2] Inference (forward-backward)")
    P_I, _ = infer_infectious_probability(
        G, Y_obs, model_params, patient_zero=patient_zero, smooth=True,
    )
    plot_posterior_heatmap(P_I, output_dir)
    plot_network_posterior(G, P_I, output_dir, seed=config.seed)

    example_node, example_t = 0, min(10, Y_obs.shape[0] - 1)
    p_inf = query_node_infectious(
        G, Y_obs, model_params, node=example_node, time=example_t,
        patient_zero=patient_zero,
    )
    if X_true is not None:
        true_state = ["S", "E", "I", "R"][X_true[example_t, example_node]]
        print(f"\n  Query: P(node {example_node} is infectious at t={example_t}) = {p_inf:.3f}")
        print(f"  (Approx. true state at t={example_t}: {true_state})")
    else:
        print(f"\n  Query: P(node {example_node} is infectious at t={example_t}) = {p_inf:.3f}")

    # --- Phase 3: Learning (EM) ---
    print("\n[Phase 3] Learning (EM algorithm)")
    if data_source == "synthetic":
        print(f"  True parameters: beta={model_params.beta}, sigma={model_params.sigma}, gamma={model_params.gamma}")
    else:
        print("  Learning parameters from real partial test observations (no ground-truth beta/sigma/gamma).")
    init_params = ModelParams(beta=0.10, sigma=0.10, gamma=0.10)
    learned, history = em_learn(
        G, Y_obs, init_params,
        n_iter=20 if quick else 30,
        patient_zero=patient_zero,
    )
    print(f"  Learned: beta={learned.beta:.3f}, sigma={learned.sigma:.3f}, gamma={learned.gamma:.3f}")
    true_arr = model_params.as_array() if data_source == "synthetic" else None
    plot_em_convergence(history, output_dir, true_params=true_arr)

    # --- Phase 4: Sensitivity experiments (synthetic only) ---
    if not quick and data_source == "synthetic":
        print("\n[Phase 4] Sensitivity experiments")
        experiment_beta_sensitivity(output_dir, config=config)
        experiment_test_rate_sensitivity(output_dir, config=config, true_params=model_params)
        experiment_network_topology(output_dir, config=config, params=model_params)
        print("  Figures 5-7 saved.")
    elif data_source == "real":
        print("\n[Phase 4] Skipped — sensitivity experiments require synthetic data.")
        print("  Run with --data synthetic to generate Figures 5-7.")

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
        "--data", choices=("real", "synthetic"), default="real",
        help="Data source: real Geneva contact tracing (default) or synthetic simulation",
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
        config = SimConfig(n_nodes=30, data_source=args.data)
        model_params = ModelParams()
        if args.data == "real":
            bundle = load_real_data(config)
            G, Y_obs = bundle.graph, bundle.Y_obs
            pz = bundle.patient_zero
        else:
            G = make_contact_network(config.n_nodes, config.network_kind, config.seed)
            _, Y_obs = simulate_epidemic(G, model_params, config)
            pz = config.patient_zero
        p = query_node_infectious(
            G, Y_obs, model_params, node=node, time=time, patient_zero=pz,
        )
        print(f"P(node {node} is infectious at t={time} | observations) = {p:.4f}")
        return

    run_pipeline(args.output, quick=args.quick, data_source=args.data)


if __name__ == "__main__":
    main()
