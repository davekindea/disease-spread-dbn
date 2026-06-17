#!/usr/bin/env python3
"""
Test the SEIR DBN with custom parameters.

Examples
--------
  # Default parameters on real data
  python test_model.py

  # Set epidemic parameters manually
  python test_model.py --beta 0.30 --sigma 0.20 --gamma 0.10

  # Query: P(person 3 infected on day 15 | all tests)
  python test_model.py --beta 0.25 --query 3 15

  # Compare several beta values (synthetic data works best)
  python test_model.py --data synthetic --compare-beta 0.1 0.3 0.5

  # Run inference + EM learning from your starting parameters
  python test_model.py --beta 0.15 --sigma 0.20 --gamma 0.10 --learn

  # Test quality of tests (sensitivity / specificity)
  python test_model.py --sensitivity 0.85 --specificity 0.90 --query 0 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.config import ModelParams, SimConfig, STATE_IDX
from src.corona_data import load_corona_for_config
from src.inference import infer_infectious_probability, query_node_infectious
from src.learning import em_learn
from src.network import make_contact_network
from src.simulation import epidemic_counts, simulate_epidemic
from src.visualization import plot_posterior_heatmap


def _load(data: str, n_nodes: int):
    config = SimConfig(n_nodes=n_nodes, data_source=data)
    if data == "real":
        bundle = load_corona_for_config(config)
        return bundle.graph, bundle.Y_obs, bundle.X_true, bundle.patient_zero, bundle.metadata
    G = make_contact_network(config.n_nodes, config.network_kind, config.seed)
    params = ModelParams()
    X, Y = simulate_epidemic(G, params, config)
    meta = {"n_nodes": G.number_of_nodes(), "n_timesteps": Y.shape[0], "n_positive_tests": int((Y == 1).sum())}
    return G, Y, X, config.patient_zero, meta


def _print_params(params: ModelParams) -> None:
    print("  beta (transmission)     :", params.beta)
    print("  sigma (E -> I)          :", params.sigma)
    print("  gamma (I -> R)          :", params.gamma)
    print("  test sensitivity        :", params.sensitivity)
    print("  test specificity        :", params.specificity)


def run_single(args) -> None:
    params = ModelParams(
        beta=args.beta,
        sigma=args.sigma,
        gamma=args.gamma,
        sensitivity=args.sensitivity,
        specificity=args.specificity,
    )
    G, Y, X, pz, meta = _load(args.data, args.nodes)

    print("=" * 60)
    print("TEST MODEL — custom parameters")
    print("=" * 60)
    print(f"\nData: {args.data}  |  nodes={meta['n_nodes']}  days={meta['n_timesteps']}")
    print(f"Positive tests: {meta.get('n_positive_tests', '?')}")
    print("\nYour parameters:")
    _print_params(params)

    print("\n--- Inference ---")
    P_I, _ = infer_infectious_probability(G, Y, params, patient_zero=pz, smooth=True)
    print(f"Posterior P(I) shape: {P_I.shape}")
    print(f"Max P(infected) over all people/days: {P_I.max():.4f}")

    if args.query:
        node, day = args.query
        p = query_node_infectious(G, Y, params, node=node, time=day, patient_zero=pz)
        print(f"\nQuery: P(node {node} infected at day {day} | all tests) = {p:.4f}")
        if X is not None and day < X.shape[0]:
            true = ["S", "E", "I", "R"][X[day, node]]
            print(f"       (label at that day: {true})")

    learned = None
    if args.learn:
        print("\n--- EM learning (starting from your parameters) ---")
        learned, hist = em_learn(G, Y, params, n_iter=args.em_iter, patient_zero=pz, verbose=True)
        print("\nLearned parameters:")
        _print_params(learned)
        P_I_learned, _ = infer_infectious_probability(G, Y, learned, patient_zero=pz, smooth=True)
        if args.query:
            node, day = args.query
            p2 = query_node_infectious(G, Y, learned, node=node, time=day, patient_zero=pz)
            print(f"\nAfter EM — P(node {node} infected at day {day}) = {p2:.4f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plot_posterior_heatmap(P_I, out / "test")
    fig, ax = plt.subplots(figsize=(9, 4))
    if args.plot_node is not None and args.plot_node < P_I.shape[1]:
        ax.plot(P_I[:, args.plot_node], lw=2, label=f"your params (node {args.plot_node})")
        if learned is not None:
            ax.plot(P_I_learned[:, args.plot_node], "--", lw=2, label="after EM")
        ax.set_xlabel("day")
        ax.set_ylabel("P(infected)")
        ax.set_title(f"P(infected) over time — node {args.plot_node}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(out / "test" / f"Pi_node{args.plot_node}.png", dpi=150)
        plt.close(fig)

    print(f"\nFigures saved to: {(out / 'test').resolve()}")
    print("=" * 60)


def run_compare_beta(args) -> None:
    """Compare P(I) heatmaps for different beta values."""
    G, Y, _, pz, meta = _load(args.data, args.nodes)
    betas = args.compare_beta
    n = len(betas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    print("=" * 60)
    print("COMPARE beta values")
    print("=" * 60)
    for ax, beta in zip(axes, betas):
        params = ModelParams(
            beta=beta, sigma=args.sigma, gamma=args.gamma,
            sensitivity=args.sensitivity, specificity=args.specificity,
        )
        P_I, _ = infer_infectious_probability(G, Y, params, patient_zero=pz, smooth=True)
        im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"beta = {beta}\nmax P(I) = {P_I.max():.2f}")
        ax.set_xlabel("day")
        ax.set_ylabel("person")
        fig.colorbar(im, ax=ax, fraction=0.046)

    out = Path(args.output) / "test"
    out.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Effect of transmission rate beta ({args.data} data)", y=1.02)
    plt.tight_layout()
    path = out / "compare_beta.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.resolve()}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test SEIR DBN with your own parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--beta", type=float, default=0.30, help="Transmission rate (default 0.30)")
    parser.add_argument("--sigma", type=float, default=0.20, help="E->I rate (default 0.20)")
    parser.add_argument("--gamma", type=float, default=0.10, help="I->R rate (default 0.10)")
    parser.add_argument("--sensitivity", type=float, default=0.90, help="P(pos test | infected)")
    parser.add_argument("--specificity", type=float, default=0.95, help="P(neg test | not infected)")
    parser.add_argument("--data", choices=("real", "synthetic"), default="real")
    parser.add_argument("--nodes", type=int, default=30, help="Max people (real data subgraph)")
    parser.add_argument("--query", nargs=2, type=int, metavar=("NODE", "DAY"), help="P(infected | tests)")
    parser.add_argument("--plot-node", type=int, default=0, help="Plot P(I) over time for this person")
    parser.add_argument("--learn", action="store_true", help="Run EM from your parameters")
    parser.add_argument("--em-iter", type=int, default=25, help="EM iterations if --learn")
    parser.add_argument(
        "--compare-beta", nargs="+", type=float, metavar="BETA",
        help="Compare multiple beta values side by side",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.compare_beta:
        run_compare_beta(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
