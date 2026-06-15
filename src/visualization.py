"""Figures for the PGM epidemic DBN project."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .config import STATES, STATE_IDX


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_network(G: nx.Graph, output_dir: Path, seed: int = 42) -> Path:
    """Figure 0: contact network layout."""
    out = _ensure_dir(output_dir)
    pos = nx.spring_layout(G, seed=seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw(
        G, pos, with_labels=True, node_color="#a8d4f0",
        node_size=450, font_size=8, edge_color="#888888", ax=ax,
    )
    ax.set_title("Contact Network")
    fig.tight_layout()
    path = out / "fig0_network.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_epidemic_curve(counts: dict[str, list[int]], output_dir: Path) -> Path:
    """Figure 1: simulated epidemic curve."""
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(counts["I"], label="Infectious (I)", linewidth=2, color="#c0392b")
    ax.plot(counts["E"], label="Exposed (E)", linewidth=1.5, linestyle="--", color="#e67e22")
    ax.plot(counts["R"], label="Recovered (R)", linewidth=1.5, linestyle=":", color="#27ae60")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Number of individuals")
    ax.set_title("Figure 1: Simulated Epidemic Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig1_epidemic_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_posterior_heatmap(P_I: np.ndarray, output_dir: Path) -> Path:
    """Figure 2: P(Infectious) over people x time."""
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        P_I.T, aspect="auto", origin="lower",
        cmap="Reds", vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Individual")
    ax.set_title("Figure 2: P(Infectious | Observations)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(I)")
    fig.tight_layout()
    path = out / "fig2_heatmap_P_I.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_network_posterior(
    G: nx.Graph,
    P_I: np.ndarray,
    output_dir: Path,
    seed: int = 42,
) -> Path:
    """Figure 3: network colored by max posterior P(I)."""
    out = _ensure_dir(output_dir)
    pos = nx.spring_layout(G, seed=seed)
    max_P = P_I.max(axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=max_P, cmap=plt.cm.Reds,
        vmin=0.0, vmax=1.0, node_size=500, ax=ax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    cbar = fig.colorbar(nodes, ax=ax)
    cbar.set_label("max_t P(I)")
    ax.set_title("Figure 3: Network Colored by Posterior P(Infectious)")
    fig.tight_layout()
    path = out / "fig3_network_posterior.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_em_convergence(
    history: np.ndarray,
    output_dir: Path,
    true_params: np.ndarray | None = None,
) -> Path:
    """Figure 4: EM parameter convergence."""
    out = _ensure_dir(output_dir)
    labels = ["beta", "sigma", "gamma"]
    colors = ["#2980b9", "#8e44ad", "#16a085"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(history[:, idx], label=f"learned {label}", color=color, linewidth=2)
        if true_params is not None:
            ax.axhline(
                true_params[idx], color=color, linestyle=":", alpha=0.6,
                label=f"true {label}",
            )
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Parameter value")
    ax.set_title("Figure 4: EM Convergence")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig4_em_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sensitivity(
    x_values: list,
    y_values: list,
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    filename: str,
) -> Path:
    """Generic sensitivity plot for Phase 4 experiments."""
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_values, y_values, marker="o", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
