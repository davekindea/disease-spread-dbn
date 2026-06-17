"""Shared state, data loading, and plotting helpers for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ModelParams,
    SimConfig,
    OBS_NEG,
    OBS_POS,
    OBS_MISSING,
    STATES,
    STATE_IDX,
    N_STATES,
)
from src.corona_data import load_corona_for_config
from src.inference import forward_filter, backward_smooth, infer_infectious_probability
from src.learning import _expected_pair_counts, m_step
from src.model import build_dbn_structure, transition_matrix
from src.simulation import epidemic_counts

OBS_LABEL = {OBS_POS: "positive (+)", OBS_NEG: "negative (−)", OBS_MISSING: "missing (?)"}


@st.cache_data(show_spinner="Loading Geneva COVID-19 data…")
def load_geneva_data(n_nodes: int):
    config = SimConfig(n_nodes=n_nodes, data_source="real")
    bundle = load_corona_for_config(config)
    meta = {
        "label": bundle.dataset_name,
        "n_nodes": bundle.metadata["n_nodes"],
        "n_days": bundle.metadata["n_timesteps"],
        "n_pos": bundle.metadata["n_positive_tests"],
    }
    pos = nx.spring_layout(bundle.graph, seed=42)
    counts = epidemic_counts(bundle.X_true)
    return bundle.graph, bundle.Y_obs, bundle.X_true, counts, bundle.patient_zero, meta, pos


def init_session_state() -> None:
    defaults = {
        "em_history": None,
        "learned_params": None,
        "em_init_params": None,
        "P_I_cache": None,
        "P_I_params_key": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def params_key(params: ModelParams) -> tuple:
    return (params.beta, params.sigma, params.gamma, params.sensitivity, params.specificity)


def get_P_I(G, Y, params, pz):
    key = params_key(params)
    if st.session_state.P_I_cache is not None and st.session_state.P_I_params_key == key:
        return st.session_state.P_I_cache
    P_I, _ = infer_infectious_probability(G, Y, params, patient_zero=pz, smooth=True)
    st.session_state.P_I_cache = P_I
    st.session_state.P_I_params_key = key
    return P_I


def invalidate_inference_cache() -> None:
    st.session_state.P_I_cache = None
    st.session_state.P_I_params_key = None


def render_sidebar() -> tuple:
    """Global settings sidebar. Returns (G, Y, X, counts, pz, meta, pos, params)."""
    init_session_state()
    st.sidebar.header("Global settings")
    st.sidebar.subheader("Model parameters")
    beta = st.sidebar.slider("β transmission", 0.01, 0.80, 0.30, 0.01)
    sigma = st.sidebar.slider("σ E→I", 0.01, 0.80, 0.20, 0.01)
    gamma = st.sidebar.slider("γ I→R", 0.01, 0.99, 0.10, 0.01)
    params = ModelParams(beta=beta, sigma=sigma, gamma=gamma)

    G, Y, X, counts, pz, meta, pos = load_geneva_data(30)
    st.sidebar.divider()
    st.sidebar.markdown("**Dataset**")
    st.sidebar.markdown(
        f"{meta['label']}  \n"
        f"{meta['n_nodes']} people · {meta['n_days']} days · {meta['n_pos']} positive tests  \n"
        f"Index case: person **{pz}**"
    )
    return G, Y, X, counts, pz, meta, pos, params


# ── Plotting helpers ──────────────────────────────────────────────────────────

def fig_dbn_plate():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    def box(x, y, lab, fc):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - 0.45, y - 0.25), 0.9, 0.5,
                boxstyle="round,pad=0.02", fc=fc, ec="k", lw=1.5,
            )
        )
        ax.text(x, y, lab, ha="center", va="center", fontsize=10, fontweight="bold")

    box(1, 1.5, r"$X_j^{t-1}$", "#aed6f1")
    box(3, 1.5, r"$X_i^{t-1}$", "#aed6f1")
    box(5, 1.5, r"$X_i^{t}$", "#aed6f1")
    box(5, 0.4, r"$Y_i^{t}$", "#abebc6")
    ax.annotate("", xy=(5, 0.72), xytext=(5, 1.22), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(5, 1.22), xytext=(3.45, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(3.45, 1.5), xytext=(1.45, 1.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#e74c3c"))
    ax.text(2.5, 1.9, "transmission along contact (β)", ha="center", fontsize=9, color="#e74c3c")
    ax.text(5.55, 0.4, "test / symptom", fontsize=9)
    ax.text(2, 2.15, "slice t−1", fontsize=12, fontweight="bold")
    ax.text(4.5, 2.15, "slice t", fontsize=12, fontweight="bold")
    ax.set_title("2-slice Dynamic Bayesian Network (unrolled over days)")
    fig.tight_layout()
    return fig


def fig_seir_compartments():
    fig, ax = plt.subplots(figsize=(10, 3))
    states_pos = {"S": 0, "E": 1, "I": 2, "R": 3}
    colors = {"S": "#3498db", "E": "#f39c12", "I": "#e74c3c", "R": "#27ae60"}
    labels = {"S": "Susceptible", "E": "Exposed", "I": "Infected", "R": "Recovered"}
    for s, x in states_pos.items():
        ax.add_patch(plt.Circle((x, 0.55), 0.32, color=colors[s], ec="k", lw=2))
        ax.text(x, 0.55, s, ha="center", va="center", fontsize=13, fontweight="bold", color="white")
        ax.text(x, 0.05, labels[s], ha="center", fontsize=9)
    for (x0, x1, lbl) in [(0.32, 0.68, "contact (β)"), (1.32, 1.68, "σ"), (2.32, 2.68, "γ")]:
        ax.annotate("", xy=(x1, 0.55), xytext=(x0, 0.55), arrowprops=dict(arrowstyle="->", lw=2.5))
        ax.text((x0 + x1) / 2, 0.88, lbl, ha="center", fontsize=9)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")
    ax.set_title("SEIR latent states (per individual)")
    fig.tight_layout()
    return fig


def fig_transition_matrices(params: ModelParams):
    T_s = transition_matrix([], params)
    T_i = transition_matrix([0.8, 0.6], params)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, T, title in zip(axes, [T_s, T_i], ["No infectious neighbors", "Infectious neighbors present"]):
        im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_STATES))
        ax.set_yticks(range(N_STATES))
        ax.set_xticklabels(STATES)
        ax.set_yticklabels(STATES)
        ax.set_xlabel("next state")
        ax.set_ylabel("prev state")
        ax.set_title(title)
        for i in range(N_STATES):
            for j in range(N_STATES):
                ax.text(
                    j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if T[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=axes.ravel().tolist()[-1], fraction=0.046)
    fig.suptitle(r"CPT: $P(X^t \mid X^{t-1}, \text{neighbors})$", y=1.02)
    fig.tight_layout()
    return fig


def fig_observation_matrix(Y: np.ndarray, day_start: int = 0, day_end: int | None = None,
                           person_start: int = 0, person_end: int | None = None):
    day_end = Y.shape[0] if day_end is None else day_end
    person_end = Y.shape[1] if person_end is None else person_end
    sub = Y[day_start:day_end, person_start:person_end]
    cmap_data = np.where(sub == OBS_POS, 1.0, np.where(sub == OBS_NEG, 0.0, 0.5))
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(cmap_data.T, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xlabel("day t")
    ax.set_ylabel("person i")
    ax.set_title("Observation matrix Y (red=positive, blue=negative, grey=missing)")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["negative", "missing", "positive"])
    fig.tight_layout()
    return fig


def plotly_network(G: nx.Graph, pos: dict, node_colors: list | None = None,
                   highlight: int | None = None, colorbar_title: str = ""):
    if node_colors is None:
        node_colors = ["#a8d4f0"] * G.number_of_nodes()

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none", showlegend=False,
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    labels = [str(n) for n in G.nodes()]
    sizes = [22 if n == highlight else 16 for n in G.nodes()]
    marker = dict(
        size=sizes,
        color=node_colors,
        colorscale="Reds",
        cmin=0,
        cmax=1,
        line=dict(width=[3 if n == highlight else 1 for n in G.nodes()], color="#333"),
        colorbar=dict(title=colorbar_title) if isinstance(node_colors[0], (int, float)) else None,
        showscale=isinstance(node_colors[0], (int, float)),
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=labels, textposition="top center",
        hovertext=[f"person {n}" for n in G.nodes()],
        hoverinfo="text",
        marker=marker,
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=480,
    )
    return fig


def fig_posterior_heatmap(P_I: np.ndarray, Y: np.ndarray | None = None,
                          highlight: tuple[int, int] | None = None):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_xlabel("day")
    ax.set_ylabel("person")
    ax.set_title("P(Infected | all test/symptom observations)")
    fig.colorbar(im, ax=ax, label="P(I)")
    if Y is not None:
        for t in range(Y.shape[0]):
            for i in range(Y.shape[1]):
                if Y[t, i] == OBS_POS:
                    ax.plot(t, i, "wo", ms=5, mec="k", mew=0.8)
    if highlight is not None:
        day, person = highlight
        ax.axhline(person, color="#2ecc71", lw=1.5, alpha=0.8)
        ax.axvline(day, color="#2ecc71", lw=1.5, alpha=0.8)
        ax.plot(day, person, "g*", ms=14, mec="k")
    fig.tight_layout()
    return fig


def fig_network_posterior(G, pos, P_I: np.ndarray, day: int | None = None):
    if day is not None:
        colors = P_I[day]
        title = f"Network colored by P(infected) on day {day}"
        cbar_label = "P(I)"
    else:
        colors = P_I.max(axis=0)
        title = "Network colored by max P(infected)"
        cbar_label = "max P(I)"
    fig, ax = plt.subplots(figsize=(8, 6))
    art = nx.draw_networkx_nodes(
        G, pos, node_color=colors, cmap=plt.cm.Reds,
        vmin=0, vmax=1, node_size=650, ax=ax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    fig.colorbar(art, ax=ax, label=cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_em_convergence(history: np.ndarray, true_params: np.ndarray | None = None):
    labels = ["beta", "sigma", "gamma"]
    colors = ["#2980b9", "#8e44ad", "#16a085"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(history[:, idx], "o-", label=label, color=color, lw=2, ms=4)
        if true_params is not None:
            ax.axhline(true_params[idx], color=color, linestyle=":", alpha=0.6, label=f"true {label}")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("parameter value")
    ax.set_title("EM parameter convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_params_comparison(init_p: ModelParams, learned_p: ModelParams):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["β\ntransmission", "σ\nE→I", "γ\nI→R"]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, [init_p.beta, init_p.sigma, init_p.gamma], w, label="initial", color="#bdc3c7")
    ax.bar(x + w / 2, [learned_p.beta, learned_p.sigma, learned_p.gamma], w, label="learned", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_title("Initial vs learned epidemic parameters")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def correlation_score(P_I: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float | None:
    true_I = (X == STATE_IDX["I"]).astype(float)
    obs_mask = Y != OBS_MISSING
    if obs_mask.sum() <= 1 or np.std(P_I[obs_mask]) < 1e-8:
        return None
    return float(np.corrcoef(P_I[obs_mask], true_I[obs_mask])[0, 1])


def fig_evaluation_dashboard(G, pos, counts, P_I, history, pz, X=None):
    max_P = P_I.max(axis=0)
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=380, font_size=7, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[pz], node_color="#e74c3c", node_size=420, ax=ax)
    ax.set_title("A  Contact network")

    ax = fig.add_subplot(gs[0, 1])
    stack = np.array([counts[s] for s in STATES])
    ax.stackplot(
        range(stack.shape[1]), stack, labels=list(STATES),
        colors=["#3498db", "#f39c12", "#e74c3c", "#27ae60"], alpha=0.8,
    )
    ax.set_title("B  SEIR compartments")
    ax.legend(fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_title("C  P(Infected) heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    if history is not None:
        for idx, (lab, col) in enumerate(zip(["beta", "sigma", "gamma"], ["#2980b9", "#8e44ad", "#16a085"])):
            ax.plot(history[:, idx], label=lab, color=col, lw=2)
        ax.set_title("D  EM learning")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Run EM on Page 4", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D  EM learning")
        ax.axis("off")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(counts["I"], "r-", lw=2, label="true infected")
    ax.plot(P_I.sum(axis=1), "b--", lw=2, label="Σ P(I) inferred")
    ax.set_xlabel("day")
    ax.set_title("E  Epidemic: true vs inferred total")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    art = nx.draw_networkx_nodes(
        G, pos, node_color=max_P, cmap=plt.cm.Reds, vmin=0, vmax=1, node_size=400, ax=ax,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    fig.colorbar(art, ax=ax, fraction=0.046, label="max P(I)")
    ax.set_title("F  Network posterior map")

    fig.suptitle("Project evaluation dashboard", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def run_em_streaming(G, Y, init_params, pz, n_iter, chart_placeholder):
    """Run EM one iteration at a time, updating the convergence plot."""
    params = init_params
    history = [params.as_array()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lines = {}
    cols = {"beta": "#2980b9", "sigma": "#8e44ad", "gamma": "#16a085"}
    for idx, lab in enumerate(["beta", "sigma", "gamma"]):
        lines[lab], = ax.plot([], [], "o-", label=lab, color=cols[lab], lw=2, ms=4)
    ax.set_xlim(0, n_iter)
    ax.set_ylim(0.01, 0.99)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("parameter value")
    ax.set_title("EM parameter convergence (live)")
    ax.legend()
    ax.grid(alpha=0.3)

    for k in range(n_iter):
        alpha = forward_filter(G, Y, params, patient_zero=pz)
        gamma = backward_smooth(G, Y, alpha, params)
        stats = _expected_pair_counts(gamma, G)
        params = m_step(stats, params)
        history.append(params.as_array())
        xs = np.arange(len(history))
        for idx, lab in enumerate(["beta", "sigma", "gamma"]):
            lines[lab].set_data(xs, [h[idx] for h in history])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        chart_placeholder.pyplot(fig)

    plt.close(fig)
    return params, np.array(history)
