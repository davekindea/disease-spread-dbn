"""Forward-backward inference for the SEIR DBN (Inference pillar)."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .config import STATE_IDX, N_STATES, OBS_MISSING, ModelParams
from .model import transition_matrix, emission_likelihood


def _neighbor_infectious_probs(belief_prev: np.ndarray, G: nx.Graph, i: int) -> list[float]:
    """P(X_j^{t-1} = I) for each neighbor j (mean-field coupling)."""
    return [float(belief_prev[j, STATE_IDX["I"]]) for j in G.neighbors(i)]


def forward_filter(
    G: nx.Graph,
    Y: np.ndarray,
    params: ModelParams,
    patient_zero: int = 0,
) -> np.ndarray:
    """
    Forward filtering: alpha[t, i, s] = P(X_i^t = s | Y^{1:t}).

    Uses mean-field approximation for neighbor coupling.
    """
    T, n = Y.shape
    alpha = np.zeros((T, n, N_STATES))

    # Initial slice: everyone susceptible except diffuse prior on patient zero
    alpha[0, :, STATE_IDX["S"]] = 1.0
    alpha[0, patient_zero, :] = 1.0 / N_STATES

    for t in range(1, T):
        for i in range(n):
            nb_probs = _neighbor_infectious_probs(alpha[t - 1], G, i)
            T_mat = transition_matrix(nb_probs, params)
            pred = alpha[t - 1, i] @ T_mat
            pred *= emission_likelihood(Y[t, i], params)
            total = pred.sum()
            alpha[t, i] = pred / total if total > 0 else np.ones(N_STATES) / N_STATES

    # Update t=0 with emission if observed
    for i in range(n):
        if Y[0, i] != OBS_MISSING:
            lik = emission_likelihood(Y[0, i], params)
            alpha[0, i] *= lik
            total = alpha[0, i].sum()
            alpha[0, i] /= total if total > 0 else 1.0

    return alpha


def backward_smooth(
    G: nx.Graph,
    Y: np.ndarray,
    alpha: np.ndarray,
    params: ModelParams,
) -> np.ndarray:
    """
    Backward smoothing: gamma[t, i, s] = P(X_i^t = s | Y^{1:T}).

    Combines forward messages with backward beta messages.
    """
    T, n, _ = alpha.shape
    beta = np.ones((T, n, N_STATES))

    for t in range(T - 2, -1, -1):
        for i in range(n):
            nb_probs = _neighbor_infectious_probs(alpha[t], G, i)
            T_mat = transition_matrix(nb_probs, params)
            # Expected future compatibility
            future = beta[t + 1, i] * alpha[t + 1, i]
            future /= future.sum() + 1e-12
            score = T_mat @ future
            beta[t, i] = np.clip(score, 1e-12, None)
        beta[t] /= beta[t].sum(axis=1, keepdims=True) + 1e-12

    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True) + 1e-12
    return gamma


def infer_infectious_probability(
    G: nx.Graph,
    Y: np.ndarray,
    params: ModelParams,
    patient_zero: int = 0,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Answer: P(X_i^t = I | observations).

    Returns (P_I, beliefs) where P_I has shape (T, n).
    """
    alpha = forward_filter(G, Y, params, patient_zero=patient_zero)
    beliefs = backward_smooth(G, Y, alpha, params) if smooth else alpha
    P_I = beliefs[:, :, STATE_IDX["I"]]
    return P_I, beliefs


def query_node_infectious(
    G: nx.Graph,
    Y: np.ndarray,
    params: ModelParams,
    node: int,
    time: int | None = None,
    patient_zero: int = 0,
) -> float | np.ndarray:
    """
    Query: given symptom/test observations, P(node is infectious).

    If time is None, returns P(I) at each timestep for that node.
    """
    P_I, _ = infer_infectious_probability(
        G, Y, params, patient_zero=patient_zero, smooth=True
    )
    if time is not None:
        return float(P_I[time, node])
    return P_I[:, node]
