"""Forward simulation of SEIR dynamics and noisy partial observations."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .config import STATES, STATE_IDX, N_STATES, OBS_MISSING, ModelParams, SimConfig
from .model import transition_distribution, emission_likelihood


def _neighbor_infectious_flags(X_prev: np.ndarray, G: nx.Graph, i: int) -> list[float]:
    """Hard 0/1 indicators: neighbor j is infectious at previous timestep."""
    return [
        1.0 if STATES[X_prev[j]] == "I" else 0.0
        for j in G.neighbors(i)
    ]


def simulate_epidemic(
    G: nx.Graph,
    params: ModelParams,
    config: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate latent SEIR states and partial test observations.

    Returns
    -------
    X_true : ndarray (T, n) int state indices
    Y_obs : ndarray (T, n) int observations (OBS_NEG, OBS_POS, OBS_MISSING)
    """
    rng = np.random.default_rng(config.seed)
    n = G.number_of_nodes()
    T = config.n_timesteps

    X = np.full((T, n), STATE_IDX["S"], dtype=int)
    Y = np.full((T, n), OBS_MISSING, dtype=int)

    X[0, config.patient_zero] = STATE_IDX["I"]

    for t in range(1, T):
        for i in range(n):
            prev = STATES[X[t - 1, i]]
            flags = _neighbor_infectious_flags(X[t - 1], G, i)
            p = transition_distribution(prev, flags, params)
            X[t, i] = int(rng.choice(N_STATES, p=p))

    for t in range(T):
        for i in range(n):
            if rng.random() > config.test_probability:
                continue
            true_state = STATES[X[t, i]]
            # Sample from emission: positive if I with sensitivity, else specificity
            if true_state == "I":
                y = 1 if rng.random() < params.sensitivity else 0
            else:
                y = 1 if rng.random() > params.specificity else 0
            Y[t, i] = y

    return X, Y


def epidemic_counts(X: np.ndarray) -> dict[str, list[int]]:
    """Count S, E, I, R over time."""
    T = X.shape[0]
    counts = {s: [] for s in STATES}
    for t in range(T):
        for s in STATES:
            counts[s].append(int((X[t] == STATE_IDX[s]).sum()))
    return counts
