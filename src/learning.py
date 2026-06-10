"""EM parameter learning for the SEIR DBN (Learning pillar)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import STATE_IDX, ModelParams

if TYPE_CHECKING:
    import networkx as nx
from .inference import forward_filter, backward_smooth


def _expected_pair_counts(
    gamma: np.ndarray,
    G: nx.Graph,
) -> dict[str, float]:
    """
    Compute expected sufficient statistics from smoothed beliefs.

    Uses soft transition counts: xi[t, s, s'] ≈ gamma[t, s] * gamma[t+1, s'].
    """
    T_steps, n, _ = gamma.shape
    stats = {
        "E_total": 0.0,
        "E_to_I": 0.0,
        "I_total": 0.0,
        "I_to_R": 0.0,
        "S_exposed_total": 0.0,
        "S_to_E": 0.0,
    }

    for t in range(T_steps - 1):
        for i in range(n):
            q_t = gamma[t, i]
            q_tp1 = gamma[t + 1, i]

            stats["E_total"] += q_t[STATE_IDX["E"]]
            stats["I_total"] += q_t[STATE_IDX["I"]]
            stats["E_to_I"] += q_t[STATE_IDX["E"]] * q_tp1[STATE_IDX["I"]]
            stats["I_to_R"] += q_t[STATE_IDX["I"]] * q_tp1[STATE_IDX["R"]]

            has_exposure = any(
                gamma[t, j, STATE_IDX["I"]] > 0.01 for j in G.neighbors(i)
            )
            if has_exposure:
                stats["S_exposed_total"] += q_t[STATE_IDX["S"]]
                stats["S_to_E"] += q_t[STATE_IDX["S"]] * q_tp1[STATE_IDX["E"]]

    return stats


def m_step(stats: dict[str, float], params: ModelParams) -> ModelParams:
    """Re-estimate beta, sigma, gamma from expected counts."""
    sigma = np.clip(stats["E_to_I"] / (stats["E_total"] + 1e-8), 0.01, 0.99)
    gamma = np.clip(stats["I_to_R"] / (stats["I_total"] + 1e-8), 0.01, 0.99)
    beta = np.clip(stats["S_to_E"] / (stats["S_exposed_total"] + 1e-8), 0.01, 0.99)

    return ModelParams(
        beta=float(beta),
        sigma=float(sigma),
        gamma=float(gamma),
        sensitivity=params.sensitivity,
        specificity=params.specificity,
    )


def em_learn(
    G: nx.Graph,
    Y: np.ndarray,
    params_init: ModelParams,
    n_iter: int = 30,
    patient_zero: int = 0,
    verbose: bool = True,
) -> tuple[ModelParams, np.ndarray]:
    """
    EM algorithm to learn beta, sigma, gamma from partial observations.

    Returns learned parameters and history array of shape (n_iter+1, 3).
    """
    params = params_init
    history = [params.as_array()]

    for k in range(n_iter):
        alpha = forward_filter(G, Y, params, patient_zero=patient_zero)
        gamma = backward_smooth(G, Y, alpha, params)
        stats = _expected_pair_counts(gamma, G)
        params = m_step(stats, params)
        history.append(params.as_array())
        if verbose:
            print(
                f"  EM iter {k + 1:2d}: "
                f"beta={params.beta:.3f}, sigma={params.sigma:.3f}, gamma={params.gamma:.3f}"
            )

    return params, np.array(history)
