"""SEIR conditional probability tables and DBN representation."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .config import STATES, STATE_IDX, N_STATES, ModelParams


def infection_probability(
    infectious_neighbor_probs: list[float],
    beta: float,
) -> float:
    """
    Noisy-OR infection probability for a susceptible individual.

    P(E | S, neighbors) = 1 - prod_j (1 - beta * P(X_j = I))
    """
    if not infectious_neighbor_probs:
        return 0.0
    survival = 1.0
    for p_i in infectious_neighbor_probs:
        survival *= 1.0 - beta * np.clip(p_i, 0.0, 1.0)
    return float(1.0 - survival)


def transition_distribution(
    prev_state: str,
    infectious_neighbor_probs: list[float],
    params: ModelParams,
) -> np.ndarray:
    """
    P(X^t | X^{t-1} = prev_state, neighbor infectiousness).

    Returns a length-N_STATES probability vector over [S, E, I, R].
    """
    p = np.zeros(N_STATES)
    if prev_state == "R":
        p[STATE_IDX["R"]] = 1.0
        return p
    if prev_state == "I":
        p[STATE_IDX["I"]] = 1.0 - params.gamma
        p[STATE_IDX["R"]] = params.gamma
        return p
    if prev_state == "E":
        p[STATE_IDX["E"]] = 1.0 - params.sigma
        p[STATE_IDX["I"]] = params.sigma
        return p
    # Susceptible
    pe = infection_probability(infectious_neighbor_probs, params.beta)
    p[STATE_IDX["S"]] = 1.0 - pe
    p[STATE_IDX["E"]] = pe
    return p


def transition_matrix(
    infectious_neighbor_probs: list[float],
    params: ModelParams,
) -> np.ndarray:
    """
    Full transition matrix T[s_prev, s_next] for one individual.

    Marginalizes over the given neighbor infectiousness (mean-field).
    """
    T = np.zeros((N_STATES, N_STATES))
    for s_prev, name in enumerate(STATES):
        T[s_prev, :] = transition_distribution(name, infectious_neighbor_probs, params)
    return T


def emission_likelihood(
    observation: int,
    params: ModelParams,
) -> np.ndarray:
    """
    P(Y^t | X^t) as a vector over states.

    observation: OBS_NEG, OBS_POS, or OBS_MISSING (returns ones).
    """
    from .config import OBS_MISSING, OBS_NEG, OBS_POS

    if observation == OBS_MISSING:
        return np.ones(N_STATES)

    lik = np.ones(N_STATES)
    sens = params.sensitivity
    spec = params.specificity
    for s_idx, state in enumerate(STATES):
        if state == "I":
            lik[s_idx] = sens if observation == OBS_POS else (1.0 - sens)
        else:
            lik[s_idx] = (1.0 - spec) if observation == OBS_POS else spec
    return lik


def build_dbn_structure(G: nx.Graph) -> dict:
    """
    Describe the 2-time-slice DBN structure (Representation pillar).

    Returns adjacency information for documentation and pgmpy export.
    """
    n = G.number_of_nodes()
    intra_edges = [(f"X_{i}_t", f"Y_{i}_t") for i in range(n)]
    temporal_self = [(f"X_{i}_t1", f"X_{i}_t2") for i in range(n)]
    temporal_contact = [
        (f"X_{j}_t1", f"X_{i}_t2")
        for i in range(n)
        for j in G.neighbors(i)
    ]
    return {
        "latent_variables": [f"X_{i}" for i in range(n)],
        "observed_variables": [f"Y_{i}" for i in range(n)],
        "intra_slice_edges": intra_edges,
        "inter_slice_edges": temporal_self + temporal_contact,
        "description": (
            "2-slice DBN: X_i^{t-1} -> X_i^t (within-person dynamics), "
            "X_j^{t-1} -> X_i^t for j in neighbors(i) (transmission), "
            "X_i^t -> Y_i^t (symptom/test observation)."
        ),
    }


def export_pgmpy_dbn(G: nx.Graph, params: ModelParams):
    """
    Build a pgmpy DynamicBayesianNetwork skeleton for the representation pillar.

    Note: Full TabularCPDs for network coupling are exponential in degree;
    inference in this project uses tractable mean-field forward-backward.
    """
    try:
        from pgmpy.models import DynamicBayesianNetwork as DBN
    except ImportError as exc:
        raise ImportError("pgmpy is required for DBN export.") from exc

    dbn = DBN()
    n = G.number_of_nodes()

    for i in range(n):
        dbn.add_edges_with_time_slice(
            [(f"X_{i}", f"Y_{i}")],
            timesteps=2,
        )
        dbn.add_edges_with_time_slice(
            [(f"X_{i}", f"X_{i}")],
            timesteps=2,
        )
        for j in G.neighbors(i):
            dbn.add_edges_with_time_slice(
                [(f"X_{j}", f"X_{i}")],
                timesteps=2,
            )

    return dbn
