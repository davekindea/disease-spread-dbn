"""Load real-world epidemic data for the DBN pipeline."""

from __future__ import annotations

import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import OBS_MISSING, OBS_POS, STATE_IDX, SimConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

GENEVA_BASE_URL = (
    "https://raw.githubusercontent.com/PersonalDataIO/GEgraph/master/data"
)
GENEVA_FILES = {
    "suivi": "redcap_suivi.csv",
    "entourage": "redcap_entourage.csv",
}


@dataclass
class RealDataBundle:
    """Container for a real-data epidemic instance."""

    graph: nx.Graph
    Y_obs: np.ndarray
    X_true: np.ndarray | None
    node_ids: list[int]
    dates: list[pd.Timestamp]
    dataset_name: str
    patient_zero: int = 0
    metadata: dict = field(default_factory=dict)


def _download_geneva_raw() -> None:
    """Fetch Geneva contact-tracing CSVs if not cached locally."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in GENEVA_FILES.values():
        dest = RAW_DIR / filename
        if dest.exists():
            continue
        url = f"{GENEVA_BASE_URL}/{filename}"
        print(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)


def _build_adjacency(suivi: pd.DataFrame, entourage: pd.DataFrame) -> dict[int, set[int]]:
    """Build undirected adjacency from contact-tracing tables."""
    adj: dict[int, set[int]] = {}

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for _, row in entourage.dropna(subset=["contact_record_id"]).iterrows():
        other = int(row["contact_record_id"])
        source = row.get("record_id_pos")
        if pd.isna(source):
            source = row.get("record_id_entourage")
        if pd.isna(source):
            continue
        add_edge(int(source), other)

    for _, row in suivi.dropna(subset=["contact_record_id", "record_id_pos"]).iterrows():
        add_edge(int(row["record_id_pos"]), int(row["contact_record_id"]))

    return adj


def _positive_test_dates(suivi: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """Earliest positive test date per person."""
    dated = suivi.dropna(subset=["date_res", "record_id_pos"]).copy()
    dated["date_res"] = pd.to_datetime(dated["date_res"])
    grouped = dated.groupby("record_id_pos")["date_res"].min()
    return {int(pid): ts for pid, ts in grouped.items()}


def _bfs_subset(adj: dict[int, set[int]], seed: int, max_nodes: int) -> set[int]:
    """Collect a connected subgraph by breadth-first expansion."""
    nodes = {seed}
    queue: deque[int] = deque([seed])
    while queue and len(nodes) < max_nodes:
        current = queue.popleft()
        for neighbor in adj.get(current, ()):
            if neighbor not in nodes:
                nodes.add(neighbor)
                queue.append(neighbor)
    return nodes


def _pick_seed(adj: dict[int, set[int]], pos_dates: dict[int, pd.Timestamp]) -> int:
    """Choose an early infected individual with several contacts."""
    early = [pid for pid, dt in pos_dates.items() if dt < pd.Timestamp("2020-04-01")]
    ranked = sorted(
        ((pid, len(adj.get(pid, []))) for pid in early if adj.get(pid)),
        key=lambda x: -x[1],
    )
    if not ranked:
        return max(pos_dates, key=lambda k: len(adj.get(k, [])))
    return ranked[0][0]


def load_geneva_contact_tracing(
    max_nodes: int = 30,
    pad_days_before: int = 7,
    pad_days_after: int = 21,
    recovery_days: int = 14,
) -> RealDataBundle:
    """
    Load Geneva COVID-19 contact-tracing data.

    Returns a contact network, partial positive-test observations, and
    approximate latent states for evaluation.
    """
    _download_geneva_raw()
    suivi = pd.read_csv(RAW_DIR / GENEVA_FILES["suivi"])
    entourage = pd.read_csv(RAW_DIR / GENEVA_FILES["entourage"])

    adj = _build_adjacency(suivi, entourage)
    pos_dates = _positive_test_dates(suivi)
    seed = _pick_seed(adj, pos_dates)
    subset = _bfs_subset(adj, seed, max_nodes)

    node_ids = sorted(subset)
    id_to_idx = {pid: i for i, pid in enumerate(node_ids)}

    G = nx.Graph()
    G.add_nodes_from(range(len(node_ids)))
    for u in subset:
        for v in adj.get(u, ()):
            if v in subset and u < v:
                G.add_edge(id_to_idx[u], id_to_idx[v])

    subgraph_pos = [pos_dates[pid] for pid in node_ids if pid in pos_dates]
    if not subgraph_pos:
        raise ValueError("No positive tests found in selected subgraph.")

    t0 = min(subgraph_pos) - timedelta(days=pad_days_before)
    t1 = max(subgraph_pos) + timedelta(days=pad_days_after)
    dates = list(pd.date_range(t0, t1, freq="D"))
    T = len(dates)
    n = len(node_ids)
    date_to_t = {d.normalize(): i for i, d in enumerate(pd.to_datetime(dates))}

    Y = np.full((T, n), OBS_MISSING, dtype=int)
    X = np.full((T, n), STATE_IDX["S"], dtype=int)

    for pid in node_ids:
        i = id_to_idx[pid]
        if pid not in pos_dates:
            continue
        test_day = pos_dates[pid].normalize()
        if test_day not in date_to_t:
            continue
        t_pos = date_to_t[test_day]
        Y[t_pos, i] = OBS_POS

        expose_start = max(0, t_pos - 5)
        recover_start = min(T, t_pos + recovery_days)
        for t in range(T):
            if t < expose_start:
                X[t, i] = STATE_IDX["S"]
            elif t < t_pos:
                X[t, i] = STATE_IDX["E"]
            elif t < recover_start:
                X[t, i] = STATE_IDX["I"]
            else:
                X[t, i] = STATE_IDX["R"]

    patient_zero = id_to_idx[seed]

    return RealDataBundle(
        graph=G,
        Y_obs=Y,
        X_true=X,
        node_ids=node_ids,
        dates=dates,
        dataset_name="Geneva COVID-19 contact tracing (GEgraph)",
        patient_zero=patient_zero,
        metadata={
            "seed_person_id": seed,
            "n_nodes": n,
            "n_timesteps": T,
            "n_positive_tests": int((Y == OBS_POS).sum()),
            "date_start": str(dates[0].date()),
            "date_end": str(dates[-1].date()),
            "source": "https://github.com/PersonalDataIO/GEgraph",
        },
    )


def load_real_data(config: SimConfig) -> RealDataBundle:
    """Load real data according to simulation config."""
    try:
        from .corona_data import load_corona_for_config
        return load_corona_for_config(config)
    except Exception:
        return load_geneva_contact_tracing(max_nodes=config.n_nodes)
