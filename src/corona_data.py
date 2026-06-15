"""
COVID-19 (Corona) epidemic dataset for the PGM DBN project.

Primary source: Geneva contact-tracing records (GEgraph) — individual-level
network + positive PCR tests from the 2020 COVID-19 pandemic.

Secondary context: Our World in Data — Switzerland national case curve.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from .config import SimConfig
from .real_data import (
    RealDataBundle,
    _bfs_subset,
    _build_adjacency,
    _pick_seed,
    _positive_test_dates,
)

CORONA_DIR = Path(__file__).resolve().parent.parent / "data" / "corona"
GENEVA_BASE_URL = "https://raw.githubusercontent.com/PersonalDataIO/GEgraph/master/data"
GENEVA_FILES = {"suivi": "redcap_suivi.csv", "entourage": "redcap_entourage.csv"}
OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)


def download_corona_dataset(verbose: bool = True) -> Path:
    """Download COVID-19 contact-tracing CSVs into data/corona/."""
    CORONA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in GENEVA_FILES.values():
        dest = CORONA_DIR / filename
        if dest.exists():
            continue
        url = f"{GENEVA_BASE_URL}/{filename}"
        if verbose:
            print(f"  Downloading COVID-19 dataset: {filename} ...")
        urllib.request.urlretrieve(url, dest)
    return CORONA_DIR


def load_corona_contact_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw COVID-19 contact-tracing tables."""
    download_corona_dataset()
    suivi = pd.read_csv(CORONA_DIR / GENEVA_FILES["suivi"])
    entourage = pd.read_csv(CORONA_DIR / GENEVA_FILES["entourage"])
    return suivi, entourage


def load_corona_owid_context(location: str = "Switzerland") -> pd.DataFrame:
    """Load national COVID-19 case curve for epidemiological context."""
    cache = CORONA_DIR / "owid-covid-context.csv"
    if not cache.exists():
        if True:
            print("  Downloading OWID COVID-19 context data ...")
        df = pd.read_csv(OWID_URL)
        df.to_csv(cache, index=False)
    else:
        df = pd.read_csv(cache)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["location"] == location].sort_values("date")


def load_corona_dataset(
    max_nodes: int = 30,
    pad_days_before: int = 7,
    pad_days_after: int = 21,
) -> RealDataBundle:
    """
    Load integrated COVID-19 (Corona) dataset for the DBN pipeline.

    Returns contact network, test observations Y, and approximate SEIR states.
    """
    download_corona_dataset()
    suivi = pd.read_csv(CORONA_DIR / GENEVA_FILES["suivi"])
    entourage = pd.read_csv(CORONA_DIR / GENEVA_FILES["entourage"])

    # Reuse core logic from real_data but read from corona folder
    import networkx as nx
    import numpy as np
    from datetime import timedelta
    from .config import OBS_MISSING, OBS_POS, STATE_IDX

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
        raise ValueError("No positive COVID-19 tests in selected subgraph.")

    t0 = min(subgraph_pos) - timedelta(days=pad_days_before)
    t1 = max(subgraph_pos) + timedelta(days=pad_days_after)
    dates = list(pd.date_range(t0, t1, freq="D"))
    T, n = len(dates), len(node_ids)
    date_to_t = {d.normalize(): i for i, d in enumerate(pd.to_datetime(dates))}

    Y = np.full((T, n), OBS_MISSING, dtype=int)
    X = np.full((T, n), STATE_IDX["S"], dtype=int)
    recovery_days = 14

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

    return RealDataBundle(
        graph=G,
        Y_obs=Y,
        X_true=X,
        node_ids=node_ids,
        dates=dates,
        dataset_name="COVID-19 (Corona) — Geneva contact tracing",
        patient_zero=id_to_idx[seed],
        metadata={
            "dataset": "COVID-19 / Corona",
            "seed_person_id": seed,
            "n_nodes": n,
            "n_timesteps": T,
            "n_positive_tests": int((Y == OBS_POS).sum()),
            "date_start": str(dates[0].date()),
            "date_end": str(dates[-1].date()),
            "source": "https://github.com/PersonalDataIO/GEgraph",
            "context": "OWID Switzerland national COVID-19 cases",
        },
    )


def load_corona_for_config(config: SimConfig) -> RealDataBundle:
    """Entry point used by run.py and the notebook."""
    return load_corona_dataset(max_nodes=config.n_nodes)
