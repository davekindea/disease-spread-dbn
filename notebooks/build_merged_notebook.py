"""Build ONE merged notebook: reference static BN + Corona SEIR DBN project."""
import json
import re
from pathlib import Path

NOTEBOOKS = Path(__file__).parent
REF_PATH = NOTEBOOKS / "REFERENCE_END_TO_END_PGM_PIPELINE.ipynb"
OUT_PATH = NOTEBOOKS / "PGM_Complete_EndToEnd_Pipeline.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in text.split("\n")],
    }


def extract_ref_cells() -> list[dict]:
    """Pull code/markdown sources from reference notebook (no outputs)."""
    if not REF_PATH.exists():
        return []
    nb = json.loads(REF_PATH.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] not in ("markdown", "code"):
            continue
        src = "".join(c.get("source", []))
        if c["cell_type"] == "code" and src.strip() == "pip install pgmpy":
            continue  # skip duplicate pip
        out.append({"cell_type": c["cell_type"], "source": src})
    return out


def relabel_ref_part_headers(src: str) -> str:
    """Prefix reference PART headers as Module 1."""
    src = re.sub(r"^## PART (\d)", r"## MODULE 1 — PART \1", src, flags=re.M)
    src = re.sub(r"^\*\*(\d-[A-Z])", r"**M1-\1", src, flags=re.M)
    return src


CELLS: list[dict] = []

# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED HEADER
# ═══════════════════════════════════════════════════════════════════════════
CELLS.append(md("""# Complete PGM End-to-End Pipeline (Single Notebook)

**Merged notebook** — everything in one place:

| Module | Topic | Dataset |
|--------|-------|---------|
| **Module 1** | Static Bayesian Network (course reference) | Breast Cancer Wisconsin |
| **Module 2** | Dynamic Bayesian Network (your project) | COVID-19 (Corona) contact tracing |

**Libraries:** pgmpy, scikit-learn, networkx, numpy, pandas, matplotlib

---

### Project goal (Module 2)

> Model disease spread using a DBN with latent **SEIR** states and **COVID test** observations. Answer: *P(node i is infectious | all test observations)?*

---

Run all cells **top to bottom**."""))

CELLS.append(code("pip install pgmpy networkx matplotlib numpy pandas scipy scikit-learn"))

CELLS.append(md("""LECTURE OUTLINE (complete pipeline)

**MODULE 1 — Static BN (reference pattern)**
* PART 1  DATA & PREPROCESSING   – Breast Cancer Wisconsin, discretisation
* PART 2  REPRESENTATION          – Bayesian Network theory
* PART 3  STRUCTURE LEARNING      – Hill-Climb + BIC
* PART 4  PARAMETER LEARNING      – Maximum Likelihood Estimation (MLE)
* PART 5  INFERENCE               – Variable Elimination
* PART 6  EVALUATION              – classification accuracy

**MODULE 2 — Dynamic BN (COVID-19 / Corona project)**
* PART 1  DATA & PREPROCESSING   – Corona contact tracing + OWID context
* PART 2  REPRESENTATION          – 2-time-slice DBN, SEIR CPTs
* PART 3  MODEL STRUCTURE         – contact network from Corona data
* PART 4  PARAMETER LEARNING      – EM algorithm (beta, sigma, gamma)
* PART 5  INFERENCE               – Variable Elimination + forward-backward
* PART 6  EVALUATION & FIGURES    – epidemic curves, heatmaps, dashboard"""))

CELLS.append(code("""import warnings
warnings.filterwarnings("ignore")
import textwrap
import sys
from pathlib import Path

_cwd = Path.cwd()
PROJECT_ROOT = _cwd if (_cwd / "src").is_dir() else (_cwd.parent if (_cwd.parent / "src").is_dir() else _cwd)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

# scikit-learn (Module 1)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# pgmpy (both modules)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination
from pgmpy.models import DynamicBayesianNetwork as DBN

# Project src (Module 2)
from src.config import ModelParams, SimConfig, STATES, STATE_IDX
from src.config import OBS_MISSING, OBS_POS, OBS_NEG
from src.corona_data import (
    download_corona_dataset, load_corona_dataset,
    load_corona_contact_tables, load_corona_owid_context, CORONA_DIR,
)
from src.model import build_dbn_structure, transition_distribution, emission_likelihood, export_pgmpy_dbn
from src.network import network_summary
from src.inference import infer_infectious_probability, query_node_infectious
from src.learning import em_learn
from src.simulation import epidemic_counts

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Project root:", PROJECT_ROOT)
print("Output dir  :", OUTPUT_DIR)"""))

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1 — from reference notebook
# ═══════════════════════════════════════════════════════════════════════════
CELLS.append(md("""---

# MODULE 1 — Static Bayesian Network (Course Reference)

  Dataset  : sklearn Breast Cancer Wisconsin (569 samples, 30 features)

  Model    : Discrete Bayesian Network

  Goal     : Learn DAG structure, fit CPDs, run Variable Elimination inference

---"""))

ref_cells = extract_ref_cells()
skip_until_part1 = True
for rc in ref_cells:
    src = rc["source"]
    # Skip reference duplicates: header, outline, imports
    if skip_until_part1:
        if "## PART 1" in src and "PREPROCESSING" in src:
            skip_until_part1 = False
        else:
            continue
    # Stop before we'd duplicate Module 2 content — reference ends at PART 6
    if rc["cell_type"] == "markdown":
        CELLS.append(md(relabel_ref_part_headers(src)))
    else:
        CELLS.append(code(src))

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2 — Corona SEIR DBN project
# ═══════════════════════════════════════════════════════════════════════════
CELLS.append(md("""---

# MODULE 2 — Dynamic Bayesian Network (COVID-19 / Corona Project)

  Dataset  : COVID-19 Geneva contact tracing + OWID Switzerland context

  Model    : 2-time-slice SEIR Dynamic Bayesian Network on a contact network

  Goal     : Representation + Inference + EM learning on real Corona epidemic data

---"""))

# Import build_notebook module 2 cells by reading build_notebook.py logic inline
# We append Module 2 parts directly here for maintainability

M2 = []

M2.append(md("## MODULE 2 — PART 1 – DATA & PREPROCESSING"))

M2.append(md("""**M2-1-A  Load the COVID-19 (Corona) dataset**

* Contact-tracing CSVs: individuals, close contacts, positive PCR test dates
* OWID Switzerland: national COVID-19 case curve for context"""))

M2.append(code("""print(textwrap.dedent(\"\"\"
MODULE 2 — COVID-19 (CORONA) DATASET
  Latent X_i^t  = SEIR state (Susceptible, Exposed, Infectious, Recovered)
  Observed Y_i^t = COVID test (positive / negative / missing)
  Network        = documented close contacts from tracing
\"\"\"))

download_corona_dataset()
suivi, entourage = load_corona_contact_tables()
print(f"Tracing records : {len(suivi):,}  |  Contact rows : {len(entourage):,}")
print(f"Data folder     : {CORONA_DIR}")
print(suivi[["record_id_pos","date_res","contact_record_id"]].dropna(subset=["date_res"]).head(3))

owid = load_corona_owid_context("Switzerland")
owid_sub = owid[(owid["date"]>="2020-02-01")&(owid["date"]<="2020-06-30")]
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(owid_sub["date"], owid_sub["new_cases"], color="#c0392b", lw=1.5)
ax.set_title("COVID-19 Switzerland — daily new cases (OWID)"); ax.grid(alpha=0.3)
plt.xticks(rotation=30); plt.tight_layout(); plt.show()"""))

M2.append(md("**M2-1-B  Build outbreak subgraph and observation matrix Y**"))

M2.append(code("""bundle = load_corona_dataset(max_nodes=30)
G, Y_obs, X_true = bundle.graph, bundle.Y_obs, bundle.X_true
patient_zero, meta = bundle.patient_zero, bundle.metadata
params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)

print(f"Dataset      : {bundle.dataset_name}")
print(f"Nodes / days : {meta['n_nodes']} people, {meta['n_timesteps']} timesteps")
print(f"Date range   : {meta['date_start']} -> {meta['date_end']}")
print(f"Pos. tests   : {meta['n_positive_tests']}  |  Matrix {Y_obs.shape}")
print(f"Patient zero : node {patient_zero}")"""))

M2.append(md("**M2-1-C  List positive COVID test observations**"))

M2.append(code("""obs_rows = []
for t in range(Y_obs.shape[0]):
    for i in range(Y_obs.shape[1]):
        if Y_obs[t,i] == OBS_POS:
            obs_rows.append({"node":i, "person_id":bundle.node_ids[i],
                "day_t":t, "date":str(bundle.dates[t].date()), "test":"POSITIVE"})
display(pd.DataFrame(obs_rows))
counts = epidemic_counts(X_true)
print(f"Peak infectious (approx): {max(counts['I'])}")"""))

M2.append(md("## MODULE 2 — PART 2 – REPRESENTATION  (Dynamic Bayesian Network)"))

M2.append(code("""print(textwrap.dedent(\"\"\"
A Dynamic Bayesian Network (DBN) unrolls a 2-time-slice template:

  X_i^{t-1} -> X_i^t       within-person SEIR dynamics
  X_j^{t-1} -> X_i^t       transmission (contact network)
  X_i^t     -> Y_i^t       COVID test emission

SEIR latent states: S, E, I, R
Parameters: beta (transmission), sigma (E->I), gamma (I->R)
\"\"\"))"""))

M2.append(md("## MODULE 2 — PART 3 – MODEL STRUCTURE"))

M2.append(code("""structure = build_dbn_structure(G)
summary = network_summary(G)
print(structure["description"])
print(summary)
dbn = export_pgmpy_dbn(G, params)
print(f"pgmpy DBN: {len(list(dbn.nodes()))} node variables, {len(list(dbn.edges()))} edges")

pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8,6))
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=550, font_size=9, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#e74c3c", node_size=650, ax=ax)
ax.set_title("FIGURE M2-0 — COVID-19 Contact Network (red = patient zero)")
plt.tight_layout(); fig.savefig(OUTPUT_DIR/"fig0_network.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(9,4.5))
ax.plot(counts["I"], label="Infectious", lw=2.5, color="#c0392b")
ax.plot(counts["E"], label="Exposed", ls="--", color="#e67e22")
ax.set_title("FIGURE M2-1 — Epidemic Curve"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUTPUT_DIR/"fig1_epidemic_curve.png", dpi=150); plt.show()"""))

M2.append(md("## MODULE 2 — PART 4 – PARAMETER LEARNING  (EM Algorithm)"))

M2.append(code("""print(textwrap.dedent(\"\"\"
Module 1 used MLE on observed features.  Module 2 uses EM because
SEIR states are LATENT — we only see COVID test results.

E-step: forward-backward inference -> soft state counts
M-step: re-estimate beta, sigma, gamma
\"\"\"))
init_p = ModelParams(0.10, 0.10, 0.10)
learned, history = em_learn(G, Y_obs, init_p, n_iter=25, patient_zero=patient_zero, verbose=True)
print(f"Learned: beta={learned.beta:.3f} sigma={learned.sigma:.3f} gamma={learned.gamma:.3f}")

fig, ax = plt.subplots(figsize=(9,4.5))
for idx, (lab,col) in enumerate(zip(["beta","sigma","gamma"],["#2980b9","#8e44ad","#16a085"])):
    ax.plot(history[:,idx], label=lab, color=col, lw=2)
ax.set_title("FIGURE M2-4 — EM Convergence"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUTPUT_DIR/"fig4_em_convergence.png", dpi=150); plt.show()"""))

M2.append(md("## MODULE 2 — PART 5 – INFERENCE"))

M2.append(code("""print(textwrap.dedent(\"\"\"
Module 1: Variable Elimination on static BN.
Module 2: VE on a COVID test model + forward-backward on full network.

PROJECT QUERY: P(X_i^t = Infectious | all COVID test observations)
\"\"\"))

# pgmpy VE demo (parallel to Module 1 Part 5)
covid_bn = DiscreteBayesianNetwork([("Health","COVID_Test")])
covid_bn.add_cpds(
    TabularCPD("Health", 2, [[0.7],[0.3]], state_names={"Health":["NotInfectious","Infectious"]}),
    TabularCPD("COVID_Test", 2, [[0.95,0.10],[0.05,0.90]],
        evidence=["Health"], evidence_card=[2],
        state_names={"COVID_Test":["Negative","Positive"],"Health":["NotInfectious","Infectious"]}),
)
assert covid_bn.check_model()
infer_ve = VariableElimination(covid_bn)
print("P(Health | Positive test):"); print(infer_ve.query(["Health"], evidence={"COVID_Test":"Positive"}))

# Full temporal inference on Corona data
P_I, beliefs = infer_infectious_probability(G, Y_obs, params, patient_zero=patient_zero, smooth=True)
print(f"\\nFull network P(I) shape: {P_I.shape}")

rows = []
for node, t in [(patient_zero,5),(patient_zero,15),(1,10),(5,20)]:
    if t >= Y_obs.shape[0]: continue
    p = query_node_infectious(G, Y_obs, params, node=node, time=t, patient_zero=patient_zero)
    rows.append({"node":node,"day":t,"P(infectious|Y)":round(p,4),
        "test":{OBS_POS:"+",OBS_NEG:"-",OBS_MISSING:"?"}[Y_obs[t,node]]})
display(pd.DataFrame(rows))"""))

M2.append(code("""fig, ax = plt.subplots(figsize=(11,5))
im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax.set_title("FIGURE M2-2 — P(Infectious | COVID Observations)")
fig.colorbar(im, ax=ax, label="P(I)")
for t in range(Y_obs.shape[0]):
    for i in range(Y_obs.shape[1]):
        if Y_obs[t,i]==OBS_POS: ax.plot(t,i,"wo",ms=4,mec="k",mew=0.5)
plt.tight_layout(); fig.savefig(OUTPUT_DIR/"fig2_heatmap_P_I.png", dpi=150); plt.show()

max_P = P_I.max(axis=0)
fig, ax = plt.subplots(figsize=(8,6))
nodes = nx.draw_networkx_nodes(G, pos, node_color=max_P, cmap=plt.cm.Reds, vmin=0, vmax=1, node_size=600, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
fig.colorbar(nodes, ax=ax, label="max P(I)")
ax.set_title("FIGURE M2-3 — Network by Posterior P(Infectious)")
plt.tight_layout(); fig.savefig(OUTPUT_DIR/"fig3_network_posterior.png", dpi=150); plt.show()"""))

M2.append(md("## MODULE 2 — PART 6 – EVALUATION & DASHBOARD"))

M2.append(code("""fig = plt.figure(figsize=(14,10))
gs = gridspec.GridSpec(2,2, figure=fig, hspace=0.35, wspace=0.3)
ax0 = fig.add_subplot(gs[0,0])
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=350, font_size=7, ax=ax0)
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#e74c3c", node_size=400, ax=ax0)
ax0.set_title("A  Contact Network")
ax1 = fig.add_subplot(gs[0,1])
ax1.plot(counts["I"], color="#c0392b", lw=2, label="I"); ax1.plot(counts["E"], color="#e67e22", ls="--", label="E")
ax1.legend(fontsize=8); ax1.set_title("B  Epidemic Curve"); ax1.grid(alpha=0.3)
ax2 = fig.add_subplot(gs[1,0])
im = ax2.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax2.set_title("C  P(I) Heatmap"); fig.colorbar(im, ax=ax2, fraction=0.046)
ax3 = fig.add_subplot(gs[1,1])
for idx,(lab,col) in enumerate(zip(["beta","sigma","gamma"],["#2980b9","#8e44ad","#16a085"])):
    ax3.plot(history[:,idx], label=lab, color=col, lw=2)
ax3.set_title("D  EM Convergence"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
fig.suptitle("MODULE 2 — COVID-19 DBN Dashboard", fontsize=14, y=1.01)
plt.tight_layout(); fig.savefig(OUTPUT_DIR/"fig_dashboard.png", dpi=150, bbox_inches="tight"); plt.show()"""))

M2.append(md("""---

# FINAL SUMMARY — Complete Pipeline

| Module | PGM pillars | Method |
|--------|-------------|--------|
| **1 — Static BN** | Representation, Structure Learning, MLE, VE Inference | Breast Cancer dataset |
| **2 — Dynamic BN** | Representation, EM Learning, Belief Propagation | COVID-19 Corona dataset |

### Module 2 project query answered

> *Given COVID test observations across the contact network, what is P(node i is infectious)?*

See Module 2 Part 5 query table and Figures M2-2, M2-3.

**End of complete pipeline.** All figures saved to `outputs/notebook/`."""))

for c in M2:
    CELLS.append(c)


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": CELLS,
    }
    OUT_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Merged notebook: {len(CELLS)} cells -> {OUT_PATH}")


if __name__ == "__main__":
    main()
