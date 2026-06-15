"""Build PGM epidemic notebook matching END_TO_END_PGM_PIPELINE.ipynb style."""
import json
import shutil
from pathlib import Path

NOTEBOOKS = Path(__file__).parent
REF = Path(r"c:\Users\davek\Downloads\Telegram Desktop\END_TO_END_PGM_PIPELINE.ipynb")


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


C = []

# ── Cell 0: install (same as reference) ──────────────────────────────────────
C.append(code("pip install pgmpy networkx matplotlib numpy pandas scipy"))

# ── Cell 1: dataset header (reference style) ───────────────────────────────
C.append(md("""  Dataset  : COVID-19 (Corona) — Geneva contact tracing + OWID Switzerland context

  Model    : Dynamic Bayesian Network (SEIR) on a contact network

  Library  : pgmpy (latest), networkx, numpy, pandas, matplotlib"""))

# ── Cell 2: lecture outline (reference style) ──────────────────────────────
C.append(md("""LECTURE OUTLINE

* PART 1  DATA & PREPROCESSING   – load COVID-19 (Corona) dataset, build network & observation matrix

* PART 2  REPRESENTATION          – understand the 2-time-slice DBN and SEIR CPTs (pgmpy)

* PART 3  MODEL STRUCTURE         – define DBN graph from Corona contact network

* PART 4  PARAMETER LEARNING      – estimate beta, sigma, gamma via EM algorithm

* PART 5  INFERENCE               – Variable Elimination (pgmpy) + forward-backward on Corona data

* PART 6  EVALUATION & FIGURES    – epidemic curves, posterior heatmaps, EM convergence"""))

# ── Cell 3: imports (reference style) ─────────────────────────────────────
C.append(code("""import warnings
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
import networkx as nx

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

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

# pgmpy (same as reference notebook)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DynamicBayesianNetwork as DBN

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Project root:", PROJECT_ROOT)"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 1
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("## PART 1 – DATA & PREPROCESSING"))

C.append(md("""**1-A  Load the COVID-19 (Corona) dataset**

* Primary: Geneva contact-tracing CSVs (individual network + positive PCR tests, 2020)
* Secondary: OWID Switzerland national case curve for epidemiological context
* Saved under `data/corona/` — same integration pattern as the course reference notebook"""))

C.append(code("""print(textwrap.dedent(\"\"\"
COVID-19 (CORONA) DATASET — integrated for this PGM project
  Primary  : Geneva contact tracing (GEgraph) — individuals, contacts, test dates
  Context  : Our World in Data — Switzerland daily new cases

Maps to the DBN:
  Latent X_i^t  = SEIR infection state (hidden)
  Observed Y_i^t = COVID test result (positive / negative / missing)
  Network edges  = documented close contacts during tracing
\"\"\"))

download_corona_dataset()
suivi, entourage = load_corona_contact_tables()

print(f"COVID tracing — suivi rows     : {len(suivi):,}")
print(f"COVID tracing — entourage rows : {len(entourage):,}")
print(f"Files location                 : {CORONA_DIR}")
print(f"\\nFirst 3 positive-test records:")
print(suivi[["record_id_pos", "date_res", "contact_record_id"]].dropna(subset=["date_res"]).head(3).to_string())

# National COVID-19 context (Switzerland)
owid = load_corona_owid_context("Switzerland")
owid_sub = owid[(owid["date"] >= "2020-02-01") & (owid["date"] <= "2020-06-30")]
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(owid_sub["date"], owid_sub["new_cases"], color="#c0392b", lw=1.5)
ax.set_title("COVID-19 (Corona) — Switzerland daily new cases (OWID context)")
ax.set_xlabel("Date"); ax.set_ylabel("New cases"); ax.grid(alpha=0.3)
plt.xticks(rotation=30); plt.tight_layout(); plt.show()"""))

C.append(md("""**1-B  Build epidemic subgraph**

* Pick an early infected person (March 2020) with several contacts as **patient zero**
* Expand by breadth-first search to ~30 connected individuals
* This gives one local outbreak cluster — the **population** for our DBN"""))

C.append(code("""bundle = load_corona_dataset(max_nodes=30)

G            = bundle.graph
Y_obs        = bundle.Y_obs
X_true       = bundle.X_true
patient_zero = bundle.patient_zero
meta         = bundle.metadata

print(f"Dataset name        : {bundle.dataset_name}")
print(f"Subgraph nodes      : {meta['n_nodes']}")
print(f"Time steps (days)   : {meta['n_timesteps']}")
print(f"Date range          : {meta['date_start']} -> {meta['date_end']}")
print(f"Patient zero        : node {patient_zero} (person ID {meta['seed_person_id']})")
print(f"Positive tests      : {meta['n_positive_tests']}")
print(f"Observation matrix  : {Y_obs.shape}  (time x people)")
print(f"Non-missing entries : {(Y_obs >= 0).sum()} / {Y_obs.size}  ({100*(Y_obs>=0).mean():.1f}%)")"""))

C.append(md("""**1-C  Observation matrix Y**

* Rows = time (days), columns = individuals
* Each cell is a test/symptom observation or missing
* This is the **evidence** fed into inference and learning"""))

C.append(code("""obs_rows = []
for t in range(Y_obs.shape[0]):
    for i in range(Y_obs.shape[1]):
        if Y_obs[t, i] == OBS_POS:
            obs_rows.append({
                "node": i, "person_id": bundle.node_ids[i],
                "day_t": t, "date": str(bundle.dates[t].date()), "Y": "POSITIVE",
            })
obs_df = pd.DataFrame(obs_rows)
print("Positive test observations:")
display(obs_df)

fig, ax = plt.subplots(figsize=(10, 3))
ax.imshow((Y_obs >= 0).T, aspect="auto", cmap="Greys", origin="lower")
ax.set_xlabel("Day index"); ax.set_ylabel("Person (node)")
ax.set_title("Observation sparsity: white = any test, black = missing")
plt.tight_layout(); plt.show()"""))

C.append(md("""**1-D  SEIR state summary (approximate, for evaluation only)**

* True SEIR states are **latent** — not given to the model
* We approximate states from test dates only to plot Figure 1
* Inference never uses X_true"""))

C.append(code("""counts = epidemic_counts(X_true)
print(f"Peak infectious (approx): {max(counts['I'])} at day t={counts['I'].index(max(counts['I']))}")
for s in STATES:
    print(f"  Total person-days in {s}: {sum(counts[s])}")"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 2
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("## PART 2 – REPRESENTATION  (Dynamic Bayesian Network)"))

C.append(code("""print(textwrap.dedent(\"\"\"
A Dynamic Bayesian Network (DBN) extends a Bayesian Network over TIME.

  - Nodes  = random variables at each time step
             X_i^t : latent SEIR state of person i at day t
             Y_i^t : observed test/symptom result
  - Edges  = temporal + contact dependencies
             X_i^{t-1} -> X_i^t     (within-person SEIR dynamics)
             X_j^{t-1} -> X_i^t     (transmission along contact)
             X_i^t     -> Y_i^t     (test emission)
  - CPTs   = Conditional Probability Tables P(child | parents)

The 2-time-slice template unrolls over T days to form the full model.

SEIR latent states per person:
  S = Susceptible   E = Exposed   I = Infectious   R = Recovered

Key parameters:
  beta  = transmission rate per infectious contact
  sigma = E -> I progression rate
  gamma = I -> R recovery rate

This is Pillar 1 of PGMs: REPRESENTATION.
\"\"\"))"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 3
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("## PART 3 – MODEL STRUCTURE  (DBN graph + CPTs)"))

C.append(code("""print(textwrap.dedent(\"\"\"
Unlike the static BN in the breast-cancer notebook (where structure is
LEARNED via Hill-Climbing), our epidemic DBN structure is SPECIFIED by
epidemiology and the contact network:

  - Network edges come from real contact-tracing data (Part 1)
  - SEIR transition rules define within-person edges
  - Test emission model defines X -> Y edges

Noisy-OR transmission for a susceptible person i:
  P(E | S, neighbors) = 1 - prod_j (1 - beta * P(X_j = I))

Emission model (imperfect tests):
  sensitivity = P(positive | infectious)
  specificity = P(negative | not infectious)
\"\"\"))

structure = build_dbn_structure(G)
summary   = network_summary(G)
params    = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)

print("DBN description:")
print(" ", structure["description"])
print(f"\\nNetwork: {summary['n_nodes']} nodes, {summary['n_edges']} edges, "
      f"avg degree {summary['avg_degree']:.2f}")

print("\\nExample CPT — susceptible person, one infectious neighbor:")
p = transition_distribution("S", [1.0], params)
for s, prob in zip(STATES, p):
    print(f"  P(next={s}) = {prob:.3f}")

# pgmpy DBN skeleton (like reference uses pgmpy.models)
dbn = export_pgmpy_dbn(G, params)
print(f"\\npgmpy DynamicBayesianNetwork nodes (sample): {list(dbn.nodes())[:8]} ...")
print(f"pgmpy DBN edges (sample): {list(dbn.edges())[:6]} ...")"""))

C.append(md("""**3-A  Visualise contact network (Figure 0)**

* Each node = individual; each edge = documented close contact
* Red node = patient zero (outbreak seed)"""))

C.append(code("""pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=550,
        font_size=9, edge_color="#666", ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#e74c3c",
                       node_size=650, ax=ax)
ax.set_title("FIGURE 0 — COVID-19 Contact Network (Corona dataset)\\nred = patient zero")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig0_network.png", dpi=150)
plt.show()

print(textwrap.dedent(\"\"\"
FIGURE 0 — HOW TO READ:
  - Nodes are individuals in the outbreak cluster.
  - Edges are real traced contacts; disease may spread along them.
  - The DBN couples X_j^{t-1} -> X_i^t for each edge (i,j).
  - Patient zero (red) is the epidemiological seed of this subgraph.
\"\"\"))"""))

C.append(md("""**3-B  Epidemic curve (Figure 1)**

* Shows approximate SEIR counts over time
* Peak of infectious (red) = height of the outbreak wave"""))

C.append(code("""fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(counts["I"], label="Infectious (I)", lw=2.5, color="#c0392b")
ax.plot(counts["E"], label="Exposed (E)", ls="--", color="#e67e22")
ax.plot(counts["R"], label="Recovered (R)", ls=":", color="#27ae60")
ax.set_xlabel("Day index"); ax.set_ylabel("Count")
ax.set_title("FIGURE 1 — Epidemic Curve (approximate SEIR from real tests)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_epidemic_curve.png", dpi=150)
plt.show()

print(textwrap.dedent(\"\"\"
FIGURE 1 — HOW TO READ:
  - Red line (I): how many people are infectious each day — the epidemic peak.
  - Orange (E): incubating; not yet spreading but infected.
  - Green (R): recovered; removed from transmission.
  - For real data these are APPROXIMATE (derived from test dates for plotting).
\"\"\"))"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 4
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("## PART 4 – PARAMETER LEARNING  (EM Algorithm)"))

C.append(code("""print(textwrap.dedent(\"\"\"
PARAMETER LEARNING estimates beta, sigma, gamma when they are unknown.

In the breast-cancer notebook, parameters were learned by MLE counting
frequencies in a static BN.  Here we use the EM algorithm because
SEIR states are LATENT (hidden):

  E-step : run forward-backward inference
           compute soft state assignments gamma_t(i,s) = P(X_i^t=s | Y, theta)

  M-step : re-estimate beta, sigma, gamma from expected transition counts
           sigma = expected(E->I) / expected(time in E)
           gamma = expected(I->R) / expected(time in I)
           beta  = expected(S->E | exposed) / expected(S | exposed)

Repeat until convergence.  This is Pillar 3: LEARNING.
\"\"\"))

init_params = ModelParams(beta=0.10, sigma=0.10, gamma=0.10)
print(f"Initial guess: beta={init_params.beta}, sigma={init_params.sigma}, gamma={init_params.gamma}")
print("\\nRunning EM (25 iterations)...\\n")
learned, history = em_learn(G, Y_obs, init_params, n_iter=25,
                            patient_zero=patient_zero, verbose=True)
print(f"\\nLearned: beta={learned.beta:.3f}, sigma={learned.sigma:.3f}, gamma={learned.gamma:.3f}")"""))

C.append(md("""**4-A  EM convergence (Figure 4)**"""))

C.append(code("""labels = ["beta (transmission)", "sigma (E->I)", "gamma (I->R)"]
colors = ["#2980b9", "#8e44ad", "#16a085"]
fig, ax = plt.subplots(figsize=(9, 4.5))
for idx, (lab, col) in enumerate(zip(labels, colors)):
    ax.plot(history[:, idx], label=f"learned {lab}", color=col, lw=2.5)
ax.set_xlabel("EM iteration"); ax.set_ylabel("Parameter value")
ax.set_title("FIGURE 4 — EM Parameter Convergence (real data)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_em_convergence.png", dpi=150)
plt.show()

print(textwrap.dedent(\"\"\"
FIGURE 4 — HOW TO READ:
  - Each line tracks one learned parameter across EM iterations.
  - Early iterations: large changes as EM finds better explanations.
  - Flat tail: convergence — parameters stabilised.
  - On real data there is no ground-truth dashed line; discuss plausibility.
\"\"\"))"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 5
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("## PART 5 – INFERENCE  (Variable Elimination + Forward–Backward)"))

C.append(code("""print(textwrap.dedent(\"\"\"
INFERENCE = answering probabilistic queries using the fitted model.

Like the reference notebook (breast-cancer BN), we use pgmpy Variable Elimination
for a small COVID test model.  For the full temporal network we use
forward-backward belief propagation on the Corona dataset.

CENTRAL PROJECT QUERY:
  P(X_i^t = Infectious | all COVID test observations)
\"\"\"))

# --- 5-0  pgmpy Variable Elimination demo (same library as reference notebook) ---
# Simplified single-person chain: Health -> COVID_Test
covid_bn = DiscreteBayesianNetwork([("Health", "COVID_Test")])
cpd_health = TabularCPD("Health", 2, [[0.7], [0.3]], state_names={"Health": ["NotInfectious", "Infectious"]})
cpd_test = TabularCPD(
    "COVID_Test", 2,
    [[0.95, 0.10],   # P(neg | NotInf), P(neg | Inf)
     [0.05, 0.90]],   # P(pos | NotInf), P(pos | Inf)
    evidence=["Health"], evidence_card=[2],
    state_names={"COVID_Test": ["Negative", "Positive"], "Health": ["NotInfectious", "Infectious"]},
)
covid_bn.add_cpds(cpd_health, cpd_test)
assert covid_bn.check_model()
infer_ve = VariableElimination(covid_bn)

q_prior = infer_ve.query(["Health"])
print("5-0a  Prior P(Health) [no evidence]:")
print(q_prior, "\\n")

q_pos = infer_ve.query(["Health"], evidence={"COVID_Test": "Positive"})
print("5-0b  Posterior P(Health | Positive COVID test):")
print(q_pos)
map_h = infer_ve.map_query(["Health"], evidence={"COVID_Test": "Positive"})
print(f"5-0c  MAP: most likely health state given positive test = {map_h['Health']}\\n")

# --- Full network inference on integrated Corona dataset ---
print("Running forward-backward on full COVID-19 contact network...")
P_I, beliefs = infer_infectious_probability(
    G, Y_obs, params, patient_zero=patient_zero, smooth=True)
print(f"Posterior P(I) shape: {P_I.shape},  range: [{P_I.min():.3f}, {P_I.max():.3f}]")"""))

C.append(md("""**5-A  Marginal query — prior infectiousness (no test evidence on that day)**

* Query P(X_i^t = I) before seeing tests at time t
* Compare with posterior after adding evidence in 5-B"""))

C.append(code("""t_example = min(10, Y_obs.shape[0] - 1)
node_example = patient_zero
p_prior_approx = beliefs[t_example, node_example, STATE_IDX["I"]]
print(f"Posterior P(node {node_example} infectious at t={t_example}) = {p_prior_approx:.4f}")
print(f"Observation that day: ", end="")
y = Y_obs[t_example, node_example]
print({OBS_POS:"POSITIVE", OBS_NEG:"NEGATIVE", OBS_MISSING:"missing"}[y])
print(f"Approx true SEIR state: {STATES[X_true[t_example, node_example]]}")"""))

C.append(md("""**5-B  Posterior given test observations across the network**

* The model uses ALL tests from ALL individuals at ALL times
* Positive tests increase P(I) for that person and often for neighbors too"""))

C.append(code("""query_rows = []
for node, t, note in [
    (patient_zero, 5,  "seed, early"),
    (patient_zero, 15, "seed, mid"),
    (1, 10, "neighbor"),
    (5, 20, "other"),
]:
    if t >= Y_obs.shape[0]: continue
    p = query_node_infectious(G, Y_obs, params, node=node, time=t, patient_zero=patient_zero)
    y = Y_obs[t, node]
    query_rows.append({
        "node": node, "day_t": t,
        "date": str(bundle.dates[t].date()),
        "P(infectious|Y)": round(p, 4),
        "observation": {OBS_POS:"+", OBS_NEG:"-", OBS_MISSING:"?"}[y],
        "approx_SEIR": STATES[X_true[t, node]],
        "note": note,
    })
print("Posterior queries P(X_i^t = I | all observations):")
display(pd.DataFrame(query_rows))"""))

C.append(md("""**5-C  MAP-style query — is this person most likely infectious?**

* MAP = state with highest posterior probability
* Here: infectious if P(I) > 0.5"""))

C.append(code("""for node, t in [(patient_zero, 10), (1, 10), (5, 20)]:
    if t >= Y_obs.shape[0]: continue
    p = float(P_I[t, node])
    map_state = "INFECTIOUS" if p > 0.5 else "NOT INFECTIOUS"
    print(f"Node {node}, day {t}: P(I)={p:.3f}  ->  MAP: {map_state}")"""))

C.append(md("""**5-D  Posterior heatmap (Figure 2)**

* Full posterior P(I) for every person and every day
* White dots = observed positive tests"""))

C.append(code("""fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax.set_xlabel("Day index"); ax.set_ylabel("Person (node)")
ax.set_title("FIGURE 2 — P(Infectious | All Observations)")
fig.colorbar(im, ax=ax, label="P(I)")
for t in range(Y_obs.shape[0]):
    for i in range(Y_obs.shape[1]):
        if Y_obs[t, i] == OBS_POS:
            ax.plot(t, i, "wo", ms=4, mec="k", mew=0.5)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_heatmap_P_I.png", dpi=150)
plt.show()

print(textwrap.dedent(\"\"\"
FIGURE 2 — HOW TO READ:
  - Dark red = high P(infectious); white = low.
  - X-axis: time; Y-axis: individuals.
  - White dots: days with a POSITIVE test (hard evidence).
  - Bright bands BEFORE a dot: smoothing infers infection before diagnosis.
  - Dark rows: person likely never infectious in this window.
\"\"\"))"""))

C.append(md("""**5-E  Network posterior (Figure 3)**

* Each node coloured by max_t P(X_i^t = I | Y)"""))

C.append(code("""max_P = P_I.max(axis=0)
fig, ax = plt.subplots(figsize=(8, 6))
nodes = nx.draw_networkx_nodes(G, pos, node_color=max_P, cmap=plt.cm.Reds,
                               vmin=0, vmax=1, node_size=600, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#8e44ad", node_size=700, ax=ax)
fig.colorbar(nodes, ax=ax, label="max_t P(I)")
ax.set_title("FIGURE 3 — Network by max posterior P(Infectious)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_network_posterior.png", dpi=150)
plt.show()

print(textwrap.dedent(\"\"\"
FIGURE 3 — HOW TO READ:
  - Colour = highest infectious probability that person reached during the outbreak.
  - Dark red nodes: most likely infected at some point.
  - Purple node: patient zero (seed).
  - Use this figure to identify who was most affected in the cluster.
\"\"\"))"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 6
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("## PART 6 – EVALUATION & SUMMARY FIGURES"))

C.append(code("""print(textwrap.dedent(\"\"\"
EVALUATION for this epidemic DBN project:

  1. Does the model produce sensible posteriors aligned with positive tests?
  2. Does EM converge on real partial observations?
  3. Can we answer: P(node i infectious | observations)?

Below: a 2x2 dashboard of all key figures for your report.
\"\"\"))"""))

C.append(code("""fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: network
ax0 = fig.add_subplot(gs[0, 0])
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=350,
        font_size=7, ax=ax0)
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#e74c3c", node_size=400, ax=ax0)
ax0.set_title("A  Contact Network (Fig 0)")

# Panel B: epidemic curve
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(counts["I"], color="#c0392b", lw=2, label="I")
ax1.plot(counts["E"], color="#e67e22", ls="--", label="E")
ax1.set_xlabel("Day"); ax1.set_ylabel("Count")
ax1.set_title("B  Epidemic Curve (Fig 1)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# Panel C: heatmap
ax2 = fig.add_subplot(gs[1, 0])
im = ax2.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax2.set_xlabel("Day"); ax2.set_ylabel("Person")
ax2.set_title("C  P(I) Heatmap (Fig 2)")
fig.colorbar(im, ax=ax2, fraction=0.046)

# Panel D: EM convergence
ax3 = fig.add_subplot(gs[1, 1])
for idx, (lab, col) in enumerate(zip(["beta","sigma","gamma"], ["#2980b9","#8e44ad","#16a085"])):
    ax3.plot(history[:, idx], label=lab, color=col, lw=2)
ax3.set_xlabel("EM iter"); ax3.set_title("D  EM Convergence (Fig 4)")
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

fig.suptitle("COVID-19 (Corona) DBN — End-to-End Dashboard", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard saved to", OUTPUT_DIR / "fig_dashboard.png")"""))

C.append(md("""### Final summary — three PGM pillars

| Pillar | What we did in this notebook |
|--------|------------------------------|
| **Representation** | 2-time-slice SEIR DBN on COVID-19 (Corona) contact network; pgmpy DBN export |
| **Inference** | pgmpy Variable Elimination + forward-backward on Corona test observations |
| **Learning** | EM algorithm on integrated COVID-19 dataset |

### Project query answered

> *Given symptom/test observations across a network, what is P(node i is currently infectious)?*

See **Part 5-B** query table and **Figures 2–3**.

### Report checklist

- [ ] Figure 0: contact network
- [ ] Figure 1: epidemic curve
- [ ] Figure 2: P(I) heatmap
- [ ] Figure 3: network posterior
- [ ] Figure 4: EM convergence
- [ ] Dashboard: `fig_dashboard.png`

**Reference notebook style:** `notebooks/REFERENCE_END_TO_END_PGM_PIPELINE.ipynb` (breast-cancer static BN example from course)."""))

# ── optional synthetic validation ──────────────────────────────────────────
C.append(md("## APPENDIX – Synthetic data validation (optional)"))

C.append(code("""from src.network import make_contact_network
from src.simulation import simulate_epidemic

print(textwrap.dedent(\"\"\"
OPTIONAL: run synthetic data to validate EM when true parameters are known.
Same pipeline as real data; compare learned vs true beta, sigma, gamma.
\"\"\"))

syn_cfg = SimConfig(n_nodes=20, n_timesteps=40, test_probability=0.7, seed=42)
true_p  = ModelParams(0.30, 0.20, 0.10)
G_syn   = make_contact_network(syn_cfg.n_nodes, syn_cfg.network_kind, syn_cfg.seed)
_, Y_syn = simulate_epidemic(G_syn, true_p, syn_cfg)
learned_syn, hist_syn = em_learn(G_syn, Y_syn, ModelParams(0.1,0.1,0.1), n_iter=25, verbose=False)

fig, ax = plt.subplots(figsize=(8, 4))
for idx, name in enumerate(["beta","sigma","gamma"]):
    ax.plot(hist_syn[:, idx], lw=2, label=f"learned {name}")
    ax.axhline(true_p.as_array()[idx], ls=":", label=f"true {name}")
ax.set_xlabel("EM iter"); ax.legend(fontsize=8); ax.set_title("Synthetic: EM vs true parameters")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print("True:", true_p, " Learned:", learned_syn)"""))


def main():
    out = NOTEBOOKS / "PGM_Epidemic_DBN_EndToEnd.ipynb"
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": C,
    }
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(C)} cells -> {out}")

    if REF.exists():
        ref_dest = NOTEBOOKS / "REFERENCE_END_TO_END_PGM_PIPELINE.ipynb"
        shutil.copy2(REF, ref_dest)
        print(f"Copied reference -> {ref_dest}")


if __name__ == "__main__":
    main()
