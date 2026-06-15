"""Build PGM_EndToEnd_Pipeline.ipynb — Disease Spread DBN end-to-end pipeline."""
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parent
OUT = NOTEBOOKS / "PGM_EndToEnd_Pipeline.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in t.split("\n")]}


def code(t):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in t.split("\n")],
    }


C = []

C.append(code("pip install pgmpy networkx matplotlib numpy pandas scipy"))

C.append(md("""# Disease Spread Modeling Using Dynamic Bayesian Networks

## Project description

This project models the **temporal spread of infectious diseases** through a population using **Dynamic Bayesian Networks (DBNs)**.

Individual infection states — **Susceptible, Exposed, Infected, and Recovered** — are **latent variables**. **Observations** correspond to reported symptoms or test results (positive, negative, or missing).

The project covers the **three core pillars of PGMs**:

| Pillar | What we implement |
|--------|-------------------|
| **Representation** | Graph structure (contact network + 2-time-slice DBN) and conditional probability tables (SEIR + test model) |
| **Inference** | Belief propagation (forward-backward) and variable elimination for probabilistic queries |
| **Learning** | EM algorithm to estimate epidemic parameters from real-world contact-tracing data |

### Expected outcome

A working DBN that answers:

> *Given a set of symptom/test observations across a network of individuals, what is the probability that a given node is currently infected?*

### Data & tools

  Dataset  : COVID-19 Geneva contact tracing (real epidemic data, 2020)

  Model    : SEIR Dynamic Bayesian Network on a contact network

  Library  : pgmpy, networkx, numpy, pandas, matplotlib"""))

C.append(md("""## Pipeline outline

* **PART 1**  DATA & PREPROCESSING — load epidemic data, build contact network and observation matrix

* **PART 2**  REPRESENTATION — DBN structure, latent SEIR states, observation model

* **PART 3**  GRAPH STRUCTURE — contact network as spatial DBN template

* **PART 4**  PARAMETER LEARNING — EM algorithm (β, σ, γ)

* **PART 5**  INFERENCE — belief propagation & variable elimination queries

* **PART 6**  EVALUATION — epidemic visualisation and posterior analysis"""))

C.append(code("""import warnings
warnings.filterwarnings("ignore")
import textwrap
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").is_dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from src.config import ModelParams, STATES, STATE_IDX, OBS_POS, OBS_NEG, OBS_MISSING
from src.corona_data import (
    download_corona_dataset,
    load_corona_dataset,
    load_corona_contact_tables,
    load_corona_owid_context,
    CORONA_DIR,
)
from src.model import (
    build_dbn_structure, export_pgmpy_dbn, transition_matrix, emission_likelihood,
)
from src.network import network_summary
from src.inference import infer_infectious_probability, query_node_infectious
from src.learning import em_learn
from src.simulation import epidemic_counts

OUT = ROOT / "outputs" / "notebook"
OUT.mkdir(parents=True, exist_ok=True)
print("Project root:", ROOT)
print("Output dir  :", OUT)"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 1
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("""## PART 1 – DATA & PREPROCESSING

Real-world epidemic data comes from **COVID-19 contact tracing** in Geneva: who tested positive, when, and who had close contact with whom.

We extract a **connected outbreak subgraph** and build the **observation matrix** `Y[t, i]` = test/symptom report for person `i` on day `t`."""))

C.append(md("""**1-A  Load contact-tracing records**

* `redcap_suivi.csv` — PCR test dates and results
* `redcap_entourage.csv` — documented close contacts between individuals"""))

C.append(code("""download_corona_dataset()
suivi, entourage = load_corona_contact_tables()

print(f"Tracing records : {len(suivi):,}")
print(f"Contact rows    : {len(entourage):,}")
print(f"Data folder     : {CORONA_DIR}")
display(suivi[["record_id_pos", "date_res", "contact_record_id"]].dropna(subset=["date_res"]).head(5))

# FIGURE 1-A — national epidemic context
owid = load_corona_owid_context("Switzerland")
owid_sub = owid[(owid["date"] >= "2020-02-01") & (owid["date"] <= "2020-06-30")]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(owid_sub["date"], owid_sub["new_cases"], alpha=0.3, color="#c0392b")
ax.plot(owid_sub["date"], owid_sub["new_cases"], color="#c0392b", lw=1.8)
ax.set_title("FIGURE 1-A — Daily new COVID-19 cases in Switzerland (epidemic context)")
ax.set_ylabel("new cases"); ax.grid(alpha=0.3)
plt.xticks(rotation=30); plt.tight_layout()
fig.savefig(OUT / "fig1a_epidemic_context.png", dpi=150); plt.show()"""))

C.append(md("""**1-B  Build contact network and observation matrix**

Each **node** is a person; each **edge** is a documented close contact. The matrix `Y` encodes test/symptom observations over time."""))

C.append(code("""bundle = load_corona_dataset(max_nodes=30)
G, Y, X = bundle.graph, bundle.Y_obs, bundle.X_true
pz, meta = bundle.patient_zero, bundle.metadata
params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)
pos = nx.spring_layout(G, seed=42)
counts = epidemic_counts(X)

print(f"People / days   : {meta['n_nodes']} people, {meta['n_timesteps']} days")
print(f"Period          : {meta['date_start']} -> {meta['date_end']}")
print(f"Positive tests  : {meta['n_positive_tests']}")
print(f"Y shape         : {Y.shape}  (time x people)")
print(f"Missing rate    : {(Y == OBS_MISSING).mean()*100:.0f}%")
print(f"Index case      : node {pz}")

# FIGURE 1-B — contact network
fig, ax = plt.subplots(figsize=(8, 6))
deg = dict(G.degree())
node_sizes = [300 + 80 * deg[n] for n in G.nodes()]
nc = [deg[n] for n in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=nc, cmap=plt.cm.Blues, node_size=node_sizes, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[pz], node_color="#e74c3c", node_size=700, ax=ax)
fig.colorbar(nodes, ax=ax, label="degree")
ax.set_title("FIGURE 1-B — Population contact network (red = index case)")
plt.tight_layout(); fig.savefig(OUT / "fig1b_network.png", dpi=150); plt.show()

# FIGURE 1-C — degree distribution
degrees = [d for _, d in G.degree()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(degrees, bins=range(min(degrees), max(degrees) + 2), color="#3498db", edgecolor="white")
axes[0].set_xlabel("degree"); axes[0].set_ylabel("count")
axes[0].set_title("Degree histogram"); axes[0].grid(alpha=0.3, axis="y")
adj = nx.to_numpy_array(G)
im = axes[1].imshow(adj, cmap="Greys")
axes[1].set_title("Adjacency matrix"); axes[1].set_xlabel("person"); axes[1].set_ylabel("person")
fig.colorbar(im, ax=axes[1], fraction=0.046)
plt.suptitle("FIGURE 1-C — Network statistics", y=1.02)
plt.tight_layout(); fig.savefig(OUT / "fig1c_network_stats.png", dpi=150); plt.show()"""))

C.append(md("""**1-C  Visualise sparse test/symptom observations**"""))

C.append(code("""# FIGURE 1-D — observation heatmap
cmap_data = np.full(Y.shape, 0.5)
cmap_data[Y == OBS_NEG] = 0.0
cmap_data[Y == OBS_POS] = 1.0
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(cmap_data.T, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_xlabel("day t"); ax.set_ylabel("person i")
ax.set_title("FIGURE 1-D — Observation matrix Y (red=positive, blue=negative, grey=missing)")
cb = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1])
cb.ax.set_yticklabels(["negative", "missing", "positive"])
plt.tight_layout(); fig.savefig(OUT / "fig1d_observations.png", dpi=150); plt.show()

# FIGURE 1-E — tests per day + observation type pie
tests_per_day = [(Y[t] == OBS_POS).sum() for t in range(Y.shape[0])]
n_pos = (Y == OBS_POS).sum(); n_neg = (Y == OBS_NEG).sum(); n_miss = (Y == OBS_MISSING).sum()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(range(len(tests_per_day)), tests_per_day, color="#e74c3c", alpha=0.85)
axes[0].set_xlabel("day"); axes[0].set_ylabel("# positive tests")
axes[0].set_title("Positive tests per day"); axes[0].grid(alpha=0.3, axis="y")
axes[1].pie([n_pos, n_neg, n_miss], labels=["positive", "negative", "missing"],
            colors=["#e74c3c", "#3498db", "#bdc3c7"], autopct="%1.1f%%", startangle=90)
axes[1].set_title("Observation breakdown")
plt.suptitle("FIGURE 1-E — Test observation summary", y=1.02)
plt.tight_layout(); fig.savefig(OUT / "fig1e_test_summary.png", dpi=150); plt.show()

rows = [{"node": i, "day": t, "date": str(bundle.dates[t].date())}
        for t in range(Y.shape[0]) for i in range(Y.shape[1]) if Y[t,i] == OBS_POS]
display(pd.DataFrame(rows))
print(f"Peak infected count (labels): {max(counts['I'])}")"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("""## PART 2 – REPRESENTATION  (Dynamic Bayesian Network)

**Pillar 1 — Representation:** we define the DBN graph and conditional probability tables.

* **Latent variables** `X_i^t` ∈ {S, E, I, R} — infection state of person `i` at time `t`
* **Observed variables** `Y_i^t` — test/symptom report (positive / negative / missing)
* **Parameters** β (transmission), σ (E→I), γ (I→R), plus test sensitivity & specificity"""))

C.append(code("""print(textwrap.dedent(\"\"\"
Dynamic Bayesian Network (DBN):

  Unrolls a 2-time-slice template across days t = 0, 1, 2, ...

  Within person:   X_i^{t-1} -> X_i^t     (SEIR dynamics)
  Between people:  X_j^{t-1} -> X_i^t     (disease transmission along contacts)
  Observation:     X_i^t     -> Y_i^t     (noisy test / symptom report)

  Joint distribution factorises over slices using Markov property.
\"\"\"))

# FIGURE 2-A — SEIR compartment diagram
fig, ax = plt.subplots(figsize=(10, 3.2))
states_pos = {"S": 0, "E": 1, "I": 2, "R": 3}
colors = {"S": "#3498db", "E": "#f39c12", "I": "#e74c3c", "R": "#27ae60"}
labels = {"S": "Susceptible", "E": "Exposed", "I": "Infected", "R": "Recovered"}
for s, x in states_pos.items():
    ax.add_patch(plt.Circle((x, 0.55), 0.32, color=colors[s], ec="k", lw=2))
    ax.text(x, 0.55, s, ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(x, 0.05, labels[s], ha="center", fontsize=9)
for (x0, x1, lbl) in [(0.32, 0.68, "contact (beta)"), (1.32, 1.68, "sigma"), (2.32, 2.68, "gamma")]:
    ax.annotate("", xy=(x1, 0.55), xytext=(x0, 0.55), arrowprops=dict(arrowstyle="->", lw=2.5))
    ax.text((x0+x1)/2, 0.88, lbl, ha="center", fontsize=9)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.2, 1.2); ax.axis("off")
ax.set_title("FIGURE 2-A — Latent SEIR states (per individual)")
plt.tight_layout(); fig.savefig(OUT / "fig2a_seir.png", dpi=150); plt.show()

# FIGURE 2-B — 2-time-slice DBN plate
fig, ax = plt.subplots(figsize=(11, 4.5))
def bbox(x, y, lab, fc):
    ax.add_patch(mpatches.FancyBboxPatch((x-0.4, y-0.22), 0.8, 0.44, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=1.5))
    ax.text(x, y, lab, ha="center", va="center", fontsize=9, fontweight="bold")
bbox(1, 1.5, "X_i^{t-1}", "#aed6f1"); bbox(3, 1.5, "X_i^{t}", "#aed6f1")
bbox(0, 1.5, "X_j^{t-1}", "#aed6f1"); bbox(3, 0.35, "Y_i^{t}", "#abebc6")
ax.annotate("", xy=(3, 0.62), xytext=(3, 1.22), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(3, 1.22), xytext=(1.4, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(1.4, 1.5), xytext=(0.4, 1.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#e74c3c"))
ax.text(2, 1.85, "transmission along contact", ha="center", fontsize=9, color="#e74c3c")
ax.text(3.55, 0.35, "test / symptom", fontsize=9)
ax.text(0.5, 2.1, "slice t-1", fontsize=12, fontweight="bold")
ax.text(2.8, 2.1, "slice t", fontsize=12, fontweight="bold")
ax.set_xlim(-0.8, 4.2); ax.set_ylim(0, 2.3); ax.axis("off")
ax.set_title("FIGURE 2-B — 2-time-slice DBN template (person i, neighbor j)")
plt.tight_layout(); fig.savefig(OUT / "fig2b_dbn_plate.png", dpi=150); plt.show()

# FIGURE 2-C — observation model P(Y|X)
fig, ax = plt.subplots(figsize=(7, 4))
states = ["S", "E", "I", "R"]
p_pos = [emission_likelihood(OBS_POS, params)[STATE_IDX[s]] for s in states]
p_neg = [emission_likelihood(OBS_NEG, params)[STATE_IDX[s]] for s in states]
x = np.arange(4); w = 0.35
ax.bar(x - w/2, p_pos, w, label="P(test + | state)", color="#e74c3c")
ax.bar(x + w/2, p_neg, w, label="P(test - | state)", color="#3498db")
ax.set_xticks(x); ax.set_xticklabels(states)
ax.set_ylim(0, 1.05); ax.set_ylabel("probability")
ax.set_title("FIGURE 2-C — Observation model P(Y | X)  (test sensitivity / specificity)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); fig.savefig(OUT / "fig2c_emission.png", dpi=150); plt.show()

# FIGURE 2-D — SEIR compartment sizes over time
fig, ax = plt.subplots(figsize=(9, 4.5))
stack = np.array([counts[s] for s in ["S", "E", "I", "R"]])
ax.stackplot(range(stack.shape[1]), stack, labels=["S","E","I","R"],
             colors=["#3498db","#f39c12","#e74c3c","#27ae60"], alpha=0.75)
ax.set_xlabel("day"); ax.set_ylabel("# individuals")
ax.set_title("FIGURE 2-D — Population compartments over time (stacked SEIR)")
ax.legend(loc="center right"); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUT / "fig2d_seir_stack.png", dpi=150); plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — GRAPH STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("""## PART 3 – GRAPH STRUCTURE  (Contact Network + DBN Template)

The **spatial structure** of the DBN is the **contact network** from tracing data. The **temporal structure** is the standard 2-time-slice SEIR template applied to every person and every contact edge."""))

C.append(code("""structure = build_dbn_structure(G)
summary = network_summary(G)
print(structure["description"])
print(summary)

dbn = export_pgmpy_dbn(G, params)
n_intra = len(dbn.get_intra_edges())
n_inter = len(list(dbn.get_inter_edges()))
print(f"pgmpy DBN: {len(list(dbn.nodes()))} variables, {len(list(dbn.edges()))} edges")
print(f"  intra-slice (X->Y): {n_intra}   inter-slice (transitions): {n_inter}")

# FIGURE 3-A — spatial structure
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=550, font_size=9,
        edge_color="#7f8c8d", width=1.5, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[pz], node_color="#e74c3c", node_size=700, ax=ax)
ax.set_title("FIGURE 3-A — DBN spatial structure = contact network")
plt.tight_layout(); fig.savefig(OUT / "fig3a_structure.png", dpi=150); plt.show()

# FIGURE 3-B — index case neighborhood
sub_nodes = {pz} | set(G.neighbors(pz))
sub = G.subgraph(sub_nodes).copy()
spos = nx.spring_layout(sub, seed=7)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nx.draw(sub, spos, with_labels=True, node_color="#a8d4f0", node_size=900,
        font_size=11, width=2.5, edge_color="#e74c3c", ax=axes[0])
nx.draw_networkx_nodes(sub, spos, nodelist=[pz], node_color="#e74c3c", node_size=1000, ax=axes[0])
axes[0].set_title("Index case + direct contacts")
ax2 = axes[1]; ax2.set_xlim(-0.5, 3.5); ax2.set_ylim(-0.3, 2.6); ax2.axis("off")
ax2.text(1.5, 2.35, "DBN temporal edges for one person", ha="center", fontweight="bold", fontsize=11)
for x, y, lab, fc in [(0.5,1.6,"X^{t-1}","#aed6f1"), (2.5,1.6,"X^{t}","#aed6f1"), (2.5,0.4,"Y^{t}","#abebc6")]:
    ax2.add_patch(mpatches.FancyBboxPatch((x-0.35,y-0.18),0.7,0.36,boxstyle="round",fc=fc,ec="k"))
    ax2.text(x,y,lab,ha="center",va="center",fontsize=10)
ax2.annotate("", xy=(2.5,1.35), xytext=(2.5,0.65), arrowprops=dict(arrowstyle="->",lw=2))
ax2.annotate("", xy=(2.15,1.6), xytext=(0.85,1.6), arrowprops=dict(arrowstyle="->",lw=2))
ax2.text(1.5, 1.95, "SEIR transition", ha="center", fontsize=9)
ax2.text(2.85, 0.4, "observed test", fontsize=9)
plt.suptitle("FIGURE 3-B — Local subgraph and temporal edges", y=1.02)
plt.tight_layout(); fig.savefig(OUT / "fig3b_local_dbn.png", dpi=150); plt.show()

# FIGURE 3-C — epidemic on this network
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(range(len(counts["I"])), counts["I"], alpha=0.25, color="#e74c3c")
ax.plot(counts["I"], color="#e74c3c", lw=2.5, label="Infected (I)")
ax.plot(counts["E"], color="#f39c12", ls="--", lw=2, label="Exposed (E)")
ax.plot(counts["S"], color="#3498db", ls=":", lw=1.5, label="Susceptible (S)")
ax.set_xlabel("day"); ax.set_ylabel("count"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("FIGURE 3-C — Epidemic dynamics on the contact subgraph")
plt.tight_layout(); fig.savefig(OUT / "fig3c_epidemic.png", dpi=150); plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — LEARNING
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("""## PART 4 – PARAMETER LEARNING  (EM Algorithm)

**Pillar 3 — Learning:** SEIR states are **latent**; only tests/symptoms are observed. The **EM algorithm** estimates β, σ, and γ:

* **E-step:** run inference to get soft estimates of latent states
* **M-step:** update transmission parameters from expected counts"""))

C.append(code("""init_p = ModelParams(0.10, 0.10, 0.10)
learned, history = em_learn(G, Y, init_p, n_iter=30, patient_zero=pz, verbose=True)
print(f"Learned: beta={learned.beta:.3f}  sigma={learned.sigma:.3f}  gamma={learned.gamma:.3f}")

# FIGURE 4-A — EM convergence
fig, ax = plt.subplots(figsize=(9, 4.5))
cols = {"beta": "#2980b9", "sigma": "#8e44ad", "gamma": "#16a085"}
for idx, lab in enumerate(["beta", "sigma", "gamma"]):
    ax.plot(history[:, idx], "o-", label=lab, color=cols[lab], lw=2, ms=4)
ax.set_xlabel("EM iteration"); ax.set_ylabel("parameter value")
ax.set_title("FIGURE 4-A — EM parameter convergence")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUT / "fig4a_em.png", dpi=150); plt.show()

# FIGURE 4-B — before / after bar chart
fig, ax = plt.subplots(figsize=(7, 4))
names = ["beta\\n(transmission)", "sigma\\n(E->I)", "gamma\\n(I->R)"]
x = np.arange(3); w = 0.35
ax.bar(x-w/2, [init_p.beta, init_p.sigma, init_p.gamma], w, label="initial", color="#bdc3c7")
ax.bar(x+w/2, [learned.beta, learned.sigma, learned.gamma], w, label="learned", color="#2980b9")
ax.set_xticks(x); ax.set_xticklabels(names); ax.legend()
ax.set_title("FIGURE 4-B — Initial vs learned epidemic parameters")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); fig.savefig(OUT / "fig4b_params.png", dpi=150); plt.show()

# FIGURE 4-C — transition matrix heatmap
T_s = transition_matrix([], params)
T_i = transition_matrix([0.8, 0.6], params)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, T, title in zip(axes, [T_s, T_i], ["No infectious neighbors", "Infectious neighbors present"]):
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(STATES); ax.set_yticklabels(STATES)
    ax.set_xlabel("next state"); ax.set_ylabel("prev state")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if T[i,j] > 0.5 else "black")
fig.colorbar(im, ax=axes.ravel().tolist()[-1], fraction=0.046)
plt.suptitle("FIGURE 4-C — CPT: P(X^t | X^{t-1}, neighbors)  [representation]", y=1.02)
plt.tight_layout(); fig.savefig(OUT / "fig4c_cpt.png", dpi=150); plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("""## PART 5 – INFERENCE  (Belief Propagation & Variable Elimination)

**Pillar 2 — Inference:** answer probabilistic queries about **latent infection states** given all test/symptom observations.

**Main query:**

> P(node i is **infected** at time t | all observations across the network)"""))

C.append(code("""# Small BN: variable elimination (single person, single test)
test_bn = DiscreteBayesianNetwork([("InfectionState", "TestResult")])
test_bn.add_cpds(
    TabularCPD("InfectionState", 2, [[0.7], [0.3]],
               state_names={"InfectionState": ["NotInfected", "Infected"]}),
    TabularCPD("TestResult", 2, [[0.95, 0.10], [0.05, 0.90]],
               evidence=["InfectionState"], evidence_card=[2],
               state_names={"TestResult": ["Negative", "Positive"],
                            "InfectionState": ["NotInfected", "Infected"]}),
)
assert test_bn.check_model()
ve = VariableElimination(test_bn)
post = ve.query(["InfectionState"], evidence={"TestResult": "Positive"})
p_inf = float(post.values[1])
print("Single-person query: P(Infected | positive test) =", round(p_inf, 3))

# FIGURE 5-A — VE result
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(["Not infected", "Infected"], [1-p_inf, p_inf], color=["#3498db", "#e74c3c"])
axes[0].set_ylim(0, 1); axes[0].set_title("P(Infection | positive test)")
for i, v in enumerate([1-p_inf, p_inf]):
    axes[0].text(i, v+0.02, f"{v:.2f}", ha="center")
# schematic inference flow
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(5, 5.3, "Forward-backward inference on full DBN", ha="center", fontweight="bold")
for x, lab in [(1, "Y^0"), (4, "Y^1"), (7, "Y^2"), (9.5, "...")]:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.6,2.5),1.2,0.8,boxstyle="round",fc="#abebc6",ec="k"))
    ax.text(x, 2.9, lab, ha="center", fontsize=9)
for x in [1, 4, 7]:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.6,4),1.2,0.8,boxstyle="round",fc="#aed6f1",ec="k"))
    ax.text(x, 4.4, "X", ha="center", fontsize=10, fontweight="bold")
ax.annotate("", xy=(5, 1.5), xytext=(5, 2.4), arrowprops=dict(arrowstyle="<->", lw=2))
ax.text(5, 1.1, "forward filter  +  backward smooth", ha="center", fontsize=9)
ax.set_title("Belief propagation over time")
plt.suptitle("FIGURE 5-A — Inference methods", y=1.02)
plt.tight_layout(); fig.savefig(OUT / "fig5a_inference.png", dpi=150); plt.show()

# Full network inference
P_I, beliefs = infer_infectious_probability(G, Y, learned, patient_zero=pz, smooth=True)
print(f"Posterior P(I) shape: {P_I.shape}  (days x people)")"""))

C.append(md("""**5-B  Answer infection queries on the full network**"""))

C.append(code("""query_rows = []
for node, day in [(pz, 5), (pz, 15), (1, 10), (5, 20), (10, 25)]:
    if day >= Y.shape[0]: continue
    p = query_node_infectious(G, Y, learned, node=node, time=day, patient_zero=pz)
    sym = {OBS_POS: "+", OBS_NEG: "-", OBS_MISSING: "?"}[Y[day, node]]
    query_rows.append({"person": node, "day": day, "P(infected|all tests)": round(p, 4), "observed test": sym})
display(pd.DataFrame(query_rows))

# FIGURE 5-B — posterior heatmap
fig, ax = plt.subplots(figsize=(11, 5.5))
im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax.set_xlabel("day"); ax.set_ylabel("person")
ax.set_title("FIGURE 5-B — P(Infected | all test/symptom observations)")
fig.colorbar(im, ax=ax, label="P(I)")
for t in range(Y.shape[0]):
    for i in range(Y.shape[1]):
        if Y[t,i] == OBS_POS: ax.plot(t, i, "wo", ms=5, mec="k", mew=0.8)
plt.tight_layout(); fig.savefig(OUT / "fig5b_heatmap.png", dpi=150); plt.show()

# FIGURE 5-C — posterior time series for selected individuals
fig, ax = plt.subplots(figsize=(10, 4.5))
for node, col in [(pz, "#e74c3c"), (1, "#2980b9"), (5, "#27ae60"), (10, "#8e44ad")]:
    if node < P_I.shape[1]:
        ax.plot(P_I[:, node], label=f"person {node}", color=col, lw=2)
        ax.axvline(x=np.argmax(P_I[:, node]), color=col, ls=":", alpha=0.5)
ax.set_xlabel("day"); ax.set_ylabel("P(infected)")
ax.set_title("FIGURE 5-C — Infection probability over time (selected people)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUT / "fig5c_timeseries.png", dpi=150); plt.show()

# FIGURE 5-D — network map coloured by max P(infected)
max_P = P_I.max(axis=0)
fig, ax = plt.subplots(figsize=(8, 6))
art = nx.draw_networkx_nodes(G, pos, node_color=max_P, cmap=plt.cm.Reds, vmin=0, vmax=1, node_size=650, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
fig.colorbar(art, ax=ax, label="max P(infected)")
ax.set_title("FIGURE 5-D — Who is most likely infected? (network posterior)")
plt.tight_layout(); fig.savefig(OUT / "fig5d_network.png", dpi=150); plt.show()

# FIGURE 5-E — true infected state vs posterior (sample person)
true_I = (X == STATE_IDX["I"]).astype(float)
fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(range(P_I.shape[0]), P_I[:, pz], alpha=0.35, color="#e74c3c", label="P(I) inferred")
ax.step(range(true_I.shape[0]), true_I[:, pz], where="mid", color="k", lw=2, label="true state (I=1)")
ax.set_xlabel("day"); ax.set_ylabel("probability / state")
ax.set_title(f"FIGURE 5-E — Inferred vs true infection state (person {pz})")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(OUT / "fig5e_compare.png", dpi=150); plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
C.append(md("""## PART 6 – EVALUATION

We evaluate whether the DBN successfully tracks the epidemic: compartment dynamics, posterior heatmaps, parameter learning, and alignment between inferred infection probability and ground-truth labels."""))

C.append(code("""true_I = (X == STATE_IDX["I"]).astype(float)
# evaluation at timesteps with any observation
obs_mask = Y != OBS_MISSING
if obs_mask.sum() > 1 and np.std(P_I[obs_mask]) > 1e-8:
    corr = np.corrcoef(P_I[obs_mask], true_I[obs_mask])[0, 1]
    print(f"Correlation P(I) vs true infected state: {corr:.3f}")

# FIGURE 6-A — full dashboard
fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=380, font_size=7, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[pz], node_color="#e74c3c", node_size=420, ax=ax)
ax.set_title("A  Contact network")

ax = fig.add_subplot(gs[0, 1])
stack = np.array([counts[s] for s in ["S","E","I","R"]])
ax.stackplot(range(stack.shape[1]), stack, labels=["S","E","I","R"],
             colors=["#3498db","#f39c12","#e74c3c","#27ae60"], alpha=0.8)
ax.set_title("B  SEIR compartments"); ax.legend(fontsize=7, loc="upper right")

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax.set_title("C  P(Infected) heatmap"); fig.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[1, 1])
for idx, (lab, col) in enumerate(zip(["beta","sigma","gamma"],["#2980b9","#8e44ad","#16a085"])):
    ax.plot(history[:, idx], label=lab, color=col, lw=2)
ax.set_title("D  EM learning"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 0])
ax.plot(counts["I"], "r-", lw=2, label="true infected")
ax.plot(P_I.sum(axis=1), "b--", lw=2, label="sum P(I) inferred")
ax.set_xlabel("day"); ax.set_title("E  Epidemic: true vs inferred total")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 1])
art = nx.draw_networkx_nodes(G, pos, node_color=max_P, cmap=plt.cm.Reds, vmin=0, vmax=1, node_size=400, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
fig.colorbar(art, ax=ax, fraction=0.046, label="max P(I)")
ax.set_title("F  Network posterior map")

fig.suptitle("FIGURE 6 — Project evaluation dashboard", fontsize=14, y=1.01)
plt.tight_layout(); fig.savefig(OUT / "fig6_dashboard.png", dpi=150, bbox_inches="tight"); plt.show()
print("All figures saved to", OUT)"""))

C.append(md("""### Project summary

| Part | PGM pillar | What we did |
|------|------------|-------------|
| 1 | Data | Loaded real epidemic tracing data; built Y observation matrix |
| 2 | **Representation** | Defined DBN, SEIR latent states, test observation model |
| 3 | **Representation** | Contact network as spatial DBN structure |
| 4 | **Learning** | EM algorithm estimated β, σ, γ from partial observations |
| 5 | **Inference** | Belief propagation answered P(infected \| all tests) |
| 6 | Evaluation | Visualised epidemic dynamics and posterior quality |

**Key query answered:**

> Given symptom/test observations across the network, what is the probability that a given node is currently **infected**?

*End of pipeline.*"""))


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": C,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(C)} cells -> {OUT}")


if __name__ == "__main__":
    main()
