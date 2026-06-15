"""Regenerate the expanded end-to-end notebook."""
import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


CELLS = []

# --- Title & intro ---
CELLS.append(md(r"""# Disease Spread Modeling Using Dynamic Bayesian Networks

**PGM Course Project — Complete End-to-End Notebook with Explanations**

---

## What is this notebook?

This notebook walks through the **entire PGM project** from raw epidemic data to final figures and conclusions. It is written so you can follow every step for your **report, presentation, or exam**.

You will learn:
1. **What data we use** and how it maps to the DBN model
2. **How to represent** disease spread as a graphical model (**Pillar 1: Representation**)
3. **How to run inference** and answer *"Is person i infectious?"* (**Pillar 2: Inference**)
4. **How to learn parameters** with EM when β, σ, γ are unknown (**Pillar 3: Learning**)
5. **How to interpret every figure** for your write-up

---

## Official project description

> This project models the **temporal spread of infectious diseases through a population** using **Dynamic Bayesian Networks (DBNs)**. Individual infection states — **Susceptible, Exposed, Infectious, and Recovered** — are **latent variables**. **Observations** correspond to **reported symptoms or test results**. The project covers **representation**, **inference** (belief propagation), and **learning** (EM) on **simulated or real-world epidemic data**.

### The central probabilistic query

> *Given a set of symptom/test observations across a network of individuals, what is the probability that a given node is currently infectious?*

$$P(X_i^t = \text{Infectious} \mid \{\text{all test observations}\})$$

---

## Glossary

| Term | Plain-English meaning |
|------|----------------------|
| **DBN** | A Bayesian network unrolled over time — captures how states evolve day by day |
| **Latent variable** | Hidden infection state (SEIR) — we never observe it directly, only infer it |
| **Observation** | Symptom report or diagnostic test (positive / negative / missing) |
| **CPT** | Conditional Probability Table — local rules like "if neighbor is infectious, I might become exposed" |
| **Belief propagation** | Algorithm that propagates probability messages; forward–backward is the DBN version |
| **EM algorithm** | Learns unknown parameters by alternating inference (E-step) and re-estimation (M-step) |
| **β (beta)** | How easily disease transmits along one contact |
| **σ (sigma)** | How fast exposed people become infectious |
| **γ (gamma)** | How fast infectious people recover |

---

## Roadmap

| Part | Topic | PGM pillar |
|------|-------|------------|
| 0 | Setup | — |
| 1 | Data description & loading | Data |
| 2 | Representation (graph + CPTs) | **Representation** |
| 3 | Inference & figures | **Inference** |
| 4 | Learning with EM | **Learning** |
| 5 | Synthetic validation (optional) | Learning |
| 6 | Report summary | — |"""))

CELLS.append(md("""## Part 0 — Setup

**What this cell does:** Adds the project folder to Python's path, imports all modules from `src/`, and turns on inline plotting.

**Before running:** Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

**Tip:** Run cells **top to bottom** the first time. Later you can re-run individual sections."""))

CELLS.append(code("""import sys
from pathlib import Path

_cwd = Path.cwd()
if (_cwd / "src").is_dir():
    PROJECT_ROOT = _cwd
elif (_cwd.parent / "src").is_dir():
    PROJECT_ROOT = _cwd.parent
else:
    PROJECT_ROOT = _cwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

%matplotlib inline
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["font.size"] = 11

from src.config import ModelParams, SimConfig, STATES, STATE_IDX
from src.config import OBS_MISSING, OBS_POS, OBS_NEG
from src.real_data import load_real_data, RAW_DIR, GENEVA_FILES, _download_geneva_raw
from src.model import build_dbn_structure, transition_distribution, emission_likelihood
from src.network import network_summary
from src.inference import infer_infectious_probability, query_node_infectious, forward_filter
from src.learning import em_learn
from src.simulation import epidemic_counts

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Project root:", PROJECT_ROOT)
print("Figures saved to:", OUTPUT_DIR)"""))

# --- Part 1 Data ---
CELLS.append(md(r"""---

# Part 1 — Data Description (Real-World Epidemic Data)

## 1.1 Why do we need data?

A PGM needs **evidence** — the observations $Y$ — to perform inference. Without data, the model is only a mathematical structure (representation). With data, we can:

- **Infer** hidden SEIR states
- **Answer queries** like $P(\text{infectious} \mid \text{tests})$
- **Learn** unknown parameters via EM

Our project description requires **simulated or real-world epidemic data**. This notebook uses **real-world data**.

---

## 1.2 The dataset: Geneva COVID-19 contact tracing

| Property | Value |
|----------|-------|
| **Name** | GEgraph — Geneva contact tracing graph |
| **Source** | [PersonalDataIO/GEgraph](https://github.com/PersonalDataIO/GEgraph) |
| **When** | COVID-19 pandemic, Canton of Geneva, Switzerland, 2020 |
| **What it contains** | De-identified individuals, close-contact links, positive test dates |

### Why this dataset fits our project

| Project requirement | How Geneva data satisfies it |
|---------------------|------------------------------|
| Population of **individuals** | Each row/person is one individual (not county aggregates) |
| **Contact network** | Edges from manual contact-tracing interviews |
| **Latent infection state** | True SEIR state is never recorded — only inferred |
| **Symptom/test observations** | Positive PCR test dates = reported test results |
| **Temporal spread** | Tests occur on different calendar days → time series |

### Important limitations (mention in your report!)

1. **Sparse testing** — most person-days have no test (observation = missing)
2. **Only positives recorded clearly** — negatives are often unrecorded
3. **Subgraph** — we study ~30 connected people, not the full population
4. **SEIR is approximate** — real disease progression is more complex than 4 states

These limitations are normal in real epidemiology and make the PGM approach valuable: we **reason under uncertainty**.

---

## 1.3 Mapping data fields to the DBN

```
Real world                    DBN model
──────────                    ─────────
Person i                  →   Node i in graph G
"Were you in contact?"    →   Edge (i, j)
SEIR health state         →   Latent variable X_i^t  (HIDDEN)
PCR test on date d        →   Observation Y_i^t = positive
No test that day          →   Observation Y_i^t = missing
```

### SEIR states (latent — hidden from inference)

| State | Epidemiological meaning | Can transmit? |
|-------|------------------------|---------------|
| **S** Susceptible | Never infected (or not yet) | No |
| **E** Exposed | Infected, incubating | Usually no |
| **I** Infectious | Can spread to contacts | **Yes** |
| **R** Recovered | Cleared infection | No |

### Observation values (what the model sees)

| Code | Name | Meaning |
|------|------|---------|
| `1` | POSITIVE | Test/symptom indicates infection |
| `0` | NEGATIVE | Test indicates no infection |
| `-1` | MISSING | No test or report that day |"""))

CELLS.append(md("""## 1.4 Raw data files

Two CSV files are downloaded to `data/raw/`:

| File | Role |
|------|------|
| `redcap_suivi.csv` | Follow-up records — includes **date of positive result** (`date_res`) |
| `redcap_entourage.csv` | **Contact links** — who was near whom |

Run the next cell to preview them."""))

CELLS.append(code("""_download_geneva_raw()

suivi = pd.read_csv(RAW_DIR / GENEVA_FILES["suivi"])
entourage = pd.read_csv(RAW_DIR / GENEVA_FILES["entourage"])

print("=== suivi: follow-up & test dates ===")
print(f"Total rows: {len(suivi):,}")
print("Key columns:")
print("  record_id_pos     → infected person ID")
print("  date_res          → date of positive test result")
print("  contact_record_id → ID of a contact they met")
display(suivi.head(4))

print("\\n=== entourage: contact network edges ===")
print(f"Total rows: {len(entourage):,}")
print("Key columns:")
print("  record_id_pos / record_id_entourage → person A")
print("  contact_record_id                  → person B (contact)")
display(entourage.head(4))"""))

CELLS.append(md(r"""## 1.5 Building our epidemic subgraph

We cannot fit the full Geneva dataset (thousands of people) into a classroom DBN without approximations. Instead we:

1. **Pick a seed** — an early infected person (March 2020) with several contacts
2. **Expand by BFS** — add neighbors until we have ~30 people (one connected component)
3. **Build graph $G$** — nodes = people, edges = documented contacts
4. **Build matrix $Y$** — shape `(T days, n people)` with positive tests on the right days

**Intuition:** We zoom into one **outbreak cluster** like epidemiologists do when investigating a local wave."""))

CELLS.append(code("""config = SimConfig(n_nodes=30, data_source="real")
bundle = load_real_data(config)

G = bundle.graph
Y_obs = bundle.Y_obs
X_true = bundle.X_true
patient_zero = bundle.patient_zero
meta = bundle.metadata

print("=" * 55)
print("LOADED EPIDEMIC INSTANCE")
print("=" * 55)
print("Dataset:     ", bundle.dataset_name)
print("People:      ", meta["n_nodes"])
print("Time steps:  ", meta["n_timesteps"], "days")
print("Date range:  ", meta["date_start"], "->", meta["date_end"])
print("Seed (patient zero): node", patient_zero, "| person ID", meta["seed_person_id"])
print("Positive tests:", meta["n_positive_tests"])
print("Observation matrix Y shape:", Y_obs.shape, "(time x people)")
print("Non-missing entries:", (Y_obs >= 0).sum(), "/", Y_obs.size,
      f"({100 * (Y_obs >= 0).mean():.1f}% filled)")
print("=" * 55)"""))

CELLS.append(md("""### 1.6 List of observed positive tests

Each row below is **evidence** the DBN uses. Days without a row for person *i* mean $Y_i^t = \\text{missing}$."""))

CELLS.append(code("""obs_records = []
for t in range(Y_obs.shape[0]):
    for i in range(Y_obs.shape[1]):
        if Y_obs[t, i] == OBS_POS:
            obs_records.append({
                "node": i,
                "person_id": bundle.node_ids[i],
                "day_t": t,
                "calendar_date": str(bundle.dates[t].date()),
                "Y": "POSITIVE (+)",
            })

obs_df = pd.DataFrame(obs_records)
print(f"Total positive observations: {len(obs_df)}")
display(obs_df)

# Sparsity visualization
fig, ax = plt.subplots(figsize=(10, 3))
ax.imshow((Y_obs >= 0).T, aspect="auto", cmap="Greys", origin="lower")
ax.set_xlabel("Day index")
ax.set_ylabel("Person (node)")
ax.set_title("Where we have ANY test observation (white = observed, black = missing)")
plt.tight_layout()
plt.show()"""))

CELLS.append(md("""**How to read the sparsity plot:**
- **White pixels** = we have a test result that day for that person
- **Black pixels** = missing — the model must **infer** infection using neighbors and dynamics
- Real contact tracing is almost always this sparse — that is why PGMs are useful!"""))

# --- Part 2 Representation ---
CELLS.append(md(r"""---

# Part 2 — Representation (PGM Pillar 1)

## 2.1 What does "representation" mean?

In PGMs, **representation** means defining:
1. **Which variables exist** (latent $X$ and observed $Y$)
2. **How they connect** (the graph / DBN structure)
3. **Local probability rules** (CPTs)

Once representation is fixed, inference and learning are **mechanical** — they operate on this structure.

---

## 2.2 The 2-time-slice Dynamic Bayesian Network

A DBN repeats the same **slice** at every timestep:

```
Slice at time t-1          Slice at time t
─────────────────          ───────────────

  X_j^{t-1} ──────────────→ X_i^t ───→ Y_i^t
       ↑                         ↑
  X_i^{t-1} ──────────────→ (self-transition)
```

### Three types of edges

| Edge | Meaning | Epidemiology |
|------|---------|--------------|
| $X_i^{t-1} \to X_i^t$ | Your state tomorrow depends on your state today | SEIR progression: S→E→I→R |
| $X_j^{t-1} \to X_i^t$ | Your state depends on neighbors' past states | Transmission along contacts |
| $X_i^t \to Y_i^t$ | Test depends on current health | Positive if infectious (usually) |

### Why "2-time-slice"?

We only need **two consecutive time slices** to describe the DBN template; unrolling it over $T$ days gives the full model. This is standard in DBN textbooks (Koller & Friedman, Ch. 6).

---

## 2.3 SEIR compartment transitions (within-person)

These are the **self-transitions** $X_i^{t-1} \to X_i^t$:

| From | To | Rate / rule |
|------|-----|-------------|
| S | E | Infected by a neighbor (depends on β and neighbor's $P(I)$) |
| E | I | Incubation ends (rate σ) |
| I | R | Recovery (rate γ) |
| R | R | Stays recovered (absorbing) |

**Key idea:** Susceptible people only become exposed through **contact** with infectious neighbors — that is why the **network structure matters**."""))

CELLS.append(code("""structure = build_dbn_structure(G)
summary = network_summary(G)

print("DBN STRUCTURE (textual description):")
print(" ", structure["description"])
print()
print("CONTACT NETWORK STATISTICS:")
for k, v in summary.items():
    print(f"  {k}: {v}")"""))

CELLS.append(md(r"""## 2.4 Conditional Probability Tables (CPTs)

CPTs encode **local** probability rules. Our model has three epidemiological parameters:

| Symbol | Name | Typical role |
|--------|------|--------------|
| **β** | beta | Transmission probability per infectious neighbor |
| **σ** | sigma | Daily probability E → I |
| **γ** | gamma | Daily probability I → R |

### Transmission: noisy-OR model

If person $i$ is susceptible, each infectious neighbor $j$ independently tries to infect them with strength $\beta \cdot P(X_j^{t-1}=I)$. Combined:

$$P(X_i^t = E \mid X_i^{t-1}=S) = 1 - \prod_{j \in \text{neighbors}(i)} \bigl(1 - \beta \cdot P(X_j^{t-1}=I)\bigr)$$

**Intuition:** More infectious neighbors → higher chance of exposure. Noisy-OR is a classic PGM trick to keep the CPT tractable.

### Emission: diagnostic test model

| True state | P(positive test) | P(negative test) |
|------------|------------------|------------------|
| Infectious (I) | sensitivity (0.90) | 1 − sensitivity |
| Not infectious (S,E,R) | 1 − specificity | specificity (0.95) |

**Intuition:** Tests are imperfect — false positives and false negatives exist."""))

CELLS.append(code("""params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)

print("EXAMPLE CPT: Susceptible person, ONE infectious neighbor (P(I)=1.0)")
print("-" * 55)
p = transition_distribution("S", [1.0], params)
for state, prob in zip(STATES, p):
    bar = "#" * int(prob * 40)
    print(f"  P(next={state}) = {prob:.3f}  {bar}")

print("\\nEMISSION LIKELIHOOD P(Y | X) for each true state:")
print("-" * 55)
for state in STATES:
    ep = emission_likelihood(OBS_POS, params)[STATE_IDX[state]]
    en = emission_likelihood(OBS_NEG, params)[STATE_IDX[state]]
    print(f"  True state {state}: P(pos)={ep:.2f}, P(neg)={en:.2f}")"""))

CELLS.append(md("""---

## Figure 0 — Contact Network

### What this figure shows
The **population structure**: individuals (nodes) and documented close contacts (edges).

### How to read it
- **Each circle** = one person in our subgraph
- **Each line** = a traced close contact (disease may spread along this link)
- **Labels** = node indices (0, 1, 2, …) used by the code

### Why it matters for the DBN
The graph determines **who can infect whom**. A person with many neighbors (high degree) is more exposed. The DBN couples each $X_i^t$ to the previous states of graph neighbors.

### What to write in your report
> "Figure 0 shows the contact network extracted from Geneva COVID-19 tracing data. The DBN transmission edges follow this structure: $X_j^{t-1} \to X_i^t$ for each contact $(i,j)$." """))

CELLS.append(code("""pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="#a8d4f0", node_size=550,
        font_size=9, edge_color="#666666", ax=ax)
# Highlight patient zero
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#e74c3c",
                       node_size=600, ax=ax)
ax.set_title("Figure 0: Contact Network\\n(red = patient zero / outbreak seed)", fontsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig0_network.png", dpi=150)
plt.show()
print(f"Patient zero is node {patient_zero} (red) — the outbreak seed in this subgraph.")"""))

CELLS.append(md("""---

## Figure 1 — Epidemic Curve

### What this figure shows
How many people are in each **SEIR compartment** over time.

### How to read it
| Line | Color | Meaning |
|------|-------|---------|
| **I** (infectious) | Red solid | People who can transmit — look for the **peak** |
| **E** (exposed) | Orange dashed | Infected but not yet infectious |
| **R** (recovered) | Green dotted | Cleared infection |

### Important note for real data
These SEIR counts are **approximated** from test dates for visualization only. During **inference**, the model never sees true SEIR states — only sparse test results.

### Why it matters
Shows our subgraph captures a realistic outbreak pattern (rise and fall of infections). Validates that the data window is meaningful.

### What to write in your report
> "Figure 1 shows the approximate epidemic curve. The infectious compartment peaks at day $t = \ldots$, consistent with a localized COVID-19 cluster in Geneva, March 2020." """))

CELLS.append(code("""counts = epidemic_counts(X_true)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(counts["I"], label="Infectious (I)", linewidth=2.5, color="#c0392b")
ax.plot(counts["E"], label="Exposed (E)", linestyle="--", color="#e67e22")
ax.plot(counts["R"], label="Recovered (R)", linestyle=":", color="#27ae60")
ax.set_xlabel("Time step (days from start of observation window)")
ax.set_ylabel("Number of individuals")
ax.set_title("Figure 1: Epidemic Curve (approximate SEIR from real test dates)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_epidemic_curve.png", dpi=150)
plt.show()

peak_t = counts["I"].index(max(counts["I"]))
print(f"Peak infectious: {max(counts['I'])} people at day t={peak_t}")
print(f"Calendar date at peak: {bundle.dates[peak_t].date()}")"""))

# --- Part 3 Inference ---
CELLS.append(md(r"""---

# Part 3 — Inference (PGM Pillar 2)

## 3.1 What is inference?

**Inference** = computing posterior probabilities of **latent** variables given **observations**:

$$P(X \mid Y) = \frac{P(Y \mid X) \, P(X)}{P(Y)}$$

In our project:

$$P(X_i^t = I \mid \{\text{all tests}\})$$

We cannot observe SEIR directly — we only see occasional tests. Inference **fills in the gaps** using:
- SEIR dynamics (CPTs)
- Network structure (who contacts whom)
- All test results (past and future, after smoothing)

---

## 3.2 Forward–backward belief propagation

Our implementation uses **forward filtering + backward smoothing** — the standard exact inference method for chain-structured temporal models. This is the DBN equivalent of **belief propagation**.

### Forward pass (filtering)

At each day $t$, compute:

$$\alpha_t(i,s) = P(X_i^t = s \mid Y^{1:t})$$

**Steps for each person $i$:**
1. **Predict:** combine yesterday's belief with SEIR transition and neighbor infectiousness
2. **Update:** multiply by likelihood of today's test $Y_i^t$ (skip if missing)

**Intuition:** Forward pass answers *"What do I believe **so far** given tests up to today?"*

### Backward pass (smoothing)

$$\gamma_t(i,s) = P(X_i^t = s \mid Y^{1:T}) \propto \alpha_t(i,s) \cdot \beta_t(i,s)$$

**Intuition:** A positive test **3 days later** tells us we were likely infectious earlier too. Smoothing uses **future** evidence.

### Mean-field neighbor approximation

Exact DBN inference on a network is intractable (exponential in degree). We use **mean-field**:

$$P(X_j^{t-1} = I) \approx \gamma_{t-1}(j, I)$$

Each neighbor contributes its marginal infectious probability — a standard trade-off between accuracy and tractability."""))

CELLS.append(code("""print("Running forward-backward inference...")
P_I, beliefs = infer_infectious_probability(
    G, Y_obs, params, patient_zero=patient_zero, smooth=True,
)
print("Done.")
print(f"Posterior P(I) shape: {P_I.shape}  (days x people)")
print(f"Value range: [{P_I.min():.4f}, {P_I.max():.4f}]")
print(f"Mean P(I) across all person-days: {P_I.mean():.4f}")"""))

CELLS.append(md("""---

## Figure 2 — Posterior Heatmap $P(\text{Infectious})$

### What this figure shows
For **every person** and **every day**, the probability they are in the **Infectious (I)** state.

### How to read it
| Axis | Meaning |
|------|---------|
| Horizontal (x) | Time — day index |
| Vertical (y) | Person — node index |
| Color intensity | $P(X_i^t = I)$ — white = 0, dark red = 1 |

**White dots** (in the plot below) mark days with an **observed positive test**.

### Patterns to look for
1. **Bright rows** near positive tests — model agrees with evidence
2. **Bright regions before a test** — smoothing infers infection before diagnosis
3. **Dark rows** — person likely never infectious
4. **Spread to neighbors** — bright bands on connected nodes shortly after seed infection

### Why it matters
This is the **direct answer** to the project query, visualized for everyone at all times.

### Report sentence template
> "Figure 2 displays $P(X_i^t = I \mid Y)$ for all individuals and timesteps. High-probability regions align with observed positive tests (marked), while unobserved periods are inferred from network transmission dynamics." """))

CELLS.append(code("""fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(P_I.T, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1)
ax.set_xlabel("Day index (time)")
ax.set_ylabel("Person (node index)")
ax.set_title("Figure 2: P(Infectious | All Observations)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("P(I)")

for t in range(Y_obs.shape[0]):
    for i in range(Y_obs.shape[1]):
        if Y_obs[t, i] == OBS_POS:
            ax.plot(t, i, "wo", markersize=5, markeredgecolor="black", markeredgewidth=0.6)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_heatmap_P_I.png", dpi=150)
plt.show()
print("White dots = observed POSITIVE tests on that person-day.")"""))

CELLS.append(md("""---

## Figure 3 — Network Colored by Posterior

### What this figure shows
Each person colored by their **maximum** infectious probability over the entire outbreak window:

$$\max_t \; P(X_i^t = I \mid Y)$$

### How to read it
| Color | Meaning |
|-------|---------|
| White / pale | Person was likely never infectious |
| Dark red | Person was very likely infectious at some point |

### Why it matters
Summarizes **who was most affected** in one view. Useful for identifying:
- The outbreak **seed** (patient zero)
- **Hub nodes** with many contacts who may have spread disease
- **Peripheral nodes** who escaped infection

### Report sentence template
> "Figure 3 maps $\max_t P(X_i^t = I \mid Y)$ onto the contact network, highlighting individuals most likely to have been infectious during the Geneva cluster." """))

CELLS.append(code("""max_P = P_I.max(axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
nodes = nx.draw_networkx_nodes(G, pos, node_color=max_P, cmap=plt.cm.Reds,
                               vmin=0, vmax=1, node_size=600, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=[patient_zero], node_color="#8e44ad",
                       node_size=650, ax=ax)
cbar = fig.colorbar(nodes, ax=ax)
cbar.set_label("max over time: P(I)")
ax.set_title("Figure 3: Network by Posterior P(Infectious)\\n(purple = patient zero)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_network_posterior.png", dpi=150)
plt.show()

ranking = sorted(enumerate(max_P), key=lambda x: -x[1])[:5]
print("Top 5 nodes by max P(I):")
for node, p in ranking:
    print(f"  node {node:2d} (person {bundle.node_ids[node]}): max P(I) = {p:.3f}")"""))

CELLS.append(md(r"""## 3.3 Answering the project query (step by step)

**Query:** Given symptom/test observations across the network, what is $P(\text{node } i \text{ is infectious at time } t)$?

**Procedure:**
1. Load observations $Y$ into the DBN
2. Run forward–backward inference → get $\gamma_t(i, I)$
3. Read off $P(X_i^t = I \mid Y) = \gamma_t(i, I)$

The table below shows example queries with:
- The **posterior** from the model
- The **observation** that day (positive / missing)
- The **approximate true state** (for comparison only — not used by the model)"""))

CELLS.append(code("""examples = [
    (patient_zero, 5,  "Seed person, early outbreak"),
    (patient_zero, 15, "Seed person, mid outbreak"),
    (1, 10, "Neighbor of seed"),
    (5, 20, "Another individual"),
]

rows = []
for node, t, note in examples:
    if t >= Y_obs.shape[0]:
        continue
    p = query_node_infectious(G, Y_obs, params, node=node, time=t, patient_zero=patient_zero)
    y = Y_obs[t, node]
    y_str = {OBS_POS: "POSITIVE", OBS_NEG: "NEGATIVE", OBS_MISSING: "missing"}[y]
    approx = STATES[X_true[t, node]]
    rows.append({
        "node": node,
        "person_id": bundle.node_ids[node],
        "day_t": t,
        "date": str(bundle.dates[t].date()),
        "P(infectious | Y)": round(p, 4),
        "observation": y_str,
        "approx_true_SEIR": approx,
        "note": note,
    })

display(pd.DataFrame(rows))"""))

CELLS.append(md("""### How to interpret query results

| Situation | Expected P(I) |
|-----------|---------------|
| Positive test that day | High (often > 0.7) |
| Missing test, but neighbors tested positive recently | Moderate (network propagation) |
| Missing test, far from outbreak | Low |
| Negative test | Low (but not zero — false negatives exist) |

**Remember:** On real data we rarely know the true SEIR state. The posterior is our **best probabilistic estimate** under the model assumptions."""))

# --- Part 4 Learning ---
CELLS.append(md(r"""---

# Part 4 — Learning (PGM Pillar 3)

## 4.1 Why learning?

In practice we do not know $\beta$, $\sigma$, $\gamma$. **Learning** estimates them from data.

Our project uses the **EM (Expectation–Maximization) algorithm** — the standard approach for PGMs with latent variables.

---

## 4.2 EM algorithm — intuition

EM alternates two steps until parameters stabilize:

| Step | Name | What it does |
|------|------|--------------|
| **E** | Expectation | Run inference → get soft assignments $\gamma_t(i,s) \approx P(X_i^t=s \mid Y, \theta)$ |
| **M** | Maximization | Recompute $\beta, \sigma, \gamma$ to best explain expected transitions |

**Why EM works:** We never observe SEIR states, but we can count **expected** transitions (e.g., how often S→E, E→I, I→R) using posterior beliefs from the E-step.

### M-step formulas (simplified)

$$\sigma^{\text{new}} = \frac{\#\text{expected } E \to I}{\#\text{expected in } E}$$
$$\gamma^{\text{new}} = \frac{\#\text{expected } I \to R}{\#\text{expected in } I}$$
$$\beta^{\text{new}} \propto \frac{\#\text{expected } S \to E \text{ with exposure}}{\#\text{expected susceptible with exposure}}$$

---

## 4.3 Learning on real data vs synthetic

| | Real data | Synthetic data |
|---|-----------|----------------|
| Ground-truth $\beta,\sigma,\gamma$ | **Unknown** | Known (set by simulator) |
| Goal | Find parameters that explain tests | Recover true parameters |
| Validation | Compare learned values to literature | Compare learned vs true |

On **real Geneva data**, EM finds parameters that best fit the observed positive tests under our SEIR+network model."""))

CELLS.append(code("""init_params = ModelParams(beta=0.10, sigma=0.10, gamma=0.10)
print("INITIAL PARAMETER GUESS (deliberately wrong):")
print(f"  beta={init_params.beta}, sigma={init_params.sigma}, gamma={init_params.gamma}")
print()
print("Running EM for 25 iterations...")
learned, history = em_learn(G, Y_obs, init_params, n_iter=25, patient_zero=patient_zero, verbose=True)
print()
print("FINAL LEARNED PARAMETERS:")
print(f"  beta={learned.beta:.3f}  (transmission)")
print(f"  sigma={learned.sigma:.3f} (E -> I)")
print(f"  gamma={learned.gamma:.3f} (I -> R)")"""))

CELLS.append(md("""---

## Figure 4 — EM Convergence

### What this figure shows
How $\beta$, $\sigma$, and $\gamma$ change across EM iterations.

### How to read it
| Line color | Parameter |
|------------|-----------|
| Blue | $\beta$ — transmission |
| Purple | $\sigma$ — incubation / E→I |
| Teal | $\gamma$ — recovery I→R |

- **Steep changes early** = EM quickly adjusts from bad initial guess
- **Flat tail** = convergence (parameters stabilized)

### On real data
There are no dashed "true" reference lines — we do not know ground truth. Discuss whether learned values are **epidemiologically plausible**.

### Report sentence template
> "Figure 4 shows EM convergence on Geneva contact-tracing data. Parameters stabilize after approximately $N$ iterations, demonstrating successful learning from partial observations without direct access to latent SEIR states." """))

CELLS.append(code("""labels = ["beta (transmission)", "sigma (E->I)", "gamma (I->R)"]
colors = ["#2980b9", "#8e44ad", "#16a085"]

fig, ax = plt.subplots(figsize=(9, 4.5))
for idx, (label, color) in enumerate(zip(labels, colors)):
    ax.plot(history[:, idx], label=f"learned {label}", color=color, linewidth=2.5)
ax.set_xlabel("EM iteration")
ax.set_ylabel("Parameter value")
ax.set_title("Figure 4: EM Parameter Convergence (Real Geneva Data)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_em_convergence.png", dpi=150)
plt.show()"""))

# --- Part 5 Synthetic ---
CELLS.append(md("""---

# Part 5 — Optional: Synthetic Data Validation

The project description allows **simulated or real-world** data. Synthetic data lets us **validate** EM because we know the true $\beta$, $\sigma$, $\gamma$.

**When to include this in your report:** Show that when ground truth is known, EM recovers parameters reasonably well — then argue the same algorithm applies to real data where truth is hidden."""))

CELLS.append(code("""from src.network import make_contact_network
from src.simulation import simulate_epidemic

syn_config = SimConfig(n_nodes=20, n_timesteps=40, test_probability=0.7, seed=42)
true_params = ModelParams(beta=0.30, sigma=0.20, gamma=0.10)
G_syn = make_contact_network(syn_config.n_nodes, syn_config.network_kind, syn_config.seed)
X_syn, Y_syn = simulate_epidemic(G_syn, true_params, syn_config)

learned_syn, hist_syn = em_learn(
    G_syn, Y_syn, ModelParams(0.1, 0.1, 0.1), n_iter=25, verbose=False,
)

fig, ax = plt.subplots(figsize=(9, 4.5))
for idx, (name, color) in enumerate(zip(["beta", "sigma", "gamma"], colors)):
    ax.plot(hist_syn[:, idx], color=color, linewidth=2, label=f"learned {name}")
    ax.axhline(true_params.as_array()[idx], color=color, linestyle=":", alpha=0.7, label=f"true {name}")
ax.set_xlabel("EM iteration")
ax.set_ylabel("Parameter value")
ax.set_title("Synthetic Data: EM Converges Toward Known True Parameters")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("True:   ", true_params)
print("Learned:", learned_syn)"""))

# --- Part 6 Summary ---
CELLS.append(md(r"""---

# Part 6 — Summary & Report Guide

## 6.1 Three PGM pillars — what we did

| Pillar | What we built | Code module |
|--------|---------------|-------------|
| **Representation** | 2-slice DBN, SEIR CPTs, contact network | `src/model.py` |
| **Inference** | Forward–backward belief propagation | `src/inference.py` |
| **Learning** | EM for $\beta$, $\sigma$, $\gamma$ | `src/learning.py` |

---

## 6.2 Figure reference (copy into your report)

| Figure | File | One-sentence caption |
|--------|------|---------------------|
| **0** | `fig0_network.png` | Contact network of individuals from Geneva COVID-19 tracing; edges represent documented close contacts. |
| **1** | `fig1_epidemic_curve.png` | Approximate SEIR epidemic curve showing rise and fall of infectious individuals over time. |
| **2** | `fig2_heatmap_P_I.png` | Posterior $P(X_i^t = I \mid Y)$ for all individuals and days; white dots mark positive tests. |
| **3** | `fig3_network_posterior.png` | Contact network colored by $\max_t P(X_i^t = I \mid Y)$, identifying most affected individuals. |
| **4** | `fig4_em_convergence.png` | EM learning convergence for transmission, progression, and recovery parameters. |

---

## 6.3 Suggested report structure

1. **Introduction** — disease spread on networks; why DBNs
2. **Data** — Geneva tracing; mapping to latent/observed variables
3. **Representation** — DBN plate, SEIR CPTs, parameters
4. **Inference** — forward–backward; answer key query with examples
5. **Learning** — EM derivation sketch; convergence figure
6. **Results** — Figures 0–4 with interpretation
7. **Discussion** — sparsity, mean-field approximation, limitations
8. **Conclusion** — project successfully answers $P(\text{infectious} \mid \text{observations})$

---

## 6.4 Key takeaways

1. **Latent SEIR states** are never directly observed — the DBN infers them from sparse tests
2. **Network structure** couples individuals — disease spreads along contact edges
3. **Inference** answers the project's central query probabilistically
4. **EM learning** estimates epidemiological parameters without seeing true states
5. **Real data** (Geneva) demonstrates practical applicability; **synthetic data** validates the pipeline

---

## 6.5 Assumptions to acknowledge

- Mean-field neighbor coupling (tractable but approximate)
- SEIR is a simplification of real COVID biology
- Real data has mostly positive tests recorded, few explicit negatives
- Subgraph of ~30 people is a local cluster, not full population

---

**End of notebook.** Re-run all cells before submitting your report to regenerate figures in `outputs/notebook/`."""))


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
    out = Path(__file__).parent / "PGM_Epidemic_DBN_EndToEnd.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(CELLS)} cells to {out}")


if __name__ == "__main__":
    main()
