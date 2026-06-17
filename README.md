# Disease Spread Modeling Using Dynamic Bayesian Networks

## Project description

This project models the **temporal spread of infectious diseases** through a population using **Dynamic Bayesian Networks (DBNs)**.

Individual infection states — **Susceptible, Exposed, Infected, and Recovered** — are **latent variables**. **Observations** correspond to reported symptoms or test results.

The project covers the **three core pillars of PGMs**:

| Pillar | Implementation |
|--------|----------------|
| **Representation** | DBN graph structure + SEIR conditional probability tables (`src/model.py`) |
| **Inference** | Belief propagation and variable elimination (`src/inference.py`) |
| **Learning** | EM algorithm for epidemic parameters (`src/learning.py`) |

### Expected outcome

> Given symptom/test observations across a network of individuals, what is the probability that a given node is currently **infected**?

## Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/PGM_EndToEnd_Pipeline.ipynb
```

```bash
python run.py
python run.py --query 0 10
```

Regenerate notebook: `python notebooks/build_notebook.py`

## Test with your own parameters

Use `test_model.py` to set β, σ, γ and run inference:

```bash
# Your parameters + one query
python test_model.py --beta 0.25 --sigma 0.20 --gamma 0.10 --query 0 10

# Also run EM learning from your starting values
python test_model.py --beta 0.15 --sigma 0.20 --gamma 0.10 --learn

# Compare different transmission rates
python test_model.py --data synthetic --compare-beta 0.1 0.3 0.5

# Test quality of COVID tests
python test_model.py --sensitivity 0.85 --specificity 0.90 --query 3 15
```

Results: terminal numbers + figures in `outputs/test/`.

## Interactive web app — 6-page demo

For **course demonstrations** with supervisors:

```bash
pip install -r requirements.txt
streamlit run app.py
```

| Page | Content |
|------|---------|
| **1 · Overview** | Problem statement, 3 PGM pillars, DBN diagram |
| **2 · Data Explorer** | Interactive contact network (Plotly), observation heatmap, filters |
| **3 · Model & Parameters** | SEIR diagram, CPT heatmap, live β/σ/γ sliders |
| **4 · EM Learning** | Run EM, live convergence plot, before/after params |
| **5 · Inference** | P(infected) query, posterior heatmap, network map |
| **6 · Evaluation** | Full Fig 6 dashboard + correlation metric |

## Data

COVID-19 Geneva contact tracing → `data/corona/` (auto-downloaded on first run).

## Layout

```
pgm/
├── notebooks/PGM_EndToEnd_Pipeline.ipynb
├── src/          # model, inference, learning, corona_data
├── data/corona/
└── outputs/
```
