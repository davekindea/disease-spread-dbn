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
