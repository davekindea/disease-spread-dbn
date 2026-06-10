# Disease Spread Modeling Using Dynamic Bayesian Networks

PGM course project modeling infectious disease spread on a contact network using a **2-time-slice Dynamic Bayesian Network (DBN)** with SEIR latent states and noisy test observations.

## Three PGM Pillars

| Pillar | Module | Description |
|--------|--------|-------------|
| **Representation** | `src/model.py` | 2-slice DBN structure, SEIR CPTs with parameters β, σ, γ |
| **Inference** | `src/inference.py` | Forward-backward filtering/smoothing → P(infectious \| observations) |
| **Learning** | `src/learning.py` | EM algorithm to estimate β, σ, γ from partial observations |

## Project Structure

```
pgm/
├── run.py                  # Main entry point
├── requirements.txt
├── src/
│   ├── config.py           # Constants and parameters
│   ├── network.py          # Contact network generation
│   ├── model.py            # CPTs and DBN structure
│   ├── simulation.py       # Epidemic simulation
│   ├── inference.py        # Forward-backward inference
│   ├── learning.py         # EM parameter learning
│   ├── visualization.py    # Report figures
│   └── experiments.py      # Sensitivity analyses
└── outputs/                # Generated figures (after running)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Full pipeline (all figures)
python run.py

# Quick test run
python run.py --quick

# Answer a specific query
python run.py --query 0 10
# → P(node 0 is infectious at t=10 | observations)
```

## Generated Figures

| Figure | File | Description |
|--------|------|-------------|
| 0 | `fig0_network.png` | Contact network |
| 1 | `fig1_epidemic_curve.png` | Simulated SEIR epidemic curve |
| 2 | `fig2_heatmap_P_I.png` | P(Infectious) heatmap (people × time) |
| 3 | `fig3_network_posterior.png` | Network colored by posterior P(I) |
| 4 | `fig4_em_convergence.png` | EM parameter convergence |
| 5 | `fig5_sensitivity_beta.png` | Peak infections vs β |
| 6 | `fig6_sensitivity_test_rate.png` | EM error vs test rate |
| 7 | `fig7_sensitivity_topology.png` | Epidemic size vs topology |

## Model Summary

**Latent variables:** \(X_i^t \in \{S, E, I, R\}\) — infection state of person \(i\) at time \(t\).

**Observations:** \(Y_i^t\) — symptom/test result (positive, negative, or missing).

**2-time-slice edges:**
- \(X_i^{t-1} \to X_i^t\) — within-person SEIR dynamics
- \(X_j^{t-1} \to X_i^t\) for neighbors \(j\) — disease transmission
- \(X_i^t \to Y_i^t\) — noisy test emission

**Key query:** Given partial test observations across the network, what is \(P(X_i^t = I \mid Y)\)?

## Parameters

| Symbol | Name | Default | Role |
|--------|------|---------|------|
| β | beta | 0.30 | Transmission rate per infectious contact |
| σ | sigma | 0.20 | Exposed → Infectious rate |
| γ | gamma | 0.10 | Infectious → Recovered rate |

## Notes

- Inference uses a **mean-field approximation** for neighbor coupling (tractable for arbitrary networks).
- The `export_pgmpy_dbn()` function in `model.py` builds a pgmpy DBN skeleton for the representation write-up.
- EM learns β, σ, γ; test sensitivity/specificity are held fixed.
