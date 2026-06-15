# Data — Alignment with Project Description

This project requires epidemic data where:

1. **Individuals** form a population connected by a **contact network**
2. **Latent variables** are SEIR infection states (hidden from the model at inference time)
3. **Observations** are **reported symptoms or test results** (partial, noisy)

Both supported data modes satisfy these requirements.

---

## Real-world epidemic data (default)

### Source

[Geneva COVID-19 contact tracing (GEgraph)](https://github.com/PersonalDataIO/GEgraph)  
De-identified records from the Canton of Geneva, Switzerland, 2020.

### Mapping to the DBN model

| Project requirement | Geneva dataset |
|---------------------|----------------|
| Population of individuals | Subgraph of ~30 de-identified persons from an early outbreak cluster |
| Contact network | Undirected edges from close-contact tracing records |
| Latent SEIR states \(X_i^t\) | Not directly observed; inferred by forward–backward belief propagation |
| Symptom/test observations \(Y_i^t\) | **Positive PCR test dates** (`date_res` field) mapped to daily timesteps |
| Missing observations | Days without a recorded test → missing (\(Y_i^t = ?\)) |

### Files

| File | Content |
|------|---------|
| `redcap_suivi.csv` | Follow-up records including positive test dates |
| `redcap_entourage.csv` | Contact links between infected individuals and their contacts |

Downloaded automatically to `data/raw/` on first run by `src/real_data.py`.

### Why this is appropriate

- **Individual-level** data (not aggregate county counts)
- **Real outbreak** from the COVID-19 pandemic
- **Network structure** from actual contact tracing
- **Test results** as observations, matching the project description

---

## Simulated epidemic data (optional)

```bash
python run.py --data synthetic
```

### Mapping to the DBN model

| Project requirement | Synthetic generator |
|---------------------|---------------------|
| Population of individuals | \(n\) nodes on a Watts–Strogatz (or ER/BA) contact network |
| Latent SEIR states \(X_i^t\) | Simulated by SEIR transition rules; ground truth kept for validation |
| Symptom/test observations \(Y_i^t\) | Noisy positive/negative tests with configurable missingness |
| Parameters β, σ, γ | Known ground truth for validating EM learning |

### Why this is appropriate

- Enables **controlled experiments** and sensitivity analyses (Figures 5–7)
- Validates inference and EM when true latent states and parameters are known
- Matches the proposal: *"simulated **or** real-world epidemic data"*

---

## Observation encoding

| Value | Meaning | Project interpretation |
|-------|---------|------------------------|
| `0` | Negative test | Reported symptom/test: not infectious |
| `1` | Positive test | Reported symptom/test: likely infectious |
| `-1` | Missing | No symptom report or test at this timestep |

Emission probabilities (sensitivity / specificity) are defined in `src/config.py` and `src/model.py`.
