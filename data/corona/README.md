# COVID-19 (Corona) Dataset

Integrated dataset for the PGM epidemic DBN project.

## Primary: Geneva COVID-19 contact tracing

| File | Description |
|------|-------------|
| `redcap_suivi.csv` | Follow-up + **positive PCR test dates** |
| `redcap_entourage.csv` | **Contact network** edges |

**Source:** [GEgraph](https://github.com/PersonalDataIO/GEgraph) — Canton of Geneva, Switzerland, 2020 pandemic.

## Secondary: OWID national context

| File | Description |
|------|-------------|
| `owid-covid-context.csv` | Switzerland daily new cases (cached from Our World in Data) |

## Usage

```python
from src.corona_data import load_corona_dataset, download_corona_dataset
download_corona_dataset()
bundle = load_corona_dataset(max_nodes=30)
```

Files download automatically to this folder on first run.
