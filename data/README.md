# Real-world datasets

## Geneva COVID-19 contact tracing (default)

**Source:** [PersonalDataIO/GEgraph](https://github.com/PersonalDataIO/GEgraph)  
**License:** See repository for data use terms.

De-identified contact-tracing records from the Canton of Geneva, Switzerland (2020).
Includes:

- **Contact network** — edges between infected individuals and their close contacts
- **Test dates** — dates of positive PCR results (`date_res` in `redcap_suivi.csv`)

Files are downloaded automatically on first run into `data/raw/`.

## Synthetic data (optional)

Use `python run.py --data synthetic` to generate a simulated SEIR epidemic
on a Watts–Strogatz network instead.
