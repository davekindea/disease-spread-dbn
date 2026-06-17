"""Page 1 — Overview / Introduction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app_shared import fig_dbn_plate, render_sidebar
from src.model import build_dbn_structure

G, Y, X, counts, pz, meta, pos, params = render_sidebar()

st.title("Disease Spread Modeling with Dynamic Bayesian Networks")
st.markdown(
    """
    This project models **temporal disease spread** through a contact network using a
    **SEIR Dynamic Bayesian Network (DBN)**. Given partial test/symptom observations,
    we answer probabilistic queries about **latent infection states**.
    """
)

st.subheader("Problem statement")
st.markdown(
    """
    > Given symptom/test observations across a network of individuals, what is the
    > probability that a given node is currently **infected**?

    Individuals move through latent states **Susceptible → Exposed → Infected → Recovered**.
    Tests provide noisy, partial observations. The DBN couples **within-person SEIR dynamics**
    with **between-person transmission** along contact edges.
    """
)

st.subheader("The three PGM pillars")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 1 · Representation")
    st.markdown(
        "DBN graph structure, SEIR latent states, and conditional probability tables "
        "(`src/model.py`)."
    )
with c2:
    st.markdown("#### 2 · Inference")
    st.markdown(
        "Forward–backward belief propagation answers P(infected | observations) "
        "(`src/inference.py`)."
    )
with c3:
    st.markdown("#### 3 · Learning")
    st.markdown(
        "EM algorithm estimates epidemic parameters β, σ, γ from partial tests "
        "(`src/learning.py`)."
    )

st.divider()
st.subheader("DBN structure")
left, right = st.columns([1.1, 1])
with left:
    fig = fig_dbn_plate()
    st.pyplot(fig)
    plt.close(fig)
with right:
    dbn = build_dbn_structure(G)
    st.markdown(
        f"""
        **Latent variables:** {', '.join(dbn['latent_variables'][:3])}… (one per person)

        **Observed variables:** {', '.join(dbn['observed_variables'][:3])}… (test results)

        **Within-slice edges:** {len(dbn['intra_slice_edges'])} (state → observation)

        **Between-slice edges:** {len(dbn['inter_slice_edges'])} (SEIR dynamics + transmission)

        {dbn['description']}
        """
    )

st.caption("Use the sidebar to switch datasets · Pick a page from the top-left menu")
