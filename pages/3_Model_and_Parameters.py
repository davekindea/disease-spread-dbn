"""Page 3 — Model & Parameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app_shared import fig_seir_compartments, fig_transition_matrices, render_sidebar
from src.config import ModelParams

G, Y, X, counts, pz, meta, pos, params = render_sidebar()

st.title("Model & Parameters")
st.markdown("**Representation pillar** — SEIR dynamics, CPTs, and live parameter tweaking.")

left, right = st.columns([1, 1])
with left:
    st.subheader("SEIR compartment diagram")
    fig = fig_seir_compartments()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Tweak epidemic parameters")
    beta = st.slider("β transmission", 0.01, 0.80, params.beta, 0.01, key="model_beta")
    sigma = st.slider("σ E→I", 0.01, 0.80, params.sigma, 0.01, key="model_sigma")
    gamma = st.slider("γ I→R", 0.01, 0.99, params.gamma, 0.01, key="model_gamma")
    page_params = ModelParams(beta=beta, sigma=sigma, gamma=gamma)
    st.latex(r"P(E \mid S, \text{neighbors}) = 1 - \prod_j (1 - \beta \cdot P(X_j = I))")
    st.markdown(
        f"Live CPT below uses **β={page_params.beta:.2f}** · **σ={page_params.sigma:.2f}** · "
        f"**γ={page_params.gamma:.2f}** (sidebar controls inference on other pages)"
    )

st.subheader("Transition matrix heatmap (Fig 4-C)")
fig_t = fig_transition_matrices(page_params)
st.pyplot(fig_t)
plt.close(fig_t)

with st.expander("Emission model P(test | state)"):
    st.markdown(
        f"""
        | State | P(positive test) | P(negative test) |
        |-------|------------------|------------------|
        | S, E, R | {1 - page_params.specificity:.2f} | {page_params.specificity:.2f} |
        | I | {page_params.sensitivity:.2f} | {1 - page_params.sensitivity:.2f} |

        Sensitivity = {page_params.sensitivity:.0%} · Specificity = {page_params.specificity:.0%}
        """
    )

with st.expander("DBN structure summary"):
    from src.model import build_dbn_structure
    dbn = build_dbn_structure(G)
    st.json({
        "n_people": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "intra_slice_edges": len(dbn["intra_slice_edges"]),
        "inter_slice_edges": len(dbn["inter_slice_edges"]),
        "description": dbn["description"],
    })
