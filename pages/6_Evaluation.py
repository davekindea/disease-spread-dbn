"""Page 6 — Evaluation Dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app_shared import (
    correlation_score,
    fig_evaluation_dashboard,
    get_P_I,
    render_sidebar,
)

G, Y, X, counts, pz, meta, pos, params = render_sidebar()

st.title("Evaluation Dashboard")
st.markdown("Full project dashboard (Fig 6) — epidemic tracking, posteriors, and learning quality.")

use_learned = (
    st.session_state.learned_params is not None
    and st.checkbox("Use EM-learned parameters for inference", value=False)
)
eval_params = st.session_state.learned_params if use_learned else params
P_I = get_P_I(G, Y, eval_params, pz)
corr = correlation_score(P_I, X, Y)

m1, m2, m3, m4 = st.columns(4)
m1.metric("People", meta["n_nodes"])
m2.metric("Days", meta["n_days"])
m3.metric("Positive tests", meta["n_pos"])
if corr is not None:
    m4.metric("Correlation P(I) vs true state", f"{corr:.3f}")
else:
    m4.metric("Correlation P(I) vs true state", "N/A")

st.subheader("Full evaluation dashboard (Fig 6)")
fig = fig_evaluation_dashboard(
    G, pos, counts, P_I, st.session_state.em_history, pz, X,
)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
    **Panel guide**

    - **A** Contact network with index case highlighted
    - **B** True SEIR compartment counts over time
    - **C** Posterior heatmap P(infected | observations)
    - **D** EM parameter convergence (run EM on Page 4)
    - **E** True infected count vs Σ P(I) inferred
    - **F** Network colored by max posterior infection probability
    """
)

if corr is not None:
    if corr >= 0.7:
        st.success(f"Strong alignment between inferred and true infection states (r = {corr:.3f}).")
    elif corr >= 0.4:
        st.info(f"Moderate alignment (r = {corr:.3f}) — expected with sparse testing.")
    else:
        st.warning(f"Weak alignment (r = {corr:.3f}) — try EM learning or adjust parameters.")
