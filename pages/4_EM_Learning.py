"""Page 4 — EM Learning."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app_shared import (
    fig_em_convergence,
    fig_params_comparison,
    invalidate_inference_cache,
    render_sidebar,
    run_em_streaming,
)
from src.config import ModelParams

G, Y, X, counts, pz, meta, pos, params = render_sidebar()

st.title("EM Learning")
st.markdown(
    "**Learning pillar** — estimate β, σ, γ from partial test observations using the EM algorithm."
)

c1, c2, c3 = st.columns(3)
with c1:
    init_beta = st.number_input("Initial β", 0.01, 0.99, 0.10, 0.01)
with c2:
    init_sigma = st.number_input("Initial σ", 0.01, 0.99, 0.10, 0.01)
with c3:
    init_gamma = st.number_input("Initial γ", 0.01, 0.99, 0.10, 0.01)

n_iter = st.slider("EM iterations", 5, 40, 25)

run = st.button("Run EM", type="primary", use_container_width=True)

if run:
    init_params = ModelParams(beta=init_beta, sigma=init_sigma, gamma=init_gamma)
    st.session_state.em_init_params = init_params
    chart_slot = st.empty()
    progress = st.progress(0, text="Running EM…")

    with st.spinner("E-step / M-step iterations…"):
        learned, history = run_em_streaming(G, Y, init_params, pz, n_iter, chart_slot)

    progress.progress(1.0, text="EM complete")
    st.session_state.learned_params = learned
    st.session_state.em_history = history
    invalidate_inference_cache()
    st.success(
        f"Learned: β={learned.beta:.3f}, σ={learned.sigma:.3f}, γ={learned.gamma:.3f}"
    )

if st.session_state.em_history is not None:
    st.divider()
    st.subheader("Convergence (Fig 4-A)")
    fig = fig_em_convergence(st.session_state.em_history)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Before / after comparison (Fig 4-B)")
    fig_b = fig_params_comparison(st.session_state.em_init_params, st.session_state.learned_params)
    st.pyplot(fig_b)
    plt.close(fig_b)

    lc1, lc2, lc3 = st.columns(3)
    init = st.session_state.em_init_params
    learned = st.session_state.learned_params
    lc1.metric("β", f"{learned.beta:.3f}", delta=f"{learned.beta - init.beta:+.3f}")
    lc2.metric("σ", f"{learned.sigma:.3f}", delta=f"{learned.sigma - init.sigma:+.3f}")
    lc3.metric("γ", f"{learned.gamma:.3f}", delta=f"{learned.gamma - init.gamma:+.3f}")
else:
    st.info("Set initial parameters and click **Run EM** to watch convergence in real time.")
