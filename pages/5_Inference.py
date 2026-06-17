"""Page 5 — Inference."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app_shared import (
    OBS_LABEL,
    fig_network_posterior,
    fig_posterior_heatmap,
    get_P_I,
    render_sidebar,
)
from src.config import OBS_NEG, OBS_POS, STATES
from src.inference import query_node_infectious, query_seir_posterior, query_with_extra_evidence

G, Y, X, counts, pz, meta, pos, params = render_sidebar()

n_people, n_days = G.number_of_nodes(), Y.shape[0]

st.title("Inference")
st.markdown(
    "**Inference pillar** — answer P(person *i* is infected on day *t* | all observations)."
)

add_ev = st.checkbox(
    "Add hypothetical test evidence (optional)",
    value=False,
    help="Check this only if you want to see what happens after adding a fake test result.",
)

with st.form("inference_query"):
    fc1, fc2 = st.columns(2)
    with fc1:
        person = st.number_input("Person", 0, n_people - 1, int(pz), 1)
    with fc2:
        day = st.number_input("Day", 0, n_days - 1, min(10, n_days - 1), 1)

    ev_person = ev_day = ev_result = None
    if add_ev:
        st.markdown("**Hypothetical test to add**")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            ev_person = st.number_input("Evidence person", 0, n_people - 1, 1, 1)
        with ec2:
            ev_day = st.number_input("Evidence day", 0, n_days - 1, 5, 1)
        with ec3:
            ev_result = st.selectbox("Evidence result", ["positive (+)", "negative (−)"])

    submitted = st.form_submit_button("Run inference", type="primary", use_container_width=True)

if submitted or "inference_ran" not in st.session_state:
    st.session_state.inference_ran = True

    if submitted:
        query_person, query_day = int(person), int(day)
    else:
        query_person, query_day = int(pz), min(10, n_days - 1)

    extra = None
    if submitted and add_ev and ev_person is not None:
        obs = OBS_POS if "positive" in ev_result else OBS_NEG
        extra = [(int(ev_person), int(ev_day), obs)]

    if extra:
        p_inf = query_with_extra_evidence(
            G, Y, params, query_person, query_day, extra, patient_zero=pz,
        )
        Y_used = Y.copy()
        for pers, d, obs in extra:
            Y_used[d, pers] = obs
        evidence_note = (
            f"all dataset tests + hypothetical test on person {ev_person}, "
            f"day {ev_day} ({ev_result})"
        )
    else:
        p_inf = query_node_infectious(G, Y, params, query_person, query_day, patient_zero=pz)
        Y_used = Y
        evidence_note = "all observed tests in the dataset"

    P_I = get_P_I(G, Y_used, params, pz)
    seir = query_seir_posterior(G, Y_used, params, query_person, query_day, pz)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric(
        f"P(person {query_person} infected on day {query_day} | evidence)",
        f"{p_inf:.1%}",
    )
    m2.metric("P(NOT infected)", f"{1 - p_inf:.1%}")

    st.caption(f"Evidence used: {evidence_note}")

    st.latex(
        rf"P(X_{{{query_person}}}^{{{query_day}}} = \text{{Infected}} \mid \text{{observations}}) "
        rf"= {p_inf:.4f}"
    )

    cols = st.columns(4)
    for col, state in zip(cols, STATES):
        with col:
            st.metric(f"P({state})", f"{seir[state]:.1%}")

    st.subheader("Full posterior heatmap (Fig 5-B)")
    fig_hm = fig_posterior_heatmap(P_I, Y_used, highlight=(query_day, query_person))
    st.pyplot(fig_hm)
    plt.close(fig_hm)

    st.subheader("Network map by infection probability (Fig 5-D)")
    map_day = st.slider("Color network by day", 0, n_days - 1, query_day, key="inf_map_day")
    fig_net = fig_network_posterior(G, pos, P_I, day=map_day)
    st.pyplot(fig_net)
    plt.close(fig_net)

    st.subheader(f"P(infected) over time — person {query_person}")
    fig_ts, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(P_I[:, query_person], lw=2.5, color="#e74c3c")
    ax.axvline(query_day, color="#2ecc71", ls="--", lw=2)
    ax.scatter([query_day], [p_inf], s=120, color="#2ecc71", zorder=5, edgecolors="k")
    ax.set_xlabel("Day")
    ax.set_ylabel("P(infected | evidence)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig_ts.tight_layout()
    st.pyplot(fig_ts)
    plt.close(fig_ts)

    st.caption(f"Test at query: {OBS_LABEL[Y[query_day, query_person]]} · Index case: person {pz}")

else:
    st.info("Enter a person and day, then click **Run inference**.")
