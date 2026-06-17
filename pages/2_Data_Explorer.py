"""Page 2 — Data Explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from app_shared import (
    OBS_LABEL,
    fig_observation_matrix,
    plotly_network,
    render_sidebar,
)
from src.config import OBS_POS

G, Y, X, counts, pz, meta, pos, params = render_sidebar()

st.title("Data Explorer")
st.markdown(f"**{meta['label']}** — explore the contact network and observation matrix.")

n_people, n_days = G.number_of_nodes(), Y.shape[0]

c1, c2, c3 = st.columns(3)
with c1:
    focus_person = st.slider("Focus person", 0, n_people - 1, int(pz))
with c2:
    focus_day = st.slider("Focus day", 0, n_days - 1, min(10, n_days - 1))
with c3:
    color_by = st.selectbox("Network color", ["Uniform", "Tests on focus day", "Positive tests (max)"])

tab_net, tab_obs = st.tabs(["Contact network", "Observation matrix"])

with tab_net:
  st.subheader("Interactive contact network")
  if color_by == "Uniform":
      colors = ["#a8d4f0"] * n_people
  elif color_by == "Tests on focus day":
      from src.config import OBS_NEG, OBS_MISSING
      colors = []
      for i in range(n_people):
          y = Y[focus_day, i]
          if y == OBS_POS:
              colors.append(1.0)
          elif y == OBS_NEG:
              colors.append(0.3)
          else:
              colors.append(0.0)
  else:
      colors = [(Y[:, i] == OBS_POS).any() * 1.0 for i in range(n_people)]

  fig_p = plotly_network(
      G, pos, node_colors=colors, highlight=focus_person,
      colorbar_title="test signal" if color_by != "Uniform" else "",
  )
  st.plotly_chart(fig_p, use_container_width=True)
  st.caption(f"Highlighted: person **{focus_person}** · {G.degree(focus_person)} contacts")

with tab_obs:
  st.subheader("Observation matrix (Fig 1-D)")
  d1, d2, p1, p2 = st.columns(4)
  with d1:
      day_start = st.number_input("Day from", 0, n_days - 1, 0)
  with d2:
      day_end = st.number_input("Day to", day_start + 1, n_days, n_days)
  with p1:
      person_start = st.number_input("Person from", 0, n_people - 1, 0)
  with p2:
      person_end = st.number_input("Person to", person_start + 1, n_people, n_people)

  fig = fig_observation_matrix(Y, day_start, day_end, person_start, person_end)
  st.pyplot(fig)
  plt.close(fig)

  st.markdown("#### Observations at selected (person, day)")
  obs_val = Y[focus_day, focus_person]
  st.metric(
      f"Person {focus_person}, day {focus_day}",
      OBS_LABEL[obs_val],
  )

  pos_tests = [
      {"person": i, "day": t}
      for t in range(Y.shape[0])
      for i in range(Y.shape[1])
      if Y[t, i] == OBS_POS
  ]
  if pos_tests:
      st.dataframe(pos_tests, use_container_width=True, height=200)
