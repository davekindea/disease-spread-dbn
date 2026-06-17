"""
PGM Disease Spread DBN — multi-page Streamlit demo.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="PGM Disease Spread DBN",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_Overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/2_Data_Explorer.py", title="Data Explorer", icon="🔍"),
    st.Page("pages/3_Model_and_Parameters.py", title="Model & Parameters", icon="⚙️"),
    st.Page("pages/4_EM_Learning.py", title="EM Learning", icon="📈"),
    st.Page("pages/5_Inference.py", title="Inference", icon="🎯"),
    st.Page("pages/6_Evaluation.py", title="Evaluation", icon="📊"),
]

pg = st.navigation(pages)
pg.run()
