import streamlit as st

from boundary_shared import sidebar_boundary_uploader
from start_aid.embedded import render_start_aid

st.set_page_config(page_title="Start Aid", layout="wide")

st.title("Start Aid (XML only)")

with st.sidebar:
    boundary_df = sidebar_boundary_uploader()

# Marks are parsed from the same XML and stored in session_state by sidebar_boundary_uploader()
marks_df = st.session_state.get("marks_df", None)

deck, out = render_start_aid(
    boundary_df=boundary_df,
    marks_df=marks_df,
)

if deck is not None:
    st.pydeck_chart(deck, width="stretch")
    if out and out.get("results_html"):
        st.markdown(out["results_html"], unsafe_allow_html=True)
