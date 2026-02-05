import streamlit as st

from boundary_shared import sidebar_boundary_uploader
from routeur.embedded import render_routeur_simplifie

st.set_page_config(page_title="Routeur", layout="wide")
st.title("Routeur (XML + polaires only)")

with st.sidebar:
    boundary_df = sidebar_boundary_uploader()

# marks_df est fourni par sidebar_boundary_uploader() via st.session_state
marks_df = st.session_state.get("marks_df", None)

deck, out = render_routeur_simplifie(
    boundary_df=boundary_df,
    marks_df=marks_df,
)

if deck is not None:
    st.pydeck_chart(deck, width="stretch")
else:
    st.info("Charge un XML dans la sidebar pour afficher la carto et les résultats.")
