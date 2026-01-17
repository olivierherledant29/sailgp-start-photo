import streamlit as st

from app_start_photo3 import render_start_photo
from start_aid.embedded import render_start_aid

st.set_page_config(page_title="Replay", layout="wide")

out_main = render_start_photo(page_title="SailGP – Replay (RACE_START_COUNT FRA)", mode_override="Replay")

# Widgets Start Aid en sidebar (après ceux du module principal)
with st.sidebar:
    st.markdown("---")
    deck_aid, out_aid = render_start_aid(out_main.get("boundary"), out_main.get("marks"))

# Affichage Start Aid sous la carto principale
if deck_aid is not None:
    st.subheader("Start Aid")
    st.pydeck_chart(deck_aid, width="stretch")
    if out_aid and out_aid.get("results_html"):
        st.markdown(out_aid["results_html"], unsafe_allow_html=True)
