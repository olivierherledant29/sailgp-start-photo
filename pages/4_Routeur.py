import streamlit as st

from boundary_shared import sidebar_boundary_uploader
from routeur.embedded import render_routeur_simplifie

st.set_page_config(page_title="Routeur", layout="wide")

st.title("Routeur (XML only)")

with st.sidebar:
    boundary_df = sidebar_boundary_uploader()

marks_df = st.session_state.get("marks_df", None)

decks, outs = render_routeur_simplifie(boundary_df=boundary_df, marks_df=marks_df)
if not decks or decks.get("deck1") is None:
    st.stop()


def _render_vmg_info(out: dict):
    info = out.get("vmg_info", []) or []
    for it in info:
        st.caption(
            f"VMG {it['leg']} — {it['group']} : "
            f"TWA_vmg={float(it['TWA']):.0f}° ; BSP_vmg={float(it['BSP']):.1f} km/h"
        )


st.subheader("First DW from M1")
st.pydeck_chart(decks["deck1"], width="stretch")
_render_vmg_info(outs.get("out1", {}))

st.subheader("FULL UPWIND")
st.pydeck_chart(decks["deck2"], width="stretch")
_render_vmg_info(outs.get("out2", {}))

st.subheader("FULL DOWNWIND")
st.pydeck_chart(decks["deck3"], width="stretch")
_render_vmg_info(outs.get("out3", {}))
