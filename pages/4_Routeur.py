import subprocess
import streamlit as st

from boundary_shared import sidebar_boundary_uploader
from routeur.embedded import render_routeur_simplifie

st.set_page_config(page_title="Routeur", layout="wide")


def _show_deploy_debug():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        st.sidebar.caption(f"Git SHA: {sha}")
    except Exception as e:
        st.sidebar.caption(f"Git SHA unavailable: {e}")


def _render_vmg_info(out: dict):
    info = out.get("vmg_info", []) or []
    for it in info:
        st.caption(
            f"VMG {it['leg']} — {it['group']} : "
            f"TWA_vmg={float(it['TWA']):.0f}° ; BSP_vmg={float(it['BSP']):.1f} km/h"
        )


_show_deploy_debug()

st.title("Routeur")

with st.sidebar:
    boundary_df = sidebar_boundary_uploader()

marks_df = st.session_state.get("marks_df", None)

decks, outs = render_routeur_simplifie(boundary_df=boundary_df, marks_df=marks_df)
if not decks or decks.get("deck1") is None:
    st.stop()

MAP_WIDTH = 1200
MAP_HEIGHT = 700

st.subheader("First DW from M1")
st.pydeck_chart(decks["deck1"], width=MAP_WIDTH, height=MAP_HEIGHT)
_render_vmg_info(outs.get("out1", {}))

st.subheader("FULL UPWIND")
st.pydeck_chart(decks["deck2"], width=MAP_WIDTH, height=MAP_HEIGHT)
_render_vmg_info(outs.get("out2", {}))

st.subheader("FULL DOWNWIND")
st.pydeck_chart(decks["deck3"], width=MAP_WIDTH, height=MAP_HEIGHT)
_render_vmg_info(outs.get("out3", {}))