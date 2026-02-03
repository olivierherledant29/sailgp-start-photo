import streamlit as st
import pandas as pd
import numpy as np

if "size_buffer_BDY_m" not in st.session_state:
    st.session_state["size_buffer_BDY_m"] = 15.0


from app_start_photo3 import render_start_photo
from start_aid.embedded import render_start_aid

st.set_page_config(page_title="Replay", layout="wide")

out_main = render_start_photo(
    page_title="SailGP – Replay (RACE_START_COUNT FRA)",
    mode_override="Replay",
)

cross_df = out_main.get("cross_df")


def _fmt_line(boat_code: str, label: str):
    if cross_df is None or cross_df.empty:
        return f"{label}: pas de croisement intersection détecté"

    r = cross_df[cross_df["boat"] == boat_code]
    if r.empty:
        return f"{label}: pas de croisement intersection détecté"

    ttk = pd.to_numeric(r["ttk_s"].iloc[0], errors="coerce")
    d_pin = pd.to_numeric(r["d_pin_m"].iloc[0], errors="coerce")
    bsp = pd.to_numeric(r["bsp_kmph"].iloc[0], errors="coerce")

    parts = []
    if np.isfinite(ttk):
        parts.append(f"{int(round(float(ttk)))} TTK")
    if np.isfinite(d_pin):
        parts.append(f"{int(round(float(d_pin)))}m d_pin")
    if np.isfinite(bsp):
        parts.append(f"{int(round(float(bsp)))} kmph BSP")

    if not parts:
        return f"{label}: pas de croisement intersection détecté"

    return f"{label}: " + " / ".join(parts)

boat_ref = out_main.get("boat_ref", "FRA")

line_ref = _fmt_line(boat_ref, boat_ref)
line_avg = _fmt_line("AVG_fleet", "fleet_avg")


st.markdown(
    f"""
    <div style="margin:6px 0 10px 0;">
        <span style="color:#0064FF;font-weight:600">{line_ref}</span><br/>
        <span style="color:#DC1E1E;font-weight:600">{line_avg}</span>

  
    </div>
    """,
    unsafe_allow_html=True,
)

# Widgets Start Aid en sidebar
with st.sidebar:
    st.markdown("---")
    deck_aid, out_aid = render_start_aid(
        out_main.get("boundary"),
        out_main.get("marks"),
    )

# Affichage Start Aid sous la carto principale
if deck_aid is not None:
    st.subheader("Start Aid")
    st.pydeck_chart(deck_aid, width="stretch")
    if out_aid and out_aid.get("results_html"):
        st.markdown(out_aid["results_html"], unsafe_allow_html=True)
