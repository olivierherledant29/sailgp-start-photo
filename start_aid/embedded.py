from __future__ import annotations

import streamlit as st
import pandas as pd

from start_aid.geo import make_context_from_boundary, to_xy_marks_and_polys, compute_PI_xy
from start_aid.model import compute_all_geometry_and_times
from start_aid.viz import build_deck


def boundary_df_to_latlon(boundary_df: pd.DataFrame):
    if boundary_df is None or boundary_df.empty:
        return []
    df = boundary_df.dropna(subset=["lat", "lon"]).copy()
    return [(float(r.lat), float(r.lon)) for _, r in df.iterrows()]


def marks_df_to_marks_ll(marks_df: pd.DataFrame):
    out = {}
    if marks_df is None or marks_df.empty:
        return out

    df = marks_df.dropna(subset=["mark", "lat", "lon"]).copy()

    for _, r in df.iterrows():
        mark = str(r["mark"])
        lat = float(r["lat"])
        lon = float(r["lon"])

        if abs(lat) > 90 or abs(lon) > 180:
            lat *= 1e-7
            lon *= 1e-7

        out[mark] = (lat, lon)

    return out


def render_start_aid(boundary_df: pd.DataFrame, marks_df: pd.DataFrame):
    st.markdown(
        "<div style='padding:8px;border-radius:8px;border:1px solid #2E7D32;background:#E8F5E9;'>"
        "<b>Start Aid</b></div>",
        unsafe_allow_html=True,
    )

    # ----------------------------
    # NOUVEAU : buffer boundary
    # ----------------------------
    st.subheader("Boundary")
    boundary_buffer_m = st.number_input(
        "Buffer boundary (m)",
        min_value=0.0,
        max_value=200.0,
        value=15.0,
        step=1.0,
        help="Distance de sécurité autour de la boundary",
    )

    PI_m = st.number_input("PI (m)", min_value=-500.0, max_value=500.0, value=60.0, step=1.0)
    TWD = st.number_input("TWD (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

    st.subheader("Angles")
    TWA_port = st.number_input("TWA_port (°)", min_value=40.0, max_value=160.0, value=60.0, step=1.0)
    TWA_UW = st.number_input("TWA_UW (°)", min_value=30.0, max_value=80.0, value=45.0, step=1.0)

    st.subheader("Start points")
    X_percent = st.slider(
        "SP : X% depuis SL2 vers SL1 (0=SL2, 100=SL1)",
        min_value=0, max_value=100, value=50, step=1
    )

    st.subheader("Vitesses & temps")
    BSP1 = st.number_input("BSP_approche_BAB (km/h)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    BSP2 = st.number_input("BSP_retour_TRIB (km/h)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    M_lost = st.number_input("M_lost (s)", min_value=0.0, max_value=120.0, value=12.0, step=1.0)
    TTS_intersection = st.number_input("TTS_intersection (s)", min_value=0.0, max_value=300.0, value=60.0, step=1.0)

    boundary_latlon = boundary_df_to_latlon(boundary_df)
    marks_ll = marks_df_to_marks_ll(marks_df)

    if not boundary_latlon:
        st.info("Start Aid : boundary non chargée (XML manquant).")
        return None, None

    if "SL1" not in marks_ll or "SL2" not in marks_ll:
        st.info("Start Aid : SL1 / SL2 indisponibles dans les marques Influx.")
        return None, None

    ctx = make_context_from_boundary(boundary_latlon)

    # ⬇️ buffer dynamique ici
    geom = to_xy_marks_and_polys(
        ctx,
        marks_ll,
        boundary_latlon,
        boundary_buffer_m,
    )

    PI_xy = compute_PI_xy(geom["SL1_xy"], geom["SL2_xy"], PI_m)

    out = compute_all_geometry_and_times(
        ctx=ctx,
        geom=geom,
        PI_xy=PI_xy,
        TWD=TWD,
        TWA_port=TWA_port,
        TWA_UW=TWA_UW,
        BSP_approche_BAB=BSP1,
        BSP_retour_TRIB=BSP2,
        M_lost=M_lost,
        X_percent=X_percent,
        TTS_intersection=TTS_intersection,
    )

    deck = build_deck(ctx, geom, PI_xy, out)
    return deck, out
