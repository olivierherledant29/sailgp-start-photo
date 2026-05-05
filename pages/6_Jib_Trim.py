from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from influx_io import (
    ALL_BOATS,
    get_cfg,
    load_channels_timeseries,
)


st.set_page_config(page_title="Jib Trim", layout="wide")
st.title("Jib Trim")


# -----------------------
# Channels
# -----------------------
CH_BSP = "BOAT_SPEED_km_h_1"
CH_TWA = "TWA_MHU_SGP_deg"

CH_LEEWAY = "LEEWAY_COR_deg"
CH_JIB_LEAD = "PER_JIB_LEAD_pct"
CH_JIB_LEAD_ANGLE = "ANGLE_JIB_SHT_deg"
CH_LOAD_JIB_SHEET = "LOAD_JIB_SHEET_kgf"
CH_LOAD_JIB_CUNNO = "LOAD_JIB_CUNNO_kgf"
CH_PRES_JIB_CUNNO = "PRES_JIB_CUNNO_bar"
CH_PRES_JIB_SHEET = "PRES_JIB_SHT_bar"
CH_VMG = "VMG_km_h_1"
CH_TARGET_VMG = "TARG_VMG_km_h_1"
CH_YAW_RATE = "RATE_YAW_deg_s_1"

JIB_CHANNELS = [
    CH_BSP,
    CH_TWA,
    CH_LEEWAY,
    CH_JIB_LEAD,
    CH_JIB_LEAD_ANGLE,
    CH_LOAD_JIB_SHEET,
    CH_LOAD_JIB_CUNNO,
    CH_PRES_JIB_CUNNO,
    CH_PRES_JIB_SHEET,
    CH_VMG,
    CH_TARGET_VMG,
    CH_YAW_RATE,
]


# -----------------------
# Colors
# -----------------------
TEAM_COLORS = {
    "FRA": "#0064FF",
    "AUS": "#00A651",
    "ESP": "#FF8C00",
    "SWE": "#FFD400",
    "GBR": "#7A3DB8",
    "USA": "#DC1E1E",
    "NZL": "#111111",
    "DEN": "#B00020",
    "CAN": "#E31B23",
    "GER": "#555555",
    "ITA": "#00AEEF",
    "SUI": "#A0A0A0",
    "BRA": "#009739",
}


def _utc_dt(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _add_vmg_target_pct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vmg = _safe_num(out, CH_VMG)
    target = _safe_num(out, CH_TARGET_VMG)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["VMG_TARGET_pct"] = np.where(
            np.abs(target) > 1e-9,
            100.0 * vmg / target,
            np.nan,
        )

    out["VMG_TARGET_pct"] = pd.to_numeric(out["VMG_TARGET_pct"], errors="coerce")
    return out


def _filter_data(
    df: pd.DataFrame,
    mode_twa: str,
    bsp_min: float,
    yaw_rate_abs_max: float,
    vmg_target_pct_min: float,
) -> pd.DataFrame:
    out = df.copy()

    for col in JIB_CHANNELS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _add_vmg_target_pct(out)

    # BSP min
    if CH_BSP in out.columns:
        out = out[_safe_num(out, CH_BSP) >= float(bsp_min)]

    # Yaw rate max en absolu
    if CH_YAW_RATE in out.columns:
        out = out[_safe_num(out, CH_YAW_RATE).abs() <= float(yaw_rate_abs_max)]

    # VMG target %
    out = out[_safe_num(out, "VMG_TARGET_pct") >= float(vmg_target_pct_min)]

    # TWA filter
    if mode_twa != "All" and CH_TWA in out.columns:
        twa_abs = _safe_num(out, CH_TWA).abs()

        if mode_twa == "UW only":
            out = out[(twa_abs > 35.0) & (twa_abs < 70.0)]

        elif mode_twa == "DW only":
            out = out[(twa_abs > 110.0) & (twa_abs < 165.0)]

    return out.reset_index(drop=True)


def _scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_mode: str,
):
    d = df.dropna(subset=[x, y]).copy()

    if d.empty:
        st.info(f"Aucune donnée disponible pour : {title}")
        return

    hover_cols = [
        "time_utc",
        "boat",
        CH_BSP,
        CH_TWA,
        CH_YAW_RATE,
        "VMG_TARGET_pct",
        CH_VMG,
        CH_TARGET_VMG,
    ]

    hover_cols = [c for c in hover_cols if c in d.columns]

    if color_mode == "Team":
        fig = px.scatter(
            d,
            x=x,
            y=y,
            color="boat",
            color_discrete_map=TEAM_COLORS,
            hover_data=hover_cols,
            title=title,
            opacity=0.72,
        )
    else:
        fig = px.scatter(
            d,
            x=x,
            y=y,
            color="VMG_TARGET_pct",
            hover_data=hover_cols,
            title=title,
            opacity=0.72,
        )
        fig.update_coloraxes(colorbar_title="% VMG target")

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="Team",
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# Sidebar controls
# -----------------------
cfg = get_cfg()

with st.sidebar:
    st.header("Jib Trim controls")

    time_mode = st.radio(
        "Plage de temps",
        ["Test data Rio", "Last X minutes"],
        index=0,
    )

    if time_mode == "Test data Rio":
        start_utc = _utc_dt(2026, 4, 12, 19, 23, 0)
        stop_utc = _utc_dt(2026, 4, 12, 19, 25, 0)
        minutes = int((stop_utc - start_utc).total_seconds() / 60)
        st.caption("Test data Rio : 12/04/2026 19:23 → 19:25 UTC")
    else:
        last_minutes = st.slider(
            "Last X minutes",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
        )
        stop_utc = datetime.now(timezone.utc)
        start_utc = stop_utc - timedelta(minutes=int(last_minutes))
        minutes = int(last_minutes)

    every = "5s" if minutes > 10 else "1s"
    st.caption(f"Agrégation Influx : {every}")

    default_boats = [b for b in ["FRA", "AUS", "GBR", "ESP", "SWE"] if b in ALL_BOATS]

    boats = st.multiselect(
        "Teams",
        ALL_BOATS,
        default=default_boats,
    )

    st.markdown("---")
    st.subheader("Filtres data")

    mode_twa = st.radio(
        "TWA filter",
        ["All", "UW only", "DW only"],
        index=0,
    )

    st.caption("UW : 35 < abs(TWA) < 70")
    st.caption("DW : 110 < abs(TWA) < 165")

    bsp_min = st.slider(
        "BSP mini",
        min_value=0,
        max_value=80,
        value=0,
        step=1,
    )

    yaw_rate_abs_max = st.slider(
        "Yaw rate max |deg/s|",
        min_value=0,
        max_value=40,
        value=40,
        step=1,
    )

    vmg_target_pct_min = st.slider(
        "Target VMG % min",
        min_value=0,
        max_value=120,
        value=0,
        step=1,
    )

    st.markdown("---")
    color_mode = st.radio(
        "Coloration des points",
        ["Team", "% VMG target"],
        index=0,
    )


# -----------------------
# Main
# -----------------------
if not boats:
    st.warning("Sélectionne au moins un team.")
    st.stop()

with st.spinner("Chargement des données Jib Trim..."):
    df_raw = load_channels_timeseries(
        cfg=cfg,
        boats=boats,
        channels=JIB_CHANNELS,
        start_utc=start_utc,
        stop_utc=stop_utc,
        every=every,
        level_expr="strm|mdss|mdss_fast|raw",
        agg_fn="mean",
    )

if df_raw.empty:
    st.warning("Aucune donnée retournée par Influx sur cette plage.")
    st.stop()

df = _filter_data(
    df_raw,
    mode_twa=mode_twa,
    bsp_min=bsp_min,
    yaw_rate_abs_max=yaw_rate_abs_max,
    vmg_target_pct_min=vmg_target_pct_min,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Points bruts", f"{len(df_raw):,}".replace(",", " "))
c2.metric("Points filtrés", f"{len(df):,}".replace(",", " "))
c3.metric("Début UTC", start_utc.strftime("%H:%M:%S"))
c4.metric("Fin UTC", stop_utc.strftime("%H:%M:%S"))
c5.metric("Pas", every)

if df.empty:
    st.warning("Aucune donnée après filtres.")
    st.stop()

with st.expander("Aperçu data filtrée", expanded=False):
    st.dataframe(df.head(500), use_container_width=True)


# -----------------------
# Plots
# -----------------------
p1, p2 = st.columns(2)

with p1:
    _scatter(
        df,
        x=CH_LOAD_JIB_SHEET,
        y=CH_JIB_LEAD,
        title="Jib lead vs Jib sheet load",
        color_mode=color_mode,
    )

with p2:
    _scatter(
        df,
        x=CH_LOAD_JIB_CUNNO,
        y=CH_LOAD_JIB_SHEET,
        title="Jib load vs Jib cunno load",
        color_mode=color_mode,
    )

p3, p4 = st.columns(2)

with p3:
    _scatter(
        df,
        x=CH_LEEWAY,
        y=CH_JIB_LEAD,
        title="Jib lead vs Leeway",
        color_mode=color_mode,
    )

with p4:
    _scatter(
        df,
        x=CH_PRES_JIB_CUNNO,
        y=CH_PRES_JIB_SHEET,
        title="Pressure jib sheet vs Pressure jib cunno",
        color_mode=color_mode,
    )