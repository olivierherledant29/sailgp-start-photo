from __future__ import annotations

from datetime import datetime, date, time, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from influx_io import (
    ALL_BOATS,
    get_cfg,
    load_channels_timeseries,
)


st.set_page_config(page_title="FC Data", layout="wide")
st.title("FC Data")


# -----------------------
# Channels
# -----------------------
REF_BOAT = "FRA"

CH_BSP = "BOAT_SPEED_km_h_1"
CH_TWA = "TWA_MHU_SGP_deg"
CH_VMG = "VMG_km_h_1"
CH_TARGET_VMG = "TARG_VMG_km_h_1"
CH_YAW_RATE = "RATE_YAW_deg_s_1"

CH_CANT_PORT = "ANGLE_DB_CANT_P_deg"
CH_CANT_STBD = "ANGLE_DB_CANT_S_deg"
CH_RIDE_HEIGHT = "LENGTH_RH_BOW_mm"
CH_RUDDER_AVG = "ANGLE_RUD_AVG_deg"

FC_CHANNELS = [
    CH_BSP,
    CH_TWA,
    CH_VMG,
    CH_TARGET_VMG,
    CH_YAW_RATE,
    CH_CANT_PORT,
    CH_CANT_STBD,
    CH_RIDE_HEIGHT,
    CH_RUDDER_AVG,
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


def _combine_utc(d: date, t: time) -> datetime:
    return datetime.combine(d, t).replace(tzinfo=timezone.utc)


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

    for col in FC_CHANNELS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _add_vmg_target_pct(out)

    if CH_BSP in out.columns:
        out = out[_safe_num(out, CH_BSP) >= float(bsp_min)]

    if CH_YAW_RATE in out.columns:
        out = out[_safe_num(out, CH_YAW_RATE).abs() <= float(yaw_rate_abs_max)]

    out = out[_safe_num(out, "VMG_TARGET_pct") >= float(vmg_target_pct_min)]

    if mode_twa != "All" and CH_TWA in out.columns:
        twa_abs = _safe_num(out, CH_TWA).abs()

        if mode_twa == "UW only":
            out = out[(twa_abs > 35.0) & (twa_abs < 70.0)]

        elif mode_twa == "DW only":
            out = out[(twa_abs > 110.0) & (twa_abs < 165.0)]

    return out.reset_index(drop=True)


def _scatter(df: pd.DataFrame, x: str, y: str, title: str, color_mode: str):
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
            range_color=[0, 150],
            hover_data=hover_cols,
            title=title,
            opacity=0.72,
        )
        fig.update_coloraxes(colorbar_title="% VMG target", cmin=0, cmax=150)

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="Team",
    )

    st.plotly_chart(fig, use_container_width=True)


def _plot_twa_bsp_timeseries(df: pd.DataFrame):
    d = df[df["boat"].astype(str) == REF_BOAT].copy()

    if d.empty:
        st.info(f"Aucune donnée TWA/BSP disponible pour {REF_BOAT}.")
        return

    if CH_TWA not in d.columns or CH_BSP not in d.columns:
        st.info(f"Channels TWA/BSP manquants pour {REF_BOAT}.")
        return

    d[CH_TWA] = pd.to_numeric(d[CH_TWA], errors="coerce")
    d[CH_BSP] = pd.to_numeric(d[CH_BSP], errors="coerce")
    d = d.dropna(subset=["time_utc", CH_TWA, CH_BSP])

    if d.empty:
        st.info(f"Aucune donnée TWA/BSP exploitable pour {REF_BOAT}.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d["time_utc"],
            y=d[CH_TWA],
            mode="lines",
            name="TWA deg",
            yaxis="y1",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["time_utc"],
            y=d[CH_BSP],
            mode="lines",
            name="BSP km/h",
            yaxis="y2",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        title=f"TWA / BSP time series – {REF_BOAT}",
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(title="Time UTC"),
        yaxis=dict(title="TWA deg", side="left"),
        yaxis2=dict(
            title="BSP km/h",
            side="right",
            overlaying="y",
            showgrid=False,
        ),
        legend_title_text="Channel",
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# Sidebar controls
# -----------------------
cfg = get_cfg()

DEFAULT_START = _utc_dt(2026, 4, 12, 18, 33, 0)
DEFAULT_STOP = _utc_dt(2026, 4, 12, 18, 35, 0)

with st.sidebar:
    st.header("FC Data controls")

    time_mode = st.radio(
        "Plage de temps",
        ["Time range", "Last X minutes"],
        index=1,
    )

    if time_mode == "Time range":
        st.caption("Default : Rio data test")

        start_date = st.date_input(
            "Start date UTC",
            value=DEFAULT_START.date(),
            key="fc_start_date",
        )
        start_time_min = st.time_input(
            "Start time UTC",
            value=DEFAULT_START.time().replace(second=0),
            step=timedelta(minutes=1),
            key="fc_start_time_min",
        )
        start_second = st.number_input(
            "Start seconds UTC",
            min_value=0,
            max_value=59,
            value=DEFAULT_START.second,
            step=1,
            key="fc_start_second",
        )

        stop_date = st.date_input(
            "Stop date UTC",
            value=DEFAULT_STOP.date(),
            key="fc_stop_date",
        )
        stop_time_min = st.time_input(
            "Stop time UTC",
            value=DEFAULT_STOP.time().replace(second=0),
            step=timedelta(minutes=1),
            key="fc_stop_time_min",
        )
        stop_second = st.number_input(
            "Stop seconds UTC",
            min_value=0,
            max_value=59,
            value=DEFAULT_STOP.second,
            step=1,
            key="fc_stop_second",
        )

        start_utc = _combine_utc(
            start_date,
            start_time_min.replace(second=int(start_second)),
        )
        stop_utc = _combine_utc(
            stop_date,
            stop_time_min.replace(second=int(stop_second)),
        )

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

    if stop_utc <= start_utc:
        st.error("Stop UTC doit être après Start UTC.")
        st.stop()

    minutes = (stop_utc - start_utc).total_seconds() / 60.0
    every = "5s" if minutes > 10 else "1s"

    st.caption(f"Start UTC : {start_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Stop UTC : {stop_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Agrégation Influx : {every}")
    st.caption(f"Bateau référence toujours inclus : {REF_BOAT}")

    selectable_boats = [b for b in ALL_BOATS if b != REF_BOAT]
    default_other_boats = [
        b for b in ["AUS", "GBR", "ESP", "SWE"] if b in selectable_boats
    ]

    selected_other_boats = st.multiselect(
        "Teams supplémentaires",
        selectable_boats,
        default=default_other_boats,
    )

    boats = [REF_BOAT] + selected_other_boats

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
with st.spinner("Chargement des données FC..."):
    df_raw = load_channels_timeseries(
        cfg=cfg,
        boats=boats,
        channels=FC_CHANNELS,
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
# Scatter plots
# -----------------------
p1, p2 = st.columns(2)

with p1:
    _scatter(
        df,
        x=CH_BSP,
        y=CH_CANT_PORT,
        title="Cant port vs BSP",
        color_mode=color_mode,
    )

with p2:
    _scatter(
        df,
        x=CH_BSP,
        y=CH_CANT_STBD,
        title="Cant stbd vs BSP",
        color_mode=color_mode,
    )

p3, p4 = st.columns(2)

with p3:
    _scatter(
        df,
        x=CH_BSP,
        y=CH_RUDDER_AVG,
        title="Rudder AVG vs BSP",
        color_mode=color_mode,
    )

with p4:
    _scatter(
        df,
        x=CH_BSP,
        y=CH_RIDE_HEIGHT,
        title="Ride height vs BSP",
        color_mode=color_mode,
    )

st.subheader("Rudder AVG vs TWA")

_scatter(
    df,
    x=CH_TWA,
    y=CH_RUDDER_AVG,
    title="Rudder AVG vs TWA",
    color_mode=color_mode,
)


# -----------------------
# FRA time series
# -----------------------
st.markdown("---")
st.subheader("Reference boat time series")

_plot_twa_bsp_timeseries(df_raw)