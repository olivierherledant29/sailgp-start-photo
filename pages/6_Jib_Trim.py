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


st.set_page_config(page_title="Jib Trim", layout="wide")
st.title("Jib Trim")


REF_BOAT = "FRA"

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

JIB_BUTTON_CHANNELS = [
    "BTN_GD_P_FB_JIB_SHEET_IN_unk",
    "BTN_GD_P_FB_JIB_SHEET_OUT_unk",
    "BTN_GD_P_JIB_CUN_IN",
    "BTN_GD_P_JIB_CUN_OUT",
    "BTN_GD_P_JIB_LEAD_IN",
    "BTN_GD_P_JIB_LEAD_OUT",
    "BTN_GD_S_FB_JIB_SHEET_IN",
    "BTN_GD_S_FB_JIB_SHEET_OUT",
    "BTN_GD_S_JIB_CUN_IN",
    "BTN_GD_S_JIB_CUN_OUT",
    "BTN_GD_S_JIB_LEAD_IN",
    "BTN_GD_S_JIB_LEAD_OUT",
    "BTN_WT_P_FB_JIB_SHEET_IN_unk",
    "BTN_WT_P_FB_JIB_SHEET_OUT_unk",
    "BTN_WT_S_FB_JIB_SHEET_IN",
    "BTN_WT_S_FB_JIB_SHEET_OUT",
]

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
    *JIB_BUTTON_CHANNELS,
]


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

    for col in JIB_CHANNELS:
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
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _button_side(channel: str) -> str:
    if "_P_" in channel:
        return "P"
    if "_S_" in channel:
        return "S"
    return "Other"


def _button_color(channel: str) -> str:
    if _button_side(channel) == "P":
        return "red"
    if _button_side(channel) == "S":
        return "green"
    return "gray"


def _short_button_label(channel: str) -> str:
    return (
        channel.replace("BTN_", "")
        .replace("_unk", "")
        .replace("_JIB_", "_")
        .replace("FB_", "")
    )


def _plot_jib_buttons_timeseries(df: pd.DataFrame):
    d_ref = df[df["boat"].astype(str) == REF_BOAT].copy()
    cols = [c for c in JIB_BUTTON_CHANNELS if c in d_ref.columns]

    if d_ref.empty or not cols:
        st.info(f"Aucun channel BTN jib disponible pour {REF_BOAT}.")
        return

    d = d_ref[["time_utc", "boat", *cols]].copy()

    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d_long = d.melt(
        id_vars=["time_utc", "boat"],
        value_vars=cols,
        var_name="channel",
        value_name="value",
    ).dropna(subset=["time_utc", "boat", "channel", "value"])

    d_long = d_long[d_long["value"] > 0]

    if d_long.empty:
        st.info(f"Aucun appui bouton jib détecté pour {REF_BOAT} sur cette plage.")
        return

    channel_y = {ch: i + 1 for i, ch in enumerate(cols)}
    d_long["y_pos"] = d_long["channel"].map(channel_y)

    fig = go.Figure()

    for channel, g in d_long.groupby("channel", sort=False):
        fig.add_trace(
            go.Scatter(
                x=g["time_utc"],
                y=g["y_pos"],
                mode="markers",
                name=_short_button_label(channel),
                marker=dict(size=9, color=_button_color(channel), symbol="circle"),
                customdata=np.stack(
                    [
                        g["channel"].astype(str),
                        g["value"].astype(float),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "time=%{x}<br>"
                    "boat=FRA<br>"
                    "channel=%{customdata[0]}<br>"
                    "value=%{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Jib button inputs – {REF_BOAT}",
        height=640,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Time UTC",
        yaxis_title="Button channel",
        yaxis=dict(
            tickmode="array",
            tickvals=list(channel_y.values()),
            ticktext=[_short_button_label(ch) for ch in cols],
        ),
        legend_title_text="Button",
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


cfg = get_cfg()

RIO_START = _utc_dt(2026, 4, 12, 18, 33, 0)
RIO_STOP = _utc_dt(2026, 4, 12, 18, 35, 0)

with st.sidebar:
    st.header("Jib Trim controls")

    time_mode = st.radio(
    "Plage de temps",
    ["Time range", "Last X minutes"],
    index=1,
    )

    if time_mode == "Time range":
        st.caption("Default : Rio data test")

        start_date = st.date_input("Start date UTC", value=RIO_START.date())
        start_time_min = st.time_input(
            "Start time UTC",
            value=RIO_START.time().replace(second=0),
            step=timedelta(minutes=1),
        )
        start_second = st.number_input(
            "Start seconds UTC", min_value=0, max_value=59, value=RIO_START.second
        )

        stop_date = st.date_input("Stop date UTC", value=RIO_STOP.date())
        stop_time_min = st.time_input(
            "Stop time UTC",
            value=RIO_STOP.time().replace(second=0),
            step=timedelta(minutes=1),
        )
        stop_second = st.number_input(
            "Stop seconds UTC", min_value=0, max_value=59, value=RIO_STOP.second
        )

        start_utc = _combine_utc(
            start_date, start_time_min.replace(second=int(start_second))
        )
        stop_utc = _combine_utc(
            stop_date, stop_time_min.replace(second=int(stop_second))
        )

    else:
        last_minutes = st.slider("Last X minutes", 1, 40, 10)
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

    mode_twa = st.radio("TWA filter", ["All", "UW only", "DW only"], index=0)
    st.caption("UW : 35 < abs(TWA) < 70")
    st.caption("DW : 110 < abs(TWA) < 165")

    bsp_min = st.slider("BSP mini", 0, 80, 0, step=1)
    yaw_rate_abs_max = st.slider("Yaw rate max |deg/s|", 0, 40, 40, step=1)
    vmg_target_pct_min = st.slider("Target VMG % min", 0, 120, 0, step=1)

    st.markdown("---")
    color_mode = st.radio("Coloration des points", ["Team", "% VMG target"], index=0)


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


st.markdown("---")
st.subheader("Reference boat time series")
_plot_twa_bsp_timeseries(df_raw)

st.markdown("---")
st.subheader("Jib button inputs")
_plot_jib_buttons_timeseries(df_raw)