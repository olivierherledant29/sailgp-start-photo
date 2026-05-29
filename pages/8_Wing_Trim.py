from __future__ import annotations

from datetime import datetime, date, time, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from influx_io import ALL_BOATS, get_cfg, load_channels_timeseries
from target_utils import (
    build_targets_for_modes,
    load_target_workbook,
    target_config_channels,
)


st.set_page_config(page_title="Wing Trim", layout="wide")
st.title("Wing Trim")


REF_BOAT = "FRA"

CH_BSP = "BOAT_SPEED_km_h_1"
CH_TWA = "TWA_MHU_SGP_deg"
CH_TWS = "TWS_MHU_SGP_km_h_1"
CH_VMG = "VMG_km_h_1"
CH_TARGET_VMG = "TARG_VMG_km_h_1"
CH_YAW_RATE = "RATE_YAW_deg_s_1"

CH_CA1_RAW = "ANGLE_CA1_deg"
CH_WING_TWIST = "ANGLE_WING_TWIST_deg"
CH_CLEW_ANGLE = "ANGLE_CLEW_deg"

COL_CA1 = "CA1"
COL_WING_TWIST = "abs_wing_twist"
COL_CLEW_ANGLE = "angle_clew"

TARGET_COLUMNS = {
    "BSP_target": 7,
    "CA1_target": 17,
    "clew_target": 19,
    "twist_target": 20,
    "TWA_target": 8,
}

TARGET_NAMES = list(TARGET_COLUMNS.keys())

WING_CHANNELS = [
    CH_BSP,
    CH_TWA,
    CH_TWS,
    CH_VMG,
    CH_TARGET_VMG,
    CH_YAW_RATE,
    CH_CA1_RAW,
    CH_WING_TWIST,
    CH_CLEW_ANGLE,
    *target_config_channels(),
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

MODE_COLORS = {
    "UW": "#0064FF",
    "DW": "#D62728",
}


def _utc_dt(y: int, m: int, d: int, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def _combine_utc(d: date, t: time) -> datetime:
    return datetime.combine(d, t).replace(tzinfo=timezone.utc)


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_CA1] = pd.to_numeric(out.get(CH_CA1_RAW), errors="coerce").abs()
    out[COL_WING_TWIST] = pd.to_numeric(out.get(CH_WING_TWIST), errors="coerce").abs()
    out[COL_CLEW_ANGLE] = pd.to_numeric(out.get(CH_CLEW_ANGLE), errors="coerce")

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


def _filter_common(df, bsp_min, yaw_rate_abs_max, vmg_target_pct_min):
    out = df.copy()

    for col in WING_CHANNELS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _add_derived_columns(out)
    out = out[_safe_num(out, CH_BSP) >= float(bsp_min)]
    out = out[_safe_num(out, CH_YAW_RATE).abs() <= float(yaw_rate_abs_max)]
    out = out[_safe_num(out, "VMG_TARGET_pct") >= float(vmg_target_pct_min)]
    return out.reset_index(drop=True)


def _filter_mode(df, mode_name):
    if df.empty or CH_TWA not in df.columns:
        return pd.DataFrame(columns=df.columns)

    twa_abs = _safe_num(df, CH_TWA).abs()

    if mode_name == "UW":
        return df[(twa_abs > 35.0) & (twa_abs < 70.0)].reset_index(drop=True)

    if mode_name == "DW":
        return df[(twa_abs > 120.0) & (twa_abs < 160.0)].reset_index(drop=True)

    return df.reset_index(drop=True)


def _target_states_label(target_overlays) -> str:
    if not target_overlays:
        return "—"
    return " + ".join(str(ov.get("state", "?")) for ov in target_overlays)


def _add_target_vline(fig, x_value, color, name):
    if np.isfinite(x_value):
        fig.add_vline(
            x=x_value,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="top",
        )


def _add_target_hline(fig, y_value, color, name):
    if np.isfinite(y_value):
        fig.add_hline(
            y=y_value,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="right",
        )


def _plot_scatter(df, x, y, title, mode_name, color_mode, target_overlays=None):
    d = df.dropna(subset=[x, y]).copy()

    if d.empty:
        st.info(f"Aucune donnée disponible pour : {mode_name} – {title}")
        return

    hover_cols = [
        "time_utc",
        "boat",
        CH_BSP,
        CH_TWA,
        CH_TWS,
        CH_YAW_RATE,
        "VMG_TARGET_pct",
        CH_VMG,
        CH_TARGET_VMG,
    ]
    hover_cols = [c for c in hover_cols if c in d.columns]

    mode_color = MODE_COLORS.get(mode_name, "#333333")
    target_label = _target_states_label(target_overlays)
    full_title = (
        f'<span style="color:{mode_color};font-weight:700">{mode_name}</span>'
        f" – {title} | targets: {target_label}"
    )

    if color_mode == "Team":
        fig = px.scatter(
            d,
            x=x,
            y=y,
            color="boat",
            color_discrete_map=TEAM_COLORS,
            hover_data=hover_cols,
            title=full_title,
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
            title=full_title,
            opacity=0.72,
        )
        fig.update_coloraxes(colorbar_title="% VMG target", cmin=0, cmax=150)

    for overlay in target_overlays or []:
        state = str(overlay.get("state", "target"))
        color = overlay.get("color", "black")
        target = overlay.get("target") or {}

        bsp_t = target.get("BSP_target", np.nan)
        ca1_t = target.get("CA1_target", np.nan)
        clew_t = target.get("clew_target", np.nan)
        twist_t = target.get("twist_target", np.nan)

        if x == CH_BSP:
            _add_target_vline(fig, bsp_t, color, f"{state} BSP")
        if x == COL_CA1:
            _add_target_vline(fig, ca1_t, color, f"{state} CA1")
        if y == COL_CA1:
            _add_target_hline(fig, ca1_t, color, f"{state} CA1")
        if x == COL_WING_TWIST:
            _add_target_vline(fig, twist_t, color, f"{state} twist")
        if y == COL_WING_TWIST:
            _add_target_hline(fig, twist_t, color, f"{state} twist")
        if x == COL_CLEW_ANGLE:
            _add_target_vline(fig, clew_t, color, f"{state} clew")
        if y == COL_CLEW_ANGLE:
            _add_target_hline(fig, clew_t, color, f"{state} clew")

        if x == COL_CA1 and y == COL_WING_TWIST and np.isfinite(ca1_t) and np.isfinite(twist_t):
            fig.add_trace(
                go.Scatter(
                    x=[ca1_t],
                    y=[twist_t],
                    mode="markers",
                    name=f"{mode_name} {state} target",
                    marker=dict(size=16, color=color, symbol="x"),
                    hovertemplate=(
                        f"{state}<br>"
                        f"CA1 target={ca1_t:.2f}<br>"
                        f"Abs twist target={twist_t:.2f}<extra></extra>"
                    ),
                )
            )

    fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        title=dict(x=0.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _first_target(target_overlays):
    if not target_overlays:
        return None
    return target_overlays[0].get("target")


def _render_mode_section(df_common, mode_name, color_mode, target_overlays):
    df_mode = _filter_mode(df_common, mode_name)
    mode_color = MODE_COLORS.get(mode_name, "#333333")
    first_target = _first_target(target_overlays)

    st.markdown(
        f'<h2 style="color:{mode_color}; margin-top:30px;">{mode_name}</h2>',
        unsafe_allow_html=True,
    )

    if target_overlays:
        st.caption(f"Targets affichées : {_target_states_label(target_overlays)}")

    if df_mode.empty:
        st.info(f"Aucune donnée pour le mode {mode_name}.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{mode_name} points", f"{len(df_mode):,}".replace(",", " "))
    c2.metric(f"{mode_name} abs TWA min", f"{_safe_num(df_mode, CH_TWA).abs().min():.1f}")
    c3.metric(f"{mode_name} abs TWA max", f"{_safe_num(df_mode, CH_TWA).abs().max():.1f}")
    c4.metric(
        f"{mode_name} target BSP",
        "—"
        if not first_target or not np.isfinite(first_target.get("BSP_target", np.nan))
        else f"{first_target['BSP_target']:.1f}",
    )

    p1, p2 = st.columns(2)
    with p1:
        _plot_scatter(
            df_mode,
            COL_CA1,
            COL_WING_TWIST,
            "Abs wing twist vs CA1",
            mode_name,
            color_mode,
            target_overlays,
        )
    with p2:
        _plot_scatter(
            df_mode,
            CH_BSP,
            COL_CLEW_ANGLE,
            "Angle clew vs BSP",
            mode_name,
            color_mode,
            target_overlays,
        )

    p3, p4 = st.columns(2)
    with p3:
        _plot_scatter(
            df_mode,
            CH_BSP,
            COL_WING_TWIST,
            "Abs wing twist vs BSP",
            mode_name,
            color_mode,
            target_overlays,
        )
    with p4:
        _plot_scatter(
            df_mode,
            CH_BSP,
            COL_CA1,
            "CA1 vs BSP",
            mode_name,
            color_mode,
            target_overlays,
        )


def _mode_segments_for_ref(df, mode_name):
    d = df[df["boat"].astype(str) == REF_BOAT].copy()

    if d.empty or CH_TWA not in d.columns:
        return []

    d = d.sort_values("time_utc").dropna(subset=["time_utc", CH_TWA])
    if d.empty:
        return []

    twa_abs = pd.to_numeric(d[CH_TWA], errors="coerce").abs()

    if mode_name == "UW":
        mask = (twa_abs > 35.0) & (twa_abs < 70.0)
    elif mode_name == "DW":
        mask = (twa_abs > 120.0) & (twa_abs < 160.0)
    else:
        return []

    d["mask"] = mask.fillna(False).astype(bool)
    d["grp"] = (d["mask"] != d["mask"].shift()).cumsum()

    return [
        (g["time_utc"].min(), g["time_utc"].max())
        for _, g in d[d["mask"]].groupby("grp")
        if pd.notna(g["time_utc"].min()) and pd.notna(g["time_utc"].max())
    ]


def _plot_twa_bsp_timeseries(df):
    d = df[df["boat"].astype(str) == REF_BOAT].copy()

    if d.empty or CH_TWA not in d.columns or CH_BSP not in d.columns:
        st.info(f"Aucune donnée TWA/BSP disponible pour {REF_BOAT}.")
        return

    d[CH_TWA] = pd.to_numeric(d[CH_TWA], errors="coerce")
    d[CH_BSP] = pd.to_numeric(d[CH_BSP], errors="coerce")
    d = d.dropna(subset=["time_utc", CH_TWA, CH_BSP])

    fig = go.Figure()

    for mode_name in ["UW", "DW"]:
        for t0, t1 in _mode_segments_for_ref(df, mode_name):
            fig.add_vrect(
                x0=t0,
                x1=t1,
                fillcolor=MODE_COLORS[mode_name],
                opacity=0.13,
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=d["time_utc"],
            y=d[CH_TWA],
            mode="lines",
            name="TWA deg",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["time_utc"],
            y=d[CH_BSP],
            mode="lines",
            name="BSP km/h",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"TWA / BSP time series – {REF_BOAT} – UW/DW windows shaded",
        height=540,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(title="Time UTC"),
        yaxis=dict(title="TWA deg", side="left"),
        yaxis2=dict(title="BSP km/h", side="right", overlaying="y", showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)


cfg = get_cfg()

DEFAULT_START = _utc_dt(2026, 4, 12, 18, 33, 0)
DEFAULT_STOP = _utc_dt(2026, 4, 12, 18, 35, 0)

with st.sidebar:
    st.header("Wing Trim controls")

    time_mode = st.radio("Plage de temps", ["Time range", "Last X minutes"], index=1)

    if time_mode == "Time range":
        start_date = st.date_input("Start date UTC", value=DEFAULT_START.date(), key="wing_start_date")
        start_time_min = st.time_input(
            "Start time UTC",
            value=DEFAULT_START.time().replace(second=0),
            step=timedelta(minutes=1),
            key="wing_start_time_min",
        )
        start_second = st.number_input("Start seconds UTC", 0, 59, DEFAULT_START.second, key="wing_start_second")
        stop_date = st.date_input("Stop date UTC", value=DEFAULT_STOP.date(), key="wing_stop_date")
        stop_time_min = st.time_input(
            "Stop time UTC",
            value=DEFAULT_STOP.time().replace(second=0),
            step=timedelta(minutes=1),
            key="wing_stop_time_min",
        )
        stop_second = st.number_input("Stop seconds UTC", 0, 59, DEFAULT_STOP.second, key="wing_stop_second")

        start_utc = _combine_utc(start_date, start_time_min.replace(second=int(start_second)))
        stop_utc = _combine_utc(stop_date, stop_time_min.replace(second=int(stop_second)))
    else:
        last_minutes = st.slider("Last X minutes", 1, 40, 20, step=1)
        stop_utc = datetime.now(timezone.utc)
        start_utc = stop_utc - timedelta(minutes=int(last_minutes))

    if stop_utc <= start_utc:
        st.error("Stop UTC doit être après Start UTC.")
        st.stop()

    minutes = (stop_utc - start_utc).total_seconds() / 60.0
    every = "5s" if minutes > 10 else "1s"

    selectable_boats = [b for b in ALL_BOATS if b != REF_BOAT]
    default_other_boats = [b for b in ["AUS", "GBR", "ESP", "SWE"] if b in selectable_boats]
    selected_other_boats = st.multiselect("Teams supplémentaires", selectable_boats, default=default_other_boats)
    boats = [REF_BOAT] + selected_other_boats

    st.markdown("---")
    st.subheader("Filtres data")
    st.caption("UW : 35 < abs(TWA) < 70")
    st.caption("DW : 120 < abs(TWA) < 160")
    bsp_min = st.slider("BSP mini", 0, 80, 30, step=1)
    yaw_rate_abs_max = st.slider("Yaw rate max |deg/s|", 0, 40, 8, step=1)
    vmg_target_pct_min = st.slider("Target VMG % min", 0, 120, 50, step=1)

    st.markdown("---")
    color_mode = st.radio("Coloration des points", ["Team", "% VMG target"], index=0)

    st.markdown("---")
    st.subheader("Targets aile")
    target_source = st.radio("Source fichier targets", ["Default file", "Upload file"], index=0)
    uploaded_targets = st.file_uploader("Upload targets .xlsx", type=["xlsx"]) if target_source == "Upload file" else None


with st.spinner("Chargement des données Wing Trim..."):
    df_raw = load_channels_timeseries(
        cfg=cfg,
        boats=boats,
        channels=WING_CHANNELS,
        start_utc=start_utc,
        stop_utc=stop_utc,
        every=every,
        level_expr="strm|mdss|mdss_fast|raw",
        agg_fn="mean",
    )

if df_raw.empty:
    st.warning("Aucune donnée retournée par Influx sur cette plage.")
    st.stop()

df_common = _filter_common(df_raw, bsp_min, yaw_rate_abs_max, vmg_target_pct_min)

if df_common.empty:
    st.warning("Aucune donnée après filtres.")
    st.stop()

target_dict = load_target_workbook(target_source, uploaded_targets)

df_fra_raw = df_raw[df_raw["boat"].astype(str) == REF_BOAT].copy()
tws_fra_mean = (
    pd.to_numeric(df_fra_raw.get(CH_TWS), errors="coerce").mean()
    if CH_TWS in df_fra_raw.columns
    else np.nan
)

target_result = build_targets_for_modes(
    df_raw=df_raw,
    ref_boat=REF_BOAT,
    target_dict=target_dict,
    target_columns=TARGET_COLUMNS,
    target_names=TARGET_NAMES,
    tws_mean=float(tws_fra_mean),
    page_key="wing",
    modes=["UW", "DW"],
)

target_by_mode = target_result.get("target_by_mode", {"UW": None, "DW": None})
target_overlays_by_mode = target_result.get("target_overlays_by_mode", {"UW": [], "DW": []})

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Points bruts", f"{len(df_raw):,}".replace(",", " "))
c2.metric("Points filtrés", f"{len(df_common):,}".replace(",", " "))
c3.metric("Début UTC", start_utc.strftime("%H:%M:%S"))
c4.metric("Fin UTC", stop_utc.strftime("%H:%M:%S"))
c5.metric("Pas", every)

t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("FRA TWS mean", "—" if not np.isfinite(tws_fra_mean) else f"{tws_fra_mean:.1f} km/h")
t2.metric("Config", target_result["selected_config"] or "—")
t3.metric("Auto config", target_result["auto_config"] or "—")
t4.metric(
    "UW BSP target",
    "—"
    if not target_by_mode["UW"] or not np.isfinite(target_by_mode["UW"].get("BSP_target", np.nan))
    else f"{target_by_mode['UW']['BSP_target']:.1f}",
)
t5.metric(
    "DW BSP target",
    "—"
    if not target_by_mode["DW"] or not np.isfinite(target_by_mode["DW"].get("BSP_target", np.nan))
    else f"{target_by_mode['DW']['BSP_target']:.1f}",
)

if target_result.get("displayed_target_states"):
    st.caption("Targets affichées : " + " + ".join(target_result["displayed_target_states"]))

with st.expander("Aperçu data filtrée", expanded=False):
    st.dataframe(df_common.head(500), use_container_width=True)

with st.expander("Aperçu targets", expanded=False):
    for mode_name in ["UW", "DW"]:
        st.markdown(f"### {mode_name} — sheet: {target_result['selected_sheet_by_mode'].get(mode_name) or '—'}")
        clean = target_result["target_clean_by_mode"][mode_name]
        if not clean.empty:
            st.dataframe(clean.head(300), use_container_width=True)
        else:
            st.info(f"Aucune table target chargée pour {mode_name}.")

_render_mode_section(df_common, "UW", color_mode, target_overlays_by_mode["UW"])
_render_mode_section(df_common, "DW", color_mode, target_overlays_by_mode["DW"])

st.markdown("---")
st.subheader("Reference boat time series")
_plot_twa_bsp_timeseries(df_raw)
