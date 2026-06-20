import os
import time
from collections import deque
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any
import re

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import timezone

def _cleanup_manual_length_clicks(ref_dt: datetime, lookback_s: int = 90) -> None:
    cutoff = ref_dt - timedelta(seconds=lookback_s)
    st.session_state["manual_plot_clicks_b"] = [
        t for t in st.session_state.get("manual_plot_clicks_b", []) if t >= cutoff
    ]
    st.session_state["manual_plot_clicks_t"] = [
        t for t in st.session_state.get("manual_plot_clicks_t", []) if t >= cutoff
    ]

def _flux_z(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

import requests
import streamlit as st
from dotenv import load_dotenv
from influx_io import get_cfg, _query_data_frame_safe

load_dotenv()
SAILGP_POI_TOKEN = os.getenv("SAILGP_POI_TOKEN")
if not SAILGP_POI_TOKEN:
    raise RuntimeError("SAILGP_POI_TOKEN not found in .env")

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    st_autorefresh = None  # type: ignore
    _HAS_AUTOREFRESH = False

st.set_page_config(page_title="Board cycles count", layout="wide")
st.title("Board cycles count")

WINDOW = 60
GRAPH_LOOKBACK_S = 120
GRAPH_LOOKAHEAD_S = 40
RED = "#ff3b30"
GREEN = "#34c759"
ORANGE = "#ff9500"
YELLOW = "#ffd60a"
BLUE = "#0a84ff"


# -----------------------------
# Generic helpers
# -----------------------------
def _fmt_mmss(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(int(seconds))
    return f"{sign}{s//60:02d}:{s%60:02d}"


def _colored_dispo(value: int, color_hex: str) -> str:
    return f"<span style='color:{color_hex}; font-weight:700'>{value}</span>"


def _result_line_html(count_b: int, tr1_b: int, tr2_b: int, dispo_b: int,
                      count_t: int, tr1_t: int, tr2_t: int, dispo_t: int) -> str:
    return f"""
    <div style="width:100%; font-size:18px; display:flex; justify-content:space-between; white-space:nowrap;">
        <div>
            Count_B:{count_b} |
            tr_1move_B:{tr1_b} |
            tr_2moves_B:{tr2_b} |
            dispo_BAB:{_colored_dispo(dispo_b, RED)}
        </div>
        <div>
            Count_T:{count_t} |
            tr_1move_T:{tr1_t} |
            tr_2moves_T:{tr2_t} |
            dispo_TRIB:{_colored_dispo(dispo_t, GREEN)}
        </div>
    </div>
    """


# -----------------------------
# Manual mode helpers
# -----------------------------
def _ensure_manual_state() -> None:
    if "press_history" not in st.session_state:
        st.session_state.press_history = {
            "babord": deque(),
            "tribord": deque(),
        }


def _cleanup(history: deque, now_ts: float) -> None:
    while history and (now_ts - history[0] > WINDOW):
        history.popleft()


def _timer_until_count(history: deque, target_count: int) -> int:
    now_ts = time.time()
    _cleanup(history, now_ts)
    n = len(history)
    if n <= target_count:
        return 0
    idx = (n - target_count) - 1
    ts_limit = history[idx]
    return max(int((ts_limit + WINDOW) - now_ts), 0)


def _manual_metrics() -> dict[str, int]:
    _ensure_manual_state()
    hist_b = st.session_state.press_history["babord"]
    hist_t = st.session_state.press_history["tribord"]
    now_ts = time.time()
    _cleanup(hist_b, now_ts)
    _cleanup(hist_t, now_ts)
    count_b = len(hist_b)
    count_t = len(hist_t)
    return {
        "count_b": count_b,
        "tr1_b": _timer_until_count(hist_b, 5),
        "tr2_b": _timer_until_count(hist_b, 4),
        "dispo_b": max(6 - count_b, 0),
        "count_t": count_t,
        "tr1_t": _timer_until_count(hist_t, 5),
        "tr2_t": _timer_until_count(hist_t, 4),
        "dispo_t": max(6 - count_t, 0),
    }



def _ensure_manual_plot_click_state() -> None:
    if "manual_plot_clicks_b" not in st.session_state:
        st.session_state.manual_plot_clicks_b = []
    if "manual_plot_clicks_t" not in st.session_state:
        st.session_state.manual_plot_clicks_t = []



def _render_manual_controls(show_line: bool = True) -> None:
    _ensure_manual_state()
    _ensure_manual_plot_click_state()

    c_left, c_right = st.columns([2.5, 4.5])

    with c_left:
        col_b, col_t = st.columns(2)

        with col_b:
            st.subheader("Babord")

            # Streamlit Cloud récent refuse columns > 1 niveau de nesting.
            # Ici on est déjà dans col_b, lui-même dans c_left.
            # Donc on utilise deux containers verticaux au lieu de st.columns(2).
            c1 = st.container()
            c2 = st.container()

            with c1:
                if st.button("+1 B", use_container_width=True):
                    ts = datetime.now(timezone.utc)
                    st.session_state.press_history["babord"].append(ts.timestamp())
                    st.session_state.setdefault("manual_plot_clicks_b", []).append(ts)

            with c2:
                if st.button("Undo B", use_container_width=True):
                    if st.session_state.press_history["babord"]:
                        st.session_state.press_history["babord"].popleft()
                        if st.session_state.get("manual_plot_clicks_b"):
                            st.session_state.manual_plot_clicks_b.pop()

        with col_t:
            st.subheader("Tribord")

            # Même correction : pas de 3e niveau de st.columns.
            c3 = st.container()
            c4 = st.container()

            with c3:
                if st.button("+1 T", use_container_width=True):
                    ts = datetime.now(timezone.utc)
                    st.session_state.press_history["tribord"].append(ts.timestamp())
                    st.session_state.setdefault("manual_plot_clicks_t", []).append(ts)

            with c4:
                if st.button("Undo T", use_container_width=True):
                    if st.session_state.press_history["tribord"]:
                        st.session_state.press_history["tribord"].popleft()
                        if st.session_state.get("manual_plot_clicks_t"):
                            st.session_state.manual_plot_clicks_t.pop()

    with c_right:
        if show_line:
            m = _manual_metrics()
            st.markdown(
                _result_line_html(
                    m["count_b"], m["tr1_b"], m["tr2_b"], m["dispo_b"],
                    m["count_t"], m["tr1_t"], m["tr2_t"], m["dispo_t"],
                ),
                unsafe_allow_html=True,
            )


def _manual_events_last_graph_window(ref_dt: datetime) -> list[dict[str, Any]]:
    _ensure_manual_state()
    start = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
    end = ref_dt + timedelta(seconds=GRAPH_LOOKAHEAD_S)
    out: list[dict[str, Any]] = []
    for side_key, side_name in [("babord", "port"), ("tribord", "starboard")]:
        for ts in list(st.session_state.press_history[side_key]):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if start <= dt <= end:
                out.append({"dt": dt, "side": side_name, "type": "manual"})
    out.sort(key=lambda x: x["dt"])
    return out




@st.fragment(run_every=1)
def _render_manual_line_fragment() -> None:
    m = _manual_metrics()
    st.markdown(
        _result_line_html(
            m["count_b"], m["tr1_b"], m["tr2_b"], m["dispo_b"],
            m["count_t"], m["tr1_t"], m["tr2_t"], m["dispo_t"],
        ),
        unsafe_allow_html=True,
    )

# -----------------------------
# Next start timer (manual mode only)
# -----------------------------
def _render_next_start_timer() -> None:
    if "ns_running" not in st.session_state:
        st.session_state.ns_running = False
    now_utc = datetime.now(timezone.utc)
    if "ns_hour" not in st.session_state:
        st.session_state.ns_hour = now_utc.hour
    if "ns_min" not in st.session_state:
        st.session_state.ns_min = (now_utc.minute + 1) % 60
    if "ns_offset_tenths" not in st.session_state:
        st.session_state.ns_offset_tenths = 0

    st.subheader("Next start timer")
    c0, c1 = st.columns([1.1, 1.0])
    with c0:
        st.metric("UTC", now_utc.strftime("%H:%M:%S"))
    with c1:
        st.session_state.ns_running = st.toggle("Timer ON", value=st.session_state.ns_running, key="ns_running_toggle")

    st.caption("Choix du prochain départ (UTC)")
    ch, cm = st.columns([1.2, 1.2])
    with ch:
        bh1, bh2, bh3 = st.columns([0.55, 1.0, 0.55])
        with bh1:
            if st.button("−", key="ns_hour_minus"):
                st.session_state.ns_hour = (int(st.session_state.ns_hour) - 1) % 24
        with bh2:
            st.session_state.ns_hour = st.number_input("Heure", 0, 23, int(st.session_state.ns_hour), key="ns_hour_input")
        with bh3:
            if st.button("+", key="ns_hour_plus"):
                st.session_state.ns_hour = (int(st.session_state.ns_hour) + 1) % 24
    with cm:
        bm1, bm2, bm3 = st.columns([0.55, 1.0, 0.55])
        with bm1:
            if st.button("−", key="ns_min_minus"):
                st.session_state.ns_min = (int(st.session_state.ns_min) - 1) % 60
        with bm2:
            st.session_state.ns_min = st.number_input("Minutes", 0, 59, int(st.session_state.ns_min), key="ns_min_input")
        with bm3:
            if st.button("+", key="ns_min_plus"):
                st.session_state.ns_min = (int(st.session_state.ns_min) + 1) % 60

    st.caption("Offset sur le décompte affiché (dixièmes de seconde)")
    co1, co2, co3, co4 = st.columns([0.9, 1.15, 0.9, 1.25])
    with co1:
        if st.button("−0.1s", key="ns_off_minus"):
            st.session_state.ns_offset_tenths = int(st.session_state.ns_offset_tenths) - 1
    with co2:
        st.session_state.ns_offset_tenths = st.number_input("Offset (x0.1s)", value=int(st.session_state.ns_offset_tenths), step=1, key="ns_off_input")
    with co3:
        if st.button("+0.1s", key="ns_off_plus"):
            st.session_state.ns_offset_tenths = int(st.session_state.ns_offset_tenths) + 1
    with co4:
        offset_s = float(st.session_state.ns_offset_tenths) / 10.0
        st.write(f"Offset = **{offset_s:+.1f}s**")

    now_utc = datetime.now(timezone.utc)
    target = now_utc.replace(hour=int(st.session_state.ns_hour), minute=int(st.session_state.ns_min), second=0, microsecond=0)
    if target < now_utc:
        target += timedelta(days=1)
    tts = (target - now_utc).total_seconds()
    tts_corr = tts + offset_s

    st.markdown(
        f"""
<div style="text-align:center; margin-top:20px; margin-bottom:10px;">
  <div style="font-size:120px; font-weight:700; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color:#00BFFF; line-height:1.0;">
    {_fmt_mmss(tts)}
  </div>
  <div style="font-size:120px; font-weight:700; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; color:#FF7F00; line-height:1.0;">
    {_fmt_mmss(tts_corr)}
  </div>
  <div style="margin-top:8px; font-size:18px; opacity:0.75;">
    Start (UTC): {target.strftime('%H:%M:%S')} &nbsp; | &nbsp; Offset: {offset_s:+.1f}s
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.ns_running and _HAS_AUTOREFRESH:
        st_autorefresh(interval=1000, key="next_start_refresh")


# -----------------------------
# POI helpers
# -----------------------------
def _ensure_poi_state() -> None:
    defaults = {
        "poi_analysis_cache": {},
        "poi_auto_refresh": True,
        "poi_refresh_seconds": 5,
        "poi_fake_date": date(2026, 3, 1),
        "poi_fake_hour": 6,
        "poi_fake_minute": 55,
        "poi_fake_play": False,
        "poi_fake_cursor_dt": datetime(2026, 3, 1, 6, 55, 0, tzinfo=timezone.utc),
        "poi_live_boat": "FRA",
        "poi_fake_boat": "ESP",
        "poi_last_tick": None,
        "poi_refresh_seconds_prev": 5,
        "poi_show_length_graph": False,
        "combo_show_length_graph": False,
        "poi_show_length_debug": False,
        "combo_show_length_debug": False,
        "manual_plot_clicks_b": [],
        "manual_plot_clicks_t": [],
        "manual_length_refresh_seconds": 4,
        "manual_length_refresh_prev": 4,
        "manual_length_boat": "FRA",
        "manual_len_poi_time_mode": "Live",
        "manual_len_poi_fake_date": date(2026, 6, 20),
        "manual_len_poi_fake_hour": 19,
        "manual_len_poi_fake_minute": 6,
        "manual_len_poi_fake_play": False,
        "manual_len_poi_fake_cursor_dt": datetime(2026, 6, 20, 19, 6, 0, tzinfo=timezone.utc),
        "manual_len_poi_last_tick": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SAILGP_POI_TOKEN}"}


def _fetch_pois(from_dt: datetime, to_dt: datetime, boat: str, poi_types: list[str]) -> list[dict[str, Any]]:
    params: list[tuple[str, str]] = [
        ("from", from_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("to", to_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("boat", boat),
    ]
    for pt in poi_types:
        params.append(("poi_type", pt))
    r = requests.get("https://api.f50.sailgp.tech/v1/pois", headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _get_board_side(poi_id: str) -> str | None:
    cache = st.session_state.poi_analysis_cache
    if poi_id in cache:
        return cache[poi_id]
    try:
        r = requests.get(f"https://api.f50.sailgp.tech/v1/pois/{poi_id}/analysis", headers=_headers(), timeout=30)
        if r.status_code != 200:
            cache[poi_id] = None
            return None
        data = r.json()
        side = (((data or {}).get("scalars") or {}).get("board_side"))
        if isinstance(side, str):
            side = side.lower()
        cache[poi_id] = side
        return side
    except Exception:
        cache[poi_id] = None
        return None


def _normalize_poi_events(raw_pois: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for p in raw_pois:
        ptype = (p.get("type") or "").lower()
        if ptype not in {"boarddrop", "boardraise", "boardmovepenalty"}:
            continue
        ts = p.get("start_datetime")
        poi_id = p.get("poi_id")
        if not ts or not poi_id:
            continue
        try:
            dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
        except Exception:
            continue
        side = _get_board_side(poi_id)
        events.append({
            "dt": dt,
            "type": ptype,
            "poi_id": poi_id,
            "side": side,
        })
    events.sort(key=lambda x: x["dt"])
    return events


def _dedup_poi_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_type_by_side: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for e in events:
        side = e.get("side")
        ptype = e.get("type")
        if side not in {"port", "starboard"}:
            out.append(e)
            continue
        if last_type_by_side.get(side) != ptype:
            out.append(e)
            last_type_by_side[side] = ptype
    return out


def _events_in_window(events: list[dict[str, Any]], ref_dt: datetime, seconds: int) -> list[dict[str, Any]]:
    start = ref_dt - timedelta(seconds=seconds)
    return [e for e in events if start <= e["dt"] <= ref_dt]


def _timer_from_events(events: list[dict[str, Any]], ref_dt: datetime, side: str, target_count: int) -> int:
    times = [e["dt"] for e in events if e.get("side") == side and e.get("type") in {"boarddrop", "boardraise"}]
    times.sort()
    n = len(times)
    if n <= target_count:
        return 0
    idx = (n - target_count) - 1
    limit_dt = times[idx]
    return max(int((limit_dt + timedelta(seconds=WINDOW) - ref_dt).total_seconds()), 0)


def _poi_metrics(events: list[dict[str, Any]], ref_dt: datetime) -> dict[str, int]:
    events_60 = _events_in_window(events, ref_dt, WINDOW)
    count_b = sum(1 for e in events_60 if e.get("side") == "port" and e.get("type") in {"boarddrop", "boardraise"})
    count_t = sum(1 for e in events_60 if e.get("side") == "starboard" and e.get("type") in {"boarddrop", "boardraise"})
    return {
        "count_b": count_b,
        "tr1_b": _timer_from_events(events_60, ref_dt, "port", 5),
        "tr2_b": _timer_from_events(events_60, ref_dt, "port", 4),
        "dispo_b": max(6 - count_b, 0),
        "count_t": count_t,
        "tr1_t": _timer_from_events(events_60, ref_dt, "starboard", 5),
        "tr2_t": _timer_from_events(events_60, ref_dt, "starboard", 4),
        "dispo_t": max(6 - count_t, 0),
    }


def _poi_summary(events: list[dict[str, Any]], ref_dt: datetime) -> dict[str, int]:
    events_10 = _events_in_window(events, ref_dt, 600)
    out = {"drop_b": 0, "drop_t": 0, "raise_b": 0, "raise_t": 0, "unknown": 0}
    for e in events_10:
        side = e.get("side")
        ptype = e.get("type")
        if ptype == "boarddrop":
            if side == "port":
                out["drop_b"] += 1
            elif side == "starboard":
                out["drop_t"] += 1
            else:
                out["unknown"] += 1
        elif ptype == "boardraise":
            if side == "port":
                out["raise_b"] += 1
            elif side == "starboard":
                out["raise_t"] += 1
            else:
                out["unknown"] += 1
    return out


def _build_comparison_figure(ref_dt: datetime, poi_events: list[dict[str, Any]], poi_metrics: dict[str, int] | None,
                             manual_events: list[dict[str, Any]] | None = None) -> go.Figure:
    x0 = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
    x1 = ref_dt + timedelta(seconds=GRAPH_LOOKAHEAD_S)

    lanes = {
        "raise": 1.0,
        "drop": 0.0,
        "penalty": -1.0,
        "manual_b": 0.25,
        "manual_t": 0.75,
    }

    fig = go.Figure()

    # POI traces only
    for ptype, yval in [("boardraise", lanes["raise"]), ("boarddrop", lanes["drop"])]:
        for side, color, name in [("port", RED, f"POI {ptype} BAB"), ("starboard", GREEN, f"POI {ptype} TRIB")]:
            pts = [e for e in poi_events if e["type"] == ptype and e.get("side") == side and x0 <= e["dt"] <= x1]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[e["dt"] for e in pts],
                    y=[yval] * len(pts),
                    mode="markers",
                    marker=dict(symbol="circle", size=10, color=color),
                    name=name,
                    hovertemplate="%{x|%H:%M:%S}<extra>" + name + "</extra>",
                ))

    penalties = [e for e in poi_events if e["type"] == "boardmovepenalty" and x0 <= e["dt"] <= x1]
    if penalties:
        fig.add_trace(go.Scatter(
            x=[e["dt"] for e in penalties],
            y=[lanes["penalty"]] * len(penalties),
            mode="markers",
            marker=dict(symbol="square", size=10, color=RED, line=dict(width=1, color="black")),
            name="POI penalty",
            hovertemplate="%{x|%H:%M:%S}<extra>POI penalty</extra>",
        ))

    # Manual overlay only in combined mode
    if manual_events is not None:
        for side, color, lane_key, name in [
            ("port", RED, "manual_b", "Manual BAB"),
            ("starboard", GREEN, "manual_t", "Manual TRIB"),
        ]:
            pts = [e for e in manual_events if e.get("side") == side and x0 <= e["dt"] <= x1]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[e["dt"] for e in pts],
                    y=[lanes[lane_key]] * len(pts),
                    mode="markers",
                    marker=dict(symbol="x", size=12, color=color, line=dict(width=2, color=color)),
                    name=name,
                    hovertemplate="%{x|%H:%M:%S}<extra>" + name + "</extra>",
                ))

    # Vertical lines
    for when, color, width in [
        (ref_dt - timedelta(minutes=2), YELLOW, 1),
        (ref_dt - timedelta(minutes=1), RED, 1),
        (ref_dt, BLUE, 2),
    ]:
        fig.add_vline(x=when, line_color=color, line_width=width)

    # Future POI timer bars
    if poi_metrics:
        for label, sec, double in [
            ("B1", poi_metrics.get("tr1_b", 0), False),
            ("T1", poi_metrics.get("tr1_t", 0), False),
            ("B2", poi_metrics.get("tr2_b", 0), True),
            ("T2", poi_metrics.get("tr2_t", 0), True),
        ]:
            if sec and sec > 0:
                x = ref_dt + timedelta(seconds=int(sec))
                if x <= x1:
                    fig.add_vline(x=x, line_color=ORANGE, line_width=1, line_dash="dash")
                    if double:
                        fig.add_vline(x=x + timedelta(milliseconds=300), line_color=ORANGE, line_width=1, line_dash="dash")
                    fig.add_annotation(x=x, y=1.25, text=label, showarrow=False, font=dict(color=ORANGE, size=10))

    tick0 = x0.replace(microsecond=0)

    # ===== Manual click markers (+1B / +1T) =====
    _cleanup_manual_length_clicks(ref_dt, lookback_s=90)

    xb = st.session_state.get("manual_plot_clicks_b", [])
    xt = st.session_state.get("manual_plot_clicks_t", [])

    if xb:
        fig.add_trace(go.Scatter(
            x=xb,
            y=[-0.2] * len(xb),
            mode="markers",
            marker=dict(
                color=RED,
                size=8,
                symbol="circle",
                line=dict(color="black", width=0.5),
            ),
            name="+1 B",
            hovertemplate="%{x|%H:%M:%S} UTC<extra>+1 B</extra>",
        ))

    if xt:
        fig.add_trace(go.Scatter(
            x=xt,
            y=[-0.2] * len(xt),
            mode="markers",
            marker=dict(
                color=GREEN,
                size=8,
                symbol="circle",
                line=dict(color="black", width=0.5),
            ),
            name="+1 T",
            hovertemplate="%{x|%H:%M:%S} UTC<extra>+1 T</extra>",
        ))

    fig.update_xaxes(
        range=[x0, x1],
        tick0=tick0,
        dtick=10000,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)",
        tickformat="%H:%M:%S",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[1.0, 0.75, 0.25, 0.0, -1.0] if manual_events is not None else [1.0, 0.0, -1.0],
        ticktext=["Raise", "Manual TRIB", "Manual BAB", "Drop", "Penalty"] if manual_events is not None else ["Raise", "Drop", "Penalty"],
        range=[-1.5, 1.5],
        showgrid=False,
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -----------------------------
# POI fragment renderer
# -----------------------------
def _compute_ref_dt(time_mode: str) -> datetime:
    refresh_s = int(st.session_state.poi_refresh_seconds)
    now = datetime.now(timezone.utc)
    if time_mode == "Live":
        if st.session_state.poi_auto_refresh:
            last_tick = st.session_state.poi_last_tick
            if last_tick is None or (now - last_tick).total_seconds() >= refresh_s:
                st.session_state.poi_last_tick = now
        return now

    # Faux live
    if st.session_state.poi_auto_refresh and st.session_state.poi_fake_play:
        last_tick = st.session_state.poi_last_tick
        if last_tick is None or (now - last_tick).total_seconds() >= refresh_s:
            st.session_state.poi_last_tick = now
            st.session_state.poi_fake_cursor_dt = st.session_state.poi_fake_cursor_dt + timedelta(seconds=refresh_s)
    return st.session_state.poi_fake_cursor_dt




def _load_length_db_timeseries(start_utc: datetime, stop_utc: datetime, boat: str):
    try:
        cfg = get_cfg()
        dfs = []
        for ch in ["LENGTH_DB_H_P_mm", "LENGTH_DB_H_S_mm"]:
            boats_regex = re.escape(boat)
            flux = f"""
from(bucket: "{cfg.bucket}")
  |> range(start: {_flux_z(start_utc)}, stop: {_flux_z(stop_utc)})
  |> filter(fn: (r) => r._measurement == "{ch}")
  |> filter(fn: (r) => r._field == "value" and r.level == "strm")
  |> filter(fn: (r) => r.boat =~ /^{boats_regex}/)
  |> keep(columns: ["_time", "_value", "boat"])
  |> rename(columns: {{_time: "time", _value: "{ch}"}})
"""
            df = _query_data_frame_safe(cfg, flux)
            if df is not None and not df.empty:
                df = df.copy()
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                dfs.append(df)

        if not dfs:
            return None

        merged = None
        for df in dfs:
            cols = [c for c in df.columns if c != "boat"]
            cur = df[cols].drop_duplicates()
            if merged is None:
                merged = cur
            else:
                merged = pd.merge(merged, cur, on="time", how="outer")

        return merged.sort_values("time").reset_index(drop=True)

    except Exception:
        return None


def _build_length_db_figure(df, ref_dt: datetime) -> go.Figure:
    fig = go.Figure()

    if "LENGTH_DB_H_P_mm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["LENGTH_DB_H_P_mm"],
            mode="lines",
            line=dict(color=RED, width=1.5),
            name="LENGTH_DB_H_P_mm",
        ))
    if "LENGTH_DB_H_S_mm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["LENGTH_DB_H_S_mm"],
            mode="lines",
            line=dict(color=GREEN, width=1.5),
            name="LENGTH_DB_H_S_mm",
        ))

    x0 = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
    x1 = ref_dt

    fig.update_xaxes(
        range=[x0, x1],
        tick0=x0.replace(microsecond=0),
        dtick=10000,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        tickformat="%H:%M:%S",
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        autorange=True,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        tickfont=dict(size=10),
    )
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=8, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
    )
    return fig



def _build_manual_length_figure(df, ref_dt: datetime) -> go.Figure:
    fig = go.Figure()

    if "LENGTH_DB_H_P_mm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=-1.0 * df["LENGTH_DB_H_P_mm"],
            mode="lines",
            line=dict(color=RED, width=1.4),
            name="-LENGTH_DB_H_P_mm",
        ))
    if "LENGTH_DB_H_S_mm" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=-1.0 * df["LENGTH_DB_H_S_mm"],
            mode="lines",
            line=dict(color=GREEN, width=1.4),
            name="-LENGTH_DB_H_S_mm",
        ))

    x0 = ref_dt - timedelta(seconds=90)
    x1 = ref_dt
    fig.add_vline(x=ref_dt - timedelta(minutes=1), line_color=RED, line_width=1)
    fig.add_vline(x=ref_dt, line_color=BLUE, line_width=2)

    fig.update_xaxes(
        range=[x0, x1],
        tick0=x0.replace(microsecond=0),
        dtick=10000,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        tickformat="%H:%M:%S",
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        autorange=True,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        tickfont=dict(size=9),
    )
    fig.update_layout(
        height=145,
        margin=dict(l=10, r=10, t=6, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=8)),
    )
    return fig


def _render_manual_length_mode() -> None:
    refresh_s = int(st.session_state.manual_length_refresh_seconds)
    if "manual_length_refresh_prev" not in st.session_state:
        st.session_state.manual_length_refresh_prev = refresh_s
    if refresh_s != int(st.session_state.manual_length_refresh_prev):
        st.session_state.manual_length_refresh_prev = refresh_s
        st.rerun()

    @st.fragment(run_every=st.session_state.manual_length_refresh_seconds)
    def _render_manual_length_fragment():
        boat = st.session_state.manual_length_boat
        ref_dt = datetime.now(timezone.utc)
        start_dt = ref_dt - timedelta(seconds=90)
        df_len = _load_length_db_timeseries(start_dt, ref_dt, boat)
        if df_len is None or getattr(df_len, "empty", True):
            st.caption("No LENGTH_DB data")
        else:
            fig_len = _build_manual_length_figure(df_len, ref_dt)
            st.plotly_chart(fig_len, use_container_width=True)

    _render_manual_length_fragment()

    c1, c2 = st.columns([1.1, 1.1])
    with c1:
        st.session_state.manual_length_refresh_seconds = st.number_input(
            "Refresh LENGTH_DB (s)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.manual_length_refresh_seconds),
            step=1,
            key="manual_length_refresh_seconds_input",
        )
    with c2:
        st.session_state.manual_length_boat = st.text_input(
            "Boat code LENGTH_DB",
            value=st.session_state.manual_length_boat,
            key="manual_length_boat_input",
        ).strip().upper() or "FRA"




def _build_poi_length_figure(
    ref_dt: datetime,
    poi_events: list[dict[str, Any]],
    poi_metrics: dict[str, int] | None,
    df_len,
) -> go.Figure:
    _ensure_manual_plot_click_state()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    x0 = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S + 10)
    x1 = ref_dt + timedelta(seconds=10)

    # --- LENGTH_DB traces first, on secondary Y
    if df_len is not None and not getattr(df_len, "empty", True):
        try:
            dfp = df_len.copy()
            if "time" in dfp.columns:
                dfp = (
                    dfp.sort_values("time")
                    .set_index("time")
                    .resample("2s")
                    .last()
                    .dropna(how="all")
                    .reset_index()
                )

            if "LENGTH_DB_H_P_mm" in dfp.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dfp["time"],
                        y=-(1 / 1000.0) * dfp["LENGTH_DB_H_P_mm"],
                        mode="lines",
                        line=dict(color=RED, width=1.2),
                        name="-DB P (m)",
                        opacity=0.75,
                    ),
                    secondary_y=True,
                )

            if "LENGTH_DB_H_S_mm" in dfp.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dfp["time"],
                        y=-(1 / 1000.0) * dfp["LENGTH_DB_H_S_mm"],
                        mode="lines",
                        line=dict(color=GREEN, width=1.2),
                        name="-DB S (m)",
                        opacity=0.75,
                    ),
                    secondary_y=True,
                )
        except Exception:
            pass

    # --- POI markers, on primary Y
    added_names = set()
    for e in poi_events:
        typ = str(e.get("type", "")).lower()
        side = str(e.get("side") or e.get("board_side") or "").lower()
        ts = e.get("dt") or e.get("ts") or e.get("time") or e.get("start_datetime")
        if ts is None:
            continue
        if not isinstance(ts, datetime):
            try:
                ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                continue

        if typ == "boardraise":
            y = 1.0
            symbol = "circle"
            label = "POI boardraise"
        elif typ == "boarddrop":
            y = 0.0
            symbol = "circle"
            label = "POI boarddrop"
        elif typ == "boardmovepenalty":
            y = -1.0
            symbol = "square"
            label = "POI penalty"
        else:
            continue

        color = RED if side in ("port", "babord") else GREEN if side in ("starboard", "tribord") else "white"
        showlegend = label not in added_names
        added_names.add(label)

        fig.add_trace(
            go.Scatter(
                x=[ts],
                y=[y],
                mode="markers",
                marker=dict(
                    color=color,
                    size=11,
                    symbol=symbol,
                    line=dict(color="black", width=0.8),
                ),
                name=label,
                showlegend=showlegend,
                hovertemplate="%{x|%H:%M:%S} UTC<extra>" + label + "</extra>",
            ),
            secondary_y=False,
        )

    # --- Vertical reference markers on primary POI axis.
    # Explicit Scatter traces are more robust than add_vline with secondary_y subplots.
    y_min_marker, y_max_marker = -1.35, 1.30

    def _add_vertical_marker(
        x_dt: datetime,
        color: str,
        width: float = 1.0,
        dash: str | None = None,
        name: str = "",
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x_dt, x_dt],
                y=[y_min_marker, y_max_marker],
                mode="lines",
                line=dict(color=color, width=width, dash=dash or "solid"),
                name=name,
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    _add_vertical_marker(ref_dt - timedelta(minutes=2), YELLOW, 1.2, name="t-2min")
    _add_vertical_marker(ref_dt - timedelta(minutes=1), RED, 1.2, name="t-1min")
    _add_vertical_marker(ref_dt, BLUE, 2.2, name="now")

    # --- Future bars from POI timers
    if poi_metrics:
        for key, width in [("tr1_b", 1), ("tr1_t", 1), ("tr2_b", 2), ("tr2_t", 2)]:
            val = int(poi_metrics.get(key, 0) or 0)
            if val > 0:
                _add_vertical_marker(ref_dt + timedelta(seconds=val), "orange", width, dash="dash", name=key)

    # --- Manual button press markers on the POI axis, fixed top lane
    # +1 Babord = red square, +1 Tribord = green square.
    # They use absolute UTC timestamps, so they scroll left with the same X axis.
    try:
        cutoff_clicks = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
        st.session_state.manual_plot_clicks_b = [
            t for t in st.session_state.get("manual_plot_clicks_b", []) if t >= cutoff_clicks
        ]
        st.session_state.manual_plot_clicks_t = [
            t for t in st.session_state.get("manual_plot_clicks_t", []) if t >= cutoff_clicks
        ]

        xb = [t for t in st.session_state.get("manual_plot_clicks_b", []) if x0 <= t <= x1]
        xt = [t for t in st.session_state.get("manual_plot_clicks_t", []) if x0 <= t <= x1]
        manual_click_y = 1.18

        if xb:
            fig.add_trace(
                go.Scatter(
                    x=xb,
                    y=[manual_click_y] * len(xb),
                    mode="markers",
                    marker=dict(
                        color=RED,
                        size=4,
                        symbol="square",
                        line=dict(color="black", width=0.7),
                    ),
                    name="+1 BAB manual",
                    hovertemplate="%{x|%H:%M:%S} UTC<extra>+1 BAB manual</extra>",
                ),
                secondary_y=False,
            )

        if xt:
            fig.add_trace(
                go.Scatter(
                    x=xt,
                    y=[manual_click_y] * len(xt),
                    mode="markers",
                    marker=dict(
                        color=GREEN,
                        size=4,
                        symbol="square",
                        line=dict(color="black", width=0.7),
                    ),
                    name="+1 TRIB manual",
                    hovertemplate="%{x|%H:%M:%S} UTC<extra>+1 TRIB manual</extra>",
                ),
                secondary_y=False,
            )
    except Exception:
        pass

    fig.update_xaxes(
        range=[x0, x1],
        tick0=x0.replace(microsecond=0),
        dtick=10000,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        tickformat="%H:%M:%S",
        tickfont=dict(size=8),
    )

    # Primary Y is fixed so POI markers cannot disappear because of DB autoscale.
    fig.update_yaxes(
        tickmode="array",
        tickvals=[1.0, 0.0, -1.0],
        ticktext=["Raise", "Drop", "Penalty"],
        range=[-1.35, 1.30],
        showgrid=False,
        tickfont=dict(size=8),
        secondary_y=False,
    )

    # Secondary Y is auto for -DB curves.
    fig.update_yaxes(
        autorange=True,
        showgrid=False,
        tickfont=dict(size=8),
        title_text="-DB (m)",
        secondary_y=True,
    )

    fig.update_layout(
        height=245,
        margin=dict(l=8, r=8, t=6, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=8)),
    )
    return fig

def _render_poi_fragment_body(combined: bool, time_mode: str, boat: str) -> None:
    ref_dt = _compute_ref_dt(time_mode)
    st.caption(f"Heure de référence UTC : {ref_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        start_graph = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
        raw_pois = _fetch_pois(start_graph, ref_dt, boat, ["boarddrop", "boardraise", "boardmovepenalty"])
        poi_events = _dedup_poi_events(_normalize_poi_events(raw_pois))
        pm = _poi_metrics(poi_events, ref_dt)
    except Exception as e:
        st.error(f"Lecture POI impossible: {e}")
        return

    if combined:
        manual_events = _manual_events_last_graph_window(ref_dt)
    else:
        manual_events = None

    if not combined:
        st.markdown(
            _result_line_html(pm["count_b"], pm["tr1_b"], pm["tr2_b"], pm["dispo_b"], pm["count_t"], pm["tr1_t"], pm["tr2_t"], pm["dispo_t"]),
            unsafe_allow_html=True,
        )

    fig = _build_comparison_figure(ref_dt, poi_events, pm, manual_events)
    st.plotly_chart(fig, use_container_width=True)

    if combined:
        st.markdown(
            _result_line_html(pm["count_b"], pm["tr1_b"], pm["tr2_b"], pm["dispo_b"], pm["count_t"], pm["tr1_t"], pm["tr2_t"], pm["dispo_t"]),
            unsafe_allow_html=True,
        )

    if st.button("Bilan 10 min", key=f"summary_btn_{'combo' if combined else 'poi'}"):
        try:
            raw_10m = _fetch_pois(ref_dt - timedelta(minutes=10), ref_dt, boat, ["boarddrop", "boardraise", "boardmovepenalty"])
            ev_10m = _dedup_poi_events(_normalize_poi_events(raw_10m))
            summary = _poi_summary(ev_10m, ref_dt)
            st.success(
                f"**Bilan 10 min** — "
                f"Drop B:{summary['drop_b']} | Drop T:{summary['drop_t']} | "
                f"Raise B:{summary['raise_b']} | Raise T:{summary['raise_t']} | Unknown:{summary['unknown']}"
            )
        except Exception as e:
            st.error(f"Bilan 10 min impossible: {e}")

    show_len_key = "combo_show_length_graph" if combined else "poi_show_length_graph"
    debug_len_key = "combo_show_length_debug" if combined else "poi_show_length_debug"

    st.session_state[show_len_key] = st.toggle(
        "Show LENGTH_DB timeseries",
        value=bool(st.session_state.get(show_len_key, False)),
        key=f"{show_len_key}_toggle",
    )
    st.session_state[debug_len_key] = st.toggle(
        "Debug LENGTH_DB",
        value=bool(st.session_state.get(debug_len_key, False)),
        key=f"{debug_len_key}_toggle",
    )

    if st.session_state.get(show_len_key, False):
        len_start = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
        len_stop = ref_dt
        df_len = _load_length_db_timeseries(len_start, len_stop, boat)

        if st.session_state.get(debug_len_key, False):
            st.caption(
                f"LENGTH_DB debug — boat={boat} | from={len_start.strftime('%Y-%m-%d %H:%M:%S')}Z "
                f"| to={len_stop.strftime('%Y-%m-%d %H:%M:%S')}Z"
            )
            if df_len is None:
                st.caption("LENGTH_DB debug — df_len is None")
            elif getattr(df_len, "empty", True):
                st.caption("LENGTH_DB debug — df_len is empty")
            else:
                st.caption(
                    f"LENGTH_DB debug — rows={len(df_len)} | cols={list(df_len.columns)} | "
                    f"first={df_len['time'].min()} | last={df_len['time'].max()}"
                )
                preview_cols = [c for c in ["time", "LENGTH_DB_H_P_mm", "LENGTH_DB_H_S_mm"] if c in df_len.columns]
                st.dataframe(df_len[preview_cols].head(10), use_container_width=True)

        if df_len is None or getattr(df_len, "empty", True):
            st.caption("No LENGTH_DB data")
        else:
            fig_len = _build_length_db_figure(df_len, ref_dt)
            st.plotly_chart(fig_len, use_container_width=True)


# -----------------------------
# POI mode wrapper (widgets stable)
# -----------------------------
def _render_poi_modes(combined: bool) -> None:
    _ensure_poi_state()
    mode_key = "combo" if combined else "poi"

    if combined:
        st.subheader("Mode Manuel + POI")
        _render_manual_controls(show_line=False)
        _render_manual_line_fragment()
        st.markdown("---")
    else:
        st.subheader("Mode POI API")

    top = st.columns([1.1, 1.2, 1.4, 1.4])
    with top[0]:
        st.radio(
            "Horloge",
            ["Live"] if combined else ["Live", "Faux live"],
            horizontal=True,
            key=f"clock_mode_{mode_key}",
        )
    current_time_mode = st.session_state[f"clock_mode_{mode_key}"]

    with top[1]:
        st.session_state.poi_auto_refresh = st.toggle(
            "Auto refresh",
            value=st.session_state.poi_auto_refresh,
            key=f"poi_auto_refresh_{mode_key}",
        )

    with top[2]:
        refresh_s = st.number_input(
            "Refresh (s)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.poi_refresh_seconds),
            key=f"poi_refresh_s_{mode_key}",
        )
        refresh_s = int(refresh_s)
        if "poi_refresh_seconds_prev" not in st.session_state:
            st.session_state.poi_refresh_seconds_prev = refresh_s
        if refresh_s != int(st.session_state.poi_refresh_seconds_prev):
            st.session_state.poi_refresh_seconds_prev = refresh_s
            st.session_state.poi_refresh_seconds = refresh_s
            st.session_state.poi_last_tick = None
            st.rerun()
        st.session_state.poi_refresh_seconds = refresh_s

    with top[3]:
        boat_value = st.session_state.poi_live_boat if current_time_mode == "Live" else st.session_state.poi_fake_boat
        boat = st.text_input(
            "Boat code",
            value=boat_value,
            key=f"boat_input_{mode_key}_{current_time_mode}",
        ).strip().upper() or "FRA"
        if current_time_mode == "Live":
            st.session_state.poi_live_boat = boat
        else:
            st.session_state.poi_fake_boat = boat

    if current_time_mode == "Faux live":
        row = st.columns([1.3, 0.9, 0.9, 1.1, 1.2])
        with row[0]:
            st.session_state.poi_fake_date = st.date_input(
                "Date",
                value=st.session_state.poi_fake_date,
                key="poi_fake_date_input",
            )
        with row[1]:
            st.session_state.poi_fake_hour = st.number_input(
                "Heure UTC",
                0,
                23,
                int(st.session_state.poi_fake_hour),
                key="poi_fake_hour_input",
            )
        with row[2]:
            st.session_state.poi_fake_minute = st.number_input(
                "Minute UTC",
                0,
                59,
                int(st.session_state.poi_fake_minute),
                key="poi_fake_minute_input",
            )
        with row[3]:
            st.session_state.poi_fake_play = st.toggle(
                "Lecture faux live",
                value=st.session_state.poi_fake_play,
                key="poi_fake_play_toggle",
            )
        with row[4]:
            if st.button("Reset faux live", use_container_width=True, key="poi_fake_reset_btn"):
                st.session_state.poi_fake_cursor_dt = datetime(
                    st.session_state.poi_fake_date.year,
                    st.session_state.poi_fake_date.month,
                    st.session_state.poi_fake_date.day,
                    int(st.session_state.poi_fake_hour),
                    int(st.session_state.poi_fake_minute),
                    0,
                    tzinfo=timezone.utc,
                )
                st.session_state.poi_last_tick = None
                st.rerun()

        selected_dt = datetime(
            st.session_state.poi_fake_date.year,
            st.session_state.poi_fake_date.month,
            st.session_state.poi_fake_date.day,
            int(st.session_state.poi_fake_hour),
            int(st.session_state.poi_fake_minute),
            0,
            tzinfo=timezone.utc,
        )
        if not st.session_state.poi_fake_play:
            st.session_state.poi_fake_cursor_dt = selected_dt

    @st.fragment(run_every=st.session_state.get("poi_refresh_seconds", 5))
    def _render_poi_fragment():
        fragment_time_mode = st.session_state[f"clock_mode_{mode_key}"]
        fragment_boat = (
            st.session_state.poi_live_boat
            if fragment_time_mode == "Live"
            else st.session_state.poi_fake_boat
        )
        _render_poi_fragment_body(combined, fragment_time_mode, fragment_boat)

    _render_poi_fragment()


# -----------------------------
# Page body
# -----------------------------
mode = st.radio("Mode", ["Manuel", "POI API", "Manuel + POI", "Manuel + LENGTH_DB", "Manuel + LENGTH_DB + POI"], horizontal=True)




def _compute_manual_len_poi_ref_dt() -> datetime:
    """Reference time for Manual + LENGTH_DB + POI mode."""
    _ensure_poi_state()
    refresh_s = int(st.session_state.get("poi_refresh_seconds", 5))
    now = datetime.now(timezone.utc)

    if st.session_state.get("manual_len_poi_time_mode", "Live") == "Live":
        return now

    if "manual_len_poi_fake_cursor_dt" not in st.session_state:
        st.session_state.manual_len_poi_fake_cursor_dt = datetime(2026, 6, 20, 19, 6, 0, tzinfo=timezone.utc)

    if st.session_state.get("manual_len_poi_fake_play", False):
        last_tick = st.session_state.get("manual_len_poi_last_tick", None)
        if last_tick is None or (now - last_tick).total_seconds() >= refresh_s:
            st.session_state.manual_len_poi_last_tick = now
            st.session_state.manual_len_poi_fake_cursor_dt = (
                st.session_state.manual_len_poi_fake_cursor_dt + timedelta(seconds=refresh_s)
            )

    return st.session_state.manual_len_poi_fake_cursor_dt


def _render_manual_len_poi_time_controls(ref_dt: datetime) -> datetime:
    """Render Live/Faux-live controls and return the reference time."""
    ctrl = st.columns([1.0, 1.1, 0.8, 0.8, 1.0, 1.0])

    with ctrl[0]:
        current = st.session_state.get("manual_len_poi_time_mode", "Live")
        st.session_state.manual_len_poi_time_mode = st.radio(
            "Horloge",
            ["Live", "Faux live"],
            horizontal=True,
            index=0 if current == "Live" else 1,
            key="manual_len_poi_clock_mode",
        )

    if st.session_state.manual_len_poi_time_mode == "Faux live":
        with ctrl[1]:
            st.session_state.manual_len_poi_fake_date = st.date_input(
                "Date",
                value=st.session_state.get("manual_len_poi_fake_date", date(2026, 6, 20)),
                key="manual_len_poi_fake_date_input",
            )
        with ctrl[2]:
            st.session_state.manual_len_poi_fake_hour = st.number_input(
                "Heure UTC",
                0,
                23,
                int(st.session_state.get("manual_len_poi_fake_hour", 19)),
                key="manual_len_poi_fake_hour_input",
            )
        with ctrl[3]:
            st.session_state.manual_len_poi_fake_minute = st.number_input(
                "Minute UTC",
                0,
                59,
                int(st.session_state.get("manual_len_poi_fake_minute", 6)),
                key="manual_len_poi_fake_minute_input",
            )
        with ctrl[4]:
            st.session_state.manual_len_poi_fake_play = st.toggle(
                "Lecture faux live",
                value=bool(st.session_state.get("manual_len_poi_fake_play", False)),
                key="manual_len_poi_fake_play_toggle",
            )
        with ctrl[5]:
            if st.button("Reset faux live", use_container_width=True, key="manual_len_poi_fake_reset"):
                st.session_state.manual_len_poi_fake_cursor_dt = datetime(
                    st.session_state.manual_len_poi_fake_date.year,
                    st.session_state.manual_len_poi_fake_date.month,
                    st.session_state.manual_len_poi_fake_date.day,
                    int(st.session_state.manual_len_poi_fake_hour),
                    int(st.session_state.manual_len_poi_fake_minute),
                    0,
                    tzinfo=timezone.utc,
                )
                st.session_state.manual_len_poi_last_tick = None
                st.rerun()

        selected_dt = datetime(
            st.session_state.manual_len_poi_fake_date.year,
            st.session_state.manual_len_poi_fake_date.month,
            st.session_state.manual_len_poi_fake_date.day,
            int(st.session_state.manual_len_poi_fake_hour),
            int(st.session_state.manual_len_poi_fake_minute),
            0,
            tzinfo=timezone.utc,
        )
        if not st.session_state.manual_len_poi_fake_play:
            st.session_state.manual_len_poi_fake_cursor_dt = selected_dt
            ref_dt = selected_dt

    st.caption(f"Heure de référence UTC : {ref_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    return ref_dt

@st.fragment(run_every=st.session_state.get("poi_refresh_seconds", 5))
def _render_manual_length_poi_fragment() -> None:
    _ensure_poi_state()
    ref_dt = _compute_manual_len_poi_ref_dt()
    boat = st.session_state.get("poi_live_boat", "FRA")
    try:
        raw = _fetch_pois(
            ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S),
            ref_dt,
            boat,
            ["boarddrop", "boardraise", "boardmovepenalty"],
        )
        poi_events = _dedup_poi_events(_normalize_poi_events(raw))
        pm = _poi_metrics(poi_events, ref_dt)
    except Exception:
        poi_events = []
        pm = {"count_b": 0, "tr1_b": 0, "tr2_b": 0, "dispo_b": 6, "count_t": 0, "tr1_t": 0, "tr2_t": 0, "dispo_t": 6}

    try:
        df_len = _load_length_db_timeseries(ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S), ref_dt, boat)
    except Exception:
        df_len = None

    fig = _build_poi_length_figure(ref_dt, poi_events, pm, df_len)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        _result_line_html(
            pm["count_b"], pm["tr1_b"], pm["tr2_b"], pm["dispo_b"],
            pm["count_t"], pm["tr1_t"], pm["tr2_t"], pm["dispo_t"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    ref_dt = _render_manual_len_poi_time_controls(ref_dt)

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.session_state.poi_live_boat = st.text_input(
            "Boat code POI/LENGTH_DB",
            value=st.session_state.get("poi_live_boat", "FRA"),
            key="manual_len_poi_boat",
        )
    with c2:
        refresh_s = st.number_input(
            "Refresh POI/LENGTH_DB (s)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("poi_refresh_seconds", 5)),
            step=1,
            key="manual_len_poi_refresh",
        )

        if "poi_refresh_seconds_prev" not in st.session_state:
            st.session_state.poi_refresh_seconds_prev = int(refresh_s)

        if int(refresh_s) != int(st.session_state.poi_refresh_seconds_prev):
            st.session_state.poi_refresh_seconds_prev = int(refresh_s)
            st.session_state.poi_refresh_seconds = int(refresh_s)
            st.rerun()

        st.session_state.poi_refresh_seconds = int(refresh_s)

if mode == "Manuel":
    _render_manual_controls(show_line=False)
    _render_manual_line_fragment()
    st.divider()
    _render_next_start_timer()
elif mode == "POI API":
    _render_poi_modes(combined=False)
elif mode == "Manuel + POI":
    _render_poi_modes(combined=True)

elif mode == "Manuel + LENGTH_DB + POI":
    _render_manual_controls(show_line=False)
    _render_manual_line_fragment()
    _render_manual_length_poi_fragment()

else:
    _render_manual_controls(show_line=False)
    _render_manual_line_fragment()
    _render_manual_length_mode()
