import os
import time
from collections import deque
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

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
GRAY = "#9aa0a6"


def _fmt_mmss(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(int(seconds))
    return f"{sign}{s//60:02d}:{s%60:02d}"


def _colored_dispo(value: int, color_hex: str) -> str:
    return f"<span style='color:{color_hex}; font-weight:700'>{value}</span>"


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


def _render_manual_controls() -> None:
    _ensure_manual_state()
    c_left, c_right = st.columns([1.8, 5.2])
    with c_left:
        st.subheader("Babord")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("+1 B", use_container_width=True):
                st.session_state.press_history["babord"].append(time.time())
        with c2:
            if st.button("Undo B", use_container_width=True):
                if st.session_state.press_history["babord"]:
                    st.session_state.press_history["babord"].popleft()

        st.subheader("Tribord")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("+1 T", use_container_width=True):
                st.session_state.press_history["tribord"].append(time.time())
        with c4:
            if st.button("Undo T", use_container_width=True):
                if st.session_state.press_history["tribord"]:
                    st.session_state.press_history["tribord"].popleft()

    with c_right:
        m = _manual_metrics()
        st.markdown(
            f"""
            <div style="width:100%; font-size:18px; display:flex; justify-content:space-between; white-space:nowrap;">
                <div>
                    Count_B:{m['count_b']} |
                    tr_1move_B:{m['tr1_b']} |
                    tr_2moves_B:{m['tr2_b']} |
                    dispo_BAB:{_colored_dispo(m['dispo_b'], RED)}
                </div>
                <div>
                    Count_T:{m['count_t']} |
                    tr_1move_T:{m['tr1_t']} |
                    tr_2moves_T:{m['tr2_t']} |
                    dispo_TRIB:{_colored_dispo(m['dispo_t'], GREEN)}
                </div>
            </div>
            """,
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
        "poi_refresh_seconds": 2,
        "poi_fake_date": date(2026, 3, 1),
        "poi_fake_hour": 6,
        "poi_fake_minute": 55,
        "poi_fake_play": False,
        "poi_fake_cursor_dt": datetime(2026, 3, 1, 6, 55, 0, tzinfo=timezone.utc),
        "poi_live_boat": "FRA",
        "poi_fake_boat": "ESP",
        "poi_dedup": False,
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
        poi_type = (p.get("type") or "").lower()
        if poi_type not in {"boarddrop", "boardraise"}:
            continue
        poi_id = p.get("poi_id")
        start_dt = p.get("start_datetime")
        if not poi_id or not start_dt:
            continue
        try:
            dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        side = _get_board_side(str(poi_id))
        if side not in {"port", "starboard"}:
            side = None
        events.append({
            "poi_id": str(poi_id),
            "dt": dt,
            "type": poi_type,
            "side": side,
        })
    events.sort(key=lambda x: x["dt"])
    return events


def _dedup_poi_events(events: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return events
    last_type_by_side: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for e in events:
        side = e.get("side")
        typ = e.get("type")
        if side not in {"port", "starboard"}:
            out.append(e)
            continue
        if last_type_by_side.get(side) == typ:
            continue
        last_type_by_side[side] = typ
        out.append(e)
    return out


def _count_side_events_last_minute(events: list[dict[str, Any]], ref_dt: datetime, side: str) -> list[datetime]:
    cutoff = ref_dt - timedelta(seconds=WINDOW)
    times = [e["dt"] for e in events if e.get("side") == side and cutoff <= e["dt"] <= ref_dt]
    times.sort()
    return times


def _timer_until_count_from_datetimes(times: list[datetime], ref_dt: datetime, target_count: int) -> int:
    n = len(times)
    if n <= target_count:
        return 0
    idx = (n - target_count) - 1
    limit_dt = times[idx] + timedelta(seconds=WINDOW)
    return max(int((limit_dt - ref_dt).total_seconds()), 0)


def _poi_metrics(events: list[dict[str, Any]], ref_dt: datetime) -> dict[str, int]:
    bab = _count_side_events_last_minute(events, ref_dt, "port")
    tri = _count_side_events_last_minute(events, ref_dt, "starboard")
    return {
        "count_b": len(bab),
        "tr1_b": _timer_until_count_from_datetimes(bab, ref_dt, 5),
        "tr2_b": _timer_until_count_from_datetimes(bab, ref_dt, 4),
        "dispo_b": max(6 - len(bab), 0),
        "count_t": len(tri),
        "tr1_t": _timer_until_count_from_datetimes(tri, ref_dt, 5),
        "tr2_t": _timer_until_count_from_datetimes(tri, ref_dt, 4),
        "dispo_t": max(6 - len(tri), 0),
    }


def _poi_summary(events: list[dict[str, Any]], ref_dt: datetime) -> dict[str, int]:
    start = ref_dt - timedelta(minutes=10)
    summary = {
        "drop_b": 0, "drop_t": 0, "raise_b": 0, "raise_t": 0, "unknown": 0,
    }
    for e in events:
        if not (start <= e["dt"] <= ref_dt):
            continue
        side = e.get("side")
        typ = e.get("type")
        if side == "port" and typ == "boarddrop":
            summary["drop_b"] += 1
        elif side == "starboard" and typ == "boarddrop":
            summary["drop_t"] += 1
        elif side == "port" and typ == "boardraise":
            summary["raise_b"] += 1
        elif side == "starboard" and typ == "boardraise":
            summary["raise_t"] += 1
        else:
            summary["unknown"] += 1
    return summary


# -----------------------------
# Graph
# -----------------------------
def _build_comparison_figure(
    ref_dt: datetime,
    poi_events: list[dict[str, Any]],
    poi_metrics: dict[str, int] | None,
    manual_events: list[dict[str, Any]] | None = None,
) -> go.Figure:
    x0 = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
    x1 = ref_dt + timedelta(seconds=GRAPH_LOOKAHEAD_S)

    fig = go.Figure()

    # Y lanes (top -> bottom via reversed axis)
    lanes = {
        "raise_t": 5,
        "manual_t": 4,
        "drop_t": 3,
        "raise_b": 2,
        "manual_b": 1,
        "drop_b": 0,
    }

    # POI traces (circles)
    for side, color in [("port", RED), ("starboard", GREEN)]:
        for typ, lane_key, name in [
            ("boardraise", "raise_b" if side == "port" else "raise_t", f"POI {'BAB' if side=='port' else 'TRIB'} raise"),
            ("boarddrop", "drop_b" if side == "port" else "drop_t", f"POI {'BAB' if side=='port' else 'TRIB'} drop"),
        ]:
            pts = [e for e in poi_events if e.get("side") == side and e.get("type") == typ and x0 <= e["dt"] <= x1]
            if not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[e["dt"] for e in pts],
                y=[lanes[lane_key]] * len(pts),
                mode="markers",
                marker=dict(symbol="circle", size=10, color=color),
                name=name,
                hovertemplate="%{x|%H:%M:%S}<extra>" + name + "</extra>",
            ))

    # Manual traces (crosses)
    if manual_events:
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

    # Vertical reference lines
    for when, color, width in [
        (ref_dt - timedelta(minutes=2), YELLOW, 1),
        (ref_dt - timedelta(minutes=1), RED, 1),
        (ref_dt, BLUE, 2),
    ]:
        fig.add_vline(x=when, line_color=color, line_width=width)

    # Future timer markers from POI only
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
                    fig.add_annotation(x=x, y=5.4, text=label, showarrow=False, font=dict(color=ORANGE, size=10))

    # 10-second grid
    tick0 = x0.replace(microsecond=0)
    fig.update_xaxes(
        range=[x0, x1],
        tick0=tick0,
        dtick=10000,  # ms on date axis = 10 s
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)",
        tickformat="%H:%M:%S",
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[5, 4, 3, 2, 1, 0],
        ticktext=["Raise T", "Manual T", "Drop T", "Raise B", "Manual B", "Drop B"],
        range=[-0.5, 5.5],
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


def _manual_events_last_graph_window(ref_dt: datetime) -> list[dict[str, Any]]:
    _ensure_manual_state()
    start = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
    out: list[dict[str, Any]] = []
    for side_key, side_name in [("babord", "port"), ("tribord", "starboard")]:
        for ts in list(st.session_state.press_history[side_key]):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if start <= dt <= ref_dt + timedelta(seconds=GRAPH_LOOKAHEAD_S):
                out.append({"dt": dt, "side": side_name, "type": "manual"})
    out.sort(key=lambda x: x["dt"])
    return out


# -----------------------------
# POI mode renderer
# -----------------------------
def _render_poi_modes(combined: bool) -> None:
    _ensure_poi_state()
    if combined:
        st.subheader("Mode Manuel + POI")
        _render_manual_controls()
        st.markdown("---")
    else:
        st.subheader("Mode POI API")

    top = st.columns([1.1, 1.2, 1.2, 1.2, 1.4])
    with top[0]:
        time_mode = st.radio("Horloge", ["Live", "Faux live"] if not combined else ["Live"], horizontal=True)
    with top[1]:
        st.session_state.poi_auto_refresh = st.toggle("Auto refresh", value=st.session_state.poi_auto_refresh, key=f"poi_auto_refresh_{'combo' if combined else 'poi'}")
    with top[2]:
        st.session_state.poi_refresh_seconds = st.number_input("Refresh (s)", 1, 30, int(st.session_state.poi_refresh_seconds), key=f"poi_refresh_s_{'combo' if combined else 'poi'}")
    with top[3]:
        st.session_state.poi_dedup = st.toggle("Dédup moves", value=st.session_state.poi_dedup, key=f"poi_dedup_{'combo' if combined else 'poi'}")
    with top[4]:
        boat = st.text_input("Boat code", value=(st.session_state.poi_live_boat if time_mode == "Live" else st.session_state.poi_fake_boat), key=f"boat_input_{'combo' if combined else time_mode}")
        if time_mode == "Live":
            st.session_state.poi_live_boat = boat
        else:
            st.session_state.poi_fake_boat = boat

    if time_mode == "Faux live":
        row = st.columns([1.3, 0.9, 0.9, 1.1, 1.2])
        with row[0]:
            st.session_state.poi_fake_date = st.date_input("Date", value=st.session_state.poi_fake_date, key="poi_fake_date_input")
        with row[1]:
            st.session_state.poi_fake_hour = st.number_input("Heure UTC", 0, 23, int(st.session_state.poi_fake_hour), key="poi_fake_hour_input")
        with row[2]:
            st.session_state.poi_fake_minute = st.number_input("Minute UTC", 0, 59, int(st.session_state.poi_fake_minute), key="poi_fake_minute_input")
        with row[3]:
            st.session_state.poi_fake_play = st.toggle("Lecture faux live", value=st.session_state.poi_fake_play, key="poi_fake_play_toggle")
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

    # Reference time
    refresh_ms = int(st.session_state.poi_refresh_seconds) * 1000
    if time_mode == "Live":
        ref_dt = datetime.now(timezone.utc)
        if st.session_state.poi_auto_refresh and _HAS_AUTOREFRESH:
            st_autorefresh(interval=refresh_ms, key=f"poi_live_refresh_{'combo' if combined else 'poi'}")
    else:
        if st.session_state.poi_auto_refresh and st.session_state.poi_fake_play and _HAS_AUTOREFRESH:
            st_autorefresh(interval=refresh_ms, key=f"poi_fake_refresh_{'combo' if combined else 'poi'}")
            ref_dt = st.session_state.poi_fake_cursor_dt
            st.session_state.poi_fake_cursor_dt = ref_dt + timedelta(seconds=int(st.session_state.poi_refresh_seconds))
        else:
            ref_dt = st.session_state.poi_fake_cursor_dt

    st.caption(f"Heure de référence UTC : {ref_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Data fetch windows
    try:
        start_graph = ref_dt - timedelta(seconds=GRAPH_LOOKBACK_S)
        raw_pois = _fetch_pois(start_graph, ref_dt, boat, ["boarddrop", "boardraise"])
        poi_events = _dedup_poi_events(_normalize_poi_events(raw_pois), bool(st.session_state.poi_dedup))
        pm = _poi_metrics(poi_events, ref_dt)
    except Exception as e:
        st.error(f"Lecture POI impossible: {e}")
        return

    if not combined:
        st.markdown(
            f"""
            <div style="width:100%; font-size:18px; display:flex; justify-content:space-between; white-space:nowrap;">
                <div>
                    Count_B:{pm['count_b']} |
                    tr_1move_B:{pm['tr1_b']} |
                    tr_2moves_B:{pm['tr2_b']} |
                    dispo_BAB:{_colored_dispo(pm['dispo_b'], RED)}
                </div>
                <div>
                    Count_T:{pm['count_t']} |
                    tr_1move_T:{pm['tr1_t']} |
                    tr_2moves_T:{pm['tr2_t']} |
                    dispo_TRIB:{_colored_dispo(pm['dispo_t'], GREEN)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    manual_events = _manual_events_last_graph_window(ref_dt) if combined else None
    fig = _build_comparison_figure(ref_dt, poi_events, pm, manual_events)
    st.plotly_chart(fig, use_container_width=True)

    if combined:
        st.markdown(
            f"""
            <div style="width:100%; font-size:18px; display:flex; justify-content:space-between; white-space:nowrap;">
                <div>
                    Count_B:{pm['count_b']} |
                    tr_1move_B:{pm['tr1_b']} |
                    tr_2moves_B:{pm['tr2_b']} |
                    dispo_BAB:{_colored_dispo(pm['dispo_b'], RED)}
                </div>
                <div>
                    Count_T:{pm['count_t']} |
                    tr_1move_T:{pm['tr1_t']} |
                    tr_2moves_T:{pm['tr2_t']} |
                    dispo_TRIB:{_colored_dispo(pm['dispo_t'], GREEN)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Bilan 10 min", key=f"summary_btn_{'combo' if combined else 'poi'}"):
        try:
            raw_10m = _fetch_pois(ref_dt - timedelta(minutes=10), ref_dt, boat, ["boarddrop", "boardraise"])
            ev_10m = _dedup_poi_events(_normalize_poi_events(raw_10m), bool(st.session_state.poi_dedup))
            summary = _poi_summary(ev_10m, ref_dt)
            st.success(
                f"**Bilan 10 min** — "
                f"Drop B:{summary['drop_b']} | Drop T:{summary['drop_t']} | "
                f"Raise B:{summary['raise_b']} | Raise T:{summary['raise_t']} | Unknown:{summary['unknown']}"
            )
        except Exception as e:
            st.error(f"Bilan 10 min impossible: {e}")


# -----------------------------
# Page body
# -----------------------------
mode = st.radio("Mode", ["Manuel", "POI API", "Manuel + POI"], horizontal=True)

if mode == "Manuel":
    _render_manual_controls()
    st.divider()
    _render_next_start_timer()
elif mode == "POI API":
    _render_poi_modes(combined=False)
else:
    _render_poi_modes(combined=True)
