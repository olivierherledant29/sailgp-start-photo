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

WINDOW = 60  # seconds
GRAPH_WINDOW = 120  # seconds
RED = "#ff3b30"
GREEN = "#34c759"
GRAY = "#9aa0a6"


# -----------------------------
# Generic helpers
# -----------------------------
def _fmt_mmss(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(int(seconds))
    return f"{sign}{s//60:02d}:{s%60:02d}"


def _colored_dispo(value: int, color_hex: str) -> str:
    return f"<span style='color:{color_hex}; font-weight:700'>{value}</span>"


def _cleanup(history: deque, now: float) -> None:
    while history and (now - history[0] > WINDOW):
        history.popleft()


def _count(history: deque) -> int:
    now = time.time()
    _cleanup(history, now)
    return len(history)


def _timer_until_count(history: deque, target_count: int) -> int:
    """Seconds until the rolling count drops to target_count.

    Example:
    - target_count=5 => former tr_B / now tr_1move
    - target_count=4 => tr_2moves
    """
    now = time.time()
    _cleanup(history, now)
    n = len(history)
    if n <= target_count:
        return 0

    idx = (n - target_count) - 1  # 0-based
    ts_limit = history[idx]
    remaining = int((ts_limit + WINDOW) - now)
    return max(remaining, 0)


def _timer_until_count_from_datetimes(times: list[datetime], ref_dt: datetime, target_count: int) -> int:
    n = len(times)
    if n <= target_count:
        return 0
    idx = (n - target_count) - 1
    limit_dt = times[idx] + timedelta(seconds=WINDOW)
    remaining = int((limit_dt - ref_dt).total_seconds())
    return max(remaining, 0)


def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------
# Next start timer
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
        st.session_state.ns_running = st.toggle("Timer ON", value=st.session_state.ns_running)

    st.caption("Choix du prochain départ (UTC)")
    ch, cm = st.columns([1.2, 1.2])

    with ch:
        bh1, bh2, bh3 = st.columns([0.55, 1.0, 0.55])
        with bh1:
            if st.button("−", key="ns_hour_minus"):
                st.session_state.ns_hour = (int(st.session_state.ns_hour) - 1) % 24
        with bh2:
            st.session_state.ns_hour = st.number_input(
                "Heure", min_value=0, max_value=23, value=int(st.session_state.ns_hour), step=1, key="ns_hour_input"
            )
        with bh3:
            if st.button("+", key="ns_hour_plus"):
                st.session_state.ns_hour = (int(st.session_state.ns_hour) + 1) % 24

    with cm:
        bm1, bm2, bm3 = st.columns([0.55, 1.0, 0.55])
        with bm1:
            if st.button("−", key="ns_min_minus"):
                st.session_state.ns_min = (int(st.session_state.ns_min) - 1) % 60
        with bm2:
            st.session_state.ns_min = st.number_input(
                "Minutes", min_value=0, max_value=59, value=int(st.session_state.ns_min), step=1, key="ns_min_input"
            )
        with bm3:
            if st.button("+", key="ns_min_plus"):
                st.session_state.ns_min = (int(st.session_state.ns_min) + 1) % 60

    st.caption("Offset sur le décompte affiché (dixièmes de seconde)")
    co1, co2, co3, co4 = st.columns([0.9, 1.15, 0.9, 1.25])
    with co1:
        if st.button("−0.1s", key="ns_off_minus"):
            st.session_state.ns_offset_tenths = int(st.session_state.ns_offset_tenths) - 1
    with co2:
        st.session_state.ns_offset_tenths = st.number_input(
            "Offset (x0.1s)",
            value=int(st.session_state.ns_offset_tenths),
            step=1,
            key="ns_off_input",
            help="+1 = +0.1s ajouté au décompte ; -1 = -0.1s",
        )
    with co3:
        if st.button("+0.1s", key="ns_off_plus"):
            st.session_state.ns_offset_tenths = int(st.session_state.ns_offset_tenths) + 1
    with co4:
        offset_s = float(st.session_state.ns_offset_tenths) / 10.0
        st.write(f"Offset = **{offset_s:+.1f}s**")

    now_utc = datetime.now(timezone.utc)
    target = now_utc.replace(
        hour=int(st.session_state.ns_hour),
        minute=int(st.session_state.ns_min),
        second=0,
        microsecond=0,
    )
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

    if st.session_state.ns_running:
        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=1000, key="next_start_refresh")
        else:
            st.info("Installe `streamlit-autorefresh` pour animer le timer en continu (1 Hz).")


# -----------------------------
# Manual mode
# -----------------------------
def _ensure_manual_state() -> None:
    if "press_history" not in st.session_state:
        st.session_state.press_history = {
            "babord": deque(),
            "tribord": deque(),
        }


def _render_manual_mode() -> None:
    _ensure_manual_state()
    c_left, c_right = st.columns([1.6, 5.4])

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
        st.subheader("Live (refresh 1 Hz)")
        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=1000, key="boardcount_refresh")
        else:
            st.info("Astuce: `pip install streamlit-autorefresh` pour un refresh 1 Hz.")

        hist_b = st.session_state.press_history["babord"]
        hist_t = st.session_state.press_history["tribord"]

        count_b = _count(hist_b)
        tr_1move_b = _timer_until_count(hist_b, 5)
        tr_2moves_b = _timer_until_count(hist_b, 4)
        dispo_b = max(6 - count_b, 0)

        count_t = _count(hist_t)
        tr_1move_t = _timer_until_count(hist_t, 5)
        tr_2moves_t = _timer_until_count(hist_t, 4)
        dispo_t = max(6 - count_t, 0)

        line = (
            f"Count_B:{count_b} | tr_1move_B:{tr_1move_b} | tr_2moves_B:{tr_2moves_b} | dispo_BAB:{_colored_dispo(dispo_b, RED)}"
            f" &nbsp;&nbsp;&nbsp; "
            f"Count_T:{count_t} | tr_1move_T:{tr_1move_t} | tr_2moves_T:{tr_2moves_t} | dispo_TRIB:{_colored_dispo(dispo_t, GREEN)}"
        )
        st.markdown(
            f"<div style='font-family: ui-monospace, monospace; font-size:12px; white-space:nowrap; overflow-x:auto'>{line}</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Debug (timestamps dans la fenêtre 60s)"):
            now_ts = time.time()
            _cleanup(hist_b, now_ts)
            _cleanup(hist_t, now_ts)
            st.write(
                {
                    "babord_count": len(hist_b),
                    "tribord_count": len(hist_t),
                    "babord_oldest_utc": datetime.fromtimestamp(hist_b[0], tz=timezone.utc).isoformat() if hist_b else None,
                    "tribord_oldest_utc": datetime.fromtimestamp(hist_t[0], tz=timezone.utc).isoformat() if hist_t else None,
                }
            )


# -----------------------------
# POIs API auto mode
# -----------------------------
def _ensure_api_state() -> None:
    if "poi_analysis_cache" not in st.session_state:
        st.session_state.poi_analysis_cache = {}
    if "poi_fake_play" not in st.session_state:
        st.session_state.poi_fake_play = False
    if "poi_auto_refresh" not in st.session_state:
        st.session_state.poi_auto_refresh = True
    if "poi_fake_date" not in st.session_state:
        st.session_state.poi_fake_date = date(2026, 3, 1)
    if "poi_live_boat" not in st.session_state:
        st.session_state.poi_live_boat = "FRA"
    if "poi_fake_boat" not in st.session_state:
        st.session_state.poi_fake_boat = "ESP"
    if "poi_summary_10m" not in st.session_state:
        st.session_state.poi_summary_10m = None
    if "poi_refresh_seconds" not in st.session_state:
        st.session_state.poi_refresh_seconds = 2
    if "poi_fake_hour" not in st.session_state:
        st.session_state.poi_fake_hour = 6
    if "poi_fake_minute" not in st.session_state:
        st.session_state.poi_fake_minute = 58
    if "poi_fake_cursor_dt" not in st.session_state:
        st.session_state.poi_fake_cursor_dt = datetime(2026, 3, 1, 6, 58, 0, tzinfo=timezone.utc)


def _get_board_side(base_url: str, headers: dict[str, str], poi_id: str) -> str | None:
    cache: dict[str, str | None] = st.session_state.poi_analysis_cache
    if poi_id in cache:
        return cache[poi_id]
    try:
        r = requests.get(f"{base_url.rstrip('/')}/v1/pois/{poi_id}/analysis", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        side = data.get("scalars", {}).get("board_side")
        cache[poi_id] = side
        return side
    except Exception:
        cache[poi_id] = None
        return None


def _fetch_board_events(base_url: str, headers: dict[str, str], boat: str, start_dt: datetime, end_dt: datetime, poi_types: list[str]) -> list[dict[str, Any]]:
    params: list[tuple[str, str]] = [
        ("from", _to_iso_z(start_dt)),
        ("to", _to_iso_z(end_dt)),
        ("boat", boat),
    ]
    for poi_type in poi_types:
        params.append(("poi_type", poi_type))

    r = requests.get(f"{base_url.rstrip('/')}/v1/pois", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()

    events: list[dict[str, Any]] = []
    for p in raw:
        poi_id = p.get("poi_id")
        if not poi_id:
            continue
        evt_dt = datetime.fromisoformat(p["start_datetime"].replace("Z", "+00:00")).astimezone(timezone.utc)
        side = _get_board_side(base_url, headers, poi_id)
        events.append(
            {
                "poi_id": poi_id,
                "type": (p.get("type") or "").lower(),
                "dt": evt_dt,
                "side": side,
                "entity": p.get("entity"),
                "display_name": p.get("display_name"),
            }
        )

    events.sort(key=lambda x: x["dt"])
    return events


def _render_board_events_chart(events_last_2m: list[dict[str, Any]], ref_dt: datetime) -> None:
    fig = go.Figure()
    y_map = {"boarddrop": 1, "boardraise": 0}
    symbol_map = {"boarddrop": "circle", "boardraise": "diamond"}
    side_order = ["port", "starboard", None]
    side_labels = {"port": "Babord", "starboard": "Tribord", None: "Unknown"}
    side_colors = {"port": RED, "starboard": GREEN, None: GRAY}

    for side in side_order:
        for evt_type in ["boarddrop", "boardraise"]:
            pts = [e for e in events_last_2m if e["side"] == side and e["type"] == evt_type]
            if not pts:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[e["dt"] for e in pts],
                    y=[y_map[evt_type] for _ in pts],
                    mode="markers",
                    name=f"{side_labels[side]} - {evt_type}",
                    marker=dict(color=side_colors[side], size=11, symbol=symbol_map[evt_type]),
                    text=[f"{e['display_name']}<br>{side_labels[side]}" for e in pts],
                    hovertemplate="%{text}<br>%{x|%H:%M:%S}<extra></extra>",
                )
            )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(range=[ref_dt - timedelta(seconds=GRAPH_WINDOW), ref_dt], title="Dernières 2 minutes (UTC)"),
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["BoardRaise", "BoardDrop"],
            title="",
            range=[-0.5, 1.5],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)



def _render_api_mode() -> None:
    _ensure_api_state()
    st.subheader("POIs API auto")
    st.caption("Lecture des POIs boarddrop/boardraise puis enrichissement via /v1/pois/{poi_id}/analysis pour récupérer board_side.")

    c1, c2 = st.columns([1.6, 1.4])
    with c1:
        base_url = st.text_input("BASE_URL", value="https://api.f50.sailgp.tech")
    with c2:
        st.text_input("API source", value="SAILGP_POI_TOKEN loaded from .env", disabled=True)

    d1, d2, d3, d4 = st.columns([1.2, 1.6, 1.1, 1.1])
    with d1:
        time_mode = st.radio("Horloge", ["Live", "Faux live"], horizontal=True)
    with d2:
        poi_types = st.multiselect(
            "POI types",
            options=["boarddrop", "boardraise"],
            default=["boarddrop", "boardraise"],
        )
    with d3:
        st.session_state.poi_auto_refresh = st.toggle(
            "Auto refresh",
            value=st.session_state.poi_auto_refresh,
            key="poi_auto_refresh_toggle",
        )
    with d4:
        st.session_state.poi_refresh_seconds = st.number_input(
            "Période refresh (s)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.poi_refresh_seconds),
            step=1,
            key="poi_refresh_seconds_input",
        )


    if time_mode == "Live":
        st.session_state.poi_live_boat = st.text_input("Boat code (live)", value=st.session_state.poi_live_boat)
        boat = st.session_state.poi_live_boat
    else:
        f1, f2, f3, f4, f5, f6 = st.columns([1.1, 0.8, 0.8, 1.0, 1.0, 1.0])
        with f1:
            st.session_state.poi_fake_date = st.date_input(
                "Date faux live",
                value=st.session_state.poi_fake_date,
                key="poi_fake_date_input",
            )
        with f2:
            st.session_state.poi_fake_hour = st.number_input(
                "Heure UTC",
                min_value=0,
                max_value=23,
                value=int(st.session_state.poi_fake_hour),
                step=1,
                key="poi_fake_hour_input",
            )
        with f3:
            st.session_state.poi_fake_minute = st.number_input(
                "Minute UTC",
                min_value=0,
                max_value=59,
                value=int(st.session_state.poi_fake_minute),
                step=1,
                key="poi_fake_minute_input",
            )
        with f4:
            st.session_state.poi_fake_boat = st.text_input(
                "Boat code (faux live)",
                value=st.session_state.poi_fake_boat,
                key="poi_fake_boat_input",
            )
        with f5:
            st.session_state.poi_fake_play = st.toggle(
                "Lecture faux live",
                value=st.session_state.poi_fake_play,
                key="poi_fake_play_toggle",
            )
        with f6:
            if st.button("Reset faux live", use_container_width=True):
                st.session_state.poi_fake_cursor_dt = datetime(
                    st.session_state.poi_fake_date.year,
                    st.session_state.poi_fake_date.month,
                    st.session_state.poi_fake_date.day,
                    int(st.session_state.poi_fake_hour),
                    int(st.session_state.poi_fake_minute),
                    0,
                    tzinfo=timezone.utc,
                )
        boat = st.session_state.poi_fake_boat

    refresh_ms = int(st.session_state.poi_refresh_seconds) * 1000

    if st.session_state.poi_auto_refresh and _HAS_AUTOREFRESH:
        if time_mode == "Live":
            ref_dt = datetime.now(timezone.utc)
            st_autorefresh(interval=refresh_ms, key="poi_live_refresh")
        elif st.session_state.poi_fake_play:
            ref_dt = st.session_state.poi_fake_cursor_dt
            st_autorefresh(interval=refresh_ms, key="poi_fake_refresh")
            st.session_state.poi_fake_cursor_dt = ref_dt + timedelta(seconds=int(st.session_state.poi_refresh_seconds))
        else:
            ref_dt = st.session_state.poi_fake_cursor_dt
    elif st.session_state.poi_auto_refresh and not _HAS_AUTOREFRESH:
        st.info("Installe `streamlit-autorefresh` pour le mode live/faux live animé.")
        if time_mode == "Live":
            ref_dt = datetime.now(timezone.utc)
        else:
            ref_dt = st.session_state.poi_fake_cursor_dt
    else:
        if time_mode == "Live":
            ref_dt = datetime.now(timezone.utc)
        else:
            ref_dt = st.session_state.poi_fake_cursor_dt

    st.caption(f"Heure de référence UTC : {ref_dt.strftime('%Y-%m-%d %H:%M:%S')}  |  Boat: {boat}")
    debug_poi = st.toggle(
        "Debug POI render path",
        value=False,
        help="Affiche des marqueurs pour voir jusqu'où le rendu va dans le bloc POIs API auto.",
    )
    if debug_poi:
        st.info("DEBUG 1: bloc POIs API auto actif")

    headers = {"Authorization": f"Bearer {SAILGP_POI_TOKEN}"}

    try:
        if debug_poi:
            st.info("DEBUG 2: appel _fetch_board_events() sur 2 minutes")
        events_2m = _fetch_board_events(
            base_url=base_url,
            headers=headers,
            boat=boat,
            start_dt=ref_dt - timedelta(seconds=GRAPH_WINDOW),
            end_dt=ref_dt,
            poi_types=poi_types,
        )
    except Exception as e:
        st.error("DEBUG ERROR: _fetch_board_events() a échoué")
        st.exception(e)
        return

    events_60s = [e for e in events_2m if e["dt"] >= ref_dt - timedelta(seconds=WINDOW)]
    if debug_poi:
        st.info(f"DEBUG 3: events_2m={len(events_2m)} | events_60s={len(events_60s)}")
    babord_times = [e["dt"] for e in events_60s if e["side"] == "port"]
    tribord_times = [e["dt"] for e in events_60s if e["side"] == "starboard"]

    count_b = len(babord_times)
    tr_1move_b = _timer_until_count_from_datetimes(babord_times, ref_dt, 5)
    tr_2moves_b = _timer_until_count_from_datetimes(babord_times, ref_dt, 4)
    dispo_b = max(6 - count_b, 0)

    count_t = len(tribord_times)
    tr_1move_t = _timer_until_count_from_datetimes(tribord_times, ref_dt, 5)
    tr_2moves_t = _timer_until_count_from_datetimes(tribord_times, ref_dt, 4)
    dispo_t = max(6 - count_t, 0)

    if debug_poi:
        st.success("DEBUG 4: section ligne de résultats atteinte")
    st.markdown(
        f"""
        <div style="
            width:100%;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size:18px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:24px;
            white-space:nowrap;
            overflow-x:auto;
        ">
            <div>
                Count_B:{count_b} |
                tr_1move_B:{tr_1move_b} |
                tr_2moves_B:{tr_2moves_b} |
                dispo_BAB:{_colored_dispo(dispo_b, RED)}
            </div>
            <div>
                Count_T:{count_t} |
                tr_1move_T:{tr_1move_t} |
                tr_2moves_T:{tr_2moves_t} |
                dispo_TRIB:{_colored_dispo(dispo_t, GREEN)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if debug_poi:
        st.success("DEBUG 5: section bouton Bilan 10 min atteinte")
    csum1, csum2 = st.columns([1.2, 4.0])
    with csum1:
        compute_summary = st.button("Bilan 10 min", use_container_width=True)
    with csum2:
        st.caption("Calcul ponctuel sur les 10 dernières minutes par rapport à l'heure de référence. Pas de refresh auto tant que tu ne réappuies pas.")

    if compute_summary:
        try:
            events_10m = _fetch_board_events(
                base_url=base_url,
                headers=headers,
                boat=boat,
                start_dt=ref_dt - timedelta(minutes=10),
                end_dt=ref_dt,
                poi_types=poi_types,
            )
            summary = {
                "ref_dt": ref_dt.isoformat(),
                "boat": boat,
                "boarddrop_port": sum(1 for e in events_10m if e["type"] == "boarddrop" and e["side"] == "port"),
                "boarddrop_starboard": sum(1 for e in events_10m if e["type"] == "boarddrop" and e["side"] == "starboard"),
                "boardraise_port": sum(1 for e in events_10m if e["type"] == "boardraise" and e["side"] == "port"),
                "boardraise_starboard": sum(1 for e in events_10m if e["type"] == "boardraise" and e["side"] == "starboard"),
                "unknown_side": sum(1 for e in events_10m if e["side"] not in ("port", "starboard")),
            }
            st.session_state.poi_summary_10m = summary
        except Exception as e:
            st.session_state.poi_summary_10m = {"error": str(e)}

    summary = st.session_state.poi_summary_10m
    if summary and summary.get("boat") == boat:
        if "error" in summary:
            st.error(f"Bilan 10 min impossible: {summary['error']}")
        else:
            st.markdown(
                (
                    "**Bilan 10 min** — "
                    f"BoardDrop Babord: {summary['boarddrop_port']} | "
                    f"BoardDrop Tribord: {summary['boarddrop_starboard']} | "
                    f"BoardRaise Babord: {summary['boardraise_port']} | "
                    f"BoardRaise Tribord: {summary['boardraise_starboard']} | "
                    f"Unknown: {summary['unknown_side']}"
                )
            )

    if debug_poi:
        st.success("DEBUG 6: rendu graphe 2 minutes")
    _render_board_events_chart(events_2m, ref_dt)

    with st.expander("Debug API / analysis"):
        unknown = [e for e in events_2m if e["side"] not in ("port", "starboard")]
        st.write(
            {
                "ref_time_utc": ref_dt.isoformat(),
                "events_last_2m": len(events_2m),
                "events_last_60s": len(events_60s),
                "unknown_side_last_2m": len(unknown),
                "analysis_cache_size": len(st.session_state.poi_analysis_cache),
            }
        )
        if events_2m:
            preview = []
            for e in events_2m[:15]:
                preview.append(
                    {
                        "time": e["dt"].isoformat(),
                        "type": e["type"],
                        "side": e["side"],
                        "poi_id": e["poi_id"],
                    }
                )
            st.json(preview)


# -----------------------------
# UI
# -----------------------------
mode = st.radio(
    "Mode",
    ["Manuel (boutons Streamlit)", "POIs API auto"],
    horizontal=True,
)

if mode == "Manuel (boutons Streamlit)":
    st.caption(
        "Dans Streamlit (navigateur), on ne peut pas capturer les touches clavier système (a/z) comme dans un script terminal. "
        "Le mode manuel ci-dessous remplace ça par des boutons + undo, avec la même logique de fenêtre glissante 60s."
    )
    _render_manual_mode()
else:
    _render_api_mode()

st.divider()
_render_next_start_timer()
