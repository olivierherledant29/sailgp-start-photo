import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    st_autorefresh = None  # type: ignore
    _HAS_AUTOREFRESH = False

st.set_page_config(page_title="Board cycles count", layout="wide")
st.title("Board cycles count")

WINDOW = 60  # seconds

# Streamlit HTML colors
RED = "#ff3b30"
GREEN = "#34c759"


def _fmt_mmss(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(int(seconds))
    return f"{sign}{s//60:02d}:{s%60:02d}"


def _render_next_start_timer() -> None:
    """Next start countdown (UTC) + display offset (0.1s steps).

    Streamlit-friendly: no infinite loop; we re-run at 1 Hz only when ON.
    """

    # -----------------------
    # Session state
    # -----------------------
    if "ns_running" not in st.session_state:
        st.session_state.ns_running = False

    now_utc = datetime.now(timezone.utc)
    if "ns_hour" not in st.session_state:
        st.session_state.ns_hour = now_utc.hour
    if "ns_min" not in st.session_state:
        st.session_state.ns_min = (now_utc.minute + 1) % 60
    if "ns_offset_tenths" not in st.session_state:
        st.session_state.ns_offset_tenths = 0  # 0.1s steps

    # -----------------------
    # Header row: clock + on/off
    # -----------------------
    c0, c1 = st.columns([1.2, 1.0])
    with c0:
        st.metric("UTC", now_utc.strftime("%H:%M:%S"))
    with c1:
        st.session_state.ns_running = st.toggle("Timer ON", value=st.session_state.ns_running)

    st.caption("Choix du prochain départ (UTC)")

    # -----------------------
    # Start selection: hour + minute with +/-
    # -----------------------
    ch, cm = st.columns([1.2, 1.2])

    with ch:
        bh1, bh2, bh3 = st.columns([0.6, 1.0, 0.6])
        with bh1:
            if st.button("−", key="ns_hour_minus"):
                st.session_state.ns_hour = (int(st.session_state.ns_hour) - 1) % 24
        with bh2:
            st.session_state.ns_hour = st.number_input(
                "Heure",
                min_value=0,
                max_value=23,
                value=int(st.session_state.ns_hour),
                step=1,
                key="ns_hour_input",
            )
        with bh3:
            if st.button("+", key="ns_hour_plus"):
                st.session_state.ns_hour = (int(st.session_state.ns_hour) + 1) % 24

    with cm:
        bm1, bm2, bm3 = st.columns([0.6, 1.0, 0.6])
        with bm1:
            if st.button("−", key="ns_min_minus"):
                st.session_state.ns_min = (int(st.session_state.ns_min) - 1) % 60
        with bm2:
            st.session_state.ns_min = st.number_input(
                "Minutes",
                min_value=0,
                max_value=59,
                value=int(st.session_state.ns_min),
                step=1,
                key="ns_min_input",
            )
        with bm3:
            if st.button("+", key="ns_min_plus"):
                st.session_state.ns_min = (int(st.session_state.ns_min) + 1) % 60

    # -----------------------
    # Offset display (0.1s steps)
    # -----------------------
    st.caption("Offset sur le décompte affiché (dixièmes de seconde)")
    co1, co2, co3, co4 = st.columns([0.9, 1.2, 0.9, 1.4])

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

    # -----------------------
    # Compute target datetime (UTC) for today; if passed => tomorrow
    # -----------------------
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
<div style="text-align:center; margin-top:20px;">

  <div style="
        font-size:120px;
        font-weight:700;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color:#00BFFF;">
        {_fmt_mmss(tts)}
  </div>

  <div style="
        font-size:120px;
        font-weight:700;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color:#FF7F00;">
        {_fmt_mmss(tts_corr)}
  </div>

  <div style="margin-top:10px; font-size:18px; opacity:0.7;">
        Offset: {offset_s:+.1f}s
  </div>

</div>
""",
    unsafe_allow_html=True
)


    # -----------------------
    # Auto-refresh 1 Hz when ON
    # -----------------------
    if st.session_state.ns_running:
        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=1000, key="next_start_refresh")
        else:
            st.info("Installe `streamlit-autorefresh` pour animer le timer en continu (1 Hz).")


def _ensure_state():
    """Initialize session_state storage."""
    if "press_history" not in st.session_state:
        st.session_state.press_history = {
            "babord": deque(),   # timestamps (float seconds)
            "tribord": deque(),
        }


def _cleanup(history: deque, now: float) -> None:
    while history and (now - history[0] > WINDOW):
        history.popleft()


def _count(history: deque) -> int:
    now = time.time()
    _cleanup(history, now)
    return len(history)


def _timer_to_five(history: deque) -> int:
    """
    Seconds until count drops to 5.
    Correct logic: timer is based on the (count-5)-th oldest event.
    """
    now = time.time()
    _cleanup(history, now)
    n = len(history)
    if n < 6:
        return 0

    # Need to lose (n-5) events; the last one to expire is index (n-5)
    idx = (n - 5) - 1  # 0-based
    ts_limit = history[idx]
    remaining = int((ts_limit + WINDOW) - now)
    return max(remaining, 0)


def _colored_dispo(value: int, color_hex: str) -> str:
    return f"<span style='color:{color_hex}; font-weight:700'>{value}</span>"


# -----------------------------
# UI
# -----------------------------
_ensure_state()

mode = st.radio(
    "Mode",
    ["Manuel (boutons Streamlit)", "POIs API (beta – lecture DB)"],
    horizontal=True,
)

st.caption(
    "Note: dans Streamlit (navigateur), on ne peut pas capturer les touches clavier système (a/z) comme dans un script terminal. "
    "Le mode manuel ci-dessous remplace ça par des boutons + undo, avec la même logique de fenêtre glissante 60s et timers."
)

if mode == "Manuel (boutons Streamlit)":
    colA, colB, colC = st.columns([1.2, 1.2, 2.6])

    with colA:
        st.subheader("Babord")
        if st.button("➕ +1 Babord", use_container_width=True):
            st.session_state.press_history["babord"].append(time.time())
        if st.button("↩️ Undo Babord (-1)", use_container_width=True):
            if st.session_state.press_history["babord"]:
                st.session_state.press_history["babord"].popleft()

    with colB:
        st.subheader("Tribord")
        if st.button("➕ +1 Tribord", use_container_width=True):
            st.session_state.press_history["tribord"].append(time.time())
        if st.button("↩️ Undo Tribord (-1)", use_container_width=True):
            if st.session_state.press_history["tribord"]:
                st.session_state.press_history["tribord"].popleft()

    with colC:
        st.subheader("Live (refresh 1 Hz)")
        st.caption("Affichage identique à ton format console, mais rendu dans Streamlit.")

        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=1000, key="boardcount_refresh")
        else:
            st.info("Astuce: `pip install streamlit-autorefresh` pour un refresh 1 Hz.")


        hist_b = st.session_state.press_history["babord"]
        hist_t = st.session_state.press_history["tribord"]

        count_b = _count(hist_b)
        tr_b = _timer_to_five(hist_b)
        dispo_b = max(6 - count_b, 0)

        count_t = _count(hist_t)
        tr_t = _timer_to_five(hist_t)
        dispo_t = max(6 - count_t, 0)

        line = (
            f"Count_B: {count_b}  |  tr_B: {tr_b}  |  dispo_BAB: {_colored_dispo(dispo_b, RED)}"
            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            f"Count_T: {count_t}  |  tr_T: {tr_t}  |  dispo_TRIB: {_colored_dispo(dispo_t, GREEN)}"
        )
        st.markdown(line, unsafe_allow_html=True)

        with st.expander("Debug (timestamps dans la fenêtre 60s)"):
            now = time.time()
            _cleanup(hist_b, now)
            _cleanup(hist_t, now)
            st.write(
                {
                    "babord_count": len(hist_b),
                    "tribord_count": len(hist_t),
                    "babord_oldest_utc": datetime.fromtimestamp(hist_b[0], tz=timezone.utc).isoformat() if hist_b else None,
                    "tribord_oldest_utc": datetime.fromtimestamp(hist_t[0], tz=timezone.utc).isoformat() if hist_t else None,
                }
            )

else:
    st.subheader("POIs API (beta)")
    st.write(
        "Objectif: remplacer les appuis manuels par des événements POI (ex: board cycles / half cycles) "
        "récupérés via l'API SailGP."
    )

    base_url = st.text_input("BASE_URL", value="https://api.f50.sailgp.tech")
    api_key = st.text_input("API key (Bearer)", value=os.getenv("SAILGP_POI_TOKEN", ""), type="password")
    race_id = st.text_input("Race ID (ex: 24112301)")
    boat = st.text_input("Boat code (ex: AUS, FRA, NZL, ...)", value="FRA")

    if st.button("Récupérer les POIs", use_container_width=True):
        if not api_key:
            st.error("Renseigne une API key (Bearer).")
        elif not race_id or not boat:
            st.error("Renseigne race_id et boat.")
        else:
            try:
                headers = {"Authorization": f"Bearer {api_key}"}
                url = f"{base_url.rstrip('/')}/v1/races/{race_id}/boats/{boat}/pois"
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                st.session_state.pois = r.json()
                st.success(f"OK — {len(st.session_state.pois)} POIs reçus.")
            except Exception as e:
                st.exception(e)

    pois = st.session_state.get("pois", [])
    if pois:
        types = sorted({p.get("type") for p in pois if p.get("type")})
        st.write("Types disponibles:", types)

        needle = st.text_input("Filtrer les types (contient)", value="board")
        filt_types = [t for t in types if needle.lower() in t.lower()] if needle else types
        st.write("Types filtrés:", filt_types)

        st.info(
            "Prochaine étape: choisir quel `type` correspond à babord/tribord (half-cycle, board-cycle, etc.), "
            "puis calculer les compteurs sur 60s et les timers comme en mode manuel."
        )

        st.write("Aperçu POIs (20 premiers):")
        st.json(pois[:20])


st.divider()
_render_next_start_timer()
