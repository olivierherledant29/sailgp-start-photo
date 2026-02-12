import os
import time
from collections import deque
from datetime import datetime, timezone

import requests
import streamlit as st

from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Board cycles count", layout="wide")
st.title("Board cycles count")

WINDOW = 60  # seconds

# Streamlit HTML colors
RED = "#ff3b30"
GREEN = "#34c759"


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
        
        st_autorefresh(interval=1000, key="boardcount_refresh")


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
