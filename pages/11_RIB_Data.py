from __future__ import annotations

from datetime import datetime, timezone, timedelta
import math

import numpy as np
import pandas as pd
import streamlit as st

from telemetry_io import get_backend, get_cfg, load_channels_timeseries


st.set_page_config(page_title="RIB data", layout="wide")
st.title("RIB data")

REF_BOAT = "FRA"

# Candidate channels. The first available/non-null channel is used where aliases exist.
CANDIDATES = {
    "BSP": ["BOAT_SPEED_km_h_1", "FILT_BOATSPEED_SGP_unk", "FILT_BOATSPEED_TM_unk"],
    "TTS": ["TIME_TC_START_s", "TRK_START_TIME_SFM_s"],
    "Tank volume": ["VOL_TANK_L", "VOL_TANK_MD4_L"],
    "Accu general": ["PRES_ACC_GEN_bar"],
    "Accu rake": ["PRES_ACC_RAKE_bar"],
    "Accu wing": ["PRES_WING_ACC_bar"],
    "Accu CA1": ["PRES_CA1_ACC_bar"],
    "Pump 1 temp": ["CPM_MOTORTEMP_1_C"],
    "Pump 2 temp": ["CPM_MOTORTEMP_2_C"],
    "Pump wing temp": ["CPM_MOTORTEMP_W_C"],
    "Drop port": ["TIME_DROP_P_s", "TIME_DROP_P_AV_s"],
    "Drop starboard": ["TIME_DROP_S_s", "TIME_DROP_S_AV_s"],
    "Battery 1": ["PER_BAT1_SOC_pct"],
    "Battery 2": ["PER_BAT2_SOC_pct"],
    "Battery 3": ["PER_BAT3_SOC_pct"],
    "Battery 4": ["PER_BAT4_SOC_pct"],
    "Battery master": ["PER_BATM_SOC_pct"],
    "Battery backup": ["PER_BAT_SOC_BACKUP_pct"],
    "Battery avg": ["PER_SOC_BAT_AVG_pct"],
    "Stuck global": ["ALARM_STUCK_BUTTON_unk", "STUCK_BUTTON_GLOBAL_unk"],
    "Stuck port": ["ALARM_GLOBAL_STUCK_PORT_unk", "ALARM_STUCK_BUT_PAN_P_unk", "PILOT_P_STUCK_unk", "WHEEL_P_STUCK_unk", "G3K_WT_P_STUCK_unk"],
    "Stuck starboard": ["ALARM_GLOBAL_STUCK_STBD_unk", "ALARM_STUCK_BUT_PAN_S_unk", "PILOT_S_STUCK_unk", "WHEEL_S_STUCK_unk", "G3K_WT_S_STUCK_unk"],
    "CA1": ["ANGLE_CA1_deg"],
    "Wing twist": ["ANGLE_WING_TWIST_deg"],
}

CYLINDERS = {
    "Cant P ext": "PRES_CANT_EXT_P_bar",
    "Cant P ret": "PRES_CANT_RET_P_bar",
    "Cant S ext": "PRES_CANT_EXT_S_bar",
    "Cant S ret": "PRES_CANT_RET_S_bar",
    "DB rake P ext": "PRES_DB_RAKE_EXT_P_bar",
    "DB rake P ret": "PRES_DB_RAKE_RET_P_bar",
    "DB rake S ext": "PRES_DB_RAKE_EXT_S_bar",
    "DB rake S ret": "PRES_DB_RAKE_RET_S_bar",
    "DB UD P ext": "PRES_DB_UD_EXT_P_bar",
    "DB UD P ret": "PRES_DB_UD_RET_P_bar",
    "DB UD S ext": "PRES_DB_UD_EXT_S_bar",
    "DB UD S ret": "PRES_DB_UD_RET_S_bar",
    "Rudder P ext": "PRES_RUD_RAKE_EXT_P_bar",
    "Rudder P ret": "PRES_RUD_RAKE_RET_P_bar",
    "Rudder S ext": "PRES_RUD_RAKE_EXT_S_bar",
    "Rudder S ret": "PRES_RUD_RAKE_RET_S_bar",
    "CA1 A": "PRES_CA1_A_bar",
    "CA1 B": "PRES_CA1_B_bar",
    "CA2 A": "PRES_CA2_A_bar",
    "CA2 B": "PRES_CA2_B_bar",
    "CA3 A": "PRES_CA3_A_bar",
    "CA3 B": "PRES_CA3_B_bar",
    "CA4 A": "PRES_CA4_A_bar",
    "CA4 B": "PRES_CA4_B_bar",
    "CA5 A": "PRES_CA5_A_bar",
    "CA5 B": "PRES_CA5_B_bar",
    "CA6 A": "PRES_CA6_A_bar",
    "CA6 B": "PRES_CA6_B_bar",
    "Jib sheet": "PRES_JIB_SHT_bar",
    "Jib cunno": "PRES_JIB_CUNNO_bar",
    "Jib lead": "PRES_JIB_LEAD_bar",
}

ALL_CHANNELS = sorted({
    ch
    for vals in CANDIDATES.values()
    for ch in vals
} | set(CYLINDERS.values()))


def _last_valid(df: pd.DataFrame, candidates: list[str]):
    if df is None or df.empty:
        return None, None
    d = df.sort_values("time_utc") if "time_utc" in df.columns else df
    for ch in candidates:
        if ch not in d.columns:
            continue
        s = pd.to_numeric(d[ch], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[-1]), ch
    return None, None


def _fmt(value, decimals=1, suffix=""):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def _metric(label, value, unit="", decimals=1, help_text=None):
    st.metric(label, _fmt(value, decimals, f" {unit}" if unit else ""), help=help_text)


def _tts_text(v):
    if v is None or not np.isfinite(v):
        return "N/A"
    sign = "−" if v < 0 else ""
    x = abs(float(v))
    m = int(x // 60)
    s = x - 60 * m
    return f"{sign}{m:02d}:{s:04.1f}"


def _load_live(cfg, boat: str, lookback_s: int = 20):
    stop = datetime.now(timezone.utc)
    start = stop - timedelta(seconds=lookback_s)
    return load_channels_timeseries(
        cfg=cfg,
        boats=[boat],
        channels=ALL_CHANNELS,
        start_utc=start,
        stop_utc=stop,
        every="500ms",
        level_expr="strm|mdss|mdss_fast|raw",
        agg_fn="last",
    )


def _section_bsp_tts(df):
    bsp, bsp_ch = _last_valid(df, CANDIDATES["BSP"])
    tts, tts_ch = _last_valid(df, CANDIDATES["TTS"])

    st.markdown(
        f"""
        <div style="text-align:center;padding-top:3vh">
          <div style="font-size:3.0rem;opacity:.65">BSP</div>
          <div style="font-size:10rem;font-weight:800;line-height:1.0">
            {_fmt(bsp, 1)}
          </div>
          <div style="font-size:2.2rem;opacity:.72">km/h</div>
          <div style="height:3vh"></div>
          <div style="font-size:2.2rem;opacity:.65">TIME TO START</div>
          <div style="font-size:6rem;font-weight:750;line-height:1.05">
            {_tts_text(tts)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Channels utilisés"):
        st.write({"BSP": bsp_ch, "TTS": tts_ch})


def _section_hydro(df):
    st.subheader("Tank / accumulators")
    tank, tank_ch = _last_valid(df, CANDIDATES["Tank volume"])
    a1, _ = _last_valid(df, CANDIDATES["Accu general"])
    a2, _ = _last_valid(df, CANDIDATES["Accu rake"])
    a3, _ = _last_valid(df, CANDIDATES["Accu wing"])
    a4, _ = _last_valid(df, CANDIDATES["Accu CA1"])
    cols = st.columns(5)
    with cols[0]: _metric("Tank volume", tank, "L", 1)
    with cols[1]: _metric("Accu general", a1, "bar", 0)
    with cols[2]: _metric("Accu rake", a2, "bar", 0)
    with cols[3]: _metric("Accu wing", a3, "bar", 0)
    with cols[4]: _metric("Accu CA1", a4, "bar", 0)

    st.subheader("Pump temperatures")
    vals = []
    for key in ["Pump 1 temp", "Pump 2 temp", "Pump wing temp"]:
        vals.append(_last_valid(df, CANDIDATES[key])[0])
    cols = st.columns(3)
    labels = ["Pump 1", "Pump 2", "Wing pump"]
    for c, lab, val in zip(cols, labels, vals):
        with c: _metric(lab, val, "°C", 1)

    st.subheader("Cylinder pressures")
    items = []
    for label, ch in CYLINDERS.items():
        val, used = _last_valid(df, [ch])
        if val is not None:
            items.append((label, val, used))

    if not items:
        st.info("Aucune pression de vérin disponible avec les channels candidats.")
    else:
        for i in range(0, len(items), 4):
            cols = st.columns(4)
            for c, (label, val, used) in zip(cols, items[i:i+4]):
                with c:
                    _metric(label, val, "bar", 0, used)

    with st.expander("Diagnostic hydro"):
        st.write("Tank channel :", tank_ch)
        st.write("Les pressions de vérins sans valeur sont masquées.")


def _section_misc(df):
    st.subheader("Board drop times")
    dp, dp_ch = _last_valid(df, CANDIDATES["Drop port"])
    ds, ds_ch = _last_valid(df, CANDIDATES["Drop starboard"])
    c1, c2 = st.columns(2)
    with c1: _metric("Port", dp, "s", 2)
    with c2: _metric("Starboard", ds, "s", 2)

    st.subheader("Battery SoC")
    battery_keys = ["Battery 1", "Battery 2", "Battery 3", "Battery 4", "Battery master", "Battery backup", "Battery avg"]
    available = []
    for key in battery_keys:
        val, ch = _last_valid(df, CANDIDATES[key])
        if val is not None:
            available.append((key.replace("Battery ", ""), val, ch))
    if available:
        cols = st.columns(min(4, len(available)))
        for i, (label, val, ch) in enumerate(available):
            with cols[i % len(cols)]:
                _metric(label, val, "%", 0, ch)
    else:
        st.info("Aucun Battery SoC disponible avec les channels candidats.")

    st.subheader("Stuck buttons")
    stuck_rows = []
    for key in ["Stuck global", "Stuck port", "Stuck starboard"]:
        val, ch = _last_valid(df, CANDIDATES[key])
        stuck_rows.append((key.replace("Stuck ", "").title(), val, ch))
    cols = st.columns(3)
    for c, (label, val, ch) in zip(cols, stuck_rows):
        with c:
            if val is None:
                st.metric(label, "N/A")
            else:
                status = "STUCK" if abs(val) > 0.5 else "OK"
                st.metric(label, status, help=f"{ch} = {val:g}")

    st.subheader("Wing")
    ca1, ca1_ch = _last_valid(df, CANDIDATES["CA1"])
    twist, twist_ch = _last_valid(df, CANDIDATES["Wing twist"])
    c1, c2 = st.columns(2)
    with c1: _metric("CA1", ca1, "°", 1, ca1_ch)
    with c2: _metric("Wing twist", twist, "°", 1, twist_ch)

    with st.expander("Channels utilisés"):
        st.write({
            "Drop port": dp_ch,
            "Drop starboard": ds_ch,
            "CA1": ca1_ch,
            "Wing twist": twist_ch,
        })


backend = get_backend()
cfg = get_cfg()

with st.sidebar:
    st.header("RIB data")
    view = st.radio("Page", ["1) BSP / TTS", "2) Hydro", "3) Misc"], index=0)
    boat = st.text_input("Boat", value=REF_BOAT).strip().upper() or REF_BOAT
    refresh_s = st.number_input("Refresh (s)", min_value=1, max_value=10, value=1, step=1)
    st.caption(f"Telemetry backend : {backend.upper()}")


@st.fragment(run_every=1)
def _live_panel():
    # The fragment runs every second. If a slower refresh is selected, only
    # reload telemetry when that interval has elapsed.
    now = datetime.now(timezone.utc)
    cache_key = "rib_data_cache"
    cache = st.session_state.get(cache_key, {})
    updated = cache.get("updated_at")
    must_load = (
        not cache
        or cache.get("boat") != boat
        or updated is None
        or (now - updated).total_seconds() >= int(refresh_s)
    )

    if must_load:
        try:
            df = _load_live(cfg, boat)
            err = None
        except Exception as exc:
            df = pd.DataFrame()
            err = f"{type(exc).__name__}: {exc}"
        st.session_state[cache_key] = {
            "updated_at": now,
            "boat": boat,
            "df": df,
            "error": err,
        }
    else:
        df = cache.get("df", pd.DataFrame())
        err = cache.get("error")

    if err:
        st.error(f"Erreur télémétrie : {err}")
        return

    if df is None or df.empty:
        st.warning(f"Aucune donnée récente pour {boat}.")
        return

    if view == "1) BSP / TTS":
        _section_bsp_tts(df)
    elif view == "2) Hydro":
        _section_hydro(df)
    else:
        _section_misc(df)

    if "time_utc" in df.columns:
        t = pd.to_datetime(df["time_utc"], utc=True, errors="coerce").max()
        if pd.notna(t):
            st.caption(f"Dernière donnée : {t.strftime('%H:%M:%S.%f')[:-3]} UTC")


_live_panel()
