from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

from telemetry_io import ALL_BOATS, get_backend, get_cfg, load_channels_timeseries

st.set_page_config(page_title="Live Targets", layout="wide")
st.title("Live Targets — telemetry vs target")

# =============================================================================
# CHANNELS — TimescaleDB sgp_telemetry
# =============================================================================
CH_BSP = "BOAT_SPEED_km_h_1"
CH_TWA = "TWA_MHU_SGP_deg"
CH_TWS = "TWS_MHU_SGP_km_h_1"
CH_LEEWAY = "LEEWAY_deg"

CH_CA1 = "ANGLE_CA1_deg"
CH_CLEW = "ANGLE_CLEW_deg"
CH_TWIST = "ANGLE_WING_TWIST_deg"

# Target table JibLead is in degrees. This is the angular jib-trim channel used
# in the existing Jib Trim page. PER_JIB_LEAD_pct is also shown as information.
CH_JIB_LEAD_ANGLE = "ANGLE_JIB_SHT_deg"
CH_JIB_LEAD_PCT = "PER_JIB_LEAD_pct"
CH_JIB_SHEET = "LOAD_JIB_SHEET_kgf"
CH_JIB_CUNNO = "LOAD_JIB_CUNNO_kgf"

CH_RUD_AVG = "ANGLE_RUD_AVG_deg"
CH_RUD_DIFF = "ANGLE_RUD_DIFF_TACK_deg"
CH_PITCH = "PITCH_deg"
CH_RH_LW = "LENGTH_RH_LW_mm"
CH_RH_P = "LENGTH_RH_P_mm"
CH_RH_S = "LENGTH_RH_S_mm"
CH_HEEL = "HEEL_deg"
CH_DB_H_P = "LENGTH_DB_H_P_mm"
CH_DB_H_S = "LENGTH_DB_H_S_mm"

CH_CANT_PORT = "ANGLE_DB_CANT_P_deg"
CH_CANT_STBD = "ANGLE_DB_CANT_S_deg"

# Automatic equipment configuration.
CH_DB_CONFIG = "MD4_SEL_DB_unk"
CH_RUD_CONFIG = "MD4_SEL_RUD_unk"
CH_WING_CONFIG = "WING_CONFIG_unk"

# Channels normally available on strm. Pitch/RH/Leeway are also queried from
# log because the raw high-rate signals are known to be present there.
STRM_CHANNELS = [
    CH_BSP, CH_TWA, CH_TWS,
    CH_CA1, CH_CLEW, CH_TWIST,
    CH_JIB_LEAD_ANGLE, CH_JIB_LEAD_PCT, CH_JIB_SHEET, CH_JIB_CUNNO,
    CH_RUD_AVG, CH_RUD_DIFF,
    CH_CANT_PORT, CH_CANT_STBD,
    CH_DB_CONFIG, CH_RUD_CONFIG, CH_WING_CONFIG,
]
LOG_CHANNELS = [CH_PITCH, CH_RH_LW, CH_RH_P, CH_RH_S, CH_LEEWAY, CH_HEEL, CH_DB_H_P, CH_DB_H_S]

DB_MAP = {1: "LAB", 2: "HSB", 3: "HSB2", 4: "LAB2"}
RUD_MAP = {1: "LARW", 2: "HSRW", 3: "HSRW2", 4: "LARW2"}
WING_MAP = {
    9: ("HAW", 18.0),
    11: ("APW", 24.0),
    15: ("LAW", 29.0),
    143: ("LAW2", 27.5),
}

# =============================================================================
# TARGET TABLES
# =============================================================================
HERE = Path(__file__).resolve().parent
# Accept either the clean names or the original uploaded names.
UW_CANDIDATES = [HERE / "Targets_S6.xlsx - UW.csv", HERE / "Targets_S6_UW.csv"]
DW_CANDIDATES = [HERE / "Targets_S6.xlsx - DW.csv", HERE / "Targets_S6_DW.csv"]

TARGET_MAP = {
    "BSP": "BSP (km/h)",
    "TWA": "TWA (deg)",
    "LEEWAY": "LEEWAY (deg)",
    "LEEWARD CANT": "CANT (deg)",
    "RUD DIFF": "DIFF (deg)",
    "RUD AVG": "AVG (deg)",
    "PITCH": "PITCH (deg)",
    "RH LEEWARD": "RH LW (mm)",
    "CA1": "CA1 (deg)",
    "CLEW": "CLEW (deg)",
    "TWIST": "TWIST (deg)",
    "JIB CUNNO": "CUNNO (kgf)",
    "JIB LEAD": "JibLead (deg)",
    "JIB SHEET": "JibSheet (kgf)",
}

UNITS = {
    "BSP": "km/h", "TWA": "°", "TWS": "km/h", "LEEWAY": "°",
    "CA1": "°", "CLEW": "°", "TWIST": "°",
    "JIB LEAD": "°", "JIB LEAD %": "%", "JIB SHEET": "kgf", "JIB CUNNO": "kgf",
    "RUD AVG": "°", "RUD DIFF": "°", "LEEWARD CANT": "°",
    "RH LEEWARD": "mm", "RH WW": "mm", "PITCH": "°",
    "DB LW": "mm", "DB WW": "mm",
}

# Absolute live-target error thresholds: green / yellow / orange / red.
DEFAULT_TOLERANCES = {
    "BSP": (2.0, 4.0, 7.0),
    "TWA": (2.0, 4.0, 7.0),
    "CA1": (1.0, 2.0, 4.0),
    "CLEW": (0.7, 1.5, 3.0),
    "TWIST": (1.5, 3.0, 6.0),
    "JIB LEAD": (0.7, 1.5, 3.0),
    "JIB SHEET": (100.0, 250.0, 450.0),
    "JIB CUNNO": (100.0, 250.0, 450.0),
    "RUD AVG": (0.5, 1.0, 2.0),
    "RUD DIFF": (0.4, 0.8, 1.5),
    "LEEWAY": (0.4, 0.8, 1.5),
    "RH LEEWARD": (50.0, 100.0, 200.0),
    "LEEWARD CANT": (1.0, 2.0, 4.0),
    "PITCH": (0.5, 1.0, 2.0),
}

COLORS = {
    "green": "#22C55E", "yellow": "#FACC15", "orange": "#F97316",
    "red": "#EF4444", "neutral": "#CBD5E1",
}


def _existing_candidate(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def load_target_csv(path_or_upload) -> pd.DataFrame:
    df = pd.read_csv(path_or_upload)
    required = {"Config", "TWS (km/h)", "Foil", "Rudder", "Wing"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Target CSV: colonnes manquantes: {sorted(missing)}")
    df = df.copy()
    for c in ["Config", "State", "Foil", "Rudder"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["Wing", "TWS (km/h)", "VMG (km/h)", "HEEL (deg)", "RH LW (mm)", "LEEWAY (deg)"] + list(TARGET_MAP.values()):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Config", "TWS (km/h)"]).reset_index(drop=True)


def load_targets(upload_uw, upload_dw):
    uw_path = _existing_candidate(UW_CANDIDATES)
    dw_path = _existing_candidate(DW_CANDIDATES)
    if upload_uw is not None:
        uw = load_target_csv(upload_uw)
    elif uw_path:
        uw = load_target_csv(uw_path)
    else:
        raise FileNotFoundError("Targets UW introuvables à côté de la page.")
    if upload_dw is not None:
        dw = load_target_csv(upload_dw)
    elif dw_path:
        dw = load_target_csv(dw_path)
    else:
        raise FileNotFoundError("Targets DW introuvables à côté de la page.")
    return uw, dw


def _state_rows(df_mode: pd.DataFrame, config: str | None = None, state: str | None = None) -> pd.DataFrame:
    d = df_mode.copy()
    if config is not None:
        d = d[d["Config"].astype(str) == str(config)]
    if state is not None and "State" in d.columns:
        d = d[d["State"].astype(str).str.strip().str.upper() == str(state).strip().upper()]
    return d


def interp_target(df_mode: pd.DataFrame, config: str, tws: float, target_col: str, clamp: bool = False) -> float:
    if target_col not in df_mode.columns or not np.isfinite(tws):
        return np.nan
    d = _state_rows(df_mode, config=config)[["TWS (km/h)", target_col]].copy()
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d["TWS (km/h)"] = pd.to_numeric(d["TWS (km/h)"], errors="coerce")
    d = d.dropna().sort_values("TWS (km/h)").drop_duplicates("TWS (km/h)")
    if d.empty:
        return np.nan
    x = d["TWS (km/h)"].to_numpy(float)
    y = d[target_col].to_numpy(float)
    if not clamp and (tws < x.min() or tws > x.max()):
        return np.nan
    tws_used = float(np.clip(tws, x.min(), x.max())) if clamp else tws
    return float(np.interp(tws_used, x, y))


def target_vmg_from_row_values(bsp: float, twa_deg: float) -> float:
    """Positive VMG magnitude from target BSP/TWA; avoids broken #ERROR! VMG cells."""
    if not np.isfinite(bsp) or not np.isfinite(twa_deg):
        return np.nan
    return float(abs(bsp * np.cos(np.deg2rad(twa_deg))))


def interpolated_state_vmg(df_mode: pd.DataFrame, state: str, foil: str, rudder: str, wing_m: float, tws: float):
    if not all([state, foil, rudder]) or not np.isfinite(wing_m) or not np.isfinite(tws):
        return np.nan, None

    d = df_mode.copy()
    d = d[
        d["State"].astype(str).str.strip().str.upper().eq(state.upper())
        & d["Foil"].astype(str).str.strip().eq(foil)
        & d["Rudder"].astype(str).str.strip().eq(rudder)
        & np.isclose(pd.to_numeric(d["Wing"], errors="coerce"), float(wing_m), atol=0.05, equal_nan=False)
    ].copy()
    if d.empty:
        return np.nan, None

    # Interpolate BSP and TWA separately, then derive VMG.
    cfgs = list(dict.fromkeys(d["Config"].dropna().astype(str)))
    if not cfgs:
        return np.nan, None
    config = cfgs[0]
    bsp = interp_target(d, config, tws, "BSP (km/h)")
    twa = interp_target(d, config, tws, "TWA (deg)")
    return target_vmg_from_row_values(bsp, twa), config


def _state_equipment_rows(df_mode: pd.DataFrame, state: str, equipment: dict) -> pd.DataFrame:
    foil, rudder, wing_m = equipment["foil"], equipment["rudder"], equipment["wing_m"]
    if foil is None or rudder is None or wing_m is None:
        return pd.DataFrame()
    d = df_mode.copy()
    return d[
        d["State"].astype(str).str.strip().str.upper().eq(state.upper())
        & d["Foil"].astype(str).str.strip().eq(foil)
        & d["Rudder"].astype(str).str.strip().eq(rudder)
        & np.isclose(pd.to_numeric(d["Wing"], errors="coerce"), float(wing_m), atol=0.05, equal_nan=False)
    ].copy()


def _state_bounds(df_mode: pd.DataFrame, state: str, equipment: dict):
    d = _state_equipment_rows(df_mode, state, equipment)
    x = pd.to_numeric(d.get("TWS (km/h)", pd.Series(dtype=float)), errors="coerce").dropna()
    return (float(x.min()), float(x.max())) if not x.empty else (np.nan, np.nan)


def _state_vmg_at(df_mode: pd.DataFrame, state: str, equipment: dict, tws: float, clamp=False):
    d = _state_equipment_rows(df_mode, state, equipment)
    if d.empty:
        return np.nan, None, np.nan
    cfgs = list(dict.fromkeys(d["Config"].dropna().astype(str)))
    if not cfgs:
        return np.nan, None, np.nan
    cfg = cfgs[0]
    lo, hi = _state_bounds(df_mode, state, equipment)
    used = float(np.clip(tws, lo, hi)) if clamp and np.isfinite(lo) and np.isfinite(hi) else tws
    bsp = interp_target(d, cfg, used, "BSP (km/h)", clamp=clamp)
    twa = interp_target(d, cfg, used, "TWA (deg)", clamp=clamp)
    return target_vmg_from_row_values(bsp, twa), cfg, used


def best_target_state(df_mode: pd.DataFrame, equipment: dict, tws: float):
    if not np.isfinite(tws):
        return None, np.nan, None, {}, np.nan
    states = ["Foiling", "H2", "H1", "H1-2B"]
    bounds = {s: _state_bounds(df_mode, s, equipment) for s in states}
    valid = {s:b for s,b in bounds.items() if np.isfinite(b[0]) and np.isfinite(b[1])}
    if not valid:
        return None, np.nan, None, {}, np.nan
    global_min = min(v[0] for v in valid.values())
    global_max = max(v[1] for v in valid.values())

    # Below target domain: H2 is forced and its lowest-TWS target is extended.
    if tws < global_min and "H2" in valid:
        vmg, cfg, used = _state_vmg_at(df_mode, "H2", equipment, tws, clamp=True)
        return "H2", vmg, cfg, {"H2":{"vmg":vmg,"config":cfg,"tws_used":used}}, used

    # Above target domain: Foiling is forced and its highest-TWS target is extended.
    if tws > global_max and "Foiling" in valid:
        vmg, cfg, used = _state_vmg_at(df_mode, "Foiling", equipment, tws, clamp=True)
        return "Foiling", vmg, cfg, {"Foiling":{"vmg":vmg,"config":cfg,"tws_used":used}}, used

    results = {}
    for state in states:
        vmg, cfg, used = _state_vmg_at(df_mode, state, equipment, tws, clamp=False)
        if np.isfinite(vmg):
            results[state] = {"vmg":vmg,"config":cfg,"tws_used":used}
    if not results:
        return None, np.nan, None, {}, np.nan
    best = max(results, key=lambda s: results[s]["vmg"])
    return best, results[best]["vmg"], results[best]["config"], results, tws

# =============================================================================
# TELEMETRY
# =============================================================================
def _safe_num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _normalize_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "time_utc" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "time_utc"})
    if "time_utc" not in df.columns:
        return pd.DataFrame()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    return df.dropna(subset=["time_utc"]).sort_values("time_utc")


def _query_level(cfg, boat, channels, start, stop, level_expr, every="1s") -> pd.DataFrame:
    if not channels:
        return pd.DataFrame()
    return _normalize_time_col(load_channels_timeseries(
        cfg,
        start_utc=start,
        stop_utc=stop,
        boats=[boat],
        channels=channels,
        every=every,
        level_expr=level_expr,
        agg_fn="mean",
    ))


def _snapshot_from_df(df: pd.DataFrame, channels: list[str], reference_time, mean_s: int) -> dict:
    out = {ch: np.nan for ch in channels}
    if df.empty:
        return out
    ref = pd.Timestamp(reference_time)
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    else:
        ref = ref.tz_convert("UTC")
    cutoff = ref - pd.Timedelta(seconds=int(mean_s))
    recent = df[(df["time_utc"] <= ref) & (df["time_utc"] >= cutoff)]
    # If the exact final seconds are sparse, use the latest available samples in
    # the query window rather than returning an entirely empty card.
    if recent.empty:
        recent = df.tail(max(1, int(mean_s)))
    for ch in channels:
        if ch in recent.columns:
            s = pd.to_numeric(recent[ch], errors="coerce").dropna()
            if not s.empty:
                out[ch] = float(s.mean())
    return out


def _config_snapshot_from_df(df: pd.DataFrame, channels: list[str], reference_time) -> dict:
    """Equipment selection is a state: use latest value, not a mean."""
    out = {ch: np.nan for ch in channels}
    if df.empty:
        return out
    ref = pd.Timestamp(reference_time)
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    else:
        ref = ref.tz_convert("UTC")
    d = df[df["time_utc"] <= ref]
    if d.empty:
        d = df
    for ch in channels:
        if ch in d.columns:
            s = pd.to_numeric(d[ch], errors="coerce").dropna()
            if not s.empty:
                out[ch] = float(s.iloc[-1])
    return out


def load_snapshot(boat: str, reference_utc: datetime, lookback_s: int, mean_s: int):
    """
    Read strm + log around reference_utc.
    Works identically for real live and fake-live historical mode.
    """
    cfg = get_cfg()
    stop = reference_utc
    start = stop - timedelta(seconds=int(lookback_s))

    # Read stream channels. Config channels are included and later treated as state.
    strm_df = _query_level(cfg, boat, STRM_CHANNELS, start, stop, "strm", every="1s")

    # High-rate raw channels. Aggregate server-side to 1 s for a light query.
    log_df = _query_level(cfg, boat, LOG_CHANNELS, start, stop, "log", every="1s")

    snap = _snapshot_from_df(strm_df, STRM_CHANNELS, stop, mean_s)
    snap.update(_snapshot_from_df(log_df, LOG_CHANNELS, stop, mean_s))
    snap.update(_config_snapshot_from_df(strm_df, [CH_DB_CONFIG, CH_RUD_CONFIG, CH_WING_CONFIG], stop))

    times = []
    for d in (strm_df, log_df):
        if not d.empty:
            times.append(d["time_utc"].max())
    last_time = max(times) if times else None
    return snap, last_time, strm_df, log_df


def leeward_cant(snapshot: dict) -> float:
    twa = _safe_num(snapshot.get(CH_TWA))
    port = _safe_num(snapshot.get(CH_CANT_PORT))
    stbd = _safe_num(snapshot.get(CH_CANT_STBD))
    if not np.isfinite(twa):
        return np.nan
    # Convention already used in the FC page.
    return port if twa > 0 else stbd


def live_values(snapshot: dict) -> dict:
    return {
        "BSP": _safe_num(snapshot.get(CH_BSP)),
        "TWA": _safe_num(snapshot.get(CH_TWA)),
        "TWS": _safe_num(snapshot.get(CH_TWS)),
        "LEEWAY": _safe_num(snapshot.get(CH_LEEWAY)),
        "CA1": _safe_num(snapshot.get(CH_CA1)),
        "CLEW": _safe_num(snapshot.get(CH_CLEW)),
        "TWIST": _safe_num(snapshot.get(CH_TWIST)),
        "JIB LEAD": _safe_num(snapshot.get(CH_JIB_LEAD_ANGLE)),
        "JIB LEAD %": _safe_num(snapshot.get(CH_JIB_LEAD_PCT)),
        "JIB SHEET": _safe_num(snapshot.get(CH_JIB_SHEET)),
        "JIB CUNNO": _safe_num(snapshot.get(CH_JIB_CUNNO)),
        "RUD AVG": _safe_num(snapshot.get(CH_RUD_AVG)),
        "RUD DIFF": _safe_num(snapshot.get(CH_RUD_DIFF)),
        "LEEWARD CANT": leeward_cant(snapshot),
        "RH LEEWARD": _safe_num(snapshot.get(CH_RH_LW)),
        "_RH_P": _safe_num(snapshot.get(CH_RH_P)),
        "_RH_S": _safe_num(snapshot.get(CH_RH_S)),
        "PITCH": _safe_num(snapshot.get(CH_PITCH)),
        "_HEEL": _safe_num(snapshot.get(CH_HEEL)),
        "_DB_H_P": _safe_num(snapshot.get(CH_DB_H_P)),
        "_DB_H_S": _safe_num(snapshot.get(CH_DB_H_S)),
    }


def tack_sign(twa: float) -> float:
    if not np.isfinite(twa) or twa == 0:
        return 1.0
    return 1.0 if twa > 0 else -1.0


def current_boat_state(vals: dict):
    """State actuel sur valeurs lissées 3 s.

    Règles opérationnelles demandées:
      RH LW > 5 mm -> Foiling (heel ignoré)
      sinon RH WW > 200 mm -> H1 family
          DB WW < 50 mm -> H1
          sinon -> H1-2B
      sinon -> H2
    """
    rh_lw = vals.get("RH LEEWARD", np.nan)
    rh_ww = vals.get("RH WW", np.nan)
    db_ww = vals.get("DB WW", np.nan)
    if not np.isfinite(rh_lw):
        return None, "RH LW manquant"
    if rh_lw > 5.0:
        return "Foiling", f"RH LW={rh_lw:.0f} mm > 5"
    if not np.isfinite(rh_ww):
        return None, "RH WW manquant"
    if rh_ww > 200.0:
        if not np.isfinite(db_ww):
            return None, "RH WW > 200 mais DB WW manquant"
        if db_ww < 50.0:
            return "H1", f"RH WW={rh_ww:.0f} mm > 200; DB WW={db_ww:.0f} mm < 50"
        return "H1-2B", f"RH WW={rh_ww:.0f} mm > 200; DB WW={db_ww:.0f} mm >= 50"
    return "H2", f"RH LW={rh_lw:.0f} mm <= 5; RH WW={rh_ww:.0f} mm <= 200"


# =============================================================================
# CONFIG DETECTION
# =============================================================================
def _round_state(v):
    x = _safe_num(v)
    return int(round(x)) if np.isfinite(x) else None


def decode_equipment(snapshot: dict):
    db_id = _round_state(snapshot.get(CH_DB_CONFIG))
    rud_id = _round_state(snapshot.get(CH_RUD_CONFIG))
    wing_id = _round_state(snapshot.get(CH_WING_CONFIG))
    foil = DB_MAP.get(db_id)
    rudder = RUD_MAP.get(rud_id)
    wing_info = WING_MAP.get(wing_id)
    wing_name = wing_info[0] if wing_info else None
    wing_m = wing_info[1] if wing_info else None
    return {
        "db_id": db_id, "foil": foil,
        "rud_id": rud_id, "rudder": rudder,
        "wing_id": wing_id, "wing_name": wing_name, "wing_m": wing_m,
    }


def find_auto_config(df_mode: pd.DataFrame, equipment: dict, preferred_state="Foiling"):
    foil, rudder, wing_m = equipment["foil"], equipment["rudder"], equipment["wing_m"]
    if foil is None or rudder is None or wing_m is None:
        return None, "channel config manquant/invalide"
    d = df_mode.copy()
    wing = pd.to_numeric(d["Wing"], errors="coerce")
    mask = (
        d["Foil"].astype(str).str.strip().eq(foil)
        & d["Rudder"].astype(str).str.strip().eq(rudder)
        & np.isclose(wing, float(wing_m), atol=0.05, equal_nan=False)
    )
    candidates = d.loc[mask].copy()
    if candidates.empty:
        return None, f"aucune target {foil}/{rudder}/{wing_m:g}m"
    # Several target states can share the same hardware. Prefer Foiling for live
    # operational comparison; manual selection remains available in sidebar.
    if "State" in candidates.columns:
        preferred = candidates[candidates["State"].astype(str).str.strip().eq(preferred_state)]
        if not preferred.empty:
            candidates = preferred
    configs = list(dict.fromkeys(candidates["Config"].astype(str)))
    if len(configs) == 1:
        return configs[0], "auto"
    return None, "plusieurs états target possibles"

# =============================================================================
# DISPLAY
# =============================================================================
def error_color(name: str, live: float, target: float) -> str:
    if not np.isfinite(live) or not np.isfinite(target) or name not in DEFAULT_TOLERANCES:
        return COLORS["neutral"]
    err = abs(live - target)
    g, y, o = DEFAULT_TOLERANCES[name]
    if err <= g: return COLORS["green"]
    if err <= y: return COLORS["yellow"]
    if err <= o: return COLORS["orange"]
    return COLORS["red"]


def fmt(v, unit):
    if not np.isfinite(v): return "—"
    if unit in ("kgf", "mm"): return f"{v:.0f}"
    return f"{v:.1f}"


def metric_card(label, live, target, unit, color, target_available=True):
    live_txt = fmt(live, unit)
    target_html = (
        f'<div style="display:inline-block;margin-top:8px;padding:4px 9px;border-radius:7px;'
        f'background:{color};color:#111827;font-size:14px;font-weight:800;">'
        f'TARGET: {fmt(target, unit)} {unit}</div>'
        if target_available else
        '<div style="display:inline-block;margin-top:8px;padding:4px 9px;border-radius:7px;'
        'background:#CBD5E1;color:#111827;font-size:13px;font-weight:700;">PAS DE TARGET TABLE</div>'
    )
    return f"""
    <div style="border:1px solid #CBD5E1;border-radius:12px;padding:12px 14px;
                margin-bottom:10px;background:rgba(255,255,255,0.03);min-height:118px;">
      <div style="font-size:14px;font-weight:700;opacity:.80;">{label}</div>
      <div style="font-size:31px;font-weight:800;line-height:1.15;margin-top:4px;">
        {live_txt} <span style="font-size:15px;font-weight:500;opacity:.65;">{unit}</span>
      </div>
      {target_html}
    </div>"""

# =============================================================================
# SIDEBAR + TARGETS
# =============================================================================
with st.sidebar:
    st.header("Live Targets")
    st.caption(f"Telemetry backend: {get_backend().upper()}")

    boat = st.selectbox("Boat", list(ALL_BOATS), index=list(ALL_BOATS).index("FRA") if "FRA" in ALL_BOATS else 0)

    data_mode = st.radio("Mode données", ["Faux live historique", "Live"], index=0)
    if data_mode == "Faux live historique":
        fake_date = st.date_input("Date faux live", value=date(2026, 9, 17))
        fake_time = st.time_input("Heure faux live", value=dtime(12, 0, 0), step=timedelta(minutes=1))
        st.caption("Heure saisie interprétée en UTC.")
        refresh_s = 0
    else:
        refresh_s = st.slider("Refresh (s)", 1, 10, 2, 1)

    mean_s = st.slider("Moyenne télémétrie (s)", 1, 10, 3, 1)
    lookback_s = st.slider("Fenêtre de lecture (s)", 10, 120, 30, 5)

    st.markdown("---")
    st.subheader("Conventions de signe")
    sign_leeway = st.checkbox("LEEWAY × sign(TWA)", value=True)
    sign_ca1 = st.checkbox("CA1 × sign(TWA)", value=True)
    sign_clew = st.checkbox("CLEW × sign(TWA)", value=True)
    sign_twist = st.checkbox("TWIST × sign(TWA)", value=True)

    st.markdown("---")
    st.subheader("Target tables")
    upload_uw = st.file_uploader("UW target CSV", type=["csv"], key="live_target_uw_v2")
    upload_dw = st.file_uploader("DW target CSV", type=["csv"], key="live_target_dw_v2")

try:
    targets_uw, targets_dw = load_targets(upload_uw, upload_dw)
except Exception as exc:
    st.error(f"Targets: {exc}")
    st.stop()

all_configs = sorted(set(targets_uw["Config"].astype(str)) | set(targets_dw["Config"].astype(str)))

# =============================================================================
# READ SNAPSHOT
# =============================================================================
if data_mode == "Live":
    reference_utc = datetime.now(timezone.utc)
else:
    reference_utc = datetime.combine(fake_date, fake_time).replace(tzinfo=timezone.utc)

try:
    snapshot, last_time, raw_strm, raw_log = load_snapshot(
        boat=boat,
        reference_utc=reference_utc,
        lookback_s=lookback_s,
        mean_s=mean_s,
    )
except Exception as exc:
    st.error("Connexion/lecture Timescale impossible. Le mode faux live utilise lui aussi les données historiques TimescaleDB.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

vals = live_values(snapshot)
twa = vals["TWA"]
tws = vals["TWS"]

# State actuel: RH/DB are smoothed over exactly 3 seconds, independently
# from the general telemetry averaging selected in the sidebar.
state3 = {}
state3.update(_snapshot_from_df(raw_strm, STRM_CHANNELS, reference_utc, 3))
state3.update(_snapshot_from_df(raw_log, LOG_CHANNELS, reference_utc, 3))
state_vals = live_values(state3)

sgn = tack_sign(twa)
if sign_leeway and np.isfinite(vals["LEEWAY"]):
    vals["LEEWAY"] *= sgn
if sign_ca1 and np.isfinite(vals["CA1"]):
    vals["CA1"] *= sgn
if sign_clew and np.isfinite(vals["CLEW"]):
    vals["CLEW"] *= sgn
if sign_twist and np.isfinite(vals["TWIST"]):
    vals["TWIST"] *= sgn

# DB leeward/windward derived from tack.
# TWA > 0 -> LW = Port; TWA < 0 -> LW = Starboard.
# Use the averaged snapshot first; if unavailable, fall back to the latest
# valid P/S sample from raw_log. This makes DB LW/WW robust to sparse channels.
def _latest_valid(df, channel):
    if df is None or df.empty or channel not in df.columns:
        return np.nan
    s = pd.to_numeric(df[channel], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

db_p = vals.get("_DB_H_P", np.nan)
db_s = vals.get("_DB_H_S", np.nan)

if not np.isfinite(db_p):
    db_p = _latest_valid(raw_log, CH_DB_H_P)
if not np.isfinite(db_s):
    db_s = _latest_valid(raw_log, CH_DB_H_S)

# Keep the recovered physical values in vals as well.
vals["_DB_H_P"] = db_p
vals["_DB_H_S"] = db_s

if np.isfinite(twa) and twa > 0:
    vals["DB LW"] = db_p
    vals["DB WW"] = db_s
elif np.isfinite(twa) and twa < 0:
    vals["DB LW"] = db_s
    vals["DB WW"] = db_p
else:
    vals["DB LW"] = np.nan
    vals["DB WW"] = np.nan

# Derived LW/WW values for state detection, using the 3 s smoothed snapshot.
sdb_p = state_vals.get("_DB_H_P", np.nan)
sdb_s = state_vals.get("_DB_H_S", np.nan)
rh_p = state_vals.get("_RH_P", np.nan)
rh_s = state_vals.get("_RH_S", np.nan)
if np.isfinite(twa) and twa > 0:
    state_vals["DB LW"], state_vals["DB WW"] = sdb_p, sdb_s
    state_vals["RH WW"] = rh_s
elif np.isfinite(twa) and twa < 0:
    state_vals["DB LW"], state_vals["DB WW"] = sdb_s, sdb_p
    state_vals["RH WW"] = rh_p
else:
    state_vals["DB LW"] = state_vals["DB WW"] = state_vals["RH WW"] = np.nan

# RH LW itself comes directly from LENGTH_RH_LW_mm, averaged over 3 s.
vals["RH WW"] = state_vals.get("RH WW", np.nan)
mode = "UW" if np.isfinite(twa) and abs(twa) < 90.0 else "DW" if np.isfinite(twa) else "?"
target_df = targets_uw if mode == "UW" else targets_dw if mode == "DW" else pd.DataFrame()

equipment = decode_equipment(snapshot)

# Current state from telemetry.
current_state, current_state_reason = current_boat_state(state_vals)

# Best state according to target VMG at current TWS and detected hardware.
best_state, best_vmg, best_config, state_vmg_results, best_tws_used = (
    best_target_state(target_df, equipment, tws)
    if mode in ("UW", "DW") else (None, np.nan, None, {}, np.nan)
)

with st.sidebar:
    st.markdown("---")
    st.subheader("Configuration target")
    auto_config_enabled = st.checkbox(
        "Détection automatique config + meilleur state",
        value=True,
        help="Choisit Wing/DB/Rudder depuis la télémétrie puis le State ayant le meilleur VMG target au TWS actuel."
    )

    mode_configs = sorted(target_df["Config"].dropna().astype(str).unique()) if not target_df.empty else all_configs

    if auto_config_enabled and best_config in mode_configs:
        config = best_config
        auto_config = best_config
        auto_reason = f"best state = {best_state}"
        st.success(f"Auto: {config}")
    else:
        auto_config, auto_reason = find_auto_config(target_df, equipment) if mode in ("UW", "DW") else (None, "TWA indisponible")
        default_idx = mode_configs.index(auto_config) if auto_config in mode_configs else 0
        config = st.selectbox("Config manuelle", mode_configs, index=default_idx)
        if auto_config_enabled:
            st.warning(f"Auto best-state non résolu: {auto_reason}")

# =============================================================================
# TARGET INTERPOLATION
# =============================================================================
targets = {name: np.nan for name in TARGET_MAP}
if mode in ("UW", "DW") and np.isfinite(tws) and config:
    edge_extend = np.isfinite(best_tws_used) and not np.isclose(best_tws_used, tws)
    target_tws = best_tws_used if edge_extend else tws
    for name, col in TARGET_MAP.items():
        targets[name] = interp_target(target_df, config, target_tws, col, clamp=edge_extend)

# TWA target follows the live tack sign.
if np.isfinite(targets.get("TWA", np.nan)) and np.isfinite(twa):
    targets["TWA"] = tack_sign(twa) * abs(targets["TWA"])

# VMG of current state using the same target table/hardware/TWS.
current_state_vmg = np.nan
current_state_config = None
if current_state and equipment["foil"] and equipment["rudder"] and equipment["wing_m"] is not None:
    current_state_vmg, current_state_config = interpolated_state_vmg(
        target_df,
        current_state,
        equipment["foil"],
        equipment["rudder"],
        float(equipment["wing_m"]),
        tws,
    )

vmg_gap_pct = np.nan
if np.isfinite(best_vmg) and best_vmg > 0 and np.isfinite(current_state_vmg):
    vmg_gap_pct = 100.0 * (best_vmg - current_state_vmg) / best_vmg
# =============================================================================
# HEADER
# =============================================================================
h1, h2, h3, h4, h5 = st.columns(5)
with h1: st.metric("Boat", boat)
with h2: st.metric("Mode", mode)
with h3: st.metric("TWS", f"{tws:.1f} km/h" if np.isfinite(tws) else "—")
with h4:
    if data_mode == "Live" and last_time is not None:
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_time)).total_seconds())
        st.metric("Data age", f"{age:.1f} s")
    else:
        st.metric("Faux live", reference_utc.strftime("%H:%M:%S"))
with h5:
    st.metric("Target state", best_state if auto_config_enabled and best_state else (str(config).split("_")[0] if config else "—"))

st.caption(
    f"Référence UTC: {reference_utc:%Y-%m-%d %H:%M:%S} • "
    f"{mode} via |TWA| {'<' if mode == 'UW' else '≥' if mode == 'DW' else '?'} 90° • "
    f"Config target: {config} • targets interpolés sur TWS"
)

# Equipment status
c1, c2, c3 = st.columns(3)
with c1: st.info(f"DB: {equipment['foil'] or '—'}  (id {equipment['db_id'] if equipment['db_id'] is not None else '—'})")
with c2: st.info(f"Rudder: {equipment['rudder'] or '—'}  (id {equipment['rud_id'] if equipment['rud_id'] is not None else '—'})")
with c3: st.info(f"Wing: {equipment['wing_name'] or '—'} / {equipment['wing_m'] if equipment['wing_m'] is not None else '—'} m  (id {equipment['wing_id'] if equipment['wing_id'] is not None else '—'})")


# State comparison.
s1, s2, s3 = st.columns(3)
with s1:
    st.metric("State actuel (moy. 3 s)", current_state or "—")
    st.caption(current_state_reason)
with s2:
    st.metric("Best state target", best_state or "—")
    if np.isfinite(best_vmg):
        st.caption(f"VMG target {best_vmg:.1f} km/h")
with s3:
    if current_state and best_state and current_state != best_state and np.isfinite(vmg_gap_pct):
        st.metric("Perte VMG vs best", f"{vmg_gap_pct:.1f} %")
        st.warning(
            f"State actuel {current_state} ≠ best {best_state}. "
            f"VMG target actuel {current_state_vmg:.1f} vs best {best_vmg:.1f} km/h."
        )
    elif current_state and best_state and current_state == best_state:
        st.metric("State", "OPTIMAL")
        st.success("State actuel = meilleur state target")
    else:
        st.metric("Écart VMG", "—")

# =============================================================================
# CARDS
# =============================================================================
display_order = [
    "BSP", "TWA", "TWS", "LEEWAY",
    "CA1", "CLEW", "TWIST", "LEEWARD CANT",
    "JIB LEAD", "JIB LEAD %", "JIB SHEET", "JIB CUNNO",
    "RUD AVG", "RUD DIFF", "RH LEEWARD", "RH WW", "PITCH",
    "DB LW", "DB WW",
]

# TWS is the interpolation axis and Jib Lead % has no direct target column.
no_target = {"TWS", "JIB LEAD %", "DB LW", "DB WW", "RH WW"}

for row_start in range(0, len(display_order), 4):
    cols = st.columns(4)
    for j, name in enumerate(display_order[row_start:row_start + 4]):
        live = vals.get(name, np.nan)
        unit = UNITS[name]
        target_available = name not in no_target and name in TARGET_MAP
        target = targets.get(name, np.nan) if target_available else np.nan
        color = error_color(name, live, target) if target_available else COLORS["neutral"]
        with cols[j]:
            st.markdown(metric_card(name, live, target, unit, color, target_available), unsafe_allow_html=True)

# =============================================================================
# DETAILS
# =============================================================================
with st.expander("Valeurs target interpolées"):
    st.caption("TWA target = sign(TWA live) × |TWA target|. LEEWAY/CA1/CLEW/TWIST live peuvent être normalisés par sign(TWA) via la sidebar.")
    rows = []
    for name in display_order:
        rows.append({
            "Parameter": name,
            "Live": vals.get(name, np.nan),
            "Target": targets.get(name, np.nan) if name in TARGET_MAP else np.nan,
            "Unit": UNITS[name],
            "Target column": TARGET_MAP.get(name, "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Mapping telemetry channels"):
    mapping = [
        ("BSP", CH_BSP, "strm"), ("TWA", CH_TWA, "strm"), ("TWS", CH_TWS, "strm"),
        ("LEEWAY", CH_LEEWAY, "log"),
        ("CA1", CH_CA1, "strm"), ("CLEW", CH_CLEW, "strm"), ("TWIST", CH_TWIST, "strm"),
        ("JIB LEAD angle", CH_JIB_LEAD_ANGLE, "strm"), ("JIB LEAD %", CH_JIB_LEAD_PCT, "strm"),
        ("JIB SHEET", CH_JIB_SHEET, "strm"), ("JIB CUNNO", CH_JIB_CUNNO, "strm"),
        ("RUD AVG", CH_RUD_AVG, "strm"), ("RUD DIFF tack", CH_RUD_DIFF, "strm"),
        ("CANT PORT", CH_CANT_PORT, "strm"), ("CANT STBD", CH_CANT_STBD, "strm"),
        ("RH LEEWARD", CH_RH_LW, "log"), ("RH PORT", CH_RH_P, "log"), ("RH STBD", CH_RH_S, "log"), ("PITCH", CH_PITCH, "log"),
        ("HEEL", CH_HEEL, "log"), ("DB H PORT", CH_DB_H_P, "log"), ("DB H STBD", CH_DB_H_S, "log"),
        ("DB CONFIG", CH_DB_CONFIG, "strm"), ("RUD CONFIG", CH_RUD_CONFIG, "strm"),
        ("WING CONFIG", CH_WING_CONFIG, "strm"),
    ]
    st.dataframe(pd.DataFrame(mapping, columns=["Display", "Telemetry channel", "Level"]), use_container_width=True, hide_index=True)

with st.expander("Debug config / données brutes"):
    st.write("Decoded equipment:", equipment)
    st.write("Current state (moyenne 3 s):", current_state, "—", current_state_reason)
    st.write("State RH:", {"RH_LW": state_vals.get("RH LEEWARD"), "RH_WW": state_vals.get("RH WW")})
    st.write("Best state:", best_state, "best VMG:", best_vmg, "TWS target utilisé:", best_tws_used, "current-state VMG:", current_state_vmg, "gap %:", vmg_gap_pct)
    st.write("State VMG candidates:", state_vmg_results)
    st.write("Auto config:", auto_config, "—", auto_reason)
    st.write("Dernier timestamp retourné:", last_time)
    st.write("STRM rows:", len(raw_strm), "LOG rows:", len(raw_log))
    if not raw_strm.empty:
        st.dataframe(raw_strm.tail(10), use_container_width=True)
    if not raw_log.empty:
        st.dataframe(raw_log.tail(10), use_container_width=True)

# =============================================================================
# AUTO REFRESH — live only
# =============================================================================
if data_mode == "Live":
    time.sleep(refresh_s)
    st.rerun()
