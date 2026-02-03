from __future__ import annotations

from datetime import datetime, date, timezone, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import time

from start_aid.geo import make_context_from_boundary
from shapely.geometry import Polygon
from pyproj import Transformer

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Influx imports (robust: some repos only have FRA-specific helpers)
from influx_io import (
    get_cfg,
    scale_latlon_always,
    ALL_BOATS,
    load_boat_snapshot,
    load_marks_snapshot,
    load_trace_boat,
    detect_starts_from_race_start_count,
    load_latlon_timeseries_1s,  # IMPORTANT: pour AVG trace + crossing
)

# Optional imports depending on influx_io generation
try:
    from influx_io import load_race_start_count_day  # generic
except Exception:
    load_race_start_count_day = None

try:
    from influx_io import load_race_start_count_day_fra  # FRA-only
except Exception:
    load_race_start_count_day_fra = None

try:
    from influx_io import load_twd_tws_boat  # generic
except Exception:
    load_twd_tws_boat = None

try:
    from influx_io import load_twd_tws_fra  # FRA-only
except Exception:
    load_twd_tws_fra = None


from viz_deck import build_deck

try:
    from boundary_shared import sidebar_boundary_uploader
except Exception:
    sidebar_boundary_uploader = None


# -----------------------
# Small wrappers (safe)
# -----------------------
def _load_race_start_count_day_safe(cfg, day_utc: date, boat: str) -> pd.DataFrame:
    """Return df(time_utc, race_start_count) or empty DF; never raises if helper missing."""
    boat = str(boat)
    if load_race_start_count_day is not None:
        try:
            return load_race_start_count_day(cfg, day_utc, boat=boat)
        except Exception:
            return pd.DataFrame(columns=["time_utc", "race_start_count"])

    # fallback FRA-only helper
    if boat == "FRA" and load_race_start_count_day_fra is not None:
        try:
            return load_race_start_count_day_fra(cfg, day_utc)
        except Exception:
            return pd.DataFrame(columns=["time_utc", "race_start_count"])

    return pd.DataFrame(columns=["time_utc", "race_start_count"])


def _load_twd_tws_safe(cfg, start_gun_utc: datetime, offset_s: int, boat: str) -> dict:
    """Return dict with keys like TWD_MHU_SGP_deg / TWS_MHU_SGP_km_h_1; never raises."""
    boat = str(boat)
    if load_twd_tws_boat is not None:
        try:
            return load_twd_tws_boat(cfg, start_gun_utc, offset_s, boat=boat, half_window_s=2)
        except Exception:
            return {}
    if boat == "FRA" and load_twd_tws_fra is not None:
        try:
            return load_twd_tws_fra(cfg, start_gun_utc, offset_s, half_window_s=2)
        except Exception:
            return {}
    return {}


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _round_to_nearest_minute(dt: datetime) -> datetime:
    dt = _to_utc(dt)
    sec = dt.second + dt.microsecond / 1e6
    if sec >= 30:
        dt = dt + timedelta(minutes=1)
    return dt.replace(second=0, microsecond=0)


def _vector_from_positions(lat0, lon0, lat1, lon1):
    if not np.isfinite([lat0, lon0, lat1, lon1]).all():
        return np.nan, np.nan

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad((lat0 + lat1) * 0.5)))

    dx = (lon1 - lon0) * m_per_deg_lon
    dy = (lat1 - lat0) * m_per_deg_lat
    dist = float(np.sqrt(dx * dx + dy * dy))
    bearing = float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)
    return bearing, dist


# -------------------------------
# Crossing SL1-SL2 (G -> D)
# -------------------------------
def _enu_meters(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y


def _rotate(x: np.ndarray, y: np.ndarray, alpha_rad: float) -> tuple[np.ndarray, np.ndarray]:
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    return x * ca - y * sa, x * sa + y * ca


def last_left_to_right_crossing(
    df_track: pd.DataFrame,
    sl1_lat: float, sl1_lon: float,
    sl2_lat: float, sl2_lon: float,
) -> dict | None:
    if df_track.empty:
        return None

    g = df_track.dropna(subset=["time_utc", "lat", "lon"]).sort_values("time_utc").reset_index(drop=True)
    if len(g) < 2:
        return None

    x1, y1 = _enu_meters(np.array([sl1_lat]), np.array([sl1_lon]), sl2_lat, sl2_lon)
    vx, vy = float(x1[0]), float(y1[0])
    alpha = float(np.arctan2(vx, vy))

    x, y = _enu_meters(g["lat"].to_numpy(float), g["lon"].to_numpy(float), sl2_lat, sl2_lon)
    xr, yr = _rotate(x, y, alpha)

    idx = None
    for i in range(len(xr) - 1):
        if (xr[i] < 0) and (xr[i + 1] >= 0):
            idx = i
    if idx is None:
        return None

    x0, x1v = xr[idx], xr[idx + 1]
    f = 0.0 if x1v == x0 else float((-x0) / (x1v - x0))
    f = max(0.0, min(1.0, f))

    t0 = pd.Timestamp(g.loc[idx, "time_utc"]).to_datetime64()
    t1 = pd.Timestamp(g.loc[idx + 1, "time_utc"]).to_datetime64()
    tn = t0 + (t1 - t0) * f

    lat_cross = float(g.loc[idx, "lat"] + f * (g.loc[idx + 1, "lat"] - g.loc[idx, "lat"]))
    lon_cross = float(g.loc[idx, "lon"] + f * (g.loc[idx + 1, "lon"] - g.loc[idx, "lon"]))
    yr_cross = float(yr[idx] + f * (yr[idx + 1] - yr[idx]))

    return {"time_utc": pd.Timestamp(tn).tz_localize("UTC"), "lat": lat_cross, "lon": lon_cross, "dist_sl2_m": yr_cross}


def _snapshot_metrics_at_time(cfg, t_utc: datetime, boat: str | None = None, eligible_codes: list[str] | None = None):
    snap = load_boat_snapshot(cfg, t_utc, 0, half_window_s=1)
    if snap is None or snap.empty:
        return np.nan, np.nan

    if boat is not None:
        r = snap[snap["boat"] == boat]
        if r.empty:
            return np.nan, np.nan
        bsp = pd.to_numeric(r["BSP_kmph"].iloc[0], errors="coerce")
        ttk = pd.to_numeric(r["TTK_s"].iloc[0], errors="coerce")
        return float(bsp) if np.isfinite(bsp) else np.nan, float(ttk) if np.isfinite(ttk) else np.nan

    if eligible_codes is not None:
        r = snap[snap["boat"].isin(eligible_codes)]
        if r.empty:
            return np.nan, np.nan
        bsp = pd.to_numeric(r["BSP_kmph"], errors="coerce").mean()
        ttk = pd.to_numeric(r["TTK_s"], errors="coerce").mean()
        return float(bsp) if np.isfinite(bsp) else np.nan, float(ttk) if np.isfinite(ttk) else np.nan

    return np.nan, np.nan


def _boundary_df_to_latlon_list(boundary_df: pd.DataFrame) -> list[tuple[float, float]]:
    if boundary_df is None or boundary_df.empty:
        return []
    df = boundary_df.dropna(subset=["lat", "lon"]).copy()
    return [(float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()]


def _compute_boundary_buffer_df(boundary_df: pd.DataFrame | None, buffer_m: float | None) -> pd.DataFrame | None:
    if boundary_df is None or boundary_df.empty:
        return None
    if buffer_m is None or (not np.isfinite(buffer_m)) or float(buffer_m) <= 0.0:
        return None

    boundary_latlon = _boundary_df_to_latlon_list(boundary_df)
    if len(boundary_latlon) < 3:
        return None

    ctx = make_context_from_boundary(boundary_latlon)
    to_utm: Transformer = ctx["to_utm"]
    to_wgs: Transformer = ctx["to_wgs"]

    boundary_xy = [to_utm.transform(lon, lat) for (lat, lon) in boundary_latlon]  # lon,lat
    poly = Polygon(boundary_xy)
    if (not poly.is_valid) or poly.area <= 0:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None

    poly_buf = poly.buffer(-float(buffer_m))
    if poly_buf.is_empty or poly_buf.area <= 0:
        return None

    coords = list(poly_buf.exterior.coords)
    rows = []
    for i, (x, y) in enumerate(coords):
        lon, lat = to_wgs.transform(x, y)
        rows.append({"seq": i, "lat": float(lat), "lon": float(lon)})

    return pd.DataFrame(rows)


def render_start_photo(page_title: str = "SailGP – Replay", mode_override: str | None = None):
    st.title(page_title)

    cfg = get_cfg()

    with st.sidebar:
        st.header("Module principal (Start Photo)")

        # --- NEW: boat_ref
        boat_ref = st.selectbox(
            "Boat de référence (boat_ref)",
            options=list(ALL_BOATS),
            index=list(ALL_BOATS).index("FRA") if "FRA" in ALL_BOATS else 0,
        )
        boat_ref = str(boat_ref)

        if mode_override is None:
            mode = st.radio("Mode", ["Replay", "Live"], horizontal=True)
        else:
            mode = mode_override

        boundary_df = None
        if sidebar_boundary_uploader is not None:
            st.markdown(
                "<div style='padding:6px;border-radius:8px;border:1px solid #999;background:#f5f5f5;'><b>XML Boundary (commun)</b></div>",
                unsafe_allow_html=True,
            )
            boundary_df = sidebar_boundary_uploader()

        if mode == "Replay":
            day = st.date_input("Jour (UTC)", value=date(2025, 11, 30))

            # --- NEW: starts detected from boat_ref (never crash)
            df_cnt = _load_race_start_count_day_safe(cfg, day, boat_ref)
            df_starts = detect_starts_from_race_start_count(df_cnt) if df_cnt is not None else pd.DataFrame()

            round_minute = st.checkbox("Arrondir à la minute UTC la plus proche", value=True)

            starts_list = []
            if df_starts is not None and (not df_starts.empty):
                for _, r in df_starts.iterrows():
                    t0 = pd.to_datetime(r["detected_time_utc"]).to_pydatetime()
                    t0 = _to_utc(t0)
                    if round_minute:
                        t0 = _round_to_nearest_minute(t0)
                    label = f"start {int(r['start_index'])} – {t0.strftime('%H:%M:%S')}Z"
                    starts_list.append((label, t0))

            st.subheader("Départ détecté")
            if starts_list:
                label = st.selectbox("Choix départ", [x[0] for x in starts_list], index=0)
                start_time = dict(starts_list)[label]
            else:
                st.info(f"Aucun départ détecté (RACE_START_COUNT {boat_ref}). Utilise le départ manuel.")
                start_time = None

            st.subheader("Départ manuel")
            manual_dt = st.text_input("Heure UTC (YYYY-MM-DD HH:MM)", value=f"{day.isoformat()} 10:17")
            manual_use = st.checkbox("Utiliser départ manuel", value=(start_time is None))

            if manual_use:
                try:
                    start_time = datetime.strptime(manual_dt.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                except Exception:
                    st.error("Format invalide. Exemple: 2025-11-30 10:17")
                    st.stop()

            offset_s = st.slider("Temps relatif au départ (s)", -180, 120, -70, step=1)

        else:
            now = datetime.now(timezone.utc)
            st.write(f"Heure UTC actuelle: **{now.strftime('%H:%M:%S')}Z**")

            auto = st.radio("Départ live", [f"Auto (now + TTS {boat_ref})", "Manuel (minute)"], horizontal=False)
            offset_s = st.slider("Offset affichage (s)", -180, 120, -70, step=1)

            if auto.startswith("Auto"):
                snap_now = load_boat_snapshot(cfg, now, 0, half_window_s=2)
                snap_now = scale_latlon_always(snap_now, "lat", "lon")

                r0 = snap_now[snap_now["boat"] == boat_ref]
                tts = float(pd.to_numeric(r0["TTS_s"].iloc[0], errors="coerce")) if (not r0.empty) else np.nan
                st.write(f"TTS {boat_ref}: **{tts:.1f}s**" if np.isfinite(tts) else f"TTS {boat_ref}: indisponible")

                if not np.isfinite(tts):
                    st.warning(f"TTS {boat_ref} indisponible → passe en départ manuel.")
                    start_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                else:
                    start_time = now + timedelta(seconds=float(tts))
                    start_time = start_time.replace(microsecond=0)
                    st.write(f"Start estimé: **{start_time.strftime('%H:%M:%S')}Z**")
            else:
                minute = st.number_input("Minute du prochain départ (0-59)", 0, 59, 16, step=1)
                start_hour = now.replace(second=0, microsecond=0, minute=int(minute))
                if start_hour < now:
                    start_hour += timedelta(hours=1)
                start_time = start_hour
                st.write(f"Start manuel: **{start_time.strftime('%H:%M:%S')}Z**")

        # --- AVG fleet exclusion (exclude all except boat_ref)
        st.divider()
        st.subheader("AVG_fleet — Filtre bateaux")
        candidates = [b for b in ALL_BOATS if b != boat_ref]
        excluded = st.multiselect(
            "Exclure du calcul AVG_fleet",
            options=candidates,
            default=[],
        )

    start_time = _to_utc(start_time)

    # Snapshot at t and t-1s
    boats_now = load_boat_snapshot(cfg, start_time, offset_s, half_window_s=1)
    boats_prev = load_boat_snapshot(cfg, start_time, offset_s - 1, half_window_s=1)
    marks = load_marks_snapshot(cfg, start_time, offset_s, half_window_s=2)

    boats_now = scale_latlon_always(boats_now, "lat", "lon")
    boats_prev = scale_latlon_always(boats_prev, "lat", "lon")
    marks = scale_latlon_always(marks, "lat", "lon")

    # TTS boat_ref (safe)
    ref_row = boats_now[boats_now["boat"] == boat_ref]
    tts_ref = pd.to_numeric(ref_row["TTS_s"].iloc[0], errors="coerce") if (not ref_row.empty) else np.nan

    def _color(boat: str):
        if boat == boat_ref:
            return [0, 100, 255, 230]
        if boat == "AVG_fleet":
            return [220, 30, 30, 230]
        return [120, 120, 120, 120]

    boats = boats_now.copy()
    boats["color"] = boats["boat"].astype(str).apply(_color)

    # Vecteurs t-1 -> t (par bateau)
    prev_map = boats_prev.set_index("boat")[["lat", "lon"]] if (boats_prev is not None and not boats_prev.empty) else pd.DataFrame()
    vec_bearing = []
    vec_dist = []
    for _, r in boats.iterrows():
        b = str(r["boat"])
        if (not prev_map.empty) and (b in prev_map.index) and pd.notna(r["lat"]) and pd.notna(r["lon"]):
            lat0, lon0 = prev_map.loc[b, "lat"], prev_map.loc[b, "lon"]
            bearing, dist = _vector_from_positions(float(lat0), float(lon0), float(r["lat"]), float(r["lon"]))
        else:
            bearing, dist = np.nan, np.nan
        vec_bearing.append(bearing)
        vec_dist.append(dist)

    boats["VEC_BEARING_deg"] = vec_bearing
    boats["VEC_DIST_m"] = vec_dist

    # --- Eligible codes cohérents partout
    excluded_set = set(map(str, excluded))
    eligible_codes = [b for b in ALL_BOATS if b != boat_ref and b not in excluded_set]

    # fleet AVG (barycentre) + vecteur barycentre (sur eligible_codes)
    fleet_now = boats_now[boats_now["boat"].isin(eligible_codes)].dropna(subset=["lat", "lon"])
    fleet_prev = boats_prev[boats_prev["boat"].isin(eligible_codes)].dropna(subset=["lat", "lon"])

    # TTK AVG (moyenne bateaux éligibles)
    ttk_avg = pd.to_numeric(
        boats_now.loc[boats_now["boat"].isin(eligible_codes), "TTK_s"],
        errors="coerce",
    ).mean()
    ttk_avg = float(ttk_avg) if np.isfinite(ttk_avg) else np.nan

    if not fleet_now.empty:
        lat1 = float(fleet_now["lat"].mean())
        lon1 = float(fleet_now["lon"].mean())

        if not fleet_prev.empty:
            lat0 = float(fleet_prev["lat"].mean())
            lon0 = float(fleet_prev["lon"].mean())
            b_avg, d_avg = _vector_from_positions(lat0, lon0, lat1, lon1)
        else:
            b_avg, d_avg = np.nan, np.nan

        avg_row = pd.DataFrame([{
            "boat": "AVG_fleet",
            "lat": lat1,
            "lon": lon1,
            "BSP_kmph": np.nan,
            "TTK_s": ttk_avg,
            "TTS_s": np.nan,
            "COG_deg": np.nan,
            "color": _color("AVG_fleet"),
            "VEC_BEARING_deg": b_avg,
            "VEC_DIST_m": d_avg,
        }])
        boats = pd.concat([boats, avg_row], ignore_index=True)

    # Trace boat_ref + AVG
    trace_rows = []
    t_abs = start_time + timedelta(seconds=int(offset_s))

    ref_trace = load_trace_boat(cfg, boat_ref, t_abs - timedelta(seconds=180), t_abs)
    if ref_trace is not None and not ref_trace.empty:
        path = ref_trace[["lon", "lat"]].values.tolist()
        trace_rows.append({"path": path, "color": [0, 100, 255, 160]})

    if eligible_codes:
        fleet_ts = load_latlon_timeseries_1s(cfg, eligible_codes, t_abs - timedelta(seconds=180), t_abs)
        if fleet_ts is not None and not fleet_ts.empty:
            fleet_ts["lat"] = pd.to_numeric(fleet_ts["lat_raw"], errors="coerce") / 1e7
            fleet_ts["lon"] = pd.to_numeric(fleet_ts["lon_raw"], errors="coerce") / 1e7
            avg_track = (
                fleet_ts.dropna(subset=["time_utc", "lat", "lon"])
                .groupby("time_utc", as_index=False)[["lat", "lon"]]
                .mean(numeric_only=True)
                .sort_values("time_utc")
                .reset_index(drop=True)
            )
            if not avg_track.empty:
                path_avg = avg_track[["lon", "lat"]].astype(float).values.tolist()
                trace_rows.append({"path": path_avg, "color": [220, 30, 30, 160]})

    trace_df = pd.DataFrame(trace_rows)

    # Crossing boat_ref + AVG_fleet (G -> D)
    cross_rows = []
    sl1 = marks.loc[marks["mark"] == "SL1"].dropna(subset=["lat", "lon"]).head(1)
    sl2 = marks.loc[marks["mark"] == "SL2"].dropna(subset=["lat", "lon"]).head(1)

    if not sl1.empty and not sl2.empty:
        sl1_lat, sl1_lon = float(sl1["lat"].iloc[0]), float(sl1["lon"].iloc[0])
        sl2_lat, sl2_lon = float(sl2["lat"].iloc[0]), float(sl2["lon"].iloc[0])

        t_start = t_abs - timedelta(seconds=180)
        t_end = t_abs

        # boat_ref crossing
        if ref_trace is not None and not ref_trace.empty:
            ref_track = ref_trace[["time_utc", "lat", "lon"]].copy()
            ref_cross = last_left_to_right_crossing(ref_track, sl1_lat, sl1_lon, sl2_lat, sl2_lon)
            if ref_cross is not None:
                cross_time = ref_cross["time_utc"].to_pydatetime()
                tti = (start_time - cross_time).total_seconds()
                bsp_kmph, ttk_s = _snapshot_metrics_at_time(cfg, cross_time, boat=boat_ref)

                cross_rows.append({
                    "boat": boat_ref,
                    "time_utc": ref_cross["time_utc"],
                    "lat": float(ref_cross["lat"]),
                    "lon": float(ref_cross["lon"]),
                    "d_pin_m": float(ref_cross["dist_sl2_m"]),
                    "tti_s": float(tti),
                    "bsp_kmph": float(bsp_kmph),
                    "ttk_s": float(ttk_s),
                    "color": [0, 100, 255, 230],
                    "label": f"{int(round(tti))}s / {int(round(ref_cross['dist_sl2_m']))}m",
                })

        # AVG crossing
        if eligible_codes:
            fleet_ts = load_latlon_timeseries_1s(cfg, eligible_codes, t_start, t_end)
            if fleet_ts is not None and not fleet_ts.empty:
                fleet_ts["lat"] = pd.to_numeric(fleet_ts["lat_raw"], errors="coerce") / 1e7
                fleet_ts["lon"] = pd.to_numeric(fleet_ts["lon_raw"], errors="coerce") / 1e7
                avg_track = (
                    fleet_ts.dropna(subset=["time_utc", "lat", "lon"])
                    .groupby("time_utc", as_index=False)[["lat", "lon"]]
                    .mean(numeric_only=True)
                    .sort_values("time_utc")
                    .reset_index(drop=True)
                )
                avg_cross = last_left_to_right_crossing(avg_track, sl1_lat, sl1_lon, sl2_lat, sl2_lon)
                if avg_cross is not None:
                    cross_time = avg_cross["time_utc"].to_pydatetime()
                    tti = (start_time - cross_time).total_seconds()
                    bsp_kmph, ttk_s = _snapshot_metrics_at_time(cfg, cross_time, eligible_codes=eligible_codes)

                    cross_rows.append({
                        "boat": "AVG_fleet",
                        "time_utc": avg_cross["time_utc"],
                        "lat": float(avg_cross["lat"]),
                        "lon": float(avg_cross["lon"]),
                        "d_pin_m": float(avg_cross["dist_sl2_m"]),
                        "tti_s": float(tti),
                        "bsp_kmph": float(bsp_kmph),
                        "ttk_s": float(ttk_s),
                        "color": [220, 30, 30, 230],
                        "label": f"{int(round(tti))}s / {int(round(avg_cross['dist_sl2_m']))}m",
                    })

    cross_df = pd.DataFrame(
        cross_rows,
        columns=["boat", "time_utc", "lat", "lon", "d_pin_m", "tti_s", "bsp_kmph", "ttk_s", "color", "label"],
    )

    # TWD/TWS boat_ref (caption, safe)
    tw = _load_twd_tws_safe(cfg, start_time, offset_s, boat_ref)
    if tw:
        st.caption(
            f"TWD {boat_ref}: {tw.get('TWD_MHU_SGP_deg', float('nan')):.1f}° — "
            f"TWS {boat_ref}: {tw.get('TWS_MHU_SGP_km_h_1', float('nan')):.1f} km/h"
        )
    else:
        st.caption(f"TWD/TWS {boat_ref}: indisponible")

    # Buffer boundary synchronisé Start Aid via session_state
    buffer_m = float(st.session_state.get("size_buffer_BDY_m", 15.0))
    boundary_buffer_df = _compute_boundary_buffer_df(boundary_df, buffer_m)

    deck = build_deck(
        boats_df=boats,
        marks_df=marks,
        cross_df=cross_df,
        trace_df=trace_df,
        boundary_df=boundary_df,
        boundary_buffer_df=boundary_buffer_df,
        tts_s=float(tts_ref) if np.isfinite(tts_ref) else np.nan,
    )

    st.pydeck_chart(deck, width="stretch")

    return {
        "boats": boats,
        "marks": marks,
        "boundary": boundary_df,
        "start_time": start_time,
        "offset_s": offset_s,
        "cfg": cfg,
        "mode": mode,
        "boat_ref": boat_ref,
        "eligible_codes": eligible_codes,
        "cross_df": cross_df,
        "trace_df": trace_df,
        "tts_ref": float(tts_ref) if np.isfinite(tts_ref) else np.nan,
        "boundary_buffer_df": boundary_buffer_df,
    }


def main():
    st.set_page_config(page_title="SailGP – Start Photo", layout="wide")
    render_start_photo(page_title="SailGP – Start Photo")


if __name__ == "__main__":
    main()
