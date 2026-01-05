from __future__ import annotations

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import streamlit as st

# Local: .env ; Cloud: secrets/env vars (dotenv optionnel)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from influx_io import (
    get_cfg,
    load_race_start_count_day_fra,
    detect_starts_from_race_start_count,
    load_boat_snapshot,
    load_marks_snapshot,
    load_avg_fleet_vector_mean_displacement,
    load_latlon_timeseries_1s,
    scale_latlon_always,
    ALL_BOATS,
)

from viz_deck import build_deck, color_for_boat


def add_avg_fleet_barycenter_with_vector(
    boats_df_deg: pd.DataFrame,
    avg_vec_bearing_deg: float,
    avg_vec_dist_m: float,
) -> pd.DataFrame:
    df = boats_df_deg.copy()
    eligible = df[df["boat"] != "FRA"].copy()
    if eligible.empty:
        return df

    avg = {
        "boat": "AVG_fleet",
        "lat": float(pd.to_numeric(eligible["lat"], errors="coerce").mean()),
        "lon": float(pd.to_numeric(eligible["lon"], errors="coerce").mean()),
        "BSP_kmph": float(pd.to_numeric(eligible["BSP_kmph"], errors="coerce").mean()),
        "TWA_deg": float(pd.to_numeric(eligible["TWA_deg"], errors="coerce").mean()),
        "TWD_deg": float(pd.to_numeric(eligible["TWD_deg"], errors="coerce").mean()),
        "COG_deg": float(pd.to_numeric(eligible["COG_deg"], errors="coerce").mean()),
        "TTK_s": float(pd.to_numeric(eligible["TTK_s"], errors="coerce").mean()),
        "TTS_s": float(pd.to_numeric(eligible["TTS_s"], errors="coerce").mean()),
        "START_RATIO": float(pd.to_numeric(eligible["START_RATIO"], errors="coerce").mean()),
        "AVG_VEC_BEARING_deg": float(avg_vec_bearing_deg),
        "AVG_VEC_DIST_m": float(avg_vec_dist_m),
    }

    avg_row = pd.DataFrame([avg])
    for c in df.columns:
        if c not in avg_row.columns:
            avg_row[c] = np.nan
    for c in avg_row.columns:
        if c not in df.columns:
            df[c] = np.nan
    avg_row = avg_row[df.columns]
    return pd.concat([df, avg_row], ignore_index=True)


def round_start_time(dt_utc: datetime, mode: str) -> datetime:
    ts = pd.Timestamp(dt_utc).tz_convert("UTC")
    if mode == "Aucun":
        return ts.to_pydatetime()
    if mode == "Minute la plus proche":
        return (ts + pd.Timedelta(seconds=30)).floor("min").to_pydatetime()
    if mode == "Minute floor":
        return ts.floor("min").to_pydatetime()
    return ts.to_pydatetime()


def _enu_meters(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    x = (lon - lon0) * m_per_deg_lon  # east
    y = (lat - lat0) * m_per_deg_lat  # north
    return x, y


def _rotate(x: np.ndarray, y: np.ndarray, alpha_rad: float) -> tuple[np.ndarray, np.ndarray]:
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return xr, yr


def last_left_to_right_crossing(
    df_track: pd.DataFrame,
    sl1_lat: float, sl1_lon: float,
    sl2_lat: float, sl2_lon: float,
) -> dict | None:
    """
    df_track: columns time_utc (tz UTC), lat, lon (degrees)
    Détecte le dernier franchissement de la droite SL1–SL2 de gauche vers droite
    dans le repère "photo" où SL1 est au-dessus de SL2 (ligne verticale).
    Renvoie: time_utc, lat, lon, dist_sl2_m (signé le long de SL2->SL1).
    """
    if df_track.empty:
        return None

    g = df_track.dropna(subset=["time_utc", "lat", "lon"]).sort_values("time_utc").reset_index(drop=True)
    if len(g) < 2:
        return None

    # ENU centré sur SL2
    x1, y1 = _enu_meters(np.array([sl1_lat]), np.array([sl1_lon]), sl2_lat, sl2_lon)
    vx, vy = float(x1[0]), float(y1[0])  # vect SL2->SL1

    # rotation alpha pour aligner SL2->SL1 sur +Y
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
    if x1v == x0:
        f = 0.0
    else:
        f = float((-x0) / (x1v - x0))
        f = max(0.0, min(1.0, f))

    t0 = pd.Timestamp(g.loc[idx, "time_utc"]).to_datetime64()
    t1 = pd.Timestamp(g.loc[idx + 1, "time_utc"]).to_datetime64()
    tn = t0 + (t1 - t0) * f

    lat_cross = float(g.loc[idx, "lat"] + f * (g.loc[idx + 1, "lat"] - g.loc[idx, "lat"]))
    lon_cross = float(g.loc[idx, "lon"] + f * (g.loc[idx + 1, "lon"] - g.loc[idx, "lon"]))
    yr_cross = float(yr[idx] + f * (yr[idx + 1] - yr[idx]))

    time_cross = pd.Timestamp(tn).tz_localize("UTC")
    return {"time_utc": time_cross, "lat": lat_cross, "lon": lon_cross, "dist_sl2_m": yr_cross}


def make_path(df_track: pd.DataFrame) -> list[list[float]]:
    g = df_track.dropna(subset=["lat", "lon"]).sort_values("time_utc")
    if g.empty:
        return []
    return g[["lon", "lat"]].astype(float).values.tolist()


def main():
    st.set_page_config(page_title="SailGP – Start Photo 3", layout="wide")
    st.title("SailGP – Départs (RACE_START_COUNT FRA) + Photo")

    try:
        cfg = get_cfg()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("1) Journée (UTC)")
        day = st.date_input("Jour (UTC)", value=datetime.now(timezone.utc).date())

        st.divider()
        st.header("2) Départ")
        mode = st.radio("Mode départ", ["Auto (RACE_START_COUNT FRA)", "Manuel (heure UTC)"], index=0)

        round_mode = None
        if mode.startswith("Auto"):
            round_mode = st.selectbox("Arrondi (Auto)", ["Aucun", "Minute la plus proche", "Minute floor"], index=0)

        st.divider()
        st.header("3) Temps relatif au départ")
        offset_s = st.slider("Offset (s) relatif au départ", min_value=-180, max_value=120, value=-70, step=1)
        st.caption("Snapshot: centre ± 5 s (10 s total)")

        st.divider()
        st.header("4) Options")
        compute_cross = st.checkbox("Calculer dernier franchissement SL1–SL2", value=True)

    # ---- départ ----
    start_time = None
    detected_info = None
    detected_time_raw = None

    if mode.startswith("Auto"):
        with st.spinner("Lecture RACE_START_COUNT_unk (FRA), mean 1s, détection incréments..."):
            df_count = load_race_start_count_day_fra(cfg, day)
            df_starts = detect_starts_from_race_start_count(df_count)

        if df_starts.empty:
            st.warning("Aucun départ détecté via RACE_START_COUNT_unk (FRA) sur cette journée.")
            with st.expander("Debug – RACE_START_COUNT (FRA)", expanded=False):
                st.dataframe(df_count.head(1500), use_container_width=True)
            st.stop()

        labels = []
        for _, r in df_starts.iterrows():
            t = pd.Timestamp(r["detected_time_utc"]).tz_convert("UTC").strftime("%H:%M:%S")
            labels.append(f"start {int(r['start_index'])}, {t} (Δ={r['count_before']:.0f}→{r['count_after']:.0f})")

        with st.sidebar:
            choice = st.selectbox("Choisir le départ détecté", labels, index=0)

        chosen_idx = labels.index(choice)
        detected_time_raw = (
            pd.Timestamp(df_starts.loc[chosen_idx, "detected_time_utc"])
            .tz_convert("UTC")
            .to_pydatetime()
            .replace(tzinfo=timezone.utc)
        )
        start_time = round_start_time(detected_time_raw, round_mode).replace(tzinfo=timezone.utc)
        detected_info = df_starts.loc[chosen_idx].to_dict()

    else:
        with st.sidebar:
            hhmmss = st.time_input("Heure départ (UTC)", value=datetime(2025, 11, 30, 10, 17, 0).time())
        start_time = datetime.combine(day, hhmmss).replace(tzinfo=timezone.utc)

    if mode.startswith("Auto"):
        st.subheader(
            f"Départ auto: détecté {detected_time_raw.strftime('%H:%M:%S')}Z → utilisé {start_time.strftime('%H:%M:%S')}Z | snapshot T{offset_s:+d}s (±5s)"
        )
        st.caption(f"RACE_START_COUNT FRA {detected_info['count_before']:.0f} → {detected_info['count_after']:.0f}")
    else:
        st.subheader(f"Départ manuel: {start_time.strftime('%Y-%m-%d %H:%M:%S')}Z | snapshot T{offset_s:+d}s (±5s)")

    # ---- snapshot + avg vector ----
    with st.spinner("Chargement snapshot + vecteur AVG_fleet..."):
        boats_df = load_boat_snapshot(cfg, start_time, offset_s=offset_s, half_window_s=5)
        marks_df = load_marks_snapshot(cfg, start_time, offset_s=offset_s, half_window_s=5)

        boats_df = scale_latlon_always(boats_df, "lat", "lon")
        marks_df = scale_latlon_always(marks_df, "lat", "lon")

        vec = load_avg_fleet_vector_mean_displacement(
            cfg, start_time, offset_s=offset_s, half_window_s=5, endpoint_half_window_s=1
        )

        boats_df = add_avg_fleet_barycenter_with_vector(
            boats_df_deg=boats_df,
            avg_vec_bearing_deg=vec.get("bearing_deg", float("nan")),
            avg_vec_dist_m=vec.get("distance_m", float("nan")),
        )

    boats_view = boats_df[
        [
            "boat", "lat", "lon",
            "BSP_kmph", "TWA_deg", "TWD_deg", "COG_deg",
            "TTK_s", "TTS_s", "START_RATIO",
            "AVG_VEC_BEARING_deg", "AVG_VEC_DIST_m",
        ]
    ].copy()
    marks_view = marks_df[["mark", "lat", "lon"]].copy()

    # ---- crossings + traces ----
    cross_df = pd.DataFrame(columns=["boat", "lat", "lon", "color"])
    trace_df = pd.DataFrame(columns=["boat", "path", "color"])
    cross_info: dict[str, dict | None] = {}

    if compute_cross:
        sl1 = marks_view.loc[marks_view["mark"] == "SL1"].dropna().head(1)
        sl2 = marks_view.loc[marks_view["mark"] == "SL2"].dropna().head(1)

        if sl1.empty or sl2.empty:
            st.warning("Impossible de calculer le franchissement: SL1/SL2 manquants.")
        else:
            sl1_lat, sl1_lon = float(sl1["lat"].iloc[0]), float(sl1["lon"].iloc[0])
            sl2_lat, sl2_lon = float(sl2["lat"].iloc[0]), float(sl2["lon"].iloc[0])

            t_start = start_time - timedelta(seconds=180)
            t_end = start_time + timedelta(seconds=int(offset_s))

            if t_end <= t_start:
                st.warning("Fenêtre crossing/trace vide (offset trop tôt).")
            else:
                # FRA track 1 Hz
                fra_ts = load_latlon_timeseries_1s(cfg, ["FRA"], t_start, t_end)
                fra_ts["lat"] = fra_ts["lat_raw"] / 1e7
                fra_ts["lon"] = fra_ts["lon_raw"] / 1e7
                fra_track = fra_ts[["time_utc", "lat", "lon"]].copy()

                fra_cross = last_left_to_right_crossing(fra_track, sl1_lat, sl1_lon, sl2_lat, sl2_lon)

                # AVG_fleet track: barycentre hors FRA à chaque seconde
                fleet = [b for b in ALL_BOATS if b != "FRA"]
                fleet_ts = load_latlon_timeseries_1s(cfg, fleet, t_start, t_end)
                if not fleet_ts.empty:
                    fleet_ts["lat"] = fleet_ts["lat_raw"] / 1e7
                    fleet_ts["lon"] = fleet_ts["lon_raw"] / 1e7
                    avg_track = (
                        fleet_ts.groupby("time_utc", as_index=False)[["lat", "lon"]]
                        .mean(numeric_only=True)
                        .sort_values("time_utc")
                        .reset_index(drop=True)
                    )
                else:
                    avg_track = pd.DataFrame(columns=["time_utc", "lat", "lon"])

                avg_cross = last_left_to_right_crossing(avg_track, sl1_lat, sl1_lon, sl2_lat, sl2_lon)

                def fmt_result(cross: dict | None) -> dict | None:
                    if cross is None:
                        return None
                    tti = (start_time - cross["time_utc"].to_pydatetime()).total_seconds()
                    return {
                        "TTI_s": float(tti),
                        "dist_sl2_m": float(cross["dist_sl2_m"]),
                        "time_utc": cross["time_utc"],
                        "lat": float(cross["lat"]),
                        "lon": float(cross["lon"]),
                    }

                cross_info["FRA"] = fmt_result(fra_cross)
                cross_info["AVG_fleet"] = fmt_result(avg_cross)

                # Cross points (cercles) — même couleur que les bateaux
                rows_c = []
                for boat in ["FRA", "AVG_fleet"]:
                    r = cross_info.get(boat)
                    if r is None:
                        continue
                    rows_c.append(
                        {
                            "boat": boat,
                            "lat": r["lat"],
                            "lon": r["lon"],
                            "color": color_for_boat(boat),
                        }
                    )
                cross_df = pd.DataFrame(rows_c)

                # Traces (PathLayer) — discrètes
                fra_path = make_path(fra_track)
                avg_path = make_path(avg_track)

                rows_t = []
                if fra_path:
                    rows_t.append({"boat": "FRA", "path": fra_path, "color": color_for_boat("FRA")})
                if avg_path:
                    rows_t.append({"boat": "AVG_fleet", "path": avg_path, "color": color_for_boat("AVG_fleet")})
                trace_df = pd.DataFrame(rows_t)

    # ---- boîte infos (compacte) ----
    if compute_cross:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("**Dernier franchissement SL1–SL2 (G→D)**")
            for boat in ["FRA", "AVG_fleet"]:
                r = cross_info.get(boat)
                if r is None:
                    st.write(f"{boat}: n/a")
                else:
                    st.write(f"{boat}: TTI={r['TTI_s']:.1f}s | dSL2={r['dist_sl2_m']:.0f}m")
        with c2:
            st.write("")  # espace

    # ---- carte ----
    st.pydeck_chart(
        build_deck(boats_view, marks_view, cross_df=cross_df, trace_df=trace_df),
        use_container_width=True,
    )

    with st.expander("Debug – vecteur AVG_fleet", expanded=False):
        st.write(vec)

    if compute_cross:
        with st.expander("Debug – crossings/traces", expanded=False):
            st.write(cross_info)
            st.dataframe(cross_df, use_container_width=True)
            st.dataframe(trace_df, use_container_width=True)


if __name__ == "__main__":
    main()
