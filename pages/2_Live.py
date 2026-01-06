from __future__ import annotations

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from influx_io import (
    get_cfg,
    load_boat_snapshot,
    load_marks_snapshot,
    load_avg_fleet_vector_mean_displacement,
    load_latlon_timeseries_1s,
    scale_latlon_always,
    ALL_BOATS,
)

from viz_deck import build_deck, color_for_boat
from xml_boundary import parse_course_limit_xml


def add_avg_fleet_barycenter_with_vector(
    boats_df_deg: pd.DataFrame,
    avg_vec_bearing_deg: float,
    avg_vec_dist_m: float,
    avg_filter: str,
) -> pd.DataFrame:
    df = boats_df_deg.copy()
    eligible = df[df["boat"] != "FRA"].copy()

    if avg_filter == "only port":
        eligible = eligible[pd.to_numeric(eligible["TWA_deg"], errors="coerce") < 0].copy()

    if eligible.empty:
        return df

    avg = {
        "boat": "AVG_fleet",
        "lat": float(pd.to_numeric(eligible["lat"], errors="coerce").mean()),
        "lon": float(pd.to_numeric(eligible["lon"], errors="coerce").mean()),
        "BSP_kmph": float(pd.to_numeric(eligible.get("BSP_kmph"), errors="coerce").mean()),
        "TWA_deg": float(pd.to_numeric(eligible.get("TWA_deg"), errors="coerce").mean()),
        "TWD_deg": float(pd.to_numeric(eligible.get("TWD_deg"), errors="coerce").mean()),
        "COG_deg": float(pd.to_numeric(eligible.get("COG_deg"), errors="coerce").mean()),
        "TTK_s": float(pd.to_numeric(eligible.get("TTK_s"), errors="coerce").mean()),
        "TTS_s": float(pd.to_numeric(eligible.get("TTS_s"), errors="coerce").mean()),
        "START_RATIO": float(pd.to_numeric(eligible.get("START_RATIO"), errors="coerce").mean()),
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


def last_left_to_right_crossing(df_track: pd.DataFrame, sl1_lat: float, sl1_lon: float, sl2_lat: float, sl2_lon: float) -> dict | None:
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


def make_path(df_track: pd.DataFrame) -> list[list[float]]:
    g = df_track.dropna(subset=["lat", "lon"]).sort_values("time_utc")
    if g.empty:
        return []
    return g[["lon", "lat"]].astype(float).values.tolist()


def main():
    st.set_page_config(page_title="Live", layout="wide")
    st.title("SailGP – Live (calé sur maintenant)")

    cfg = get_cfg()
    now_utc = datetime.now(timezone.utc)

    with st.sidebar:
        st.header("Options")
        avg_filter_ui = st.selectbox("Barycentre AVG_fleet", ["ALL except FRA", "only port"], index=0)
        avg_filter = "ALL except FRA" if avg_filter_ui.startswith("ALL") else "only port"

        st.divider()
        st.header("Boundary XML")
        xml_file = st.file_uploader("Boundary XML (CourseLimit)", type=["xml"], accept_multiple_files=False)

    boundary_df = pd.DataFrame(columns=["seq", "lat", "lon"])
    if xml_file is not None:
        boundary_df = parse_course_limit_xml(xml_file.getvalue())

    st.caption(f"Heure UTC actuelle: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}Z")

    # En live minimal ici: snapshot "maintenant"
    start_time = now_utc
    offset_s = 0

    with st.spinner("Chargement snapshot..."):
        boats_df = load_boat_snapshot(cfg, start_time, offset_s=offset_s, half_window_s=5)
        marks_df = load_marks_snapshot(cfg, start_time, offset_s=offset_s, half_window_s=5)

        boats_df = scale_latlon_always(boats_df, "lat", "lon")
        marks_df = scale_latlon_always(marks_df, "lat", "lon")

        vec = load_avg_fleet_vector_mean_displacement(cfg, start_time, offset_s=offset_s, half_window_s=5, endpoint_half_window_s=1)

        boats_df = add_avg_fleet_barycenter_with_vector(
            boats_df_deg=boats_df,
            avg_vec_bearing_deg=vec.get("bearing_deg", float("nan")),
            avg_vec_dist_m=vec.get("distance_m", float("nan")),
            avg_filter=avg_filter,
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

    # Fleet list consistent with filter
    if avg_filter == "only port":
        eligible_codes = boats_df.loc[
            (boats_df["boat"] != "FRA") & (pd.to_numeric(boats_df["TWA_deg"], errors="coerce") < 0),
            "boat",
        ].astype(str).tolist()
    else:
        eligible_codes = boats_df.loc[boats_df["boat"] != "FRA", "boat"].astype(str).tolist()

    # Cross + traces in last 180s
    cross_df = pd.DataFrame(columns=["boat", "lat", "lon", "color", "label"])
    trace_df = pd.DataFrame(columns=["boat", "path", "color"])
    cross_info: dict[str, dict | None] = {}

    sl1 = marks_view.loc[marks_view["mark"] == "SL1"].dropna().head(1)
    sl2 = marks_view.loc[marks_view["mark"] == "SL2"].dropna().head(1)

    if not sl1.empty and not sl2.empty:
        sl1_lat, sl1_lon = float(sl1["lat"].iloc[0]), float(sl1["lon"].iloc[0])
        sl2_lat, sl2_lon = float(sl2["lat"].iloc[0]), float(sl2["lon"].iloc[0])

        t_start = now_utc - timedelta(seconds=180)
        t_end = now_utc

        fra_ts = load_latlon_timeseries_1s(cfg, ["FRA"], t_start, t_end)
        fra_ts["lat"] = fra_ts["lat_raw"] / 1e7
        fra_ts["lon"] = fra_ts["lon_raw"] / 1e7
        fra_track = fra_ts[["time_utc", "lat", "lon"]].copy()
        fra_cross = last_left_to_right_crossing(fra_track, sl1_lat, sl1_lon, sl2_lat, sl2_lon)

        if eligible_codes:
            fleet_ts = load_latlon_timeseries_1s(cfg, eligible_codes, t_start, t_end)
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
        else:
            avg_track = pd.DataFrame(columns=["time_utc", "lat", "lon"])

        avg_cross = last_left_to_right_crossing(avg_track, sl1_lat, sl1_lon, sl2_lat, sl2_lon)

        def fmt_result(cross: dict | None) -> dict | None:
            if cross is None:
                return None
            tti = (now_utc - cross["time_utc"].to_pydatetime()).total_seconds()
            return {
                "TTI_s": float(tti),
                "dist_sl2_m": float(cross["dist_sl2_m"]),
                "time_utc": cross["time_utc"],
                "lat": float(cross["lat"]),
                "lon": float(cross["lon"]),
            }

        cross_info["FRA"] = fmt_result(fra_cross)
        cross_info["AVG_fleet"] = fmt_result(avg_cross)

        rows_c = []
        for boat in ["FRA", "AVG_fleet"]:
            r = cross_info.get(boat)
            if r is None:
                continue
            label = f"{r['TTI_s']:.0f}s / {r['dist_sl2_m']:.0f}m"
            rows_c.append({"boat": boat, "lat": r["lat"], "lon": r["lon"], "color": color_for_boat(boat), "label": label})
        cross_df = pd.DataFrame(rows_c)

        fra_path = make_path(fra_track)
        avg_path = make_path(avg_track)
        rows_t = []
        if fra_path:
            rows_t.append({"boat": "FRA", "path": fra_path, "color": color_for_boat("FRA")})
        if avg_path:
            rows_t.append({"boat": "AVG_fleet", "path": avg_path, "color": color_for_boat("AVG_fleet")})
        trace_df = pd.DataFrame(rows_t)

    st.pydeck_chart(
        build_deck(
            boats_view,
            marks_view,
            cross_df=cross_df,
            trace_df=trace_df,
            boundary_df=boundary_df,
        ),
        use_container_width=True,
    )

    with st.expander("Debug – boundary", expanded=False):
        st.dataframe(boundary_df.head(200), use_container_width=True)


if __name__ == "__main__":
    main()
