from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # En Streamlit Cloud, on utilise st.secrets / variables d'env, pas .env
    pass


from influx_io import (
    get_cfg,
    load_race_start_count_day_fra,
    detect_starts_from_race_start_count,
    load_boat_snapshot,
    load_marks_snapshot,
    load_avg_fleet_vector_mean_displacement,
    scale_latlon_always,
)

from viz_deck import build_deck


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
            round_mode = st.selectbox(
                "Arrondi (Auto)",
                ["Aucun", "Minute la plus proche", "Minute floor"],
                index=0,
            )

        st.divider()
        st.header("3) Temps relatif au départ")
        offset_s = st.slider(
            "Offset (s) relatif au départ",
            min_value=-180,
            max_value=120,
            value=-70,
            step=1,
        )
        st.caption("Snapshot: centre ± 5 s (10 s total)")

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
        st.subheader(
            f"Départ manuel: {start_time.strftime('%Y-%m-%d %H:%M:%S')}Z | snapshot T{offset_s:+d}s (±5s)"
        )

    with st.spinner("Chargement snapshot + vecteur AVG_fleet..."):
        boats_df = load_boat_snapshot(cfg, start_time, offset_s=offset_s, half_window_s=5)
        marks_df = load_marks_snapshot(cfg, start_time, offset_s=offset_s, half_window_s=5)

        boats_df = scale_latlon_always(boats_df, "lat", "lon")
        marks_df = scale_latlon_always(marks_df, "lat", "lon")

        vec = load_avg_fleet_vector_mean_displacement(
            cfg=cfg,
            start_gun_utc=start_time,
            offset_s=offset_s,
            half_window_s=5,
            endpoint_half_window_s=1,
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

    st.pydeck_chart(build_deck(boats_view, marks_view), use_container_width=True)

    with st.expander("Debug – vecteur AVG_fleet", expanded=False):
        st.write(vec)


if __name__ == "__main__":
    main()

