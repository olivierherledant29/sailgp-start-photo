#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app – Détection des départs via RACE_START_COUNT_unk + "photo" à T-70s

Workflow:
1) L'utilisateur sélectionne un jour (UTC).
2) L'app lit RACE_START_COUNT_unk sur la journée.
   Dès que la valeur augmente (diff > 0), on considère qu'un départ a été donné.
   On arrondit l'heure détectée à la minute UTC la plus proche (round-to-nearest-minute).
3) Un second sélecteur propose les départs du jour: "start 3, 12:17".
4) Une fois un départ choisi, l'app affiche la photo à T-70s (moyenne T-75 → T-65),
   avec la même représentation que précédemment (bouées colorées + vecteurs bateaux).

Hypothèses:
- Les coordonnées GPS sont toujours en degrés * 1e7 (conversion systématique).
- RACE_START_COUNT_unk peut être taggé par boat; on agrège en prenant la valeur MAX à chaque instant.
- Les vitesses restent en km/h (SailGP).
- Token Influx dans .env: SailGP_TOKEN=...

Dépendances:
- streamlit, pandas, numpy, pydeck, influxdb-client, python-dotenv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, date, time, timezone, timedelta
import warnings
import math

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from influxdb_client import InfluxDBClient

from dotenv import load_dotenv
load_dotenv()

# Réduit le bruit des warnings de shape/pivot
try:
    from influxdb_client.client.warnings import MissingPivotFunction
    warnings.simplefilter("ignore", MissingPivotFunction)
except Exception:
    pass


# -----------------------
# CONFIG LOCALE
# -----------------------

DEFAULT_INFLUX_URL = "https://data.sailgp.tech"
DEFAULT_INFLUX_ORG = "0c2a130d50b8facc"
DEFAULT_INFLUX_BUCKET = "sailgp"
TOKEN_ENV = "SailGP_TOKEN"

GPS_SCALE = 1e7  # degrés * 1e7


# -----------------------
# Constantes projet
# -----------------------

ALL_BOATS = ["AUS", "BRA", "CAN", "DEN", "ESP", "FRA", "GBR", "GER", "ITA", "NZL", "SUI", "USA"]
MARKS = ["SL1", "SL2", "M1"]

# Channels bateaux (photo)
BOAT_CHANNELS = [
    "LATITUDE_GPS_unk",
    "LONGITUDE_GPS_unk",
    "BOAT_SPEED_km_h_1",     # km/h
    "TWA_MHU_SGP_deg",
    "TWD_MHU_SGP_deg",
    "PC_TTS_s",
    "PC_TTK_s",
    "PC_START_RATIO_unk",
]

# Marques (MDSS)
MARK_LAT_CH = "LATITUDE_MDSS_deg"
MARK_LON_CH = "LONGITUDE_MDSS_deg"

# Détection départs
START_COUNTER_CH = "RACE_START_COUNT_unk"


# -----------------------
# Types / helpers
# -----------------------

@dataclass(frozen=True)
class InfluxCfg:
    url: str
    org: str
    token: str
    bucket: str


def get_cfg() -> InfluxCfg:
    token = os.getenv(TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"Token Influx manquant. Mets-le dans .env :\n"
            f"  {TOKEN_ENV}=<TON_TOKEN>\n"
            f"Optionnel : URL / ORG / BUCKET\n"
            f"  URL={DEFAULT_INFLUX_URL}\n"
            f"  ORG={DEFAULT_INFLUX_ORG}\n"
            f"  BUCKET={DEFAULT_INFLUX_BUCKET}\n"
        )

    url = os.getenv("URL", DEFAULT_INFLUX_URL)
    org = os.getenv("ORG", DEFAULT_INFLUX_ORG)
    bucket = os.getenv("BUCKET", DEFAULT_INFLUX_BUCKET)
    return InfluxCfg(url=url, org=org, token=token, bucket=bucket)


@st.cache_resource(show_spinner=False)
def get_client(cfg: InfluxCfg) -> InfluxDBClient:
    return InfluxDBClient(
        url=cfg.url,
        token=cfg.token,
        org=cfg.org,
        verify_ssl=False,
        timeout=60_000,
    )


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def scale_latlon_always(df: pd.DataFrame, lat_col="lat", lon_col="lon") -> pd.DataFrame:
    out = df.copy()
    if lat_col in out.columns:
        out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce") / GPS_SCALE
    if lon_col in out.columns:
        out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce") / GPS_SCALE
    return out


def round_to_nearest_minute_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Round-to-nearest-minute:
    - +30s puis floor à la minute.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return (ts + pd.Timedelta(seconds=30)).floor("min")


# -----------------------
# Influx queries
# -----------------------

def query_mean_by_boat(
    cfg: InfluxCfg,
    measurement: str,
    boats: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    level_expr: str,
) -> pd.DataFrame:
    boats_or = " or ".join([f'r["boat"] == "{b}"' for b in boats])

    flux = f'''
from(bucket: "{cfg.bucket}")
  |> range(start: {iso_z(start_utc)}, stop: {iso_z(stop_utc)})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["level"] =~ /{level_expr}/)
  |> filter(fn: (r) => {boats_or})
  |> group(columns: ["boat"])
  |> mean(column: "_value")
  |> keep(columns: ["boat", "_value"])
  |> rename(columns: {{_value: "value"}})
'''
    client = get_client(cfg)
    df = client.query_api().query_data_frame(org=cfg.org, query=flux)

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame(columns=["boat", "value"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["boat", "value"])

    out = df[["boat", "value"]].copy()
    out["boat"] = out["boat"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["boat"]).reset_index(drop=True)


def load_boat_snapshot(cfg: InfluxCfg, start_gun_utc: datetime) -> pd.DataFrame:
    """
    Photo bateaux: moyenne sur 10s entre T-75 et T-65.
    """
    t0 = start_gun_utc - timedelta(seconds=75)
    t1 = start_gun_utc - timedelta(seconds=65)

    base = pd.DataFrame({"boat": ALL_BOATS})

    for ch in BOAT_CHANNELS:
        df_ch = query_mean_by_boat(
            cfg=cfg,
            measurement=ch,
            boats=ALL_BOATS,
            start_utc=t0,
            stop_utc=t1,
            level_expr="strm",
        ).rename(columns={"value": ch})
        base = base.merge(df_ch, on="boat", how="left")

    base["BSP_kmph"] = pd.to_numeric(base.get("BOAT_SPEED_km_h_1"), errors="coerce")
    base["TWA_deg"] = pd.to_numeric(base.get("TWA_MHU_SGP_deg"), errors="coerce")
    base["TWD_deg"] = pd.to_numeric(base.get("TWD_MHU_SGP_deg"), errors="coerce")
    base["TTK_s"] = pd.to_numeric(base.get("PC_TTK_s"), errors="coerce")
    base["TTS_s"] = pd.to_numeric(base.get("PC_TTS_s"), errors="coerce")
    base["START_RATIO"] = pd.to_numeric(base.get("PC_START_RATIO_unk"), errors="coerce")

    # positions (degrés*1e7 -> convert plus bas)
    base["lat"] = pd.to_numeric(base.get("LATITUDE_GPS_unk"), errors="coerce")
    base["lon"] = pd.to_numeric(base.get("LONGITUDE_GPS_unk"), errors="coerce")
    return base


def load_mark_snapshot(cfg: InfluxCfg, start_gun_utc: datetime) -> pd.DataFrame:
    """
    Photo marques SL1/SL2/M1: moyenne sur 10s entre T-75 et T-65.
    """
    t0 = start_gun_utc - timedelta(seconds=75)
    t1 = start_gun_utc - timedelta(seconds=65)

    df_lat = query_mean_by_boat(
        cfg=cfg,
        measurement=MARK_LAT_CH,
        boats=MARKS,
        start_utc=t0,
        stop_utc=t1,
        level_expr="mdss|mdss_fast|strm|raw",
    ).rename(columns={"value": "lat"})

    df_lon = query_mean_by_boat(
        cfg=cfg,
        measurement=MARK_LON_CH,
        boats=MARKS,
        start_utc=t0,
        stop_utc=t1,
        level_expr="mdss|mdss_fast|strm|raw",
    ).rename(columns={"value": "lon"})

    out = pd.DataFrame({"mark": MARKS}).rename(columns={"mark": "boat"})
    out = out.merge(df_lat, on="boat", how="left")
    out = out.merge(df_lon, on="boat", how="left")
    out = out.rename(columns={"boat": "mark"})
    return out


@st.cache_data(show_spinner=False, ttl=300)
def load_start_counter_day(cfg: InfluxCfg, day_utc: date) -> pd.DataFrame:
    """
    Charge RACE_START_COUNT_unk sur la journée UTC.
    On échantillonne à 1s via aggregateWindow(last) pour stabiliser.
    Retourne un DF long avec colonnes:
      - time_utc (Timestamp UTC)
      - boat (str ou "")
      - value (float)
    """
    start_dt = datetime.combine(day_utc, time(0, 0, 0), tzinfo=timezone.utc)
    stop_dt = start_dt + timedelta(days=1)

    flux = f'''
from(bucket: "{cfg.bucket}")
  |> range(start: {iso_z(start_dt)}, stop: {iso_z(stop_dt)})
  |> filter(fn: (r) => r["_measurement"] == "{START_COUNTER_CH}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["level"] =~ /strm|mdss|mdss_fast|raw/)
  |> aggregateWindow(every: 1s, fn: last, createEmpty: false)
  |> keep(columns: ["_time","_value","boat"])
  |> sort(columns: ["_time"])
'''
    client = get_client(cfg)
    df = client.query_api().query_data_frame(org=cfg.org, query=flux)

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["time_utc", "boat", "value"])

    out = df.copy()
    out = out.rename(columns={"_time": "time_utc", "_value": "value"})
    if "boat" not in out.columns:
        out["boat"] = ""

    out["time_utc"] = pd.to_datetime(out["time_utc"], errors="coerce")
    # assure UTC
    if out["time_utc"].dt.tz is None:
        out["time_utc"] = out["time_utc"].dt.tz_localize("UTC")
    else:
        out["time_utc"] = out["time_utc"].dt.tz_convert("UTC")

    out["boat"] = out["boat"].fillna("").astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["time_utc", "value"]).reset_index(drop=True)


def detect_starts_from_counter(df_counter: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les instants où RACE_START_COUNT_unk augmente.
    Si plusieurs séries (boats), on prend la valeur MAX à chaque instant.
    Retour:
      - start_index (1..N)
      - detected_time_utc (timestamp avant arrondi)
      - start_time_utc (timestamp arrondi à la minute)
    """
    if df_counter.empty:
        return pd.DataFrame(columns=["start_index", "detected_time_utc", "start_time_utc"])

    # Agrège par temps: max (robuste si duplications)
    g = (
        df_counter.groupby("time_utc", as_index=False)["value"]
        .max()
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    # passage en entier (souvent compteur)
    g["value_i"] = pd.to_numeric(g["value"], errors="coerce").fillna(method="ffill").fillna(0).astype(int)

    g["diff"] = g["value_i"].diff().fillna(0).astype(int)
    inc = g[g["diff"] > 0].copy()
    if inc.empty:
        return pd.DataFrame(columns=["start_index", "detected_time_utc", "start_time_utc"])

    inc["detected_time_utc"] = inc["time_utc"]
    inc["start_time_utc"] = inc["detected_time_utc"].apply(round_to_nearest_minute_utc)

    # dédoublonne (si plusieurs incréments proches qui arrondissent sur la même minute)
    inc = inc.drop_duplicates(subset=["start_time_utc"]).sort_values("start_time_utc").reset_index(drop=True)

    inc["start_index"] = np.arange(1, len(inc) + 1, dtype=int)
    return inc[["start_index", "detected_time_utc", "start_time_utc"]]


# -----------------------
# Géométrie / rotation carte
# -----------------------

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    b = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return float(b)


def compute_map_bearing_to_align_sl1_sl2(marks_df: pd.DataFrame) -> float:
    """
    On veut SL1 au-dessus de SL2 (segment SL2->SL1 vertical).
    On applique bearing = bearing(SL2->SL1) (convention testée sur ton rendu).
    """
    sl1 = marks_df.loc[marks_df["mark"] == "SL1"].head(1)
    sl2 = marks_df.loc[marks_df["mark"] == "SL2"].head(1)
    if sl1.empty or sl2.empty:
        return 0.0

    lat_sl2, lon_sl2 = float(sl2["lat"].iloc[0]), float(sl2["lon"].iloc[0])
    lat_sl1, lon_sl1 = float(sl1["lat"].iloc[0]), float(sl1["lon"].iloc[0])

    if not np.isfinite([lat_sl2, lon_sl2, lat_sl1, lon_sl1]).all():
        return 0.0

    return float(bearing_deg(lat_sl2, lon_sl2, lat_sl1, lon_sl1) % 360.0)


def offset_latlon(lat: float, lon: float, bearing: float, distance_m: float) -> tuple[float, float]:
    R = 6371000.0
    br = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_m / R) +
                     math.cos(lat1) * math.sin(distance_m / R) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(distance_m / R) * math.cos(lat1),
                             math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0)


def compute_heading_from_twd_twa(twd_deg: float, twa_deg: float) -> float:
    if not np.isfinite(twd_deg) or not np.isfinite(twa_deg):
        return np.nan
    return float((twd_deg - twa_deg) % 360.0)


# -----------------------
# Bateau virtuel AVG_fleet
# -----------------------

def add_avg_fleet(boats_df: pd.DataFrame) -> pd.DataFrame:
    df = boats_df.copy()
    eligible = df[(df["boat"] != "FRA") & (pd.to_numeric(df["TWA_deg"], errors="coerce") < 0)].copy()
    if eligible.empty:
        return df

    avg = {
        "boat": "AVG_fleet",
        "lat": float(pd.to_numeric(eligible["lat"], errors="coerce").mean()),
        "lon": float(pd.to_numeric(eligible["lon"], errors="coerce").mean()),
        "BSP_kmph": float(pd.to_numeric(eligible["BSP_kmph"], errors="coerce").mean()),
        "TWA_deg": float(pd.to_numeric(eligible["TWA_deg"], errors="coerce").mean()),
        "TWD_deg": float(pd.to_numeric(eligible["TWD_deg"], errors="coerce").mean()),
        "TTK_s": float(pd.to_numeric(eligible["TTK_s"], errors="coerce").mean()),
        "TTS_s": float(pd.to_numeric(eligible["TTS_s"], errors="coerce").mean()),
        "START_RATIO": float(pd.to_numeric(eligible["START_RATIO"], errors="coerce").mean()),
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


# -----------------------
# Rendu Pydeck (style demandé)
# -----------------------

def color_for_buoy(mark: str) -> list[int]:
    if mark in ("SL1", "SL2"):
        return [255, 105, 180]  # rose
    if mark == "M1":
        return [255, 215, 0]    # jaune
    return [200, 200, 200]


def color_for_boat(boat: str) -> list[int]:
    if boat == "FRA":
        return [30, 144, 255]   # bleu
    if boat == "AVG_fleet":
        return [255, 0, 0]      # rouge
    return [80, 80, 80]         # gris


def build_vector_layers(boats_df: pd.DataFrame) -> tuple[pdk.Layer, pdk.Layer, pdk.Layer]:
    df = boats_df.dropna(subset=["lat", "lon"]).copy()

    df["heading_deg"] = df.apply(
        lambda r: compute_heading_from_twd_twa(r.get("TWD_deg", np.nan), r.get("TWA_deg", np.nan)),
        axis=1,
    )

    v = pd.to_numeric(df.get("BSP_kmph"), errors="coerce")
    df["arrow_m"] = (60.0 + 3.0 * v.fillna(0.0)).clip(lower=60.0, upper=260.0)

    ends = df.apply(
        lambda r: offset_latlon(float(r["lat"]), float(r["lon"]), float(r["heading_deg"]), float(r["arrow_m"]))
        if np.isfinite(r.get("heading_deg", np.nan)) else (np.nan, np.nan),
        axis=1,
        result_type="expand",
    )
    df["lat2"] = ends[0]
    df["lon2"] = ends[1]

    df["color"] = df["boat"].astype(str).apply(color_for_boat)
    df["path"] = df.apply(lambda r: [[r["lon"], r["lat"]], [r["lon2"], r["lat2"]]], axis=1)

    base_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=10,          # plus petit que bouées
        get_fill_color="color",
        pickable=True,
    )

    path_layer = pdk.Layer(
        "PathLayer",
        data=df.dropna(subset=["lat2", "lon2"]),
        get_path="path",
        get_width=2,
        get_color="color",
        pickable=False,
    )

    head = df.dropna(subset=["lat2", "lon2", "heading_deg"]).copy()
    head["glyph"] = "➤"
    head_layer = pdk.Layer(
        "TextLayer",
        data=head,
        get_position='[lon2, lat2]',
        get_text="glyph",
        get_size=20,
        get_angle="heading_deg",
        get_color="color",
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        pickable=False,
    )

    return base_layer, path_layer, head_layer


def build_label_layer(boats_df: pd.DataFrame) -> pdk.Layer:
    df = boats_df.dropna(subset=["lat", "lon"]).copy()

    def _fmt(r):
        boat = str(r.get("boat", ""))
        bsp = r.get("BSP_kmph", np.nan)
        twa = r.get("TWA_deg", np.nan)
        ttk = r.get("TTK_s", np.nan)
        if not np.isfinite(bsp) or not np.isfinite(twa) or not np.isfinite(ttk):
            return boat
        return f"{boat}/{bsp:.0f}kmph/{twa:.0f}°/{ttk:.0f}s"

    df["label"] = df.apply(_fmt, axis=1)

    return pdk.Layer(
        "TextLayer",
        data=df,
        get_position='[lon, lat]',
        get_text="label",
        get_size=12,
        get_alignment_baseline="'top'",
        get_text_anchor="'start'",
        get_color=[20, 20, 20],
        pickable=False,
    )


def build_buoy_layers(marks_df: pd.DataFrame) -> tuple[pdk.Layer, pdk.Layer]:
    df = marks_df.dropna(subset=["lat", "lon"]).copy()
    df["color"] = df["mark"].astype(str).apply(color_for_buoy)

    points = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=14,
        get_fill_color="color",
        pickable=True,
    )

    labels = pdk.Layer(
        "TextLayer",
        data=df,
        get_position='[lon, lat]',
        get_text="mark",
        get_size=14,
        get_alignment_baseline="'bottom'",
        get_text_anchor="'middle'",
        get_color=[20, 20, 20],
        pickable=False,
    )

    return points, labels


def build_deck(boats_df: pd.DataFrame, marks_df: pd.DataFrame) -> pdk.Deck:
    pts = pd.concat(
        [boats_df[["lat", "lon"]].copy(), marks_df[["lat", "lon"]].copy()],
        ignore_index=True,
    ).dropna()

    if pts.empty:
        center_lat, center_lon = 0.0, 0.0
    else:
        center_lat, center_lon = float(pts["lat"].mean()), float(pts["lon"].mean())

    bearing_map = compute_map_bearing_to_align_sl1_sl2(marks_df)

    buoy_points, buoy_labels = build_buoy_layers(marks_df)
    base_layer, path_layer, head_layer = build_vector_layers(boats_df)
    boat_labels = build_label_layer(boats_df)

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        bearing=bearing_map,
        pitch=0,
    )

    tooltip = {"text": "{boat}\nBSP {BSP_kmph} kmph\nTWA {TWA_deg}°\nTTK {TTK_s}s"}

    return pdk.Deck(
        layers=[path_layer, head_layer, base_layer, boat_labels, buoy_points, buoy_labels],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v10",
    )


# -----------------------
# App Streamlit
# -----------------------

def main():
    st.set_page_config(page_title="SailGP – Starts & Photo", layout="wide")
    st.title("SailGP – Départs (RACE_START_COUNT_unk) + Photo à T-70s")

    try:
        cfg = get_cfg()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("1) Choisir la journée (UTC)")
        day = st.date_input("Jour (UTC)", value=datetime.now(timezone.utc).date())

        st.divider()
        st.header("2) Départ détecté")
        st.caption("Détection: incrément de RACE_START_COUNT_unk, arrondi à la minute UTC.")

    # Charge / détecte
    with st.spinner("Lecture RACE_START_COUNT_unk et détection des départs..."):
        df_counter = load_start_counter_day(cfg, day)
        df_starts = detect_starts_from_counter(df_counter)

    if df_starts.empty:
        st.warning("Aucun départ détecté sur cette journée (aucun incrément de RACE_START_COUNT_unk).")
        with st.expander("Debug – aperçu compteur", expanded=False):
            st.dataframe(df_counter.head(200), use_container_width=True)
        st.stop()

    # Sélecteur départ
    starts_labels = []
    for _, r in df_starts.iterrows():
        hhmm = pd.Timestamp(r["start_time_utc"]).tz_convert("UTC").strftime("%H:%M")
        starts_labels.append(f"start {int(r['start_index'])}, {hhmm}")

    with st.sidebar:
        choice = st.selectbox("Choisir le départ", starts_labels, index=0)

    # Map selection -> start_time
    chosen_idx = starts_labels.index(choice)
    start_time = pd.Timestamp(df_starts.loc[chosen_idx, "start_time_utc"]).to_pydatetime()
    start_time = start_time.replace(tzinfo=timezone.utc)

    st.subheader(f"Départ sélectionné: {choice} (UTC = {start_time.strftime('%Y-%m-%dT%H:%M:%SZ')})")

    # Photo à T-70 (moyenne T-75 → T-65)
    with st.spinner("Chargement de la photo (T-75 → T-65) ..."):
        boats_df = load_boat_snapshot(cfg, start_time)
        marks_df = load_mark_snapshot(cfg, start_time)

        boats_df = add_avg_fleet(boats_df)

        # conversion degrés*1e7 -> degrés
        boats_df = scale_latlon_always(boats_df, "lat", "lon")
        marks_df = scale_latlon_always(marks_df, "lat", "lon")

    boats_view = boats_df[["boat", "lat", "lon", "BSP_kmph", "TWA_deg", "TWD_deg", "TTK_s", "TTS_s", "START_RATIO"]].copy()
    marks_view = marks_df[["mark", "lat", "lon"]].copy()

    col_map, col_tbl = st.columns([2.2, 1.0])

    with col_map:
        st.subheader("Carte – Photo à ~T-70s")
        st.pydeck_chart(build_deck(boats_view, marks_view), use_container_width=True)

    with col_tbl:
        st.subheader("Données – snapshot")
        st.caption("Bateaux")
        st.dataframe(boats_view.sort_values("boat"), use_container_width=True, height=420)
        st.caption("Marques")
        st.dataframe(marks_view.sort_values("mark"), use_container_width=True, height=220)

    with st.expander("Debug – départs détectés", expanded=False):
        show = df_starts.copy()
        show["detected_time_utc"] = pd.to_datetime(show["detected_time_utc"]).dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        show["start_time_utc"] = pd.to_datetime(show["start_time_utc"]).dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        st.dataframe(show, use_container_width=True)


if __name__ == "__main__":
    main()
