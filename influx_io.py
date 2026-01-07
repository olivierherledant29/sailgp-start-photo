from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, date, time, timezone, timedelta
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from influxdb_client import InfluxDBClient

try:
    from influxdb_client.client.warnings import MissingPivotFunction
    warnings.simplefilter("ignore", MissingPivotFunction)
except Exception:
    pass


DEFAULT_INFLUX_URL = "https://data.sailgp.tech"
DEFAULT_INFLUX_ORG = "0c2a130d50b8facc"
DEFAULT_INFLUX_BUCKET = "sailgp"
TOKEN_ENV = "SailGP_TOKEN"

GPS_SCALE = 1e7  # degrees * 1e7

ALL_BOATS = ["AUS", "BRA", "CAN", "DEN", "ESP", "FRA", "GBR", "GER", "ITA", "NZL", "SUI", "USA", "SWE"]
MARKS = ["SL1", "SL2", "M1"]

BOAT_CHANNELS = [
    "LATITUDE_GPS_unk",
    "LONGITUDE_GPS_unk",
    "BOAT_SPEED_km_h_1",
    "TWA_MHU_SGP_deg",
    "TWD_MHU_SGP_deg",
    "TWS_MHU_SGP_km_h_1",   # <-- AJOUT
    "GPS_COG_deg",
    "PC_TTS_s",
    "PC_TTK_s",
    "PC_START_RATIO_unk",
]

MARK_LAT_CH = "LATITUDE_MDSS_deg"
MARK_LON_CH = "LONGITUDE_MDSS_deg"


@dataclass(frozen=True)
class InfluxCfg:
    url: str
    org: str
    token: str
    bucket: str


def get_cfg() -> InfluxCfg:
    token = os.getenv(TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Token Influx manquant. Mets-le dans .env : {TOKEN_ENV}=<TON_TOKEN>")
    url = os.getenv("URL", DEFAULT_INFLUX_URL)
    org = os.getenv("ORG", DEFAULT_INFLUX_ORG)
    bucket = os.getenv("BUCKET", DEFAULT_INFLUX_BUCKET)
    return InfluxCfg(url=url, org=org, token=token, bucket=bucket)


def get_client(cfg: InfluxCfg) -> InfluxDBClient:
    # verify_ssl=False pour éviter l’InsecureRequestWarning en local
    return InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org, verify_ssl=False)


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def snapshot_window_times(start_gun_utc: datetime, offset_s: int, half_window_s: int) -> tuple[datetime, datetime]:
    if start_gun_utc.tzinfo is None:
        start_gun_utc = start_gun_utc.replace(tzinfo=timezone.utc)
    center = start_gun_utc + timedelta(seconds=int(offset_s))
    t0 = center - timedelta(seconds=int(half_window_s))
    t1 = center + timedelta(seconds=int(half_window_s))
    return t0, t1


def scale_latlon_always(df: pd.DataFrame, lat_col="lat", lon_col="lon") -> pd.DataFrame:
    """
    IMPORTANT:
    - GPS_unk (bateaux) est typiquement en degrés * 1e7
    - MDSS_deg (marques) est déjà en degrés

    Donc on applique une heuristique :
      si |val| > 180 => on divise par 1e7, sinon on laisse tel quel.
    """
    out = df.copy()

    if lat_col in out.columns:
        lat = pd.to_numeric(out[lat_col], errors="coerce")
        if lat.notna().any() and float(np.nanmax(np.abs(lat.to_numpy()))) > 180.0:
            out[lat_col] = lat / GPS_SCALE
        else:
            out[lat_col] = lat

    if lon_col in out.columns:
        lon = pd.to_numeric(out[lon_col], errors="coerce")
        if lon.notna().any() and float(np.nanmax(np.abs(lon.to_numpy()))) > 180.0:
            out[lon_col] = lon / GPS_SCALE
        else:
            out[lon_col] = lon

    return out


# -----------------------
# Generic queries
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
  |> mean()
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


def query_last_by_boat(
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
  |> last()
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


# -----------------------
# Public loaders
# -----------------------

@st.cache_data(show_spinner=False, ttl=300)
def load_boat_snapshot(cfg: InfluxCfg, start_gun_utc: datetime, offset_s: int, half_window_s: int = 5) -> pd.DataFrame:
    t0, t1 = snapshot_window_times(start_gun_utc, offset_s, half_window_s)
    base = pd.DataFrame({"boat": ALL_BOATS})

    for ch in BOAT_CHANNELS:
        df_ch = query_mean_by_boat(cfg, ch, ALL_BOATS, t0, t1, "strm").rename(columns={"value": ch})
        base = base.merge(df_ch, on="boat", how="left")

    base["BSP_kmph"] = pd.to_numeric(base.get("BOAT_SPEED_km_h_1"), errors="coerce")
    base["TWA_deg"] = pd.to_numeric(base.get("TWA_MHU_SGP_deg"), errors="coerce")
    base["TWD_deg"] = pd.to_numeric(base.get("TWD_MHU_SGP_deg"), errors="coerce")

    # Ajouts demandés (lecture FRA — mais dispo si feed pour autres)
    base["TWS_kmph"] = pd.to_numeric(base.get("TWS_MHU_SGP_km_h_1"), errors="coerce")
    base["TWD_MHU_deg"] = base["TWD_deg"]

    base["COG_deg"] = pd.to_numeric(base.get("GPS_COG_deg"), errors="coerce")
    base["TTK_s"] = pd.to_numeric(base.get("PC_TTK_s"), errors="coerce")
    base["TTS_s"] = pd.to_numeric(base.get("PC_TTS_s"), errors="coerce")
    base["START_RATIO"] = pd.to_numeric(base.get("PC_START_RATIO_unk"), errors="coerce")

    base["lat"] = pd.to_numeric(base.get("LATITUDE_GPS_unk"), errors="coerce")
    base["lon"] = pd.to_numeric(base.get("LONGITUDE_GPS_unk"), errors="coerce")
    return base


@st.cache_data(show_spinner=False, ttl=300)
def load_marks_snapshot(cfg: InfluxCfg, start_gun_utc: datetime, offset_s: int, half_window_s: int = 5) -> pd.DataFrame:
    t0, t1 = snapshot_window_times(start_gun_utc, offset_s, half_window_s)

    df_lat = query_mean_by_boat(cfg, MARK_LAT_CH, MARKS, t0, t1, "mdss|mdss_fast|strm|raw").rename(columns={"value": "lat"})
    df_lon = query_mean_by_boat(cfg, MARK_LON_CH, MARKS, t0, t1, "mdss|mdss_fast|strm|raw").rename(columns={"value": "lon"})

    out = pd.DataFrame({"mark": MARKS}).rename(columns={"mark": "boat"})
    out = out.merge(df_lat, on="boat", how="left").merge(df_lon, on="boat", how="left")
    out = out.rename(columns={"boat": "mark"})
    return out


# -----------------------
# AVG_fleet vector: mean displacement (hors FRA)
# -----------------------

@st.cache_data(show_spinner=False, ttl=300)
def load_avg_fleet_vector_mean_displacement(
    cfg: InfluxCfg,
    start_gun_utc: datetime,
    offset_s: int,
    half_window_s: int = 5,
    endpoint_half_window_s: int = 1,
    included_boats: list[str] | None = None,
) -> dict[str, float]:
    boats_src = included_boats if included_boats is not None else ALL_BOATS
    boats = [b for b in boats_src if b != "FRA"]
    if not boats:
        return {"bearing_deg": float("nan"), "distance_m": float("nan")}

    t0, t1 = snapshot_window_times(start_gun_utc, offset_s, half_window_s)

    ehw = timedelta(seconds=int(endpoint_half_window_s))
    t0a, t0b = t0 - ehw, t0 + ehw
    t1a, t1b = t1 - ehw, t1 + ehw

    lat0 = query_last_by_boat(cfg, "LATITUDE_GPS_unk", boats, t0a, t0b, "strm").rename(columns={"value": "lat0"})
    lon0 = query_last_by_boat(cfg, "LONGITUDE_GPS_unk", boats, t0a, t0b, "strm").rename(columns={"value": "lon0"})
    lat1 = query_last_by_boat(cfg, "LATITUDE_GPS_unk", boats, t1a, t1b, "strm").rename(columns={"value": "lat1"})
    lon1 = query_last_by_boat(cfg, "LONGITUDE_GPS_unk", boats, t1a, t1b, "strm").rename(columns={"value": "lon1"})

    df = lat0.merge(lon0, on="boat", how="inner").merge(lat1.merge(lon1, on="boat", how="inner"), on="boat", how="inner")
    if df.empty:
        return {"bearing_deg": float("nan"), "distance_m": float("nan")}

    for c in ("lat0", "lon0", "lat1", "lon1"):
        df[c] = pd.to_numeric(df[c], errors="coerce") / GPS_SCALE
    df = df.dropna(subset=["lat0", "lon0", "lat1", "lon1"])
    if df.empty:
        return {"bearing_deg": float("nan"), "distance_m": float("nan")}

    lat_ref = float(df[["lat0", "lat1"]].stack().mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_ref)))

    dx = (df["lon1"] - df["lon0"]) * m_per_deg_lon
    dy = (df["lat1"] - df["lat0"]) * m_per_deg_lat

    dxm = float(np.nanmean(dx))
    dym = float(np.nanmean(dy))
    dist_mean = float(np.nanmean(np.sqrt(dx * dx + dy * dy)))

    if not np.isfinite([dxm, dym, dist_mean]).all():
        return {"bearing_deg": float("nan"), "distance_m": float("nan")}

    bearing = float((np.degrees(np.arctan2(dxm, dym)) + 360.0) % 360.0)
    return {"bearing_deg": bearing, "distance_m": dist_mean}


# -----------------------
# Start detection support (RACE_START_COUNT FRA)
# -----------------------

@st.cache_data(show_spinner=False, ttl=300)
def load_race_start_count_day_fra(cfg: InfluxCfg, day_utc: date) -> pd.DataFrame:
    start_utc = datetime.combine(day_utc, time(0, 0, 0), tzinfo=timezone.utc)
    stop_utc = start_utc + timedelta(days=1)

    flux = f'''
from(bucket: "{cfg.bucket}")
  |> range(start: {iso_z(start_utc)}, stop: {iso_z(stop_utc)})
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["_measurement"] == "RACE_START_COUNT_unk")
  |> filter(fn: (r) => r["boat"] == "FRA")
  |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
  |> keep(columns: ["_time","_value"])
  |> rename(columns: {{_time: "time_utc", _value: "race_start_count"}})
'''
    client = get_client(cfg)
    df = client.query_api().query_data_frame(org=cfg.org, query=flux)

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame(columns=["time_utc", "race_start_count"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["time_utc", "race_start_count"])

    df = df[["time_utc", "race_start_count"]].copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["race_start_count"] = pd.to_numeric(df["race_start_count"], errors="coerce")
    return df.sort_values("time_utc").reset_index(drop=True)


def detect_starts_from_race_start_count(df_count: pd.DataFrame) -> pd.DataFrame:
    df = df_count.dropna(subset=["race_start_count"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["start_index", "detected_time_utc", "count_before", "count_after"])

    df["prev"] = df["race_start_count"].shift(1)
    df["delta"] = df["race_start_count"] - df["prev"]
    hits = df[(df["delta"] >= 0.9) & (df["delta"] <= 1.1)].copy()

    if hits.empty:
        return pd.DataFrame(columns=["start_index", "detected_time_utc", "count_before", "count_after"])

    hits["detected_time_utc"] = hits["time_utc"]
    hits["count_before"] = hits["prev"]
    hits["count_after"] = hits["race_start_count"]

    hits = hits[["detected_time_utc", "count_before", "count_after"]].reset_index(drop=True)
    hits["start_index"] = np.arange(1, len(hits) + 1, dtype=int)
    return hits[["start_index", "detected_time_utc", "count_before", "count_after"]]


# -----------------------
# Timeseries lat/lon (1 Hz) pour crossing
# -----------------------

@st.cache_data(show_spinner=False, ttl=300)
def load_latlon_timeseries_1s(cfg: InfluxCfg, boats: list[str], start_utc: datetime, stop_utc: datetime) -> pd.DataFrame:
    boats_or = " or ".join([f'r["boat"] == "{b}"' for b in boats])
    flux = f'''
from(bucket: "{cfg.bucket}")
  |> range(start: {iso_z(start_utc)}, stop: {iso_z(stop_utc)})
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => (r["_measurement"] == "LATITUDE_GPS_unk" or r["_measurement"] == "LONGITUDE_GPS_unk"))
  |> filter(fn: (r) => r["level"] =~ /strm|mdss|mdss_fast|raw/)
  |> filter(fn: (r) => {boats_or})
  |> aggregateWindow(every: 1s, fn: last, createEmpty: false)
  |> pivot(rowKey:["_time","boat"], columnKey:["_measurement"], valueColumn:"_value")
  |> keep(columns: ["_time","boat","LATITUDE_GPS_unk","LONGITUDE_GPS_unk"])
  |> rename(columns: {{_time:"time_utc", LATITUDE_GPS_unk:"lat_raw", LONGITUDE_GPS_unk:"lon_raw"}})
'''
    client = get_client(cfg)
    df = client.query_api().query_data_frame(org=cfg.org, query=flux)

    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame(columns=["time_utc", "boat", "lat_raw", "lon_raw"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["time_utc", "boat", "lat_raw", "lon_raw"])

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df.sort_values(["boat", "time_utc"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=1)
def load_fra_tts_ttk_at_time(cfg: InfluxCfg, t_utc: datetime) -> dict[str, float]:
    """
    Retourne TTS_s et TTK_s (mean 1s) autour de t_utc pour FRA.
    """
    if t_utc.tzinfo is None:
        t_utc = t_utc.replace(tzinfo=timezone.utc)

    start_dt = t_utc - timedelta(seconds=1)
    stop_dt = t_utc + timedelta(seconds=1)

    def _get(meas: str) -> float:
        df = query_mean_by_boat(cfg, meas, ["FRA"], start_dt, stop_dt, "strm|mdss|mdss_fast|raw")
        if df.empty:
            return float("nan")
        return float(df["value"].iloc[0])

    return {
        "TTS_s": _get("PC_TTS_s"),
        "TTK_s": _get("PC_TTK_s"),
    }
