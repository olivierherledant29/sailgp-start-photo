from __future__ import annotations

import os
import warnings
from datetime import datetime, date, time, timedelta, timezone
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

try:
    load_dotenv()
except Exception:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

URL = os.getenv("URL", "https://data.sailgp.tech")
ORG = os.getenv("ORG", "0c2a130d50b8facc")
TOKEN = os.getenv("SailGP_TOKEN", "")
BUCKET = os.getenv("BUCKET", "sailgp")

ALL_BOATS = ["AUS", "BRA", "CAN", "DEN", "ESP", "FRA", "GBR", "GER", "ITA", "NZL", "SUI", "USA", "SWE"]
WINDMARK_BUOYS = ["WG1", "WG2", "LG1", "LG2", "SL1", "SL2", "M1"]

RACE_CHANNELS = ["RACE_NUM", "TRK_RACE_NUM_unk"]
RACE_CHANNEL_CANDIDATES = ["RACE_NUM", "TRK_RACE_NUM_unk"]
LEG_CHANNEL = "PC_BEACON_NUMBER_unk"
POS_CHANNELS = ["LATITUDE_GPS_unk", "LONGITUDE_GPS_unk"]
DETAIL_CHANNELS = POS_CHANNELS + RACE_CHANNELS + [LEG_CHANNEL]

WINNING_COLORS = [
    [0, 80, 255],
    [0, 180, 80],
    [255, 220, 0],
    [255, 140, 0],
    [255, 40, 0],
    [255, 0, 180],
]


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _client():
    return InfluxDBClient(url=URL, token=TOKEN, org=ORG, verify_ssl=False, timeout=120000)


def _as_df(result) -> pd.DataFrame:
    if isinstance(result, list):
        return pd.concat(result, ignore_index=True) if result else pd.DataFrame()
    return result if result is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def scan_races_day(day_start: datetime, day_end: datetime, scan_boat: str) -> pd.DataFrame:
    meas_filter = " or ".join([f'r["_measurement"] == "{ch}"' for ch in RACE_CHANNELS])
    flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: {iso_z(day_start)}, stop: {iso_z(day_end)})
  |> filter(fn: (r) => r["boat"] == "{scan_boat}")
  |> filter(fn: (r) => {meas_filter})
  |> filter(fn: (r) => r["_field"] == "value")
  |> aggregateWindow(every: 5s, fn: last, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
  |> keep(columns: ["_time", "RACE_NUM", "TRK_RACE_NUM_unk"])
  |> sort(columns: ["_time"])
'''
    try:
        df = _as_df(_client().query_api().query_data_frame(org=ORG, query=flux))
    except Exception as e:
        st.warning(f"Scan courses impossible pour {scan_boat}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"_time": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def race_summary_from_scan(df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    race_col = None
    for c in RACE_CHANNEL_CANDIDATES:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            race_col = c
            break

    if race_col is None:
        return None, pd.DataFrame()

    d = df[["time", race_col]].copy()
    d[race_col] = pd.to_numeric(d[race_col], errors="coerce")
    d = d.dropna(subset=[race_col])

    if d.empty:
        return None, pd.DataFrame()

    d["race_id"] = d[race_col].round().astype(int)
    d = d[d["race_id"] > 0]

    if d.empty:
        return race_col, pd.DataFrame()

    d = d.sort_values("time").reset_index(drop=True)
    d["block"] = (d["race_id"] != d["race_id"].shift()).cumsum()

    rows = []
    for _, g in d.groupby("block"):
        race_id = int(g["race_id"].iloc[0])
        t0 = pd.Timestamp(g["time"].min())
        t1 = pd.Timestamp(g["time"].max())
        duration_min = (t1 - t0).total_seconds() / 60.0

        if duration_min < 5.0 or duration_min > 45.0:
            continue

        rows.append({
            "race_id": race_id,
            "start": t0,
            "end": t1,
            "duration_min": duration_min,
            "n_points": len(g),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return race_col, summary

    return race_col, summary.sort_values("start").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=300)
def read_course_details(start_time: datetime, end_time: datetime, boats: list[str]) -> pd.DataFrame:
    if not boats:
        return pd.DataFrame()

    meas_filter = " or ".join([f'r["_measurement"] == "{ch}"' for ch in DETAIL_CHANNELS])
    boat_filter = " or ".join([f'r["boat"] == "{b}"' for b in boats])

    flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: {iso_z(start_time)}, stop: {iso_z(end_time)})
  |> filter(fn: (r) => {meas_filter})
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => {boat_filter})
  |> group(columns: ["boat", "_measurement"])
  |> aggregateWindow(every: 1s, fn: last, createEmpty: false)
  |> group(columns: ["boat"])
  |> pivot(rowKey:["_time", "boat"], columnKey: ["_measurement"], valueColumn: "_value")
  |> keep(columns: ["_time", "boat", "LATITUDE_GPS_unk", "LONGITUDE_GPS_unk", "RACE_NUM", "TRK_RACE_NUM_unk", "PC_BEACON_NUMBER_unk"])
  |> sort(columns: ["_time"])
'''
    try:
        df = _as_df(_client().query_api().query_data_frame(org=ORG, query=flux))
    except Exception as e:
        st.error(f"Erreur requête Influx détails course : {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"_time": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(["boat", "time"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=300)
def load_buoy_channel(meas: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    boats_or = " or ".join([f'r["boat"] == "{b}"' for b in WINDMARK_BUOYS])
    flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: {iso_z(start_time)}, stop: {iso_z(end_time)})
  |> filter(fn: (r) => r["_measurement"] == "{meas}")
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["level"] =~ /mdss|mdss_fast|strm|raw/)
  |> filter(fn: (r) => {boats_or})
  |> aggregateWindow(every: 5s, fn: mean, createEmpty: false)
  |> keep(columns: ["_time", "_value", "boat"])
  |> pivot(rowKey:["_time"], columnKey:["boat"], valueColumn:"_value")
'''
    try:
        df = _as_df(_client().query_api().query_data_frame(org=ORG, query=flux))
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"_time": "time"}).set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def load_buoy_snapshot(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    df_lat = load_buoy_channel("LATITUDE_MDSS_deg", start_time, end_time)
    df_lon = load_buoy_channel("LONGITUDE_MDSS_deg", start_time, end_time)

    if df_lat.empty or df_lon.empty:
        return pd.DataFrame(columns=["boat", "latitude", "longitude"])

    rows = []
    for b in WINDMARK_BUOYS:
        if b not in df_lat.columns or b not in df_lon.columns:
            continue

        lat_s = pd.to_numeric(df_lat[b], errors="coerce").dropna()
        lon_s = pd.to_numeric(df_lon[b], errors="coerce").dropna()
        if lat_s.empty or lon_s.empty:
            continue

        lat = float(lat_s.iloc[-1])
        lon = float(lon_s.iloc[-1])

        if abs(lat) > 1000:
            lat /= 1e7
        if abs(lon) > 1000:
            lon /= 1e7

        rows.append({"boat": b, "latitude": lat, "longitude": lon})

    return pd.DataFrame(rows)


def parse_course_limit_boundary(xml_bytes: bytes, course_limit_name: str = "Boundary"):
    root = ET.fromstring(xml_bytes)

    target = None
    for el in root.iter():
        if el.tag.lower().endswith("courselimit") and el.attrib.get("name") == course_limit_name:
            target = el
            break

    if target is None:
        return [], {}

    meta = {
        "name": target.attrib.get("name", course_limit_name),
        "colour": target.attrib.get("colour", "000000FF"),
    }

    pts = []
    for lim in target.iter():
        if not lim.tag.lower().endswith("limit"):
            continue
        lat = lim.attrib.get("Lat")
        lon = lim.attrib.get("Lon")
        seq = lim.attrib.get("SeqID")
        if lat is None or lon is None or seq is None:
            continue
        pts.append((int(seq), float(lat), float(lon)))

    if len(pts) < 3:
        return [], meta

    pts.sort(key=lambda x: x[0])
    ring = [(lat, lon) for _, lat, lon in pts]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring, meta


def _hex8_to_rgba(hex8: str, alpha: float = 1.0) -> str:
    h = (hex8 or "000000FF").strip().replace("#", "")
    if len(h) != 8:
        return f"rgba(0,0,0,{alpha})"
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def add_boundary(fig: go.Figure, ring, meta: dict, opacity: float = 0.12):
    if not ring:
        return fig

    lats = [p[0] for p in ring]
    lons = [p[1] for p in ring]
    color = _hex8_to_rgba(meta.get("colour", "000000FF"), 1.0)
    fill = _hex8_to_rgba(meta.get("colour", "000000FF"), opacity)

    fig.add_scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=3, color=color),
        fill="toself",
        fillcolor=fill,
        hoverinfo="skip",
        showlegend=False,
    )
    return fig


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["latitude"] = pd.to_numeric(out["LATITUDE_GPS_unk"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["LONGITUDE_GPS_unk"], errors="coerce")

    if out["latitude"].notna().any() and out["latitude"].abs().median() > 1000:
        out["latitude"] /= 1e7
    if out["longitude"].notna().any() and out["longitude"].abs().median() > 1000:
        out["longitude"] /= 1e7

    return out


def color_from_leg_time(t: float, t_min: float, t_max: float) -> list[int]:
    if not np.isfinite(t) or not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return WINNING_COLORS[-1]

    score_fast = 1.0 - (float(t) - float(t_min)) / (float(t_max) - float(t_min))
    score_fast = max(0.0, min(1.0, score_fast))

    x = score_fast * (len(WINNING_COLORS) - 1)
    i = int(np.floor(x))
    j = min(i + 1, len(WINNING_COLORS) - 1)
    f = x - i

    c0 = np.array(WINNING_COLORS[i], dtype=float)
    c1 = np.array(WINNING_COLORS[j], dtype=float)
    return [int(v) for v in (c0 * (1 - f) + c1 * f)]


def build_leg_segments(df: pd.DataFrame, race_id: int, race_col: str, include_open_last_leg: bool = True):
    df = df.copy()
    df[race_col] = pd.to_numeric(df[race_col], errors="coerce")
    df[LEG_CHANNEL] = pd.to_numeric(df[LEG_CHANNEL], errors="coerce")

    df = df[df[race_col].round().astype("Int64") == int(race_id)].copy()
    df = df.dropna(subset=["time", "boat", "latitude", "longitude", LEG_CHANNEL])

    if df.empty:
        return [], pd.DataFrame()

    df["leg"] = df[LEG_CHANNEL].round().astype(int)
    df = df.sort_values(["boat", "time"]).reset_index(drop=True)

    segments = []
    rows = []

    for boat, g in df.groupby("boat"):
        g = g.sort_values("time").reset_index(drop=True)
        g["block"] = (g["leg"] != g["leg"].shift()).cumsum()
        blocks = list(g.groupby("block"))

        for idx, (_, bdf) in enumerate(blocks):
            leg = int(bdf["leg"].iloc[0])
            if leg < 2:
                continue

            start_t = pd.Timestamp(bdf["time"].iloc[0])
            if idx + 1 < len(blocks):
                end_t = pd.Timestamp(blocks[idx + 1][1]["time"].iloc[0])
                incomplete = False
            else:
                end_t = pd.Timestamp(bdf["time"].iloc[-1])
                incomplete = True

            if incomplete and not include_open_last_leg:
                continue

            dt_s = (end_t - start_t).total_seconds()
            if not np.isfinite(dt_s) or dt_s <= 0:
                continue

            path = bdf[["longitude", "latitude"]].dropna().astype(float).values.tolist()
            if len(path) < 2:
                continue

            leg_type = "UW" if (leg >= 3 and leg % 2 == 1) else ("DW" if leg % 2 == 0 else "R")

            row = {
                "boat": boat,
                "leg": leg,
                "leg_type": leg_type,
                "start_time": start_t,
                "end_time": end_t,
                "dt_s": float(dt_s),
                "incomplete": bool(incomplete),
            }
            rows.append(row)
            segments.append({**row, "path": path, "color": [120, 120, 120]})

    summary = pd.DataFrame(rows)
    if summary.empty:
        return [], summary

    for leg, leg_sum in summary.groupby("leg"):
        t_min = float(leg_sum["dt_s"].min())
        t_max = float(leg_sum["dt_s"].max())
        for seg in segments:
            if int(seg["leg"]) == int(leg):
                seg["color"] = color_from_leg_time(seg["dt_s"], t_min, t_max)

    return segments, summary


def make_traces_map(segments: list[dict], buoys_df: pd.DataFrame, boundary_ring, boundary_meta, title: str):
    fig = go.Figure()

    for seg in segments:
        lons = [p[0] for p in seg["path"]]
        lats = [p[1] for p in seg["path"]]
        c = seg["color"]

        label = f"{seg['boat']} | leg {seg['leg']} | {seg['dt_s']:.1f}s"
        if seg.get("incomplete"):
            label += " | open"

        fig.add_scattermapbox(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(width=4, color=f"rgba({c[0]},{c[1]},{c[2]},0.85)"),
            hovertemplate=label + "<extra></extra>",
            showlegend=False,
        )

    if boundary_ring:
        fig = add_boundary(fig, boundary_ring, boundary_meta)

    if buoys_df is not None and not buoys_df.empty:
        fig.add_scattermapbox(
            lat=buoys_df["latitude"],
            lon=buoys_df["longitude"],
            mode="markers+text",
            marker=dict(size=10, color="black"),
            text=buoys_df["boat"],
            textposition="bottom center",
            textfont=dict(size=12, color="black"),
            hoverinfo="none",
            showlegend=False,
        )

    all_lats, all_lons = [], []
    for seg in segments:
        all_lons += [p[0] for p in seg["path"]]
        all_lats += [p[1] for p in seg["path"]]

    if buoys_df is not None and not buoys_df.empty:
        all_lats += buoys_df["latitude"].dropna().tolist()
        all_lons += buoys_df["longitude"].dropna().tolist()

    center = {"lat": np.nanmean(all_lats), "lon": np.nanmean(all_lons)} if all_lats and all_lons else {"lat": 0, "lon": 0}

    fig.update_layout(
        title=title,
        mapbox_style="carto-positron",
        mapbox_zoom=13,
        mapbox_center=center,
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


st.set_page_config(page_title="Winning Traces", layout="wide")
st.title("🏁 Trace gagnante")

if not TOKEN:
    st.error("Token Influx manquant : définir SailGP_TOKEN dans le fichier .env")
    st.stop()

st.sidebar.header("Paramètres")

day = st.sidebar.date_input("Jour UTC", value=date.today())
day_start = datetime.combine(day, time(0, 0, 0), tzinfo=timezone.utc)
day_end = day_start + timedelta(days=1)

scan_boat = st.sidebar.selectbox(
    "Bateau pour scanner les courses",
    options=ALL_BOATS,
    index=ALL_BOATS.index("FRA") if "FRA" in ALL_BOATS else 0,
)

selected_boats = st.sidebar.multiselect(
    "Bateaux à tracer",
    options=ALL_BOATS,
    default=ALL_BOATS,
)

include_open_last_leg = st.sidebar.checkbox(
    "Inclure dernier leg ouvert si pas de changement suivant",
    value=True,
)

uploaded_boundary = st.sidebar.file_uploader("Boundary XML", type=["xml"])
boundary_course_name = st.sidebar.text_input("CourseLimit name", value="Boundary")

boundary_ring = []
boundary_meta = {}
if uploaded_boundary is not None:
    try:
        boundary_ring, boundary_meta = parse_course_limit_boundary(uploaded_boundary.getvalue(), boundary_course_name)
    except Exception as e:
        st.sidebar.warning(f"Boundary XML non lue : {e}")

with st.spinner("Recherche des courses de la journée..."):
    scan_df = scan_races_day(day_start, day_end, scan_boat)

if scan_df.empty:
    st.warning(f"Aucune course détectée pour {scan_boat} sur cette journée.")
    st.stop()

race_col, races = race_summary_from_scan(scan_df)

if race_col is None or races.empty:
    st.warning("Aucun race number exploitable sur cette journée après filtrage race_id > 0 et durée 5–45 min.")
    st.dataframe(scan_df.head(50), use_container_width=True)
    st.stop()

races["label"] = races.apply(
    lambda r: f"{int(r['race_id'])} | {pd.Timestamp(r['start']).strftime('%H:%M:%S')}–{pd.Timestamp(r['end']).strftime('%H:%M:%S')}Z | {r['duration_min']:.1f} min",
    axis=1,
)

choice = st.selectbox("Course détectée", options=races["label"].tolist(), index=0)
race_row = races.loc[races["label"] == choice].iloc[0]

race_id = int(race_row["race_id"])
course_start = pd.Timestamp(race_row["start"]).to_pydatetime() - timedelta(seconds=30)
course_end = pd.Timestamp(race_row["end"]).to_pydatetime() + timedelta(seconds=30)

st.caption(
    f"Scan via **{scan_boat}** | channel course = **{race_col}** | "
    f"fenêtre chargée : {course_start.strftime('%H:%M:%S')}–{course_end.strftime('%H:%M:%S')}Z"
)

with st.spinner("Lecture trajectoires course..."):
    df_raw = read_course_details(course_start, course_end, selected_boats)

if df_raw.empty:
    st.warning("Aucune donnée de trajectoire sur la fenêtre de course.")
    st.stop()

df = normalize_positions(df_raw)
df = df.sort_values(["boat", "time"]).reset_index(drop=True)

for c in [race_col, LEG_CHANNEL]:
    if c not in df.columns:
        st.error(f"Channel manquant dans les données détaillées : {c}")
        st.dataframe(df_raw.head(30), use_container_width=True)
        st.stop()

    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df.groupby("boat")[c].ffill().bfill()

with st.spinner("Chargement bouées..."):
    buoys_df = load_buoy_snapshot(course_start, course_end)

segments, summary = build_leg_segments(
    df=df,
    race_id=race_id,
    race_col=race_col,
    include_open_last_leg=include_open_last_leg,
)

if summary.empty:
    st.warning("Aucune trace de leg exploitable pour cette course.")
    st.dataframe(df.head(50), use_container_width=True)
    st.stop()

uw_segments = [s for s in segments if s["leg_type"] == "UW"]
dw_segments = [s for s in segments if s["leg_type"] == "DW"]

st.caption(
    f"Race {race_id} | {len(uw_segments)} traces UW | {len(dw_segments)} traces DW | "
    f"rose = plus rapide, bleu = plus lent"
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Traces UW", len(uw_segments))
with col2:
    st.metric("Traces DW", len(dw_segments))

st.subheader("UPWIND — legs impairs ≥ 3")
st.plotly_chart(
    make_traces_map(uw_segments, buoys_df, boundary_ring, boundary_meta, f"Trace gagnante UW — Race {race_id}"),
    use_container_width=True,
)

st.subheader("DOWNWIND — legs pairs")
st.plotly_chart(
    make_traces_map(dw_segments, buoys_df, boundary_ring, boundary_meta, f"Trace gagnante DW — Race {race_id}"),
    use_container_width=True,
)

st.subheader("Tableau des temps de legs")
summary_show = summary.sort_values(["leg", "dt_s"]).copy()
summary_show["dt_s"] = summary_show["dt_s"].round(1)

st.dataframe(
    summary_show[["boat", "leg", "leg_type", "dt_s", "start_time", "end_time", "incomplete"]],
    use_container_width=True,
)