from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pydeck as pdk


# Basemap CARTO (pas besoin de token)
CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


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


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def compute_map_bearing_to_align_sl1_sl2(marks_df: pd.DataFrame) -> float:
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


def build_vector_layers(boats_df: pd.DataFrame) -> tuple[pdk.Layer, pdk.Layer, pdk.Layer]:
    """
    - Pour les bateaux standards: direction = COG_deg, longueur = f(vitesse).
    - Pour AVG_fleet: direction/longueur = via colonnes optionnelles:
        AVG_VEC_BEARING_deg, AVG_VEC_DIST_m
      (si absentes: fallback sur COG_deg + vitesse)
    """
    df = boats_df.dropna(subset=["lat", "lon"]).copy()

    df["color"] = df["boat"].astype(str).apply(color_for_boat)

    # Heading par défaut: COG
    df["heading_deg"] = pd.to_numeric(df.get("COG_deg"), errors="coerce")

    # Longueur par défaut (m) basée sur la vitesse (km/h)
    v = pd.to_numeric(df.get("BSP_kmph"), errors="coerce")
    df["arrow_m"] = (60.0 + 3.0 * v.fillna(0.0)).clip(lower=60.0, upper=260.0)

    # Override AVG_fleet si vecteur dispo
    is_avg = df["boat"].astype(str) == "AVG_fleet"
    if "AVG_VEC_BEARING_deg" in df.columns:
        df.loc[is_avg, "heading_deg"] = pd.to_numeric(df.loc[is_avg, "AVG_VEC_BEARING_deg"], errors="coerce")
    if "AVG_VEC_DIST_m" in df.columns:
        # On scale la distance réelle pour rester lisible (clip)
        dist = pd.to_numeric(df.loc[is_avg, "AVG_VEC_DIST_m"], errors="coerce")
        df.loc[is_avg, "arrow_m"] = (dist * 2.5).clip(lower=20.0, upper=260.0)  # scale empirique

    ends = df.apply(
        lambda r: offset_latlon(float(r["lat"]), float(r["lon"]), float(r["heading_deg"]), float(r["arrow_m"]))
        if np.isfinite(r.get("heading_deg", np.nan)) and np.isfinite(r.get("arrow_m", np.nan)) else (np.nan, np.nan),
        axis=1,
        result_type="expand",
    )
    df["lat2"] = ends[0]
    df["lon2"] = ends[1]

    df["path"] = df.apply(lambda r: [[r["lon"], r["lat"]], [r["lon2"], r["lat2"]]], axis=1)

    base_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=10,
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


def build_boat_label_layer_name_ttk_only(boats_df: pd.DataFrame) -> pdk.Layer:
    """
    Label sous le bateau: "FRA/8s"
    """
    df = boats_df.dropna(subset=["lat", "lon"]).copy()

    def _fmt(r):
        boat = str(r.get("boat", ""))
        ttk = r.get("TTK_s", np.nan)
        if np.isfinite(ttk):
            return f"{boat}/{ttk:.0f}s"
        return boat

    df["label"] = df.apply(_fmt, axis=1)

    return pdk.Layer(
        "TextLayer",
        data=df,
        get_position='[lon, lat]',
        get_text="label",
        get_size=12,
        get_alignment_baseline="'top'",   # texte "en dessous" du point (ancrage haut)
        get_text_anchor="'middle'",
        get_color=[20, 20, 20],
        pickable=False,
    )


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
    boat_labels = build_boat_label_layer_name_ttk_only(boats_df)

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        bearing=bearing_map,
        pitch=0,
    )

    tooltip = {"text": "{boat}\nTTK {TTK_s}s\nCOG {COG_deg}°"}

    return pdk.Deck(
        layers=[path_layer, head_layer, base_layer, boat_labels, buoy_points, buoy_labels],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=CARTO_POSITRON,
    )
