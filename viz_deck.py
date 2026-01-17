from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pydeck as pdk

CARTO_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


def _safe_mean(series: pd.Series, default=0.0) -> float:
    v = pd.to_numeric(series, errors="coerce")
    return float(v.mean()) if v.notna().any() else float(default)


def bearing_sl1_sl2(marks_df: pd.DataFrame) -> float:
    sl1 = marks_df.loc[marks_df["mark"] == "SL1"]
    sl2 = marks_df.loc[marks_df["mark"] == "SL2"]
    if sl1.empty or sl2.empty:
        return 0.0

    lat1, lon1 = float(sl1.iloc[0]["lat"]), float(sl1.iloc[0]["lon"])
    lat2, lon2 = float(sl2.iloc[0]["lat"]), float(sl2.iloc[0]["lon"])

    dy = lat1 - lat2
    dx = (lon1 - lon2) * math.cos(math.radians((lat1 + lat2) * 0.5))
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def build_marks_layers(marks_df: pd.DataFrame):
    df = marks_df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return []

    def color(mark):
        if mark in ("SL1", "SL2"):
            return [255, 105, 180, 230]
        if mark == "M1":
            return [255, 255, 0, 230]
        return [160, 160, 160, 200]

    df["color"] = df["mark"].apply(color)
    df["radius"] = 10

    return [
        pdk.Layer(
            "ScatterplotLayer",
            df,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            df,
            get_position="[lon, lat]",
            get_text="mark",
            get_size=12,
            get_color=[20, 20, 20, 230],
            get_pixel_offset=[0, -12],
        ),
    ]


def _make_dashed_segments(p1, p2, n_segments=22):
    x1, y1 = p1
    x2, y2 = p2
    segs = []
    for i in range(n_segments):
        if i % 2 == 0:
            a = i / n_segments
            b = (i + 1) / n_segments
            segs.append(
                [
                    [x1 + (x2 - x1) * a, y1 + (y2 - y1) * a],
                    [x1 + (x2 - x1) * b, y1 + (y2 - y1) * b],
                ]
            )
    return segs


def build_startline_layer_dashed(marks_df: pd.DataFrame):
    sl1 = marks_df.loc[marks_df["mark"] == "SL1"]
    sl2 = marks_df.loc[marks_df["mark"] == "SL2"]
    if sl1.empty or sl2.empty:
        return None

    p1 = [float(sl1.iloc[0]["lon"]), float(sl1.iloc[0]["lat"])]
    p2 = [float(sl2.iloc[0]["lon"]), float(sl2.iloc[0]["lat"])]

    segs = _make_dashed_segments(p1, p2, n_segments=26)
    df = pd.DataFrame([{"path": s} for s in segs])

    return pdk.Layer(
        "PathLayer",
        df,
        get_path="path",
        get_width=2,
        width_min_pixels=2,
        get_color=[80, 80, 80, 210],
    )


def _meters_per_deg(lat_deg: float) -> tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat_deg)))
    return m_per_deg_lat, m_per_deg_lon


def build_boats_layers(boats_df: pd.DataFrame):
    df = boats_df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return []

    base = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position="[lon, lat]",
        get_radius=4,
        get_fill_color="color",
        get_line_color=[0, 0, 0, 60],
        line_width_min_pixels=1,
        pickable=True,
    )

    vec = df.copy()
    if "VEC_BEARING_deg" in vec.columns and "VEC_DIST_m" in vec.columns:
        vec["heading"] = pd.to_numeric(vec["VEC_BEARING_deg"], errors="coerce")
        vec["dist_m"] = pd.to_numeric(vec["VEC_DIST_m"], errors="coerce")
    else:
        vec["heading"] = pd.to_numeric(vec.get("COG_deg"), errors="coerce")
        bsp = pd.to_numeric(vec.get("BSP_kmph"), errors="coerce")
        vec["dist_m"] = (bsp / 3.6).clip(lower=0.0)

    vec = vec.dropna(subset=["heading", "dist_m", "lat", "lon"]).copy()

    vector_layer = None
    if not vec.empty:
        m_per_deg_lat, _ = _meters_per_deg(float(vec["lat"].astype(float).mean()))
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(vec["lat"].astype(float)))

        theta = np.deg2rad(vec["heading"].astype(float))
        dx = vec["dist_m"].astype(float) * np.sin(theta)
        dy = vec["dist_m"].astype(float) * np.cos(theta)

        vec["lon2"] = vec["lon"].astype(float) + dx / m_per_deg_lon
        vec["lat2"] = vec["lat"].astype(float) + dy / m_per_deg_lat

        vec["path"] = vec.apply(
            lambda r: [[float(r["lon"]), float(r["lat"])], [float(r["lon2"]), float(r["lat2"])]],
            axis=1,
        )

        vector_layer = pdk.Layer(
            "PathLayer",
            vec,
            get_path="path",
            get_width=2,
            width_min_pixels=2,
            get_color="color",
        )

    def _label(r):
        ttk = r.get("TTK_s")
        if pd.notna(ttk):
            return f"{r['boat']} / {int(ttk)}s"
        return str(r["boat"])

    df["label"] = df.apply(_label, axis=1)

    labels = pdk.Layer(
        "TextLayer",
        df,
        get_position="[lon, lat]",
        get_text="label",
        get_size=11,
        get_color="color",
        get_pixel_offset=[0, 10],
    )

    layers = [base, labels]
    if vector_layer is not None:
        layers.insert(1, vector_layer)

    return layers


def build_trace_layer(trace_df: pd.DataFrame):
    if trace_df is None or trace_df.empty:
        return None
    return pdk.Layer(
        "PathLayer",
        trace_df,
        get_path="path",
        get_color="color",
        get_width=2,
        width_min_pixels=2,
    )


def build_cross_layers(cross_df: pd.DataFrame):
    if cross_df is None or cross_df.empty:
        return []
    points = pdk.Layer(
        "ScatterplotLayer",
        cross_df,
        get_position="[lon, lat]",
        get_radius=7,
        get_fill_color="color",
        pickable=True,
    )
    labels = pdk.Layer(
        "TextLayer",
        cross_df,
        get_position="[lon, lat]",
        get_text="label",
        get_size=11,
        get_color="color",
        get_pixel_offset=[0, 14],
    )
    return [points, labels]


def build_boundary_layer(boundary_df: pd.DataFrame | None):
    """
    Dessine la boundary (CourseLimit) si présente.
    On utilise un PathLayer fermé (simple et robuste).
    Attendu: colonnes lat/lon (seq optionnel).
    """
    if boundary_df is None or boundary_df.empty:
        return None

    df = boundary_df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return None

    if "seq" in df.columns:
        df = df.sort_values("seq")

    path = df[["lon", "lat"]].astype(float).values.tolist()
    if len(path) < 3:
        return None

    # fermer le polygone visuellement
    if path[0] != path[-1]:
        path = path + [path[0]]

    bdf = pd.DataFrame([{"path": path}])
    return pdk.Layer(
        "PathLayer",
        bdf,
        get_path="path",
        get_width=2,
        width_min_pixels=2,
        get_color=[40, 40, 40, 180],
    )


def build_deck(
    boats_df: pd.DataFrame,
    marks_df: pd.DataFrame,
    cross_df: pd.DataFrame | None = None,
    trace_df: pd.DataFrame | None = None,
    boundary_df: pd.DataFrame | None = None,
) -> pdk.Deck:

    lat_center = _safe_mean(pd.concat([boats_df["lat"], marks_df["lat"]], ignore_index=True))
    lon_center = _safe_mean(pd.concat([boats_df["lon"], marks_df["lon"]], ignore_index=True))

    layers = []

    # Boundary en fond (carto du haut)
    b_layer = build_boundary_layer(boundary_df)
    if b_layer is not None:
        layers.append(b_layer)

    sl_layer = build_startline_layer_dashed(marks_df)
    if sl_layer is not None:
        layers.append(sl_layer)

    trace = build_trace_layer(trace_df)
    if trace is not None:
        layers.append(trace)

    layers += build_marks_layers(marks_df)
    layers += build_boats_layers(boats_df)
    layers += build_cross_layers(cross_df)

    bearing = bearing_sl1_sl2(marks_df)

    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=13,
        bearing=bearing,
        pitch=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=CARTO_STYLE,
        tooltip={"text": "{boat}"},
    )

