from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pydeck as pdk

CARTO_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

try:
    from shapely.geometry import Polygon, LineString
except Exception:
    Polygon = None
    LineString = None


def build_boundary_buffer_layer(boundary_buffer_df: pd.DataFrame | None):
    """
    (Conservée) Dessine un buffer boundary "plein" si on te le demande ailleurs.
    Ici on garde ton code tel quel.
    """
    if boundary_buffer_df is None or boundary_buffer_df.empty:
        return None

    df = boundary_buffer_df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return None

    if "seq" in df.columns:
        df = df.sort_values("seq")

    path = df[["lon", "lat"]].astype(float).values.tolist()
    if len(path) < 3:
        return None
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
    """
    Fabrique une alternance de segments (un segment sur deux) entre 2 points.
    p1/p2 sont [lon,lat] (ou n'importe quel plan), on garde ta logique.
    """
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
        ttk = pd.to_numeric(r.get("TTK_s"), errors="coerce")
        if np.isfinite(ttk):
            return f"{r['boat']} / {int(round(float(ttk)))} TTK"
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


def _build_boundary_buffer_dashed_layer(boundary_df: pd.DataFrame | None, buffer_m: float | None):
    """
    (Conservée) Buffer intérieur (négatif) tracé en pointillé.
    Nécessite shapely; sinon, on skip sans casser l’app.
    """
    if buffer_m is None:
        return None
    try:
        buffer_m = float(buffer_m)
    except Exception:
        return None
    if not np.isfinite(buffer_m) or buffer_m <= 0:
        return None
    if boundary_df is None or boundary_df.empty:
        return None
    if Polygon is None:
        return None

    df = boundary_df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return None
    if "seq" in df.columns:
        df = df.sort_values("seq")

    lat0 = float(df["lat"].astype(float).mean())
    lon0 = float(df["lon"].astype(float).mean())
    m_per_deg_lat, m_per_deg_lon = _meters_per_deg(lat0)

    xs = (df["lon"].astype(float) - lon0) * m_per_deg_lon
    ys = (df["lat"].astype(float) - lat0) * m_per_deg_lat
    coords_xy = list(zip(xs.to_numpy(float), ys.to_numpy(float)))

    poly = Polygon(coords_xy)
    if (not poly.is_valid) or poly.area <= 0:
        poly = poly.buffer(0)
    if (not poly.is_valid) or poly.area <= 0:
        return None

    poly_buf = poly.buffer(-buffer_m)
    if poly_buf.is_empty:
        return None

    if poly_buf.geom_type == "MultiPolygon":
        poly_buf = max(list(poly_buf.geoms), key=lambda p: p.area)

    ring = list(poly_buf.exterior.coords)
    if len(ring) < 4:
        return None

    path_ll = []
    for x, y in ring:
        lon = lon0 + (x / m_per_deg_lon)
        lat = lat0 + (y / m_per_deg_lat)
        path_ll.append([float(lon), float(lat)])

    segs = []
    for i in range(len(path_ll) - 1):
        segs.extend(_make_dashed_segments(path_ll[i], path_ll[i + 1], n_segments=10))

    if not segs:
        return None

    sdf = pd.DataFrame([{"path": s} for s in segs])
    return pdk.Layer(
        "PathLayer",
        sdf,
        get_path="path",
        get_width=2,
        width_min_pixels=2,
        get_color=[40, 40, 40, 180],
    )


def _build_boundary_buffer_dashed_layer_from_df(boundary_buffer_df: pd.DataFrame | None):
    """
    NEW (robuste): dessine le buffer à partir du DF (déjà calculé côté app_start_photo3),
    donc sans shapely et sans dépendre de boundary_buffer_m.
    """
    if boundary_buffer_df is None or boundary_buffer_df.empty:
        return None

    df = boundary_buffer_df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return None
    if "seq" in df.columns:
        df = df.sort_values("seq")

    path = df[["lon", "lat"]].astype(float).values.tolist()
    if len(path) < 3:
        return None
    if path[0] != path[-1]:
        path = path + [path[0]]

    segs = []
    for i in range(len(path) - 1):
        segs.extend(_make_dashed_segments(path[i], path[i + 1], n_segments=10))

    if not segs:
        return None

    sdf = pd.DataFrame([{"path": s} for s in segs])
    return pdk.Layer(
        "PathLayer",
        sdf,
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
    boundary_buffer_df: pd.DataFrame | None = None,  # NEW: buffer déjà calculé (préféré)
    boundary_buffer_m: float | None = None,          # NEW: fallback shapely si pas de df
    tts_s: float | None = None,
) -> pdk.Deck:

    lat_center = _safe_mean(pd.concat([boats_df["lat"], marks_df["lat"]], ignore_index=True))
    lon_center = _safe_mean(pd.concat([boats_df["lon"], marks_df["lon"]], ignore_index=True))

    layers = []

    # Boundary en fond
    b_layer = build_boundary_layer(boundary_df)
    if b_layer is not None:
        layers.append(b_layer)

    # Buffer boundary (pointillé)
    buf_layer = None
    if boundary_buffer_df is not None and not boundary_buffer_df.empty:
        buf_layer = _build_boundary_buffer_dashed_layer_from_df(boundary_buffer_df)
    else:
        # fallback shapely (si tu passes boundary_buffer_m au lieu de boundary_buffer_df)
        buf_layer = _build_boundary_buffer_dashed_layer(boundary_df, boundary_buffer_m)

    if buf_layer is not None:
        layers.append(buf_layer)

    sl_layer = build_startline_layer_dashed(marks_df)
    if sl_layer is not None:
        layers.append(sl_layer)

    trace = build_trace_layer(trace_df)
    if trace is not None:
        layers.append(trace)

    layers += build_marks_layers(marks_df)
    layers += build_boats_layers(boats_df)
    layers += build_cross_layers(cross_df)

    # TTS badge (bas-gauche)
    if tts_s is not None and np.isfinite(tts_s):
        tts_txt = f"{int(round(float(tts_s)))} TTS"
        tts_df = pd.DataFrame([{
            "lat": lat_center,
            "lon": lon_center,
            "text": tts_txt,
        }])
        tts_layer = pdk.Layer(
            "TextLayer",
            tts_df,
            get_position="[lon, lat]",
            get_text="text",
            get_size=22,
            get_color=[20, 20, 20, 230],
            get_pixel_offset=[-260, 220],
            get_alignment_baseline="'top'",
        )
        layers.append(tts_layer)

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
