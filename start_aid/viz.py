import math
import numpy as np
import pandas as pd
import pydeck as pdk

from .geo import (
    xy_to_ll,
    polygon_exterior_to_lonlat_path,
    dashed_paths_from_polygon,
    dashed_paths_from_linestring_xy,
    xy_path_to_lonlat_path,
    label_offset_ll,
    heading_to_unit_vector,
    rot,
)


def build_deck(ctx, geom, PI_xy, out):
    to_utm, to_wgs = ctx["to_utm"], ctx["to_wgs"]
    centroid_lat, centroid_lon = ctx["centroid_lat"], ctx["centroid_lon"]

    SL1_ll, SL2_ll, M1_ll = geom["SL1_ll"], geom["SL2_ll"], geom["M1_ll"]
    SL1_xy, SL2_xy, M1_xy = geom["SL1_xy"], geom["SL2_xy"], geom["M1_xy"]
    poly_BDY, poly_buffer = geom["poly_BDY"], geom["poly_buffer"]

    # Map orientation (validated): bearing follows TWD
    TWD = float(out.get("TWD", 0.0))
    bearing = (TWD % 360.0)

    # Polygons (white)
    poly_BDY_path_ll = polygon_exterior_to_lonlat_path(poly_BDY, to_wgs)

    buffer_dash_data = []
    if poly_buffer is not None:
        dash_segments_xy = dashed_paths_from_polygon(poly_buffer, dash_m=25.0, gap_m=15.0)
        buffer_dash_data = [{"path": xy_path_to_lonlat_path(seg, to_wgs)} for seg in dash_segments_xy]

    # Start line (pink)
    start_line_seg_ll = [[SL1_ll[1], SL1_ll[0]], [SL2_ll[1], SL2_ll[0]]]

    # Laylines (orange dashed)
    lay_dashes = []
    for lay in [out.get("lay_vis_SL1"), out.get("lay_vis_SL2")]:
        if lay is None:
            continue
        dashes_xy = dashed_paths_from_linestring_xy(lay, dash_m=40.0, gap_m=30.0)
        for seg_xy in dashes_xy:
            lay_dashes.append({"path": xy_path_to_lonlat_path(seg_xy, to_wgs)})

    # First legs red
    first_leg_paths = out.get("first_leg_paths", [])

    # Second legs colored (per-segment layer)
    second_leg_layers = []
    for seg in out.get("traj_second_segments", []):
        second_leg_layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": seg["path"]}],
                get_path="path",
                width_scale=6,
                width_min_pixels=3,
                get_color=seg["color"],
            )
        )

    # Wind vector (blue), reversed direction, length = 0.5*|SL1-SL2|
    startline_len_m = float(np.linalg.norm(SL2_xy - SL1_xy))
    wind_len_m = 0.5 * startline_len_m
    mid_sl_xy = (SL1_xy + SL2_xy) / 2.0
    anchor_xy = (M1_xy + mid_sl_xy) / 2.0

    wind_dir = -heading_to_unit_vector(TWD)
    wind_end_xy = anchor_xy + wind_dir * wind_len_m
    anchor_ll = xy_to_ll(to_wgs, anchor_xy[0], anchor_xy[1])
    wind_end_ll = xy_to_ll(to_wgs, wind_end_xy[0], wind_end_xy[1])

    arrow_head_len = max(10.0, 0.08 * wind_len_m)
    arrow_ang = math.radians(25.0)
    u = wind_dir / (np.linalg.norm(wind_dir) + 1e-12)
    back = -u
    leg1 = rot(back, +arrow_ang) * arrow_head_len
    leg2 = rot(back, -arrow_ang) * arrow_head_len
    ah1_xy = wind_end_xy + leg1
    ah2_xy = wind_end_xy + leg2
    ah1_ll = xy_to_ll(to_wgs, ah1_xy[0], ah1_xy[1])
    ah2_ll = xy_to_ll(to_wgs, ah2_xy[0], ah2_xy[1])

    wind_paths = [
        {"path": [[anchor_ll[1], anchor_ll[0]], [wind_end_ll[1], wind_end_ll[0]]]},
        {"path": [[wind_end_ll[1], wind_end_ll[0]], [ah1_ll[1], ah1_ll[0]]]},
        {"path": [[wind_end_ll[1], wind_end_ll[0]], [ah2_ll[1], ah2_ll[0]]]},
    ]

    # Points
    PI_ll = xy_to_ll(to_wgs, PI_xy[0], PI_xy[1])
    points = [
        {"name": "SL1", "lat": SL1_ll[0], "lon": SL1_ll[1], "kind": "mark"},
        {"name": "SL2", "lat": SL2_ll[0], "lon": SL2_ll[1], "kind": "mark"},
        {"name": "M1",  "lat": M1_ll[0],  "lon": M1_ll[1],  "kind": "mark"},
        {"name": "PI",  "lat": PI_ll[0],  "lon": PI_ll[1],  "kind": "calc"},
    ]

    if out.get("M_buffer_BDY_xy") is not None:
        p = out["M_buffer_BDY_xy"]
        lat, lon = xy_to_ll(to_wgs, p[0], p[1])
        points.append({"name": "M_buffer_BDY", "lat": lat, "lon": lon, "kind": "calc"})
    if out.get("M_LL_SL1_xy") is not None:
        p = out["M_LL_SL1_xy"]
        lat, lon = xy_to_ll(to_wgs, p[0], p[1])
        points.append({"name": "M_LL_SL1", "lat": lat, "lon": lon, "kind": "calc"})

    # New endpoints on start line
    for key, label in [("SL1_7m_xy", "SL1_7m"), ("SL2_7m_xy", "SL2_7m"), ("SP_xy", "SP")]:
        if out.get(key) is not None:
            p = out[key]
            lat, lon = xy_to_ll(to_wgs, p[0], p[1])
            points.append({"name": label, "lat": lat, "lon": lon, "kind": "calc"})

    # Labels offset (white)
    label_e, label_n = 10.0, -14.0
    labels = []
    for txt, ll in [("SL1", SL1_ll), ("SL2", SL2_ll), ("M1", M1_ll)]:
        lat2, lon2 = label_offset_ll(to_utm, to_wgs, ll[0], ll[1], label_e, label_n)
        labels.append({"text": txt, "lat": lat2, "lon": lon2})

    layers = []

    layers.append(
        pdk.Layer(
            "PathLayer",
            data=[{"path": poly_BDY_path_ll}],
            get_path="path",
            width_scale=6,
            width_min_pixels=2,
            get_color=[255, 255, 255],
        )
    )

    if buffer_dash_data:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=buffer_dash_data,
                get_path="path",
                width_scale=6,
                width_min_pixels=2,
                get_color=[255, 255, 255],
            )
        )

    layers.append(
        pdk.Layer(
            "PathLayer",
            data=[{"path": start_line_seg_ll}],
            get_path="path",
            width_scale=6,
            width_min_pixels=4,
            get_color=[255, 105, 180],
        )
    )

    if lay_dashes:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=lay_dashes,
                get_path="path",
                width_scale=4,
                width_min_pixels=2,
                get_color=[255, 165, 0],
            )
        )

    if first_leg_paths:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=first_leg_paths,
                get_path="path",
                width_scale=6,
                width_min_pixels=3,
                get_color=[220, 0, 0],
            )
        )

    layers.extend(second_leg_layers)

    layers.append(
        pdk.Layer(
            "PathLayer",
            data=wind_paths,
            get_path="path",
            width_scale=6,
            width_min_pixels=3,
            get_color=[0, 90, 255],
        )
    )
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=[{"text": f"TWD {TWD:.0f}°", "lat": anchor_ll[0], "lon": anchor_ll[1]}],
            get_position="[lon, lat]",
            get_text="text",
            get_size=12,
            get_alignment_baseline="'top'",
            get_color=[0, 90, 255],
        )
    )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame(points),
            get_position="[lon, lat]",
            get_radius=7,
            pickable=True,
            get_fill_color="kind == 'mark' ? [255, 220, 0] : [255, 255, 255]",
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
        )
    )

    layers.append(
        pdk.Layer(
            "TextLayer",
            data=pd.DataFrame(labels),
            get_position="[lon, lat]",
            get_text="text",
            get_size=14,
            get_alignment_baseline="'top'",
            get_color=[255, 255, 255],
        )
    )

    view_state = pdk.ViewState(
        latitude=centroid_lat,
        longitude=centroid_lon,
        zoom=14,
        pitch=0,
        bearing=bearing,
    )

    return pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"})
