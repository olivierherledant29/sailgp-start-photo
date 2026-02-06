from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from shapely.affinity import rotate as shp_rotate

from .geo import make_context_from_boundary, to_xy_marks_and_polys, marks_ll_to_xy, xy_to_ll
from .polars import list_polar_files, load_polar_interpolator, polar_twa_candidates
from .model import compute_first_dw_from_M1
from .viz import build_deck_routeur


def _rot_xy_about(p_xy: np.ndarray, origin_xy: np.ndarray, ang_rad_ccw: float) -> np.ndarray:
    v = p_xy - origin_xy
    c, s = float(np.cos(ang_rad_ccw)), float(np.sin(ang_rad_ccw))
    vr = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)
    return origin_xy + vr


def _rot_vec(v_xy: np.ndarray, ang_rad_ccw: float) -> np.ndarray:
    c, s = float(np.cos(ang_rad_ccw)), float(np.sin(ang_rad_ccw))
    return np.array([c * v_xy[0] - s * v_xy[1], s * v_xy[0] + c * v_xy[1]], dtype=float)


def _rotate_marks_xy(marks_xy: dict, origin_xy: np.ndarray, ang_rad_ccw: float) -> dict:
    return {k: _rot_xy_about(np.array(v, dtype=float), origin_xy, ang_rad_ccw) for k, v in marks_xy.items()}


def boundary_df_to_latlon(boundary_df: pd.DataFrame):
    if boundary_df is None or boundary_df.empty:
        return []
    df = boundary_df.dropna(subset=["lat", "lon"]).copy()
    return [(float(r.lat), float(r.lon)) for _, r in df.iterrows()]


def marks_df_to_marks_ll(marks_df: pd.DataFrame):
    out = {}
    if marks_df is None or marks_df.empty:
        return out

    df = marks_df.dropna(subset=["mark", "lat", "lon"]).copy()
    for _, r in df.iterrows():
        mark = str(r["mark"])
        lat = float(r["lat"])
        lon = float(r["lon"])
        if abs(lat) > 90 or abs(lon) > 180:
            lat *= 1e-7
            lon *= 1e-7
        out[mark] = (lat, lon)
    return out


@st.cache_resource(show_spinner=False)
def _cached_polar_interpolator(abs_path_str: str):
    return load_polar_interpolator(abs_path_str)


ROUTE_COLORS = {
    "A":  [0, 255, 0],
    "B":  [255, 0, 255],
    "A'": [255, 165, 0],
    "B'": [0, 255, 255],
}


def _gate_side_label(gate: str, bias_m: float) -> str:
    if not np.isfinite(bias_m):
        return f"porte {gate} ?"

    if gate == "LW":
        if bias_m < -1.0:
            return "L gauche en descendant"
        if bias_m > 1.0:
            return "R droite en descendant"
        return "porte LW neutre"

    if bias_m < -1.0:
        return "L gauche"
    if bias_m > 1.0:
        return "R droite"
    return "porte WW neutre"


def _rgb_to_css(rgb: list[int]) -> str:
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def _build_results_html(routes: list[dict]) -> str:
    rows = []
    for r in routes:
        name = r.get("traj_name") or r.get("traj", "?")
        color = r.get("color", [255, 255, 255])
        c = _rgb_to_css(color)
        rows.append(
            "<tr>"
            f"<td style='padding:4px 8px;'><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};border:1px solid #111;'></span></td>"
            f"<td style='padding:4px 8px;font-weight:700;color:{c};'>{name}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{float(r.get('time_total_s', float('nan'))):.0f}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{float(r.get('dist_total_m', float('nan'))):.0f}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{int(r.get('n_gybes', 0))}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{float(r.get('TWA_to_dest_start_deg', float('nan'))):.0f}</td>"
            "</tr>"
        )

    return f"""
    <div style="font-size: 13px;">
      <table style="width:100%; border-collapse: collapse;">
        <thead>
          <tr>
            <th style="text-align:left; border-bottom:1px solid #666; width:24px;"></th>
            <th style="text-align:left; border-bottom:1px solid #666;">traj</th>
            <th style="text-align:right; border-bottom:1px solid #666;">time (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">dist (m)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">gybes</th>
            <th style="text-align:right; border-bottom:1px solid #666;">TWA_start (°)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def _guess_twd_from_xml_filename() -> float:
    """
    TWD_base par défaut = 0 ou 3 premiers chars du nom du XML (si digits) stocké dans boundary_shared.
    Ex: "260116_132758R3A 260 .75_race" -> 260
    """
    name = st.session_state.get("boundary_xml_name", None)
    if not isinstance(name, str) or len(name) < 3:
        return 0.0
    prefix = name.strip()[:3]
    if prefix.isdigit():
        x = float(int(prefix))
        if 0.0 <= x <= 360.0:
            return x
    return 0.0


def render_routeur_simplifie(boundary_df: pd.DataFrame, marks_df: pd.DataFrame):
    st.markdown(
        "<div style='padding:8px;border-radius:8px;border:1px solid #1565C0;background:#E3F2FD;'>"
        "<b>Routeur simplifié SailGP</b> — XML-only</div>",
        unsafe_allow_html=True,
    )

    boundary_latlon = boundary_df_to_latlon(boundary_df)
    marks_ll_all = marks_df_to_marks_ll(marks_df)

    if not boundary_latlon:
        st.info("Routeur : boundary non chargée (XML manquant).")
        return None, None

    marks_of_interest = ["SL1", "SL2", "M1", "WG1", "WG2", "LG1", "LG2", "FL1", "FL2"]
    marks_ll = {k: marks_ll_all[k] for k in marks_of_interest if k in marks_ll_all}

    missing = [m for m in ["M1", "LG1", "LG2"] if m not in marks_ll]
    if missing:
        st.error(f"Marques manquantes dans le XML: {', '.join(missing)} (requises pour 'first DW').")
        return None, None

    # Sidebar: offset only
    with st.sidebar:
        offset_TWD = st.slider(
            "offset_TWD (°)",
            min_value=-30,
            max_value=30,
            value=int(st.session_state.get("offset_TWD", 0)),
            step=2,
        )
        st.session_state["offset_TWD"] = int(offset_TWD)

    # Widgets (integers)
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.6, 1.1, 1.1])

    with c1:
        TWS_ref_kmh = st.number_input("TWS (km/h)", min_value=1, max_value=50, value=20, step=1, format="%d")

    with c2:
        if "TWD_base" not in st.session_state:
            st.session_state["TWD_base"] = float(_guess_twd_from_xml_filename())
        TWD_base = st.number_input("TWD_base (°)", min_value=0, max_value=360, value=int(st.session_state["TWD_base"]), step=1, format="%d")
        st.session_state["TWD_base"] = float(TWD_base)

    TWD = (float(TWD_base) + float(offset_TWD)) % 360.0

    with c3:
        polars_dir = Path(__file__).resolve().parent / "polars"
        fallback_dir = Path(__file__).resolve().parents[1] / "start_aid" / "polars"
        polar_files = list_polar_files(polars_dir)
        if not polar_files and fallback_dir.exists():
            polars_dir = fallback_dir
            polar_files = list_polar_files(polars_dir)
        if not polar_files:
            st.warning(f"Aucun fichier de polaires trouvé dans: {polars_dir}")
            return None, None
        polar_name = st.selectbox("Fichier de polaires", options=polar_files, index=0)
        polar_path = polars_dir / polar_name
        polar_interp = _cached_polar_interpolator(str(polar_path.resolve()))

    with c4:
        size_buffer_BDY = st.number_input("Buffer boundary (m)", min_value=0, max_value=500, value=20, step=1, format="%d")

    with c5:
        gybe_loss_s = st.number_input("Perte gybe (s)", min_value=0, max_value=30, value=5, step=1, format="%d")

    c6, c7, c8 = st.columns([1.1, 1.0, 2.9])
    with c6:
        d_rounding_mark = st.number_input("d_rounding_mark (m)", min_value=0, max_value=50, value=10, step=1, format="%d")
    with c7:
        debug = st.checkbox("Debug (events)", value=False)
    with c8:
        st.markdown(f"**TWD_base={int(TWD_base)}°** — **offset_TWD={int(offset_TWD):+d}°** — **TWD appliqué={int(TWD)}°**")

    # World geometry
    ctx = make_context_from_boundary(boundary_latlon)
    marks_for_geom = {"SL1": marks_ll.get("SL1", marks_ll["M1"]), "SL2": marks_ll.get("SL2", marks_ll["M1"]), "M1": marks_ll["M1"]}
    geom = to_xy_marks_and_polys(ctx, marks_for_geom, boundary_latlon, float(size_buffer_BDY))
    marks_xy_world = marks_ll_to_xy(ctx, marks_ll)

    # Wind-up rotation CW=(TWD-180)
    angle_deg_cw = (float(TWD) - 180.0) % 360.0
    angle_rad_ccw = np.deg2rad(angle_deg_cw)

    origin_xy = np.array([geom["poly_BDY"].centroid.x, geom["poly_BDY"].centroid.y], dtype=float)
    marks_xy_rot = _rotate_marks_xy(marks_xy_world, origin_xy, angle_rad_ccw)

    geom_rot = dict(geom)
    geom_rot["poly_BDY"] = shp_rotate(geom["poly_BDY"], float(np.rad2deg(angle_rad_ccw)), origin=(origin_xy[0], origin_xy[1]))
    geom_rot["poly_buffer"] = shp_rotate(geom["poly_buffer"], float(np.rad2deg(angle_rad_ccw)), origin=(origin_xy[0], origin_xy[1])) if geom.get("poly_buffer") is not None else None

    # M1 rounding: if currently on right, we invert by using +left_world (not -)
    if "M1" in marks_xy_world:
        left_wind = np.array([-1.0, 0.0], dtype=float)
        left_world = -1 * _rot_vec(left_wind, -angle_rad_ccw)
        # ✅ left_world (no extra inversion here). If it still goes right, flip here.
        M1_round_world = np.array(marks_xy_world["M1"], dtype=float) + float(d_rounding_mark) * left_world
        M1_round_rot = _rot_xy_about(M1_round_world, origin_xy, angle_rad_ccw)
        marks_xy_rot["M1"] = M1_round_rot

    twa_cands = polar_twa_candidates(polar_interp)

    out = compute_first_dw_from_M1(
        ctx=ctx,
        geom=geom_rot,
        marks_xy=marks_xy_rot,
        TWD=180.0,
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
    )

    routes = out.get("routes", [])
    if not routes:
        st.error("Aucune trajectoire renvoyée par le modèle.")
        return None, None

    # de-rotate for display + colors
    for r in routes:
        traj = r.get("traj", r.get("label", "?"))
        r["traj"] = traj
        r["color"] = ROUTE_COLORS.get(traj, [255, 255, 255])

        path_xy_rot = [np.array(p, dtype=float) for p in r.get("route_path_xy", [])]
        path_xy_world = [_rot_xy_about(p, origin_xy, -angle_rad_ccw) for p in path_xy_rot]
        r["route_path_ll"] = []
        for p in path_xy_world:
            lat, lon = xy_to_ll(ctx["to_wgs"], float(p[0]), float(p[1]))
            r["route_path_ll"].append([lon, lat])

        gy_xy_rot = [np.array(p, dtype=float) for p in (r.get("gybe_points_xy", []) or [])]
        gy_xy_world = [_rot_xy_about(p, origin_xy, -angle_rad_ccw) for p in gy_xy_rot]
        r["gybe_points_ll"] = []
        for p in gy_xy_world:
            lat, lon = xy_to_ll(ctx["to_wgs"], float(p[0]), float(p[1]))
            r["gybe_points_ll"].append((lat, lon))

    out["TWD"] = float(TWD)
    out["TWD_base"] = float(TWD_base)
    out["offset_TWD"] = float(offset_TWD)
    out["routes"] = routes
    out["results_html"] = _build_results_html(routes)

    # Lines above map
    st.markdown(f"**Rappel TWD** — base={int(TWD_base)}° ; offset={int(offset_TWD):+d}° ; appliqué={int(TWD)}°")
    st.markdown(
        f"**VMG DW**: TWA={out['TWA_vmg_DW_deg']:.0f}° — BSP={out['BSP_vmg_DW_kmph']:.1f} km/h — VMG={out['VMG_vmg_DW_kmph']:.1f} km/h  \n"
        f"**VMG UW**: TWA={out['TWA_vmg_UW_deg']:.0f}° — BSP={out['BSP_vmg_UW_kmph']:.1f} km/h — VMG={out['VMG_vmg_UW_kmph']:.1f} km/h"
    )

    lw_bias = float(out.get("LW_gate_bias_m", float("nan")))
    ww_bias = float(out.get("WW_gate_bias_m", float("nan")))
    st.markdown(
        f"**LW gate bias**: {lw_bias:.1f} m — *{_gate_side_label('LW', lw_bias)}*  \n"
        f"**WW gate bias**: {ww_bias:.1f} m — *{_gate_side_label('WW', ww_bias)}*"
    )

    deck = build_deck_routeur(
        ctx=ctx,
        geom=geom,
        marks_ll=marks_ll,
        marks_xy=marks_xy_world,
        route_out=out,
    )

    if debug:
        ev_rows = []
        for r in routes:
            for e in (r.get("events", []) or []):
                ee = dict(e)
                ee["traj"] = r.get("traj_name") or r.get("traj", "?")
                ev_rows.append(ee)
        if ev_rows:
            st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)

    return deck, out
