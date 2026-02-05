from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from shapely.affinity import rotate as shp_rotate

from .geo import make_context_from_boundary, to_xy_marks_and_polys, marks_ll_to_xy, xy_to_ll
from .polars import list_polar_files, load_polar_interpolator, polar_twa_candidates
from .model import compute_first_dw_from_M1  # <-- NEW (multi routes)
from .viz import build_deck_routeur


# --------------------
# Rotation helpers
# --------------------
def _rot_xy_about(p_xy: np.ndarray, origin_xy: np.ndarray, ang_rad_ccw: float) -> np.ndarray:
    """Rotate point around origin by ang_rad_ccw (CCW)."""
    v = p_xy - origin_xy
    c, s = float(np.cos(ang_rad_ccw)), float(np.sin(ang_rad_ccw))
    vr = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)
    return origin_xy + vr


def _rotate_marks_xy(marks_xy: dict, origin_xy: np.ndarray, ang_rad_ccw: float) -> dict:
    return {k: _rot_xy_about(np.array(v, dtype=float), origin_xy, ang_rad_ccw) for k, v in marks_xy.items()}


# --------------------
# DF -> dict helpers
# --------------------
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

        # Certains XML stockent en 1e-7
        if abs(lat) > 90 or abs(lon) > 180:
            lat *= 1e-7
            lon *= 1e-7

        out[mark] = (lat, lon)

    return out


@st.cache_resource(show_spinner=False)
def _cached_polar_interpolator(abs_path_str: str):
    return load_polar_interpolator(abs_path_str)


# --------------------
# Color palette for routes
# --------------------
ROUTE_COLORS = {
    "A":  [0, 255, 0],        # green
    "B":  [255, 0, 255],      # magenta
    "A'": [255, 165, 0],      # orange
    "B'": [0, 255, 255],      # cyan
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

    # WW
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
        label = r.get("label", "?")
        color = r.get("color", [255, 255, 255])
        c = _rgb_to_css(color)

        rows.append(
            "<tr>"
            f"<td style='padding:4px 8px;'><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};border:1px solid #111;'></span></td>"
            f"<td style='padding:4px 8px;font-weight:700;color:{c};'>{label}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{float(r.get('time_total_s', float('nan'))):.1f}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{float(r.get('dist_total_m', float('nan'))):.1f}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{int(r.get('n_gybes', 0))}</td>"
            f"<td style='padding:4px 8px;text-align:right;'>{float(r.get('TWA_to_dest_start_deg', float('nan'))):.1f}</td>"
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

    # --- Widgets
    st.subheader("Vent")
    TWS_ref_kmh = st.number_input("TWS (km/h)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
    TWD = st.number_input("TWD (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

    st.subheader("Boundary")
    size_buffer_BDY = st.number_input("Buffer boundary (m)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)

    st.subheader("Manœuvres")
    gybe_loss_s = st.number_input("Perte gybe (s)", min_value=0.0, max_value=30.0, value=5.0, step=1.0)

    st.subheader("Polaires")
    polars_dir = Path(__file__).resolve().parent / "polars"
    polar_files = list_polar_files(polars_dir)
    if not polar_files:
        st.warning(f"Aucun fichier de polaires trouvé dans: {polars_dir}")
        return None, None

    polar_name = st.selectbox("Fichier de polaires", options=polar_files, index=0)
    polar_path = polars_dir / polar_name
    polar_interp = _cached_polar_interpolator(str(polar_path.resolve()))

    debug = st.checkbox("Debug (events)", value=False)

    # --- Context + geometry (repère monde)
    ctx = make_context_from_boundary(boundary_latlon)

    marks_for_geom = {
        "SL1": marks_ll.get("SL1", marks_ll["M1"]),
        "SL2": marks_ll.get("SL2", marks_ll["M1"]),
        "M1": marks_ll["M1"],
    }

    geom = to_xy_marks_and_polys(ctx, marks_for_geom, boundary_latlon, float(size_buffer_BDY))
    marks_xy = marks_ll_to_xy(ctx, marks_ll)

    # ============================================================
    # REPÈRE VENT : rotation CW = (TWD - 180)  (ta règle validée)
    # => en maths: CCW = +angle_rad
    # ============================================================
    angle_deg_cw = (float(TWD) - 180.0) % 360.0
    angle_rad_ccw = np.deg2rad(angle_deg_cw)

    origin_xy = np.array([geom["poly_BDY"].centroid.x, geom["poly_BDY"].centroid.y], dtype=float)

    marks_xy_rot = _rotate_marks_xy(marks_xy, origin_xy, angle_rad_ccw)

    geom_rot = dict(geom)
    geom_rot["poly_BDY"] = shp_rotate(
        geom["poly_BDY"],
        float(np.rad2deg(angle_rad_ccw)),
        origin=(origin_xy[0], origin_xy[1]),
    )
    if geom.get("poly_buffer") is not None:
        geom_rot["poly_buffer"] = shp_rotate(
            geom["poly_buffer"],
            float(np.rad2deg(angle_rad_ccw)),
            origin=(origin_xy[0], origin_xy[1]),
        )
    else:
        geom_rot["poly_buffer"] = None

    for k in ("SL1_xy", "SL2_xy", "M1_xy"):
        if k in geom_rot and geom_rot[k] is not None:
            geom_rot[k] = _rot_xy_about(np.array(geom_rot[k], dtype=float), origin_xy, angle_rad_ccw)

    # --- Compute FIRST DW (multi routes) in repère vent normalisé
    twa_cands = polar_twa_candidates(polar_interp)

    out = compute_first_dw_from_M1(
        ctx=ctx,
        geom=geom_rot,
        marks_xy=marks_xy_rot,
        TWD=180.0,  # vent normalisé dans ce repère
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
    )

    routes = out.get("routes", [])
    if not routes:
        st.error("Aucune trajectoire renvoyée par le modèle.")
        return None, None

    # ============================================================
    # Dé-rotation pour AFFICHAGE (repère monde) + couleurs
    # ============================================================
    for r in routes:
        label = r.get("label", "?")
        r["color"] = ROUTE_COLORS.get(label, [255, 255, 255])

        # path XY (rot) -> XY world -> LL
        path_xy_rot = [np.array(p, dtype=float) for p in r.get("route_path_xy", [])]
        path_xy = [_rot_xy_about(p, origin_xy, -angle_rad_ccw) for p in path_xy_rot]
        r["route_path_ll"] = []
        for p in path_xy:
            lat, lon = xy_to_ll(ctx["to_wgs"], float(p[0]), float(p[1]))
            r["route_path_ll"].append([lon, lat])

        # gybe points
        gy_xy_rot = [np.array(p, dtype=float) for p in (r.get("gybe_points_xy", []) or [])]
        gy_xy = [_rot_xy_about(p, origin_xy, -angle_rad_ccw) for p in gy_xy_rot]
        r["gybe_points_ll"] = []
        for p in gy_xy:
            lat, lon = xy_to_ll(ctx["to_wgs"], float(p[0]), float(p[1]))
            r["gybe_points_ll"].append((lat, lon))

        # put back real TWD for viz
        r["TWD"] = float(TWD)

    # Target MLG LL (world)
    MLG_xy_world = 0.5 * (marks_xy["LG1"] + marks_xy["LG2"])
    lat, lon = xy_to_ll(ctx["to_wgs"], float(MLG_xy_world[0]), float(MLG_xy_world[1]))
    out["target_MLG_ll"] = (lat, lon)
    out["TWD"] = float(TWD)

    # --- Display: VMG + gate biases
    st.markdown(
        f"**VMG DW**: TWA={out['TWA_vmg_DW_deg']:.0f}° — BSP={out['BSP_vmg_DW_kmph']:.1f} km/h — VMG={out['VMG_vmg_DW_kmph']:.1f} km/h  \n"
        f"**VMG UW**: TWA={out['TWA_vmg_UW_deg']:.0f}° — BSP={out['BSP_vmg_UW_kmph']:.1f} km/h — VMG={out['VMG_vmg_UW_kmph']:.1f} km/h"
    )

    # Bias display (sign not finalized)
    lw_bias = float(out.get("LW_gate_bias_m", float("nan")))
    ww_bias = float(out.get("WW_gate_bias_m", float("nan")))

    lw_txt = _gate_side_label("LW", lw_bias)
    ww_txt = _gate_side_label("WW", ww_bias)

    st.markdown(
        f"**LW gate bias**: {lw_bias:.1f} m — *{lw_txt}*  \n"
        f"**WW gate bias**: {ww_bias:.1f} m — *{ww_txt}*"
    )


    # Results table with colors
    results_html = _build_results_html(routes)
    st.markdown(results_html, unsafe_allow_html=True)

    # Optional debug
    if debug:
        # concat events from all routes
        ev_rows = []
        for r in routes:
            for e in (r.get("events", []) or []):
                ee = dict(e)
                ee["traj"] = r.get("label", "?")
                ev_rows.append(ee)
        if ev_rows:
            st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)
        else:
            st.caption("Aucun event.")

    # --- Map (boundary in world + routes in world)
    deck = build_deck_routeur(
        ctx=ctx,
        geom=geom,              # world polygons
        marks_ll=marks_ll,
        marks_xy=marks_xy,      # world marks
        route_out={**out, "routes": routes},  # pass routes list to viz
    )

    return deck, {"out": out, "routes": routes}
