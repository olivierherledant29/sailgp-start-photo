from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from shapely.affinity import rotate as shp_rotate

from .geo import make_context_from_boundary, to_xy_marks_and_polys, marks_ll_to_xy, xy_to_ll
from .polars import list_polar_files, load_polar_interpolator, polar_twa_candidates
from .model import (
    compute_gate_biases,
    compute_route_group,
    wrap180,
    twa_from_twd_hdg,
    bearing_deg_xy,
)
from .viz import build_deck_routeur


def _rot_xy_about(p_xy: np.ndarray, origin_xy: np.ndarray, ang_rad_ccw: float) -> np.ndarray:
    v = p_xy - origin_xy
    c, s = float(np.cos(ang_rad_ccw)), float(np.sin(ang_rad_ccw))
    vr = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)
    return origin_xy + vr


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


# --- Colors
GROUP_BASE_COLORS = {
    "FIRST_DW_M1": [0, 255, 0],
    "FULL_UW_LG2": [0, 200, 255],
    "FULL_UW_LG1": [255, 180, 0],
    "FULL_DW_WG2": [180, 0, 255],
    "FULL_DW_WG1": [255, 80, 80],
}
TRAJ_SHADE = {"A": 1.00, "B": 0.85, "A'": 0.70, "B'": 0.55}


def _color_for(group_id: str, traj: str) -> list[int]:
    traj = str(traj).strip()
    if group_id == "FIRST_DW_M1":
        return {
            "A": [0, 255, 0],        # vert
            "B": [255, 0, 0],        # rouge
            "A'": [0, 200, 255],     # cyan
            "B'": [255, 255, 0],     # jaune
        }.get(traj, [255, 255, 255])

    base = GROUP_BASE_COLORS.get(group_id, [255, 255, 255])
    k = TRAJ_SHADE.get(traj, 1.0)
    return [int(base[0] * k), int(base[1] * k), int(base[2] * k)]


def _guess_twd_from_xml_filename() -> float:
    name = st.session_state.get("boundary_xml_name", None)
    if not isinstance(name, str) or len(name) < 3:
        return 0.0
    prefix = name.strip()[:3]
    if prefix.isdigit():
        x = float(int(prefix))
        if 0.0 <= x <= 360.0:
            return x
    return 0.0


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


# =========================================================
# Start-line overlay helpers
# =========================================================
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0], dtype=float)
    return v / n


def _point_along(a: np.ndarray, b: np.ndarray, dist_from_a: float) -> np.ndarray:
    u = _unit(b - a)
    return a + float(dist_from_a) * u


def _kmph_to_mps(v_kmph: float) -> float:
    return float(v_kmph) / 3.6


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

    required = ["M1", "LG1", "LG2", "WG1", "WG2"]
    missing = [m for m in required if m not in marks_ll]
    if missing:
        st.error(f"Marques manquantes dans le XML: {', '.join(missing)}")
        return None, None

    # Sidebar: offsets only
    with st.sidebar:
        offset_TWD = st.slider(
            "offset_TWD (°)",
            min_value=-30,
            max_value=30,
            value=int(st.session_state.get("offset_TWD", 0)),
            step=2,
        )
        st.session_state["offset_TWD"] = int(offset_TWD)

        # ✅ NEW: offset_TWS (km/h) in [-10,+10], but clamp min so TWS doesn't go below 0
        # We'll set actual slider min later once we know TWS_base; keep placeholder in session for now.
        if "offset_TWS" not in st.session_state:
            st.session_state["offset_TWS"] = 0

    # Widgets top row
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.6, 1.1])
    with c1:
        # base only
        TWS_base = st.number_input("TWS_base (km/h)", min_value=0, max_value=50, value=20, step=1, format="%d")
    with c2:
        if "TWD_base" not in st.session_state:
            st.session_state["TWD_base"] = float(_guess_twd_from_xml_filename())
        TWD_base = st.number_input(
            "TWD_base (°)",
            min_value=0,
            max_value=360,
            value=int(st.session_state["TWD_base"]),
            step=1,
            format="%d",
        )
        st.session_state["TWD_base"] = float(TWD_base)

    # now we can build the TWS offset slider with correct minimum
    with st.sidebar:
        min_off = int(max(-10, -int(TWS_base)))  # if TWS_base<10, min offset stops at -TWS_base
        offset_TWS = st.slider(
            "offset_TWS (km/h)",
            min_value=min_off,
            max_value=10,
            value=int(st.session_state.get("offset_TWS", 0)),
            step=1,
        )
        st.session_state["offset_TWS"] = int(offset_TWS)

    TWD = (float(TWD_base) + float(offset_TWD)) % 360.0
    TWS_ref_kmh = float(max(0.0, float(TWS_base) + float(offset_TWS)))

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

    c6, c7 = st.columns([1.1, 3.9])
    with c6:
        d_rounding_mark = st.number_input("d_rounding_mark (m)", min_value=0, max_value=50, value=10, step=1, format="%d")
    with c7:
        st.markdown(
            f"**TWS_base={int(TWS_base)}** — **offset_TWS={int(offset_TWS):+d}** — **TWS appliqué={int(TWS_ref_kmh)}**  \n"
            f"**TWD_base={int(TWD_base)}°** — **offset_TWD={int(offset_TWD):+d}°** — **TWD appliqué={int(TWD)}°**"
        )

    gybe_loss_s = 3.0

    # World geometry
    ctx = make_context_from_boundary(boundary_latlon)
    marks_for_geom = {
        "SL1": marks_ll.get("SL1", marks_ll["M1"]),
        "SL2": marks_ll.get("SL2", marks_ll["M1"]),
        "M1": marks_ll["M1"],
    }
    geom = to_xy_marks_and_polys(ctx, marks_for_geom, boundary_latlon, float(size_buffer_BDY))
    marks_xy_world = marks_ll_to_xy(ctx, marks_ll)

    # Rotate into calc frame (wind from top => TWD_calc=180)
    angle_deg_cw = (float(TWD) - 180.0) % 360.0
    angle_rad_ccw = np.deg2rad(angle_deg_cw)

    origin_xy = np.array([geom["poly_BDY"].centroid.x, geom["poly_BDY"].centroid.y], dtype=float)
    marks_xy_rot = _rotate_marks_xy(marks_xy_world, origin_xy, angle_rad_ccw)

    geom_rot = dict(geom)
    geom_rot["poly_BDY"] = shp_rotate(geom["poly_BDY"], float(np.rad2deg(angle_rad_ccw)), origin=(origin_xy[0], origin_xy[1]))
    geom_rot["poly_buffer"] = (
        shp_rotate(geom["poly_buffer"], float(np.rad2deg(angle_rad_ccw)), origin=(origin_xy[0], origin_xy[1]))
        if geom.get("poly_buffer") is not None
        else None
    )

    LG1 = marks_xy_rot["LG1"]
    LG2 = marks_xy_rot["LG2"]
    WG1 = marks_xy_rot["WG1"]
    WG2 = marks_xy_rot["WG2"]
    M1 = marks_xy_rot["M1"]

    MLG = 0.5 * (LG1 + LG2)
    MWG = 0.5 * (WG1 + WG2)

    LEFT = np.array([+1.0, 0.0], dtype=float)
    RIGHT = np.array([-1.0, 0.0], dtype=float)

    M1_round = M1 + float(d_rounding_mark) * LEFT
    LG2_left = LG2 + float(d_rounding_mark) * LEFT
    LG1_right = LG1 + float(d_rounding_mark) * RIGHT
    WG2_left = WG2 + float(d_rounding_mark) * LEFT
    WG1_right = WG1 + float(d_rounding_mark) * RIGHT

    twa_cands = polar_twa_candidates(polar_interp)
    biases = compute_gate_biases(marks_xy_rot, 180.0)

    # =========================================================
    # Start-line overlay (First DW only): 11 points SL_0..SL_100
    # VECTEURS SANS FLECHES
    # Référence: segment le plus long (point le plus lent) => vecteur de longueur 0
    # =========================================================
    startline_overlay = None
    if "SL1" in marks_xy_rot and "SL2" in marks_xy_rot:
        SL1 = marks_xy_rot["SL1"]
        SL2 = marks_xy_rot["SL2"]

        SL1_7m = _point_along(SL1, SL2, 7.0)
        SL2_7m = _point_along(SL2, SL1, 7.0)

        rows = []
        for i in range(11):  # 0..100%
            t = i / 10.0
            p = SL2_7m + t * (SL1_7m - SL2_7m)
            x_pct = i * 10

            v = M1 - p
            dist_m = float(np.linalg.norm(v))
            hdg = bearing_deg_xy((float(p[0]), float(p[1])), (float(M1[0]), float(M1[1])))
            twa = wrap180(twa_from_twd_hdg(180.0, hdg))

            bsp_kmph = float(polar_interp(float(TWS_ref_kmh), float(abs(twa))))
            bsp_mps = _kmph_to_mps(bsp_kmph)
            time_s = dist_m / bsp_mps if bsp_mps > 1e-9 else float("inf")

            rows.append(
                dict(
                    x_pct=int(x_pct),
                    p_xy=p,
                    hdg_deg=float(hdg),
                    twa_deg=float(twa),
                    dist_m=float(dist_m),
                    bsp_kmph=float(bsp_kmph),
                    time_s=float(time_s),
                )
            )

        # ✅ Référence = temps le plus long (point le plus lent)
        Tmax = max(r["time_s"] for r in rows if np.isfinite(r["time_s"]))

        pts_ll = []
        arrow_paths_ll = []   # PathLayer segments (just shaft)
        vectors_ll = []       # LineLayer compat (shaft)

        for r in rows:
            # ✅ dt = 0 pour le plus lent => vecteur 0
            dt = float(Tmax - r["time_s"])
            dd = _kmph_to_mps(r["bsp_kmph"]) * dt

            u_to_M1 = _unit(M1 - r["p_xy"])
            end_xy = r["p_xy"] + dd * u_to_M1

            # point itself (always)
            p_xy_world = _rot_xy_about(np.array(r["p_xy"], dtype=float), origin_xy, -angle_rad_ccw)
            plat, plon = xy_to_ll(ctx["to_wgs"], float(p_xy_world[0]), float(p_xy_world[1]))
            pts_ll.append(
                dict(
                    name=f"SL_{r['x_pct']}%",
                    lon=float(plon),
                    lat=float(plat),
                    time_s=float(r["time_s"]),
                    dt_s=float(dt),
                    dd_m=float(dd),
                    bsp_kmph=float(r["bsp_kmph"]),
                    twa_deg=float(r["twa_deg"]),
                    hdg_deg=float(r["hdg_deg"]),
                    dist_m=float(r["dist_m"]),
                )
            )

            # segment (skip if too small)
            if dd < 1e-6:
                continue

            # build one shaft segment in WORLD LL
            a_w = _rot_xy_about(np.array(r["p_xy"], dtype=float), origin_xy, -angle_rad_ccw)
            b_w = _rot_xy_about(np.array(end_xy, dtype=float), origin_xy, -angle_rad_ccw)
            alat, alon = xy_to_ll(ctx["to_wgs"], float(a_w[0]), float(a_w[1]))
            blat, blon = xy_to_ll(ctx["to_wgs"], float(b_w[0]), float(b_w[1]))

            arrow_paths_ll.append({"path": [[float(alon), float(alat)], [float(blon), float(blat)]]})
            vectors_ll.append({"lon0": float(alon), "lat0": float(alat), "lon1": float(blon), "lat1": float(blat)})

        startline_overlay = {
            "SL_points": pts_ll,
            "SL_arrow_paths": arrow_paths_ll,  # (shaft only)
            "SL_vectors": vectors_ll,          # compat
            "Tref_s": float(Tmax),
        }

    # --- Groups
    g_first_dw = compute_route_group(
        group_id="FIRST_DW_M1",
        start_xy=M1_round,
        dest_xy=MLG,
        ctx=ctx,
        geom=geom_rot,
        TWD=180.0,
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
        leg_type="DW",
        A_twa_sign=+1,
    )

    g_uw_lg2 = compute_route_group(
        group_id="FULL_UW_LG2",
        start_xy=LG2_left,
        dest_xy=MWG,
        ctx=ctx,
        geom=geom_rot,
        TWD=180.0,
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
        leg_type="UW",
        A_twa_sign=+1,
    )
    g_uw_lg1 = compute_route_group(
        group_id="FULL_UW_LG1",
        start_xy=LG1_right,
        dest_xy=MWG,
        ctx=ctx,
        geom=geom_rot,
        TWD=180.0,
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
        leg_type="UW",
        A_twa_sign=-1,
    )

    g_dw_wg2 = compute_route_group(
        group_id="FULL_DW_WG2",
        start_xy=WG2_left,
        dest_xy=MLG,
        ctx=ctx,
        geom=geom_rot,
        TWD=180.0,
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
        leg_type="DW",
        A_twa_sign=+1,
    )
    g_dw_wg1 = compute_route_group(
        group_id="FULL_DW_WG1",
        start_xy=WG1_right,
        dest_xy=MLG,
        ctx=ctx,
        geom=geom_rot,
        TWD=180.0,
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,
        twa_candidates_deg=twa_cands,
        gybe_loss_s=float(gybe_loss_s),
        leg_type="DW",
        A_twa_sign=-1,
    )

    # Convert routes to WORLD LL for viz + colors
    def _routes_to_world(group_out: dict) -> list[dict]:
        routes = group_out.get("routes", [])
        for r in routes:
            traj = str(r.get("traj", r.get("label", "?"))).strip()
            gid = str(r.get("group_id", group_out.get("group_id", "")))
            r["traj"] = traj
            r["color"] = _color_for(gid, traj)

            path_xy_rot = [np.array(p, dtype=float) for p in r.get("route_path_xy", [])]
            path_xy_world = [_rot_xy_about(p, origin_xy, -angle_rad_ccw) for p in path_xy_rot]
            r["route_path_ll"] = []
            for p in path_xy_world:
                lat, lon = xy_to_ll(ctx["to_wgs"], float(p[0]), float(p[1]))
                r["route_path_ll"].append([lon, lat])
        return routes

    routes_1 = _routes_to_world(g_first_dw)
    routes_2 = _routes_to_world(g_uw_lg2) + _routes_to_world(g_uw_lg1)
    routes_3 = _routes_to_world(g_dw_wg2) + _routes_to_world(g_dw_wg1)

    out1 = {**biases, "TWD": float(TWD), "routes": routes_1, "vmg_info": []}
    if startline_overlay is not None:
        out1["startline_overlay"] = startline_overlay

    out2 = {**biases, "TWD": float(TWD), "routes": routes_2, "vmg_info": []}
    out3 = {**biases, "TWD": float(TWD), "routes": routes_3, "vmg_info": []}

    lw_bias = float(biases.get("LW_gate_bias_m", float("nan")))
    ww_bias = float(biases.get("WW_gate_bias_m", float("nan")))
    st.markdown(
        f"**LW gate bias**: {lw_bias:.1f} m — *{_gate_side_label('LW', lw_bias)}*  \n"
        f"**WW gate bias**: {ww_bias:.1f} m — *{_gate_side_label('WW', ww_bias)}*"
    )

    deck1 = build_deck_routeur(ctx=ctx, geom=geom, marks_ll=marks_ll, marks_xy=marks_xy_world, route_out=out1)
    deck2 = build_deck_routeur(ctx=ctx, geom=geom, marks_ll=marks_ll, marks_xy=marks_xy_world, route_out=out2)
    deck3 = build_deck_routeur(ctx=ctx, geom=geom, marks_ll=marks_ll, marks_xy=marks_xy_world, route_out=out3)

    return {"deck1": deck1, "deck2": deck2, "deck3": deck3}, {"out1": out1, "out2": out2, "out3": out3}
