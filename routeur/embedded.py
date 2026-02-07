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

    # First DW: couleurs franches demandées
    if group_id == "FIRST_DW_M1":
        return {
            "A": [0, 255, 0],        # vert
            "B": [255, 0, 0],        # rouge
            "A'": [0, 200, 255],     # cyan
            "B'": [255, 255, 0],     # jaune
        }.get(traj, [255, 255, 255])

    # autres groupes : base + shade
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
# Sign logic for selecting A'/B' correctly
# =========================================================
def _twa_sign_of_segment(p0: np.ndarray, p1: np.ndarray, TWD_calc: float = 180.0) -> int:
    hdg = bearing_deg_xy((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
    twa = wrap180(twa_from_twd_hdg(float(TWD_calc), float(hdg)))
    if twa > 1e-6:
        return +1
    if twa < -1e-6:
        return -1
    return 0


def _twa_sign_first_segment_forward(path_xy: list, TWD_calc: float = 180.0) -> int:
    """First segment (start -> next) for forward routes (A/B)."""
    if not path_xy or len(path_xy) < 2:
        return 0
    p0 = np.array(path_xy[0], dtype=float)
    p1 = np.array(path_xy[1], dtype=float)
    return _twa_sign_of_segment(p0, p1, TWD_calc=TWD_calc)


def _twa_sign_depart_from_rear(path_xy: list, TWD_calc: float = 180.0) -> int:
    """
    For rear routes (A'/B') stored as dest -> start:
    departure segment must be start -> previous point (outgoing):
      p[-1] (start) -> p[-2]
    """
    if not path_xy or len(path_xy) < 2:
        return 0
    p_start = np.array(path_xy[-1], dtype=float)
    p_prev = np.array(path_xy[-2], dtype=float)
    return _twa_sign_of_segment(p_start, p_prev, TWD_calc=TWD_calc)


def _filter_group_routes_full_uw(routes: list[dict]) -> list[dict]:
    """
    FULL UW (inchangé) :
      - keep A, B
      - keep A'/B' only if sign(TWA at departure) == sign(A forward first segment)
    """
    sign_A = 0
    for r in routes:
        traj = str(r.get("traj", r.get("label", ""))).strip()
        if traj == "A":
            sign_A = _twa_sign_first_segment_forward(r.get("route_path_xy", []), TWD_calc=180.0)
            break

    out = []
    for r in routes:
        traj = str(r.get("traj", r.get("label", ""))).strip()
        if traj in ("A", "B"):
            out.append(r)
            continue
        if traj in ("A'", "B'"):
            s = _twa_sign_depart_from_rear(r.get("route_path_xy", []), TWD_calc=180.0)
            if s == sign_A:
                out.append(r)
            continue
        out.append(r)
    return out


def _filter_group_routes_full_dw(routes: list[dict]) -> list[dict]:
    """
    FULL DW (corrigé):
      - remove B
      - keep A
      - keep EXACTLY ONE of A'/B' (never 0, never 2):
          prefer the one with matching sign to A;
          if both match -> keep faster;
          if none match -> keep faster.
    """
    # sign of A (forward)
    sign_A = 0
    route_A = None
    for r in routes:
        traj = str(r.get("traj", r.get("label", ""))).strip()
        if traj == "A":
            route_A = r
            sign_A = _twa_sign_first_segment_forward(r.get("route_path_xy", []), TWD_calc=180.0)
            break

    # collect primes
    prime_routes = []
    for r in routes:
        traj = str(r.get("traj", r.get("label", ""))).strip()
        if traj in ("A'", "B'"):
            s = _twa_sign_depart_from_rear(r.get("route_path_xy", []), TWD_calc=180.0)
            prime_routes.append((r, s))

    # decide which prime to keep
    chosen_prime = None
    if prime_routes:
        matching = [r for (r, s) in prime_routes if s == sign_A]
        if len(matching) == 1:
            chosen_prime = matching[0]
        elif len(matching) >= 2:
            chosen_prime = min(matching, key=lambda rr: float(rr.get("time_total_s", 1e18)))
        else:
            # none match -> keep fastest anyway
            chosen_prime = min([r for (r, _) in prime_routes], key=lambda rr: float(rr.get("time_total_s", 1e18)))

    out = []
    for r in routes:
        traj = str(r.get("traj", r.get("label", ""))).strip()

        if traj == "B":
            continue
        if traj == "A":
            out.append(r)
            continue
        if traj in ("A'", "B'"):
            if chosen_prime is not None and r is chosen_prime:
                out.append(r)
            continue

        out.append(r)

    return out


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

    # Widgets top row (no gybe widget)
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.6, 1.1])
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

    c6, c7 = st.columns([1.1, 3.9])
    with c6:
        d_rounding_mark = st.number_input("d_rounding_mark (m)", min_value=0, max_value=50, value=10, step=1, format="%d")
    with c7:
        st.markdown(f"**TWD_base={int(TWD_base)}°** — **offset_TWD={int(offset_TWD):+d}°** — **TWD appliqué={int(TWD)}°**")

    # Gybe loss fixed to 3s
    gybe_loss_s = 3.0

    # World geometry
    ctx = make_context_from_boundary(boundary_latlon)
    marks_for_geom = {"SL1": marks_ll.get("SL1", marks_ll["M1"]), "SL2": marks_ll.get("SL2", marks_ll["M1"]), "M1": marks_ll["M1"]}
    geom = to_xy_marks_and_polys(ctx, marks_for_geom, boundary_latlon, float(size_buffer_BDY))
    marks_xy_world = marks_ll_to_xy(ctx, marks_ll)

    # Rotate into calc frame (wind from top => TWD_calc=180)
    angle_deg_cw = (float(TWD) - 180.0) % 360.0
    angle_rad_ccw = np.deg2rad(angle_deg_cw)

    origin_xy = np.array([geom["poly_BDY"].centroid.x, geom["poly_BDY"].centroid.y], dtype=float)
    marks_xy_rot = _rotate_marks_xy(marks_xy_world, origin_xy, angle_rad_ccw)

    geom_rot = dict(geom)
    geom_rot["poly_BDY"] = shp_rotate(geom["poly_BDY"], float(np.rad2deg(angle_rad_ccw)), origin=(origin_xy[0], origin_xy[1]))
    geom_rot["poly_buffer"] = shp_rotate(geom["poly_buffer"], float(np.rad2deg(angle_rad_ccw)), origin=(origin_xy[0], origin_xy[1])) if geom.get("poly_buffer") is not None else None

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

    # Apply filtering
    g_uw_lg2["routes"] = _filter_group_routes_full_uw(g_uw_lg2["routes"])   # UW unchanged
    g_uw_lg1["routes"] = _filter_group_routes_full_uw(g_uw_lg1["routes"])   # UW unchanged
    g_dw_wg2["routes"] = _filter_group_routes_full_dw(g_dw_wg2["routes"])   # DW: keep one prime
    g_dw_wg1["routes"] = _filter_group_routes_full_dw(g_dw_wg1["routes"])   # DW: keep one prime

    # =============================
    # Convert routes to WORLD LL for viz + colors
    # =============================
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

    # Final safety: ensure B never appears in FULL DW
    routes_3 = [r for r in routes_3 if str(r.get("traj", "")).strip() != "B"]

    # VMG info per card
    vmg1 = [{"group": g_first_dw["group_id"], "leg": g_first_dw["leg_type"], "TWA": g_first_dw["TWA_vmg_abs_deg"], "BSP": g_first_dw["BSP_vmg_kmph"]}]
    vmg2 = [
        {"group": g_uw_lg2["group_id"], "leg": g_uw_lg2["leg_type"], "TWA": g_uw_lg2["TWA_vmg_abs_deg"], "BSP": g_uw_lg2["BSP_vmg_kmph"]},
        {"group": g_uw_lg1["group_id"], "leg": g_uw_lg1["leg_type"], "TWA": g_uw_lg1["TWA_vmg_abs_deg"], "BSP": g_uw_lg1["BSP_vmg_kmph"]},
    ]
    vmg3 = [
        {"group": g_dw_wg2["group_id"], "leg": g_dw_wg2["leg_type"], "TWA": g_dw_wg2["TWA_vmg_abs_deg"], "BSP": g_dw_wg2["BSP_vmg_kmph"]},
        {"group": g_dw_wg1["group_id"], "leg": g_dw_wg1["leg_type"], "TWA": g_dw_wg1["TWA_vmg_abs_deg"], "BSP": g_dw_wg1["BSP_vmg_kmph"]},
    ]

    out1 = {**biases, "TWD": float(TWD), "routes": routes_1, "vmg_info": vmg1}
    out2 = {**biases, "TWD": float(TWD), "routes": routes_2, "vmg_info": vmg2}
    out3 = {**biases, "TWD": float(TWD), "routes": routes_3, "vmg_info": vmg3}

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
