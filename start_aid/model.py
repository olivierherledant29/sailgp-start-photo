import numpy as np
from shapely.geometry import LineString

from .geo import (
    heading_to_unit_vector,
    intersection_ray_with_polygon_boundary,
    line_infinite_through_points,
    xy_to_ll,
)


def _bearing_deg_xy(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """
    Bearing in degrees, 0=N, 90=E, from XY (meters) where +x=East, +y=North.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    ang = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    return float(ang)


def _twa_abs_deg(heading_deg: float, twd_deg: float) -> float:
    """
    Absolute TWA in [0,180], from heading and TWD (both degrees, 0=N).
    """
    a = (float(heading_deg) - float(twd_deg) + 360.0) % 360.0
    if a > 180.0:
        a = 360.0 - a
    return float(a)


def kmh_to_mps(v_kmh: float) -> float:
    return float(v_kmh) / 3.6


def meters_to_seconds(dist_m: float, bsp_kmh: float) -> float:
    v = kmh_to_mps(bsp_kmh)
    return float(dist_m) / v if v > 1e-6 else float("inf")


def compute_forward_intersection_between_lines(
    ray_PI: LineString,
    PI_xy: np.ndarray,
    dir_PI: np.ndarray,
    layline: LineString,
    SL1_xy: np.ndarray,
    dir_lay: np.ndarray,
):
    inter = ray_PI.intersection(layline)
    if inter.is_empty:
        return None

    pts = []
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
        pts = [g for g in inter.geoms if g.geom_type == "Point"]
    else:
        return None

    candidates = []
    for p in pts:
        pxy = np.array([p.x, p.y], dtype=float)
        v1 = pxy - PI_xy
        v2 = pxy - SL1_xy

        if float(np.dot(v1, dir_PI)) <= 1e-6:
            continue
        if float(np.dot(v2, dir_lay)) <= 1e-6:
            continue

        candidates.append((float(np.linalg.norm(v1)), p))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def rgb_to_css(rgb):
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


SECOND_COLORS = {
    ("buffer_BDY", "to_SL1"): [160, 32, 240],
    ("buffer_BDY", "to_SL2"): [0, 255, 255],
    ("buffer_BDY", "to_SP"):  [165, 42, 42],
    ("LL_SL1",     "to_SL1"): [112, 128, 144],
    ("LL_SL1",     "to_SL2"): [0, 128, 128],
    ("LL_SL1",     "to_SP"):  [138, 43, 226],
    # NEW trajectory (PI->M_PAR_SL2->SL2_7m)
    ("PAR_SL2",    "to_SL2"): [255, 140, 0],
}

def compute_startline_special_points(
    SL1_xy: np.ndarray,
    SL2_xy: np.ndarray,
    X_percent: float,
    offset_m: float = 7.0,
):
    v = SL1_xy - SL2_xy
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return SL1_xy.copy(), SL2_xy.copy(), SL2_xy.copy()

    u = v / L  # unit from SL2 -> SL1
    off = min(float(offset_m), max(0.0, 0.49 * L))

    SL1_7m = SL1_xy - u * off
    SL2_7m = SL2_xy + u * off

    x = max(0.0, min(100.0, float(X_percent)))
    SP = SL2_xy + u * (L * (x / 100.0))

    return SL1_7m, SL2_7m, SP


def compute_all_geometry_and_times(
    ctx,
    geom,
    PI_xy,
    TWD,
    TWA_port,
    TWA_UW,
    TWS_ref_kmh: float,
    polar_bsp_kmh,          # callable: (tws_kmh: float, twa_deg: float) -> float
    pol_BAB_pct: float,
    pol_TRIB_pct: float,
    M_lost,
    X_percent,
    TTS_intersection,
):
    """
    Compute geometry + timings.

    New inputs (polar-based):
      - TWS_ref_kmh: reference wind speed (km/h) used for polar interpolation
      - polar_bsp_kmh(tws_kmh, twa_deg): returns BSP in km/h (bilinear interp)
      - pol_BAB_pct / pol_TRIB_pct: multipliers in percent applied to polar BSP

    Notes:
      - Approach (BAB): uses TWA_port
      - Return (TRIB): for each (group, dest), compute the heading from M -> target,
        derive TWA_abs vs TWD, then interpolate polar BSP, then apply pol_TRIB_pct.
    """
    to_wgs = ctx["to_wgs"]
    SL1_xy, SL2_xy = geom["SL1_xy"], geom["SL2_xy"]
    poly_BDY = geom["poly_BDY"]
    poly_buffer = geom.get("poly_buffer", None)

    # Output dict early (so we can populate progressively)
    out: dict = {}

    # Headings (deg, 0=N)
    ANG_port = (float(TWD) + float(TWA_port)) % 360.0
    ANG_UW = (float(TWD) - float(TWA_UW) + 180.0) % 360.0

    # Start line (infinite, only for consistency / debug)
    _ = line_infinite_through_points(SL1_xy, SL2_xy, scale=80000.0)

    # Ray from PI on port approach
    dir_port = heading_to_unit_vector(ANG_port)
    ray_PI = LineString([tuple(PI_xy), tuple(PI_xy + dir_port * 60000.0)])

    # --- Polar-based approach speed (BAB)
    bsp_approche_pol = float(polar_bsp_kmh(float(TWS_ref_kmh), float(TWA_port)))
    bsp_approche_bab = bsp_approche_pol * (float(pol_BAB_pct) / 100.0)

    out["TWS_ref_kmh"] = float(TWS_ref_kmh)
    out["pol_BAB_pct"] = float(pol_BAB_pct)
    out["pol_TRIB_pct"] = float(pol_TRIB_pct)
    out["BSP_approche_pol_kmph"] = float(bsp_approche_pol)
    out["BSP_approche_BAB_kmph"] = float(bsp_approche_bab)

    # --- M points
    M_buffer_BDY_xy = None
    if poly_buffer is not None:
        p = intersection_ray_with_polygon_boundary(ray_PI, PI_xy, poly_buffer, dir_port)
        if p is not None:
            M_buffer_BDY_xy = np.array([p.x, p.y], dtype=float)

    dir_UW = heading_to_unit_vector(ANG_UW)
    layline_SL1 = LineString([tuple(SL1_xy), tuple(SL1_xy + dir_UW * 80000.0)])

    p_ll = compute_forward_intersection_between_lines(
        ray_PI=ray_PI,
        PI_xy=PI_xy,
        dir_PI=dir_port,
        layline=layline_SL1,
        SL1_xy=SL1_xy,
        dir_lay=dir_UW,
    )
    M_LL_SL1_xy = None
    if p_ll is not None:
        M_LL_SL1_xy = np.array([p_ll.x, p_ll.y], dtype=float)

    # --- Laylines clipped to boundary (visual)
    def layline_to_boundary(start_xy):
        r = LineString([tuple(start_xy), tuple(start_xy + dir_UW * 120000.0)])
        p = intersection_ray_with_polygon_boundary(r, start_xy, poly_BDY, dir_UW)
        if p is None:
            return None
        return LineString([tuple(start_xy), (p.x, p.y)])

    lay_vis_SL1 = layline_to_boundary(SL1_xy)
    lay_vis_SL2 = layline_to_boundary(SL2_xy)

    # --- Special points on start line
    SL1_7m_xy, SL2_7m_xy, SP_xy = compute_startline_special_points(SL1_xy, SL2_xy, X_percent, offset_m=7.0)

    # --- NEW: Manoeuvre point = intersection(ray_PI, parallel layline through SL2_7m)
    layline_SL2_7m = LineString([tuple(SL2_7m_xy), tuple(SL2_7m_xy + dir_UW * 80000.0)])
    p_par = compute_forward_intersection_between_lines(
        ray_PI=ray_PI,
        PI_xy=PI_xy,
        dir_PI=dir_port,
        layline=layline_SL2_7m,
        SL1_xy=SL2_7m_xy,
        dir_lay=dir_UW,
    )
    M_PAR_SL2_xy = None
    if p_par is not None:
        M_PAR_SL2_xy = np.array([p_par.x, p_par.y], dtype=float)

    DESTS = {
        "to_SL1": SL1_7m_xy,
        "to_SL2": SL2_7m_xy,
        "to_SP": SP_xy,
    }
    GROUPS = {"buffer_BDY": M_buffer_BDY_xy, "LL_SL1": M_LL_SL1_xy, "PAR_SL2": M_PAR_SL2_xy}

    # --- Paths (for map)
    results = []
    traj_second_segments = []
    first_leg_paths = []

    PI_ll = xy_to_ll(to_wgs, PI_xy[0], PI_xy[1])

    if M_buffer_BDY_xy is not None:
        M_ll = xy_to_ll(to_wgs, M_buffer_BDY_xy[0], M_buffer_BDY_xy[1])
        first_leg_paths.append({"path": [[PI_ll[1], PI_ll[0]], [M_ll[1], M_ll[0]]], "name": "PI->M_buffer_BDY"})
    if M_LL_SL1_xy is not None:
        M_ll = xy_to_ll(to_wgs, M_LL_SL1_xy[0], M_LL_SL1_xy[1])
        first_leg_paths.append({"path": [[PI_ll[1], PI_ll[0]], [M_ll[1], M_ll[0]]], "name": "PI->M_LL_SL1"})
    if M_PAR_SL2_xy is not None:
        M_ll = xy_to_ll(to_wgs, M_PAR_SL2_xy[0], M_PAR_SL2_xy[1])
        first_leg_paths.append({"path": [[PI_ll[1], PI_ll[0]], [M_ll[1], M_ll[0]]], "name": "PI->M_PAR_SL2"})

    # --- Evaluate trajectories
    bsp_retour_vals = []

    for gname, M_xy in GROUPS.items():
        if M_xy is None:
            continue

        d1 = float(np.linalg.norm(M_xy - PI_xy))
        t1 = meters_to_seconds(d1, float(bsp_approche_bab))

        dests_iter = DESTS.items() if gname != "PAR_SL2" else [("to_SL2", SL2_7m_xy)]

        for dname, target_xy in dests_iter:
            # Return heading and TWA (abs) for this segment
            if float(np.linalg.norm(target_xy - M_xy)) < 1e-6:
                heading_return = float("nan")
                twa_return = float("nan")
            else:
                heading_return = _bearing_deg_xy((float(M_xy[0]), float(M_xy[1])), (float(target_xy[0]), float(target_xy[1])))
                twa_return = _twa_abs_deg(heading_return, float(TWD))

            bsp_retour_pol = float(polar_bsp_kmh(float(TWS_ref_kmh), float(twa_return))) if np.isfinite(twa_return) else float("nan")
            bsp_retour_trib = bsp_retour_pol * (float(pol_TRIB_pct) / 100.0) if np.isfinite(bsp_retour_pol) else float("nan")

            d2 = float(np.linalg.norm(target_xy - M_xy))
            t2 = meters_to_seconds(d2, float(bsp_retour_trib)) if np.isfinite(bsp_retour_trib) else float("inf")

            t_total = t1 + t2 + float(M_lost)
            ttk = float(TTS_intersection) - t_total
            ttk_before_tack = ttk + float(M_lost)

            color = SECOND_COLORS.get((gname, dname), [255, 255, 255])

            M_ll = xy_to_ll(to_wgs, M_xy[0], M_xy[1])
            end_ll = xy_to_ll(to_wgs, target_xy[0], target_xy[1])

            traj_second_segments.append({
                "group": gname,
                "dest": dname,
                "color": color,
                "path": [[M_ll[1], M_ll[0]], [end_ll[1], end_ll[0]]],
            })

            results.append({
                "group": gname,
                "dest": dname,
                "t1": t1,
                "t2": t2,
                "t_total": t_total,
                "ttk_before_tack": ttk_before_tack,
                "ttk": ttk,
                "color": color,
                "heading_return_deg": heading_return,
                "twa_return_abs_deg": twa_return,
                "BSP_retour_pol_kmph": bsp_retour_pol,
                "BSP_retour_TRIB_kmph": bsp_retour_trib,
            })

            if np.isfinite(bsp_retour_trib):
                bsp_retour_vals.append(float(bsp_retour_trib))

    out["BSP_retour_TRIB_avg_kmph"] = float(np.mean(bsp_retour_vals)) if bsp_retour_vals else float("nan")

    # --- Results table HTML
    rows = []
    for r in results:
        c = rgb_to_css(r["color"])
        rows.append(
            f"<tr>"
            f"<td>{r['group']}</td>"
            f"<td>{r['dest']}</td>"
            f"<td style='text-align:right;'>{r['t1']:.1f}</td>"
            f"<td style='text-align:right;'>{r['t2']:.1f}</td>"
            f"<td style='text-align:right; color:{c}; font-weight:700;'>{r['t_total']:.1f}</td>"
            f"<td style='text-align:right; color:{c}; font-weight:700;'>{r['ttk_before_tack']:.1f}</td>"
            f"<td style='text-align:right; color:{c}; font-weight:700;'>{r['ttk']:.1f}</td>"
            f"<td style='text-align:right;'>{r['twa_return_abs_deg']:.1f}</td>"
            f"<td style='text-align:right;'>{r['BSP_retour_TRIB_kmph']:.1f}</td>"
            f"</tr>"
        )

    results_html = f"""
    <div style="font-size: 13px;">
      <table style="width:100%; border-collapse: collapse;">
        <thead>
          <tr>
            <th style="text-align:left; border-bottom:1px solid #666;">groupe</th>
            <th style="text-align:left; border-bottom:1px solid #666;">dest</th>
            <th style="text-align:right; border-bottom:1px solid #666;">t1 PI→M (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">t2 M→target (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">t_total (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">TTK_beforeTack (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">TTK (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">TWA_return (°)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">BSP_TRIB (km/h)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else ''}
        </tbody>
      </table>
      <div style="margin-top:6px; color:#aaa;">
        t_total = t1(BSP_approche_BAB={float(bsp_approche_bab):.1f} km/h) + t2(BSP_retour_TRIB=pol(TWS_ref={float(TWS_ref_kmh):.1f},TWA_return)*{float(pol_TRIB_pct):.0f}%) + M_lost({float(M_lost):.1f}s) <br/>
        TTK = TTS_intersection({float(TTS_intersection):.1f}s) − t_total <br/>
        TTK_beforeTack = TTK + M_lost
      </div>
    </div>
    """

    out.update({
        "TWD": float(TWD),
        "ANG_port": ANG_port,
        "ANG_UW": ANG_UW,
        "dir_UW": dir_UW,
        "M_buffer_BDY_xy": M_buffer_BDY_xy,
        "M_LL_SL1_xy": M_LL_SL1_xy,
        "M_PAR_SL2_xy": M_PAR_SL2_xy,
        "lay_vis_SL1": lay_vis_SL1,
        "lay_vis_SL2": lay_vis_SL2,
        "first_leg_paths": first_leg_paths,
        "traj_second_segments": traj_second_segments,
        "results": results,
        "results_html": results_html,
        "SL1_7m_xy": SL1_7m_xy,
        "SL2_7m_xy": SL2_7m_xy,
        "SP_xy": SP_xy,
    })
    return out
