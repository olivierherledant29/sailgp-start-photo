from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point as ShapelyPoint

from .geo import heading_to_unit_vector


# =========================================================
# Angles helpers
# =========================================================
def wrap180(a: float) -> float:
    """Normalize angle to (-180, 180]."""
    x = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if x == -180.0 else x


def twa_from_twd_hdg(twd: float, hdg: float) -> float:
    """
    SIGNED TWA in (-180,180] with your convention:
      TWA = TWD - HDG
    Positive => tribord, negative => babord
    """
    return wrap180(float(twd) - float(hdg))


def hdg_from_twd_twa(twd: float, twa: float) -> float:
    """
    Inverse mapping consistent with your convention:
      HDG = TWD - TWA
    """
    return (float(twd) - float(twa)) % 360.0


def _bearing_deg_xy(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """
    Bearing in degrees, 0=N, 90=E, from XY (meters) where +x=East, +y=North.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    ang = math.degrees(math.atan2(dx, dy))  # atan2(x, y) => 0 at North
    return ang % 360.0


def _twa_signed_to_target_xy(from_xy: np.ndarray, to_xy: np.ndarray, twd_deg: float) -> float:
    """Signed TWA to target using your convention TWA=TWD-HDG."""
    hdg = _bearing_deg_xy(
        (float(from_xy[0]), float(from_xy[1])),
        (float(to_xy[0]), float(to_xy[1])),
    )
    return twa_from_twd_hdg(float(twd_deg), float(hdg))


# =========================================================
# Units
# =========================================================
def kmh_to_mps(v_kmh: float) -> float:
    return float(v_kmh) / 3.6


def meters_to_seconds(dist_m: float, bsp_kmh: float) -> float:
    v = kmh_to_mps(float(bsp_kmh))
    return float(dist_m) / v if v > 1e-6 else float("inf")


# =========================================================
# Gate bias
# =========================================================
def gate_bias_m(mark1_xy: np.ndarray, mark2_xy: np.ndarray, TWD: float) -> float:
    """
    Bias = projection of segment (mark2 - mark1) onto TWD axis.
    Axis direction is heading_to_unit_vector(TWD).
    Returned in meters, signed (sign convention validated by you).
    """
    v = np.array(mark2_xy, dtype=float) - np.array(mark1_xy, dtype=float)
    u = heading_to_unit_vector(float(TWD))
    return float(np.dot(v, u))


def compute_gate_biases(marks_xy: dict[str, np.ndarray], TWD: float) -> dict:
    out = {}
    if "LG1" in marks_xy and "LG2" in marks_xy:
        out["LW_gate_bias_m"] = gate_bias_m(marks_xy["LG1"], marks_xy["LG2"], float(TWD))
    else:
        out["LW_gate_bias_m"] = float("nan")

    if "WG1" in marks_xy and "WG2" in marks_xy:
        out["WW_gate_bias_m"] = gate_bias_m(marks_xy["WG1"], marks_xy["WG2"], float(TWD))
    else:
        out["WW_gate_bias_m"] = float("nan")

    return out


# =========================================================
# VMG optima
# =========================================================
def compute_vmg_optima(TWS_ref_kmh, polar_bsp_kmh, twa_candidates_deg):
    """
    best_uw: maximize bsp*cos(twa_abs)
    best_dw: minimize bsp*cos(twa_abs)  (most negative)
    """
    best_uw = {"vmg": -1e9, "twa": float("nan"), "bsp": float("nan")}
    best_dw = {"vmg": +1e9, "twa": float("nan"), "bsp": float("nan")}

    for twa in twa_candidates_deg:
        twa_abs = abs(float(twa))
        bsp = float(polar_bsp_kmh(float(TWS_ref_kmh), float(twa_abs)))
        if not np.isfinite(bsp):
            continue
        vmg = bsp * math.cos(math.radians(twa_abs))
        if vmg > best_uw["vmg"]:
            best_uw = {"vmg": vmg, "twa": twa_abs, "bsp": bsp}
        if vmg < best_dw["vmg"]:
            best_dw = {"vmg": vmg, "twa": twa_abs, "bsp": bsp}

    return best_uw, best_dw


# =========================================================
# Core simulator (Forward + Rear)
# =========================================================
def _simulate_leg_with_flips(
    *,
    geom,
    start_xy: np.ndarray,
    dest_xy: np.ndarray,
    TWD: float,
    TWS_ref_kmh: float,
    polar_bsp_kmh,
    twa_abs_forward: float,   # TWA_vmg_DW (positive, forward reference)
    bsp_sailed_kmph: float,   # constant speed during 1 Hz stepping
    start_tack_sign: int,     # +1 tribord, -1 babord in EVALUATION convention
    gybe_loss_s: float,
    reach_m: float,
    max_steps: int,
    label: str,
    include_events: bool = True,
    feasibility_mode: str = "forward",  # "forward" | "rear"
):
    """
    forward:
      sailed TWA = ±TWA_vmg_DW
      eval TWA to dest = twa_to_dest
      direct if abs(eval) <= TWA_vmg_DW

    rear:
      sailed TWA = wrap180(±TWA_vmg_DW + 180)
      eval TWA to dest = wrap180(twa_to_dest + 180)

      IMPORTANT (your spec):
        direct rear feasible if (180 - abs(twa_to_dest)) < TWA_vmg_DW
        but abs(wrap180(twa_to_dest+180)) == 180 - abs(twa_to_dest)
        => direct if abs(eval) < TWA_vmg_DW
    """

    def twa_to_dest_signed(pos_xy: np.ndarray) -> float:
        return float(_twa_signed_to_target_xy(pos_xy, dest_xy, float(TWD)))



    poly_buffer = geom.get("poly_buffer", None)

    def sign(x: float) -> int:
        if x > 1e-9:
            return 1
        if x < -1e-9:
            return -1
        return 0

    def inside_buffer(pxy: np.ndarray) -> bool:
        if poly_buffer is None:
            return True
        return bool(poly_buffer.contains(ShapelyPoint(float(pxy[0]), float(pxy[1]))))

    events: list[dict] = []

    def log(evt: str, **kw):
        if include_events:
            events.append({"evt": evt, **kw})

    twa_abs_forward = float(abs(twa_abs_forward))

    def twa_sailed_from_sign(tack_sign: int) -> float:
        """Signed TWA that we actually sail with."""
        if feasibility_mode == "rear":
            return wrap180(float(tack_sign) * twa_abs_forward + 180.0)
        return float(tack_sign) * twa_abs_forward

    def twa_eval_to_dest(pos_xy: np.ndarray) -> float:
        """Signed TWA used ONLY for feasibility (swap180 in rear)."""
        twa_to_dest = float(_twa_signed_to_target_xy(pos_xy, dest_xy, float(TWD)))
        if feasibility_mode == "rear":
            return wrap180(twa_to_dest + 180.0)
        return twa_to_dest

    # -----------------------------
    # ONLY CHANGE: direct feasibility in REAR
    # -----------------------------
    def direct_is_feasible(twa_eval: float) -> bool:
        eps = 1e-9
        if feasibility_mode == "rear":
            # abs(twa_eval) == 180 - abs(twa_to_dest)
            # spec: (180 - abs(twa_to_dest)) < TWA_vmg_DW  => abs(twa_eval) < TWA_vmg_DW
            return abs(twa_eval) < twa_abs_forward + eps
        # forward unchanged
        return abs(twa_eval) <= twa_abs_forward + eps

    # state
    tack_sign = +1 if start_tack_sign >= 0 else -1
    twa_sailed = twa_sailed_from_sign(tack_sign)

    pos = np.array(start_xy, dtype=float).copy()
    route_path_xy = [pos.copy()]
    gybe_points_xy: list[np.ndarray] = []
    time_s = 0.0
    dist_total = 0.0

    def do_flip(at_xy: np.ndarray, reason: str, twa_eval: float | None = None):
        nonlocal tack_sign, twa_sailed, time_s

        prev = float(twa_sailed)
        tack_sign *= -1
        twa_sailed = twa_sailed_from_sign(tack_sign)

        # In this router: flip => empannage (penalty +1)
        gybe_points_xy.append(at_xy.copy())
        time_s += float(gybe_loss_s)
        log(
            "GYBE",
            label=label,
            reason=reason,
            prev_twa_sailed=prev,
            new_twa_sailed=float(twa_sailed),
            twa_eval=float(twa_eval) if twa_eval is not None else None,
            x=float(at_xy[0]),
            y=float(at_xy[1]),
            n=len(gybe_points_xy),
        )

    # -----------------
    # Direct at start?
    # -----------------
    twa_eval_start = float(twa_eval_to_dest(pos))
    if direct_is_feasible(twa_eval_start):
        # keep existing behavior (as per your request: only detection change)
        twa_to_dest = float(twa_to_dest_signed(pos))
        if abs(twa_to_dest) > 1e-6 and sign(twa_to_dest) != sign(twa_sailed):
            do_flip(pos, reason="direct_start", twa_eval=twa_eval_start)


        dist = float(np.linalg.norm(dest_xy - pos))
        bsp_direct = float(polar_bsp_kmh(float(TWS_ref_kmh), float(abs(twa_eval_start))))
        time_s += meters_to_seconds(dist, bsp_direct)
        dist_total += dist
        route_path_xy.append(np.array(dest_xy, dtype=float).copy())

        return {
            "label": label,
            "route_path_xy": route_path_xy,
            "gybe_points_xy": gybe_points_xy,
            "time_total_s": float(time_s),
            "dist_total_m": float(dist_total),
            "n_gybes": int(len(gybe_points_xy)),
            "events": events,
            "TWA_to_dest_start_deg": float(twa_eval_start),
            "feasibility_mode": feasibility_mode,
        }

    # -----------------
    # Simulation at 1 Hz
    # -----------------
    v_mps = kmh_to_mps(float(bsp_sailed_kmph))
    step_m = float(v_mps) * 1.0

    for step_i in range(int(max_steps)):

        # close enough => finish (maybe flip before direct)
        if float(np.linalg.norm(dest_xy - pos)) <= float(reach_m):
            twa_now = float(twa_eval_to_dest(pos))
            if abs(twa_now) > 1e-6 and sign(twa_now) != sign(twa_sailed):
                do_flip(pos, reason="before_direct_close", twa_eval=twa_now)

            dist_last = float(np.linalg.norm(dest_xy - pos))
            bsp_now = float(polar_bsp_kmh(float(TWS_ref_kmh), float(abs(twa_now))))
            time_s += meters_to_seconds(dist_last, bsp_now)
            dist_total += dist_last
            route_path_xy.append(np.array(dest_xy, dtype=float).copy())
            log("FINISH", label=label, step=step_i, reason="close_enough")
            break

        # advance 1s on current sailed TWA
        heading = hdg_from_twd_twa(float(TWD), float(twa_sailed))
        dir_xy = heading_to_unit_vector(heading)
        pos_next = pos + dir_xy * step_m

        # buffer hit => flip
        if not inside_buffer(pos_next):
            do_flip(pos, reason="buffer")
            continue

        # accept step
        dist_step = float(np.linalg.norm(pos_next - pos))
        dist_total += dist_step
        time_s += 1.0
        pos = pos_next
        route_path_xy.append(pos.copy())

        # direct becomes feasible?
        twa_here = float(twa_eval_to_dest(pos))
        if direct_is_feasible(twa_here):
            twa_to_dest = float(twa_to_dest_signed(pos))
            if abs(twa_to_dest) > 1e-6 and sign(twa_to_dest) != sign(twa_sailed):
                do_flip(pos, reason="before_direct_feasible", twa_eval=twa_here)


            dist_last = float(np.linalg.norm(dest_xy - pos))
            bsp_direct = float(polar_bsp_kmh(float(TWS_ref_kmh), float(abs(twa_here))))
            time_s += meters_to_seconds(dist_last, bsp_direct)
            dist_total += dist_last
            route_path_xy.append(np.array(dest_xy, dtype=float).copy())
            log("FINISH", label=label, step=step_i, reason="direct_feasible")
            break

    return {
        "label": label,
        "route_path_xy": route_path_xy,
        "gybe_points_xy": gybe_points_xy,
        "time_total_s": float(time_s),
        "dist_total_m": float(dist_total),
        "n_gybes": int(len(gybe_points_xy)),
        "events": events,
        "TWA_to_dest_start_deg": float(twa_eval_start),
        "feasibility_mode": feasibility_mode,
    }


# =========================================================
# Public API — First DW from M1 (A/B + A'/B')
# =========================================================
def compute_first_dw_from_M1(
    ctx,
    geom,
    marks_xy: dict[str, np.ndarray],
    TWD: float,
    TWS_ref_kmh: float,
    polar_bsp_kmh,
    twa_candidates_deg: list[float],
    gybe_loss_s: float = 5.0,
    reach_m: float = 5.0,
    max_steps: int = 20000,
):
    for k in ("M1", "LG1", "LG2"):
        if k not in marks_xy:
            raise ValueError(f"Missing required mark: {k}")

    M1_xy = np.array(marks_xy["M1"], dtype=float)
    LG1_xy = np.array(marks_xy["LG1"], dtype=float)
    LG2_xy = np.array(marks_xy["LG2"], dtype=float)
    MLG_xy = 0.5 * (LG1_xy + LG2_xy)

    biases = compute_gate_biases(marks_xy, float(TWD))

    best_uw, best_dw = compute_vmg_optima(float(TWS_ref_kmh), polar_bsp_kmh, twa_candidates_deg)

    # forward VMG DW
    TWA_vmg_DW_abs = float(best_dw["twa"])  # positive
    BSP_vmg_DW = float(best_dw["bsp"])
    VMG_vmg_DW = float(best_dw["vmg"])

    # UW info
    TWA_vmg_UW_abs = float(best_uw["twa"])
    BSP_vmg_UW = float(best_uw["bsp"])
    VMG_vmg_UW = float(best_uw["vmg"])

    # rear speed lookup: abs(180 - TWA_vmg_DW)
    TWA_vmg_DW_rear_abs = abs(180.0 - TWA_vmg_DW_abs)
    BSP_vmg_DW_rear = float(polar_bsp_kmh(float(TWS_ref_kmh), float(TWA_vmg_DW_rear_abs)))
    VMG_vmg_DW_rear = float(BSP_vmg_DW_rear * math.cos(math.radians(float(TWA_vmg_DW_rear_abs))))

    # Forward A / B (M1 -> MLG)
    route_A = _simulate_leg_with_flips(
        geom=geom,
        start_xy=M1_xy,
        dest_xy=MLG_xy,
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_forward=TWA_vmg_DW_abs,
        bsp_sailed_kmph=BSP_vmg_DW,
        start_tack_sign=+1,
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="A",
        include_events=True,
        feasibility_mode="forward",
    )

    route_B = _simulate_leg_with_flips(
        geom=geom,
        start_xy=M1_xy,
        dest_xy=MLG_xy,
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_forward=TWA_vmg_DW_abs,
        bsp_sailed_kmph=BSP_vmg_DW,
        start_tack_sign=-1,
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="B",
        include_events=True,
        feasibility_mode="forward",
    )

    # Rear A' / B' (MLG -> M1) — leave everything else unchanged
    route_Ap = _simulate_leg_with_flips(
        geom=geom,
        start_xy=MLG_xy,
        dest_xy=M1_xy,
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_forward=TWA_vmg_DW_abs,
        bsp_sailed_kmph=BSP_vmg_DW_rear,
        start_tack_sign=+1,
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="A'",
        include_events=True,
        feasibility_mode="rear",
    )

    route_Bp = _simulate_leg_with_flips(
        geom=geom,
        start_xy=MLG_xy,
        dest_xy=M1_xy,
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_forward=TWA_vmg_DW_abs,
        bsp_sailed_kmph=BSP_vmg_DW_rear,
        start_tack_sign=-1,
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="B'",
        include_events=True,
        feasibility_mode="rear",
    )

    return {
        "category": "first_DW_from_M1",
        "M1_xy": M1_xy.copy(),
        "MLG_xy": MLG_xy.copy(),
        **biases,

        "TWA_vmg_DW_deg": float(TWA_vmg_DW_abs),
        "BSP_vmg_DW_kmph": float(BSP_vmg_DW),
        "VMG_vmg_DW_kmph": float(VMG_vmg_DW),

        "TWA_vmg_UW_deg": float(TWA_vmg_UW_abs),
        "BSP_vmg_UW_kmph": float(BSP_vmg_UW),
        "VMG_vmg_UW_kmph": float(VMG_vmg_UW),

        "TWA_vmg_DW_rear_deg": float(TWA_vmg_DW_rear_abs),
        "BSP_vmg_DW_rear_kmph": float(BSP_vmg_DW_rear),
        "VMG_vmg_DW_rear_kmph": float(VMG_vmg_DW_rear),

        "routes": [route_A, route_B, route_Ap, route_Bp],
    }
