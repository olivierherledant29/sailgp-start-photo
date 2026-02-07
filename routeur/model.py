from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point as ShapelyPoint

from .geo import heading_to_unit_vector


# =========================================================
# Angles helpers
# =========================================================
def wrap180(a: float) -> float:
    x = (float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if x == -180.0 else x


def twa_from_twd_hdg(twd: float, hdg: float) -> float:
    # convention: TWA = TWD - HDG ; tribord if >0
    return wrap180(float(twd) - float(hdg))


def hdg_from_twd_twa(twd: float, twa: float) -> float:
    return (float(twd) - float(twa)) % 360.0


def bearing_deg_xy(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """0=N, 90=E in XY meters (+x east, +y north)."""
    x0, y0 = p0
    x1, y1 = p1
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    ang = math.degrees(math.atan2(dx, dy))
    return ang % 360.0


def _twa_signed_to_target_xy(from_xy: np.ndarray, to_xy: np.ndarray, twd_deg: float) -> float:
    hdg = bearing_deg_xy(
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
    v = np.array(mark2_xy, dtype=float) - np.array(mark1_xy, dtype=float)
    u = heading_to_unit_vector(float(TWD))
    return float(np.dot(v, u))


def compute_gate_biases(marks_xy: dict[str, np.ndarray], TWD: float) -> dict:
    out = {}
    out["LW_gate_bias_m"] = gate_bias_m(marks_xy["LG1"], marks_xy["LG2"], float(TWD)) if ("LG1" in marks_xy and "LG2" in marks_xy) else float("nan")
    out["WW_gate_bias_m"] = gate_bias_m(marks_xy["WG1"], marks_xy["WG2"], float(TWD)) if ("WG1" in marks_xy and "WG2" in marks_xy) else float("nan")
    return out


# =========================================================
# VMG optima
# =========================================================
def compute_vmg_optima(TWS_ref_kmh, polar_bsp_kmh, twa_candidates_deg):
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
# Core simulator (Forward + Rear) — generic for UW/DW
# =========================================================
def _simulate_leg_with_flips(
    *,
    geom,
    start_xy: np.ndarray,
    dest_xy: np.ndarray,
    TWD: float,
    TWS_ref_kmh: float,
    polar_bsp_kmh,
    twa_abs_ref: float,       # always POSITIVE, ref angle (VMG angle for this leg type)
    bsp_ref_kmph: float,      # speed at that ref angle
    start_tack_sign: int,     # +1 => TWA_sailed>0 ; -1 => TWA_sailed<0
    gybe_loss_s: float,
    reach_m: float,
    max_steps: int,
    label: str,
    leg_type: str,            # "DW" or "UW"
    include_events: bool = True,
    feasibility_mode: str = "forward",  # "forward" | "rear"
):
    """
    DW forward  : direct feasible if abs(twa_eval) <= twa_abs_ref
    UW forward  : direct feasible if abs(twa_eval) >= twa_abs_ref  (can't sail closer than VMG)
    rear mode   : we use the same criterion but applied to twa_eval (already swapped by +180)
    """
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

    twa_abs_ref = float(abs(twa_abs_ref))
    leg_type = str(leg_type).upper()

    def twa_sailed_from_sign(tack_sign: int) -> float:
        if feasibility_mode == "rear":
            return wrap180(float(tack_sign) * twa_abs_ref + 180.0)
        return float(tack_sign) * twa_abs_ref

    def twa_eval_to_dest(pos_xy: np.ndarray) -> float:
        twa_to_dest = float(_twa_signed_to_target_xy(pos_xy, dest_xy, float(TWD)))
        if feasibility_mode == "rear":
            return wrap180(twa_to_dest + 180.0)
        return twa_to_dest

    def direct_is_feasible(twa_eval: float) -> bool:
        eps = 1e-6
        a = abs(float(twa_eval))
        if leg_type == "UW":
            # ✅ Upwind: feasible only if destination requires TWA at least as open as VMG TWA
            return a >= (twa_abs_ref - eps)
        # DW
        return a <= (twa_abs_ref + eps)

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

    twa_eval_start = float(twa_eval_to_dest(pos))

    # direct possible at start
    if direct_is_feasible(twa_eval_start):
        twa_to_dest_world = float(_twa_signed_to_target_xy(pos, dest_xy, float(TWD)))
        if abs(twa_to_dest_world) > 1e-6 and sign(twa_to_dest_world) != sign(twa_sailed):
            do_flip(pos, reason="direct_start", twa_eval=twa_eval_start)

        dist = float(np.linalg.norm(dest_xy - pos))
        bsp_direct = float(polar_bsp_kmh(float(TWS_ref_kmh), float(abs(twa_eval_start))))
        time_s += meters_to_seconds(dist, bsp_direct)
        dist_total += dist
        route_path_xy.append(np.array(dest_xy, dtype=float).copy())

        return {
            "label": label,
            "traj": label,
            "route_path_xy": route_path_xy,
            "gybe_points_xy": gybe_points_xy,
            "time_total_s": float(time_s),
            "dist_total_m": float(dist_total),
            "n_gybes": int(len(gybe_points_xy)),
            "events": events,
            "TWA_to_dest_start_deg": float(twa_eval_start),
            "feasibility_mode": feasibility_mode,
        }

    v_mps = kmh_to_mps(float(bsp_ref_kmph))
    step_m = float(v_mps) * 1.0

    for step_i in range(int(max_steps)):

        if float(np.linalg.norm(dest_xy - pos)) <= float(reach_m):
            twa_now = float(twa_eval_to_dest(pos))
            twa_to_dest_world = float(_twa_signed_to_target_xy(pos, dest_xy, float(TWD)))
            if abs(twa_to_dest_world) > 1e-6 and sign(twa_to_dest_world) != sign(twa_sailed):
                do_flip(pos, reason="before_direct_close", twa_eval=twa_now)

            dist_last = float(np.linalg.norm(dest_xy - pos))
            bsp_now = float(polar_bsp_kmh(float(TWS_ref_kmh), float(abs(twa_now))))
            time_s += meters_to_seconds(dist_last, bsp_now)
            dist_total += dist_last
            route_path_xy.append(np.array(dest_xy, dtype=float).copy())
            log("FINISH", label=label, step=step_i, reason="close_enough")
            break

        heading = hdg_from_twd_twa(float(TWD), float(twa_sailed))
        dir_xy = heading_to_unit_vector(heading)
        pos_next = pos + dir_xy * step_m

        if not inside_buffer(pos_next):
            do_flip(pos, reason="buffer")
            continue

        dist_step = float(np.linalg.norm(pos_next - pos))
        dist_total += dist_step
        time_s += 1.0
        pos = pos_next
        route_path_xy.append(pos.copy())

        twa_here = float(twa_eval_to_dest(pos))
        if direct_is_feasible(twa_here):
            twa_to_dest_world = float(_twa_signed_to_target_xy(pos, dest_xy, float(TWD)))
            if abs(twa_to_dest_world) > 1e-6 and sign(twa_to_dest_world) != sign(twa_sailed):
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
        "traj": label,
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
# Generic group builder (A/B + A'/B') for UW or DW
# =========================================================
def compute_route_group(
    *,
    group_id: str,
    start_xy: np.ndarray,
    dest_xy: np.ndarray,
    ctx,
    geom,
    TWD: float,
    TWS_ref_kmh: float,
    polar_bsp_kmh,
    twa_candidates_deg: list[float],
    gybe_loss_s: float,
    leg_type: str,          # "DW" or "UW"
    A_twa_sign: int,        # +1 => A starts TWA>0 ; -1 => A starts TWA<0
    reach_m: float = 5.0,
    max_steps: int = 20000,
):
    best_uw, best_dw = compute_vmg_optima(float(TWS_ref_kmh), polar_bsp_kmh, twa_candidates_deg)

    leg_type_u = str(leg_type).upper()
    if leg_type_u == "DW":
        twa_abs = float(best_dw["twa"])
        bsp_ref = float(best_dw["bsp"])
        vmg_ref = float(best_dw["vmg"])
    else:
        twa_abs = float(best_uw["twa"])
        bsp_ref = float(best_uw["bsp"])
        vmg_ref = float(best_uw["vmg"])

    twa_abs_rear = abs(180.0 - twa_abs)
    bsp_ref_rear = float(polar_bsp_kmh(float(TWS_ref_kmh), float(twa_abs_rear)))
    vmg_ref_rear = float(bsp_ref_rear * math.cos(math.radians(float(twa_abs_rear))))

    route_A = _simulate_leg_with_flips(
        geom=geom,
        start_xy=np.array(start_xy, dtype=float),
        dest_xy=np.array(dest_xy, dtype=float),
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_ref=twa_abs,
        bsp_ref_kmph=bsp_ref,
        start_tack_sign=int(A_twa_sign),
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="A",
        leg_type=leg_type_u,
        include_events=True,
        feasibility_mode="forward",
    )
    route_B = _simulate_leg_with_flips(
        geom=geom,
        start_xy=np.array(start_xy, dtype=float),
        dest_xy=np.array(dest_xy, dtype=float),
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_ref=twa_abs,
        bsp_ref_kmph=bsp_ref,
        start_tack_sign=-int(A_twa_sign),
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="B",
        leg_type=leg_type_u,
        include_events=True,
        feasibility_mode="forward",
    )

    route_Ap = _simulate_leg_with_flips(
        geom=geom,
        start_xy=np.array(dest_xy, dtype=float),
        dest_xy=np.array(start_xy, dtype=float),
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_ref=twa_abs,
        bsp_ref_kmph=bsp_ref_rear,
        start_tack_sign=int(A_twa_sign),
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="A'",
        leg_type=leg_type_u,
        include_events=True,
        feasibility_mode="rear",
    )
    route_Bp = _simulate_leg_with_flips(
        geom=geom,
        start_xy=np.array(dest_xy, dtype=float),
        dest_xy=np.array(start_xy, dtype=float),
        TWD=float(TWD),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_bsp_kmh,
        twa_abs_ref=twa_abs,
        bsp_ref_kmph=bsp_ref_rear,
        start_tack_sign=-int(A_twa_sign),
        gybe_loss_s=float(gybe_loss_s),
        reach_m=float(reach_m),
        max_steps=int(max_steps),
        label="B'",
        leg_type=leg_type_u,
        include_events=True,
        feasibility_mode="rear",
    )

    for r in (route_A, route_B, route_Ap, route_Bp):
        r["group_id"] = str(group_id)
        r["leg_type"] = leg_type_u
        r["traj_name"] = f"{group_id} — {r.get('traj', r.get('label','?'))}"

    return {
        "group_id": str(group_id),
        "leg_type": leg_type_u,
        "start_xy": np.array(start_xy, dtype=float).copy(),
        "dest_xy": np.array(dest_xy, dtype=float).copy(),

        "TWA_vmg_abs_deg": float(twa_abs),
        "BSP_vmg_kmph": float(bsp_ref),
        "VMG_vmg_kmph": float(vmg_ref),

        "TWA_vmg_rear_abs_deg": float(twa_abs_rear),
        "BSP_vmg_rear_kmph": float(bsp_ref_rear),
        "VMG_vmg_rear_kmph": float(vmg_ref_rear),

        "routes": [route_A, route_B, route_Ap, route_Bp],
    }
