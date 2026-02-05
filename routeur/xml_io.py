from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from shapely.geometry import Polygon
except Exception:
    Polygon = None


# ----------------------------
# Helpers
# ----------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    m = _NUM_RE.search(s.replace(",", "."))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _get_lat_lon_from_elem(e: ET.Element) -> Optional[Tuple[float, float]]:
    """
    Returns (lat, lon) if element contains coordinates.
    Supports:
      - attributes lat/lon or latitude/longitude
      - child tags <lat>, <lon> etc.
    """
    # Attributes
    lat = _to_float(e.attrib.get("lat", None) or e.attrib.get("latitude", None))
    lon = _to_float(e.attrib.get("lon", None) or e.attrib.get("lng", None) or e.attrib.get("longitude", None))

    # Some XML store in 1e-7 (integers)
    if lat is not None and lon is not None:
        if abs(lat) > 90 or abs(lon) > 180:
            lat *= 1e-7
            lon *= 1e-7
        if abs(lat) <= 90 and abs(lon) <= 180:
            return (float(lat), float(lon))

    # Child tags
    lat_child = None
    lon_child = None
    for c in list(e):
        tag = (c.tag or "").lower()
        if tag.endswith("lat") or "latitude" in tag:
            lat_child = _to_float(c.text)
        if tag.endswith("lon") or tag.endswith("lng") or "longitude" in tag:
            lon_child = _to_float(c.text)

    if lat_child is not None and lon_child is not None:
        lat, lon = float(lat_child), float(lon_child)
        if abs(lat) > 90 or abs(lon) > 180:
            lat *= 1e-7
            lon *= 1e-7
        if abs(lat) <= 90 and abs(lon) <= 180:
            return (lat, lon)

    return None


def _collect_points_under(elem: ET.Element, max_depth: int = 6) -> List[Tuple[float, float]]:
    """
    Collect all (lat, lon) points found under elem (inclusive), limited depth for speed.
    """
    out: List[Tuple[float, float]] = []

    def rec(e: ET.Element, depth: int):
        if depth > max_depth:
            return
        ll = _get_lat_lon_from_elem(e)
        if ll is not None:
            out.append(ll)
        for c in list(e):
            rec(c, depth + 1)

    rec(elem, 0)

    # Remove exact duplicates while keeping order
    seen = set()
    uniq = []
    for lat, lon in out:
        key = (round(lat, 10), round(lon, 10))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((lat, lon))
    return uniq


def _score_boundary(points: List[Tuple[float, float]]) -> float:
    """
    Score candidate boundary: prefer many points + valid polygon area.
    """
    n = len(points)
    if n < 3:
        return -1.0

    if Polygon is None:
        return float(n)

    try:
        # Use lon,lat ordering for shapely; close ring if needed
        coords = [(p[1], p[0]) for p in points]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        area = float(poly.area) if poly.is_valid else 0.0
        if area <= 0:
            return float(n) * 0.1
        return float(n) + 1e6 * area
    except Exception:
        return float(n) * 0.1


# ----------------------------
# Main API
# ----------------------------
def parse_course_xml(xml_bytes: bytes) -> Tuple[List[Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """
    Returns:
      boundary_latlon: list[(lat, lon)]
      marks_ll: dict mark_name -> (lat, lon)

    This parser is intentionally tolerant:
      - boundary tags differ between events/files
      - marks may be nested with different naming conventions
    """
    # Robust decode/parse
    root = None
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            text = xml_bytes.decode(enc, errors="strict")
            root = ET.fromstring(text)
            break
        except Exception as e:
            last_err = e
            continue
    if root is None:
        # fallback: ET can parse bytes directly sometimes
        try:
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            raise ValueError(f"XML parse error: {e}") from e

    # ---------- 1) Try to find boundary candidates by tag heuristics
    boundary_candidates: List[List[Tuple[float, float]]] = []

    boundary_keywords = ("boundary", "limit", "polygon", "course", "area", "zone")
    for elem in root.iter():
        t = (elem.tag or "").lower()
        if any(k in t for k in boundary_keywords):
            pts = _collect_points_under(elem, max_depth=8)
            if len(pts) >= 3:
                boundary_candidates.append(pts)

    # ---------- 2) Fallback: scan whole doc for point clusters
    if not boundary_candidates:
        # Collect points for each subtree and keep the largest
        for elem in root.iter():
            pts = _collect_points_under(elem, max_depth=4)
            if len(pts) >= 6:  # boundary usually many points
                boundary_candidates.append(pts)

    boundary_latlon: List[Tuple[float, float]] = []
    if boundary_candidates:
        boundary_candidates.sort(key=_score_boundary, reverse=True)
        boundary_latlon = boundary_candidates[0]

    # ---------- 3) Marks extraction (tolerant)
    marks_of_interest = {"SL1","SL2","M1","WG1","WG2","LG1","LG2","FL1","FL2"}
    marks_ll: Dict[str, Tuple[float, float]] = {}

    # Heuristic: elements with name/id/mark attributes, or tag containing "mark"
    for elem in root.iter():
        tag = (elem.tag or "").lower()

        name = (
            elem.attrib.get("mark")
            or elem.attrib.get("name")
            or elem.attrib.get("id")
            or elem.attrib.get("label")
        )
        if name is None:
            # sometimes <Mark><Name>SL1</Name>...
            for c in list(elem):
                ct = (c.tag or "").lower()
                if ct.endswith("name") or "label" in ct:
                    if c.text and c.text.strip():
                        name = c.text.strip()
                        break

        if not name:
            continue

        name = str(name).strip()
        if name not in marks_of_interest and "mark" not in tag:
            continue

        ll = _get_lat_lon_from_elem(elem)
        if ll is None:
            # try children
            pts = _collect_points_under(elem, max_depth=4)
            ll = pts[0] if pts else None

        if ll is not None and name in marks_of_interest:
            marks_ll[name] = ll

    # ---------- 4) Hard-fail messages
    if not boundary_latlon:
        # provide some diagnostics without dumping the whole XML
        raise ValueError(
            "Boundary introuvable dans le XML (parser routeur). "
            "Le fichier semble ne pas contenir de bloc identifié 'boundary/limit/polygon/course', "
            "ou les points ne sont pas au format lat/lon attendu."
        )

    return boundary_latlon, marks_ll
