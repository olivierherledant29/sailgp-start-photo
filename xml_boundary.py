# xml_boundary.py
from __future__ import annotations

import xml.etree.ElementTree as ET
import pandas as pd


def _maybe_scale_latlon(lat: float, lon: float) -> tuple[float, float]:
    # Certains XML sont en degrés, d'autres en 1e7 (comme Influx)
    if abs(lat) > 90.0 or abs(lon) > 180.0:
        return lat / 1e7, lon / 1e7
    return lat, lon


def parse_course_limit_xml(xml_bytes: bytes) -> pd.DataFrame:
    """
    Trace uniquement CourseLimit[@name='Boundary'].
    Retourne df: ring, seq, lat, lon
    """
    root = ET.fromstring(xml_bytes)

    pts: list[tuple[int, float, float]] = []

    # STRICT: uniquement CourseLimit name="Boundary"
    for cl in root.findall(".//CourseLimit"):
        if (cl.attrib.get("name") or "").strip() != "Boundary":
            continue
        for el in cl.findall("./Limit"):
            try:
                seq = int(el.attrib.get("SeqID", "0"))
                lat = float(el.attrib["Lat"])
                lon = float(el.attrib["Lon"])
                lat, lon = _maybe_scale_latlon(lat, lon)
                pts.append((seq, lat, lon))
            except Exception:
                continue

        # On prend le premier CourseLimit Boundary trouvé
        if pts:
            break

    if not pts:
        return pd.DataFrame(columns=["ring", "seq", "lat", "lon"])

    df = pd.DataFrame(pts, columns=["seq", "lat", "lon"]).dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Split en rings si SeqID repart en arrière/égal (segments multiples)
    ring = 0
    rings = []
    prev_seq = None
    for s in df["seq"].tolist():
        if prev_seq is not None and s <= prev_seq:
            ring += 1
        rings.append(ring)
        prev_seq = s
    df.insert(0, "ring", rings)

    # Supprimer doublons consécutifs exacts
    df["_key"] = df["lat"].round(10).astype(str) + "|" + df["lon"].round(10).astype(str)
    keep = [True]
    for i in range(1, len(df)):
        keep.append(df.loc[i, "_key"] != df.loc[i - 1, "_key"])
    df = df.loc[keep].drop(columns=["_key"]).reset_index(drop=True)

    return df


def parse_marks_xml(xml_bytes: bytes) -> pd.DataFrame:
    """Parse mark positions (e.g. SL1/SL2/M1) from the SailGP XML.

    Expected structure (common in SailGP exports):
      <Course> ... <Mark Name="SL1" TargetLat="..." TargetLng="..." ... />

    Returns a DataFrame with columns: mark, lat, lon (float).
    If no marks found, returns empty DataFrame with those columns.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return pd.DataFrame(columns=["mark", "lat", "lon"])

    rows = []
    for m in root.findall(".//Mark"):
        name = (m.attrib.get("Name") or m.attrib.get("name") or "").strip()
        if not name:
            continue

        # Common attribute names
        lat_s = m.attrib.get("TargetLat") or m.attrib.get("lat") or m.attrib.get("Lat")
        lon_s = m.attrib.get("TargetLng") or m.attrib.get("lng") or m.attrib.get("Lon") or m.attrib.get("LonDeg")

        try:
            lat = float(lat_s) if lat_s is not None else float("nan")
            lon = float(lon_s) if lon_s is not None else float("nan")
        except Exception:
            lat, lon = float("nan"), float("nan")

        if not (pd.notna(lat) and pd.notna(lon)):
            continue

        rows.append({"mark": name, "lat": lat, "lon": lon})

    if not rows:
        return pd.DataFrame(columns=["mark", "lat", "lon"])

    df = pd.DataFrame(rows)
    # Keep first occurrence for duplicates
    df = df.dropna(subset=["mark", "lat", "lon"]).drop_duplicates(subset=["mark"], keep="first").reset_index(drop=True)
    return df
