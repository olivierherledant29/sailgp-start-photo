import math
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from pyproj import Transformer, CRS


def utm_crs_for_latlon(lat: float, lon: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_transformers(lat: float, lon: float):
    crs_utm = utm_crs_for_latlon(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    to_wgs = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    return to_utm, to_wgs, crs_utm


def ll_to_xy(to_utm: Transformer, lat: float, lon: float):
    x, y = to_utm.transform(lon, lat)
    return x, y


def xy_to_ll(to_wgs: Transformer, x: float, y: float):
    lon, lat = to_wgs.transform(x, y)
    return lat, lon


def make_context_from_boundary(boundary_latlon):
    centroid_lat = float(np.mean([p[0] for p in boundary_latlon]))
    centroid_lon = float(np.mean([p[1] for p in boundary_latlon]))
    to_utm, to_wgs, crs_utm = make_transformers(centroid_lat, centroid_lon)
    return {
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "to_utm": to_utm,
        "to_wgs": to_wgs,
        "crs_utm": crs_utm,
    }


def to_xy_marks_and_polys(ctx, marks_ll, boundary_latlon, size_buffer_BDY: float):
    to_utm, to_wgs = ctx["to_utm"], ctx["to_wgs"]

    SL1_ll, SL2_ll, M1_ll = marks_ll["SL1"], marks_ll["SL2"], marks_ll["M1"]
    SL1_xy = np.array(ll_to_xy(to_utm, *SL1_ll), dtype=float)
    SL2_xy = np.array(ll_to_xy(to_utm, *SL2_ll), dtype=float)
    M1_xy = np.array(ll_to_xy(to_utm, *M1_ll), dtype=float)

    boundary_xy = [ll_to_xy(to_utm, lat, lon) for (lat, lon) in boundary_latlon]
    poly_BDY = Polygon(boundary_xy)
    if not poly_BDY.is_valid:
        poly_BDY = poly_BDY.buffer(0)
    if poly_BDY.area <= 0:
        raise ValueError("Invalid Boundary polygon (area <= 0).")

    poly_buffer = poly_BDY.buffer(-float(size_buffer_BDY))
    if poly_buffer.is_empty or poly_buffer.area <= 0:
        poly_buffer = None

    return {
        "SL1_ll": SL1_ll, "SL2_ll": SL2_ll, "M1_ll": M1_ll,
        "SL1_xy": SL1_xy, "SL2_xy": SL2_xy, "M1_xy": M1_xy,
        "poly_BDY": poly_BDY,
        "poly_buffer": poly_buffer,
    }


def heading_to_unit_vector(heading_deg: float):
    r = math.radians(heading_deg % 360.0)
    return np.array([math.sin(r), math.cos(r)], dtype=float)


def compute_PI_xy(SL1_xy, SL2_xy, PI_m: float):
    v = np.array(SL1_xy) - np.array(SL2_xy)
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        raise ValueError("SL1 and SL2 are coincident or too close.")
    u = v / L
    return np.array(SL2_xy) + u * float(PI_m)


def intersection_ray_with_polygon_boundary(ray: LineString, origin_xy: np.ndarray, poly: Polygon, dir_ray: np.ndarray):
    inter = ray.intersection(poly.boundary)
    if inter.is_empty:
        return None

    pts = []
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
        pts = [g for g in inter.geoms if g.geom_type == "Point"]
    elif inter.geom_type in ("LineString", "MultiLineString"):
        try:
            coords = list(inter.coords)
            pts = [Point(coords[0]), Point(coords[-1])]
        except Exception:
            pts = []
    else:
        return None

    candidates = []
    for p in pts:
        v = np.array([p.x, p.y]) - origin_xy
        if float(np.dot(v, dir_ray)) > 1e-6:
            candidates.append((float(np.linalg.norm(v)), p))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def line_infinite_through_points(A: np.ndarray, B: np.ndarray, scale: float = 50000.0):
    v = B - A
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return LineString([tuple(A), tuple(B)])
    u = v / L
    P0 = A - u * scale
    P1 = A + u * scale
    return LineString([tuple(P0), tuple(P1)])


def dashed_paths_from_linestring_xy(line: LineString, dash_m: float = 40.0, gap_m: float = 30.0):
    total = line.length
    if total <= 1e-6:
        return []
    out = []
    s = 0.0
    step = dash_m + gap_m
    while s < total:
        s_end = min(s + dash_m, total)
        p0 = line.interpolate(s)
        p1 = line.interpolate(s_end)
        out.append([(p0.x, p0.y), (p1.x, p1.y)])
        s += step
    return out


def dashed_paths_from_polygon(poly: Polygon, dash_m: float = 25.0, gap_m: float = 15.0):
    ring = LineString(list(poly.exterior.coords))
    return dashed_paths_from_linestring_xy(ring, dash_m=dash_m, gap_m=gap_m)


def xy_path_to_lonlat_path(path_xy, to_wgs: Transformer):
    out = []
    for x, y in path_xy:
        lat, lon = xy_to_ll(to_wgs, x, y)
        out.append([lon, lat])
    return out


def polygon_exterior_to_lonlat_path(poly: Polygon, to_wgs: Transformer):
    coords = list(poly.exterior.coords)
    out = []
    for x, y in coords:
        lat, lon = xy_to_ll(to_wgs, x, y)
        out.append([lon, lat])
    return out


def label_offset_ll(to_utm: Transformer, to_wgs: Transformer, lat: float, lon: float, east_m: float, north_m: float):
    x, y = ll_to_xy(to_utm, lat, lon)
    return xy_to_ll(to_wgs, x + east_m, y + north_m)


def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])
