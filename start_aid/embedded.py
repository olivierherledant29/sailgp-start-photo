from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from start_aid.geo import make_context_from_boundary, to_xy_marks_and_polys, compute_PI_xy
from start_aid.model import compute_all_geometry_and_times
from start_aid.viz import build_deck
from start_aid.polars import list_polar_files, load_polar_interpolator


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


def guess_twd_from_xml_name() -> float:
    """
    Guess TWD from the XML filename.

    Expected pattern: 3 digits starting with 0..3 appearing right before ".<digit(s)>"
    with optional spaces: e.g. "220 .6" or "220.75"
    """
    # Try a few likely keys (Start_Aid and Routeur may differ)
    candidates = [
        st.session_state.get("boundary_xml_name", None),
        st.session_state.get("xml_name", None),
        st.session_state.get("xml_filename", None),
        st.session_state.get("boundary_filename", None),
        st.session_state.get("uploaded_xml_name", None),
        st.session_state.get("last_uploaded_xml_name", None),
    ]

    # Some pipelines store it on the DF attrs
    try:
        df = st.session_state.get("boundary_df", None)
        if hasattr(df, "attrs"):
            candidates.append(df.attrs.get("source_name", None))
            candidates.append(df.attrs.get("filename", None))
    except Exception:
        pass

    name = next((x for x in candidates if isinstance(x, str) and x.strip()), "")
    if not name:
        return 0.0

    m = re.search(r"\b([0-3]\d{2})\s*\.\s*\d+", name)
    if not m:
        return 0.0

    try:
        x = int(m.group(1))
    except Exception:
        return 0.0

    return float(x) if 0 <= x <= 360 else 0.0



@st.cache_resource(show_spinner=False)
def _cached_polar_interpolator(abs_path_str: str):
    return load_polar_interpolator(abs_path_str)


def render_start_aid(boundary_df: pd.DataFrame, marks_df: pd.DataFrame):
    st.markdown(
        "<div style='padding:8px;border-radius:8px;border:1px solid #2E7D32;background:#E8F5E9;'>"
        "<b>Start Aid</b></div>",
        unsafe_allow_html=True,
    )

    # -------------------------
    # Session state defaults
    # -------------------------
    if "size_buffer_BDY_m" not in st.session_state:
        st.session_state["size_buffer_BDY_m"] = 15.0

    if "TWD_base" not in st.session_state:
        st.session_state["TWD_base"] = float(guess_twd_from_xml_name())

    if "offset_TWD" not in st.session_state:
        st.session_state["offset_TWD"] = 0

    if "TWS_base" not in st.session_state:
        st.session_state["TWS_base"] = 40.0

    if "offset_TWS" not in st.session_state:
        st.session_state["offset_TWS"] = 0

    if "TTS_intersection" not in st.session_state:
        st.session_state["TTS_intersection"] = 60.0

    # IMPORTANT: update TWD_base when XML changes (otherwise it stays at the old value)
    xml_name = st.session_state.get("boundary_xml_name", "")
    if "last_boundary_xml_name_start_aid" not in st.session_state:
        st.session_state["last_boundary_xml_name_start_aid"] = ""

    if isinstance(xml_name, str) and xml_name and (xml_name != st.session_state["last_boundary_xml_name_start_aid"]):
        st.session_state["TWD_base"] = float(guess_twd_from_xml_name())
        st.session_state["last_boundary_xml_name_start_aid"] = xml_name

    # -------------------------
    # Sidebar: offsets + TTS_intersection (moved here)
    # -------------------------
    with st.sidebar:
        st.subheader("Offsets")

        offset_TWD = st.slider(
            "offset_TWD (°)",
            min_value=-30,
            max_value=30,
            value=int(st.session_state.get("offset_TWD", 0)),
            step=2,
        )
        st.session_state["offset_TWD"] = int(offset_TWD)

        tws_base_for_bounds = float(st.session_state.get("TWS_base", 40.0))
        min_off_tws = int(max(-10, -int(tws_base_for_bounds)))  # if base<10 => clamp to -base
        offset_TWS = st.slider(
            "offset_TWS (km/h)",
            min_value=min_off_tws,
            max_value=10,
            value=int(st.session_state.get("offset_TWS", 0)),
            step=1,
        )
        st.session_state["offset_TWS"] = int(offset_TWS)

        st.subheader("Temps")
        TTS_intersection = st.number_input(
            "TTS_intersection (s)",
            min_value=0,
            max_value=300,
            value=int(round(float(st.session_state["TTS_intersection"]))),
            step=1,
            format="%d",
        )
        st.session_state["TTS_intersection"] = float(TTS_intersection)

    # -------------------------
    # Compact widget layout (3 per row)
    # -------------------------
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        size_buffer_BDY = st.number_input(
            "Buffer boundary (m)",
            min_value=0,
            max_value=200,
            value=int(round(float(st.session_state["size_buffer_BDY_m"]))),
            step=1,
            format="%d",
        )
    with r1c2:
        PI_m = st.number_input("PI (m)", min_value=-500, max_value=500, value=60, step=1, format="%d")
    with r1c3:
        M_lost = st.number_input("M_lost (s)", min_value=0, max_value=120, value=12, step=1, format="%d")

    st.session_state["size_buffer_BDY_m"] = float(size_buffer_BDY)

    # Row with TWD_base / TWS_base / (empty) because TTS moved to sidebar
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        TWD_base = st.number_input(
            "TWD_base (°)",
            min_value=0,
            max_value=360,
            value=int(round(float(st.session_state["TWD_base"]))),
            step=1,
            format="%d",
        )
    with r2c2:
        TWS_base = st.number_input(
            "TWS_base (km/h)",
            min_value=0,
            max_value=200,
            value=int(round(float(st.session_state["TWS_base"]))),
            step=1,
            format="%d",
        )
    with r2c3:
        st.empty()

    st.session_state["TWD_base"] = float(TWD_base)
    st.session_state["TWS_base"] = float(TWS_base)

    # take TTS from sidebar state
    TTS_intersection = float(st.session_state["TTS_intersection"])

    TWD = (float(TWD_base) + float(st.session_state["offset_TWD"])) % 360.0
    TWS_ref_kmh = float(max(0.0, float(TWS_base) + float(st.session_state["offset_TWS"])))

    st.caption(
        f"TWD_base={int(TWD_base)}° — offset_TWD={int(st.session_state['offset_TWD']):+d}° — **TWD appliqué={int(TWD)}°**  |  "
        f"TWS_base={int(TWS_base)} — offset_TWS={int(st.session_state['offset_TWS']):+d} — **TWS appliqué={int(TWS_ref_kmh)} km/h**  |  "
        f"TTS_intersection={int(TTS_intersection)} s"
    )

    # Angles + Start point
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        TWA_port = st.number_input("TWA_port (°)", min_value=40, max_value=160, value=60, step=1, format="%d")
    with r3c2:
        TWA_UW = st.number_input("TWA_UW (°)", min_value=30, max_value=80, value=45, step=1, format="%d")
    with r3c3:
        X_percent = st.slider(
            "SP : X% depuis SL2 vers SL1",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
        )

    # -------------------------
    # Polaires
    # -------------------------
    st.subheader("Polaires")

    polars_dir = Path(__file__).resolve().parent / "polars"
    polar_files = list_polar_files(polars_dir)

    if not polar_files:
        st.warning(f"Aucun fichier de polaires trouvé dans: {polars_dir}")
        polar_path = None
        polar_interp = None
        pol_BAB_pct = 100
        pol_TRIB_pct = 100
    else:
        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            polar_name = st.selectbox("Fichier de polaires", options=polar_files, index=0)
        polar_path = polars_dir / polar_name
        polar_interp = _cached_polar_interpolator(str(polar_path.resolve()))

        with r4c2:
            pol_BAB_pct = st.slider("%_pol_BAB", min_value=0, max_value=120, value=100, step=1)
        with r4c3:
            pol_TRIB_pct = st.slider("%_pol_TRIB", min_value=0, max_value=120, value=100, step=1)

    if polar_interp is not None:
        bsp_approche_pol = float(polar_interp(float(TWS_ref_kmh), float(TWA_port)))
        bsp_approche_bab = bsp_approche_pol * (float(pol_BAB_pct) / 100.0)
        st.caption(
            f"BSP_approche_pol: **{bsp_approche_pol:.1f} km/h** — BSP_approche_BAB: **{bsp_approche_bab:.1f} km/h**"
        )

    # Target TTK (default 15s)
    target_TTK_beforeTack = st.number_input(
        "target_TTK_beforeTack (s)",
        min_value=-300,
        max_value=300,
        value=15,
        step=1,
        format="%d",
    )

    boundary_latlon = boundary_df_to_latlon(boundary_df)
    marks_ll = marks_df_to_marks_ll(marks_df)

    if not boundary_latlon:
        st.info("Start Aid : boundary non chargée (XML manquant).")
        return None, None

    if "SL1" not in marks_ll or "SL2" not in marks_ll:
        st.info("Start Aid : SL1 / SL2 indisponibles dans les marques Influx.")
        return None, None

    if polar_interp is None:
        st.info("Start Aid : polaire non chargée.")
        return None, None

    ctx = make_context_from_boundary(boundary_latlon)

    geom = to_xy_marks_and_polys(ctx, marks_ll, boundary_latlon, float(size_buffer_BDY))
    PI_xy = compute_PI_xy(geom["SL1_xy"], geom["SL2_xy"], float(PI_m))

    out = compute_all_geometry_and_times(
        ctx=ctx,
        geom=geom,
        PI_xy=PI_xy,
        TWD=float(TWD),
        TWA_port=float(TWA_port),
        TWA_UW=float(TWA_UW),
        TWS_ref_kmh=float(TWS_ref_kmh),
        polar_bsp_kmh=polar_interp,   # callable
        pol_BAB_pct=float(pol_BAB_pct),
        pol_TRIB_pct=float(pol_TRIB_pct),
        M_lost=float(M_lost),
        X_percent=float(X_percent),
        TTS_intersection=float(TTS_intersection),
        target_TTK_beforeTack=float(target_TTK_beforeTack),
    )

    out["size_buffer_BDY_m"] = float(size_buffer_BDY)
    out["polar_file"] = str(polar_path) if polar_path is not None else None

    out["TWD_base"] = float(TWD_base)
    out["offset_TWD"] = float(st.session_state["offset_TWD"])
    out["TWS_base"] = float(TWS_base)
    out["offset_TWS"] = float(st.session_state["offset_TWS"])
    out["TTS_intersection"] = float(TTS_intersection)
    out["target_TTK_beforeTack"] = float(target_TTK_beforeTack)

    deck = build_deck(ctx, geom, PI_xy, out)
    return deck, out
