from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from xml_boundary import parse_course_limit_xml, parse_marks_xml


LATEST_XML_JSON_URL = "https://xml.sailgp.tech/latest/json"

BOUNDARY_COLUMNS = ["ring", "seq", "lat", "lon"]
MARK_COLUMNS = ["mark", "lat", "lon"]

KNOWN_MARKS = {
    "SL1", "SL2",
    "M1",
    "LG1", "LG2",
    "WG1", "WG2",
    "FL1", "FL2",
}


def _empty_boundary() -> pd.DataFrame:
    return pd.DataFrame(columns=BOUNDARY_COLUMNS)


def _empty_marks() -> pd.DataFrame:
    return pd.DataFrame(columns=MARK_COLUMNS)


def _normalise_mark_name(name: str) -> str:
    """
    Normalise les marques virtuelles SailGP vers les noms historiques
    attendus par Start Aid / Routeur.

        VSL1 -> SL1
        VSL2 -> SL2
        VM1  -> M1
        VLG1 -> LG1
        VLG2 -> LG2
        VWG1 -> WG1
        VWG2 -> WG2
        VFL1 -> FL1
        VFL2 -> FL2
    """
    raw = str(name or "").strip().upper()

    if raw in KNOWN_MARKS:
        return raw

    if raw.startswith("V") and raw[1:] in KNOWN_MARKS:
        return raw[1:]

    return raw


@st.cache_data(show_spinner=False, ttl=10)
def _fetch_latest_xml_json() -> dict:
    response = requests.get(
        LATEST_XML_JSON_URL,
        timeout=10,
        headers={"User-Agent": "sailgp-streamlit-app/1.0"},
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Réponse Last XML invalide : objet JSON attendu.")

    return payload


def _iter_features(payload: dict):
    """
    Le endpoint renvoie :
        {"t": "...", "v": [FeatureCollection, FeatureCollection, ...]}
    """
    collections = payload.get("v", [])
    if not isinstance(collections, list):
        return

    for collection in collections:
        if not isinstance(collection, dict):
            continue

        features = collection.get("features", [])
        if not isinstance(features, list):
            continue

        for feature in features:
            if isinstance(feature, dict):
                yield feature


def _parse_last_xml_json(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convertit /latest/json vers exactement les DataFrames déjà utilisés
    par les pages existantes :

        boundary_df : ring, seq, lat, lon
        marks_df    : mark, lat, lon
    """
    boundary_points = []
    marks = {}

    for feature in _iter_features(payload):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}

        geom_type = str(geometry.get("type") or "")
        feature_type = str(properties.get("type") or "")
        name = str(properties.get("name") or "").strip()

        # ----------------------------------------------------------
        # Boundary
        # ----------------------------------------------------------
        if (
            geom_type == "LineString"
            and feature_type.lower() == "boundary"
            and name == "Boundary"
        ):
            coords = geometry.get("coordinates") or []

            for seq, xy in enumerate(coords):
                if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue

                try:
                    lon = float(xy[0])
                    lat = float(xy[1])
                except (TypeError, ValueError):
                    continue

                boundary_points.append(
                    {
                        "ring": 0,
                        "seq": seq,
                        "lat": lat,
                        "lon": lon,
                    }
                )

        # ----------------------------------------------------------
        # Marks
        # ----------------------------------------------------------
        if geom_type == "Point" and feature_type.lower() == "mark":
            coords = geometry.get("coordinates") or []
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue

            mark = _normalise_mark_name(name)
            if mark not in KNOWN_MARKS:
                continue

            try:
                lon = float(coords[0])
                lat = float(coords[1])
            except (TypeError, ValueError):
                continue

            # Le JSON peut contenir plusieurs fois LG/WG.
            # Une entrée par nom est suffisante.
            marks[mark] = {
                "mark": mark,
                "lat": lat,
                "lon": lon,
            }

    if boundary_points:
        boundary_df = pd.DataFrame(boundary_points, columns=BOUNDARY_COLUMNS)
        boundary_df = (
            boundary_df
            .dropna(subset=["lat", "lon"])
            .reset_index(drop=True)
        )
    else:
        boundary_df = _empty_boundary()

    if marks:
        preferred_order = [
            "SL1", "SL2", "M1",
            "LG1", "LG2",
            "WG1", "WG2",
            "FL1", "FL2",
        ]
        order = {name: i for i, name in enumerate(preferred_order)}

        marks_df = pd.DataFrame(list(marks.values()), columns=MARK_COLUMNS)
        marks_df["_order"] = marks_df["mark"].map(order).fillna(999)
        marks_df = (
            marks_df
            .sort_values(["_order", "mark"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    else:
        marks_df = _empty_marks()

    return boundary_df, marks_df


def _latest_metadata(payload: dict) -> dict:
    meta = {}

    timestamp = payload.get("t")
    if timestamp:
        meta["timestamp"] = timestamp

    collections = payload.get("v", [])
    if isinstance(collections, list):
        for collection in collections:
            if not isinstance(collection, dict):
                continue
            props = collection.get("properties")
            if isinstance(props, dict) and props:
                meta.update(props)
                break

    return meta


def sidebar_boundary_uploader() -> pd.DataFrame:
    """
    Source commune de parcours pour Start Aid et Routeur.

    Deux modes :
      - Upload XML : comportement historique inchangé
      - Last XML   : lecture automatique de xml.sailgp.tech/latest/json

    marks_df reste stocké dans st.session_state pour compatibilité totale
    avec les pages existantes.
    """
    st.markdown(
        "<div style='padding:8px;border-radius:8px;"
        "border:1px solid #7B1FA2;background:#F3E5F5;'>"
        "<b>Parcours commun – Boundary + Marks</b></div>",
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Source du parcours",
        ["Upload XML", "Last XML"],
        index=0,
        key="boundary_course_source",
    )

    # ==============================================================
    # MODE HISTORIQUE : UPLOAD XML
    # ==============================================================
    if source == "Upload XML":
        xml_file = st.file_uploader(
            "Boundary XML (CourseLimit name='Boundary')",
            type=["xml"],
            accept_multiple_files=False,
        )

        if xml_file is None:
            st.session_state.pop("marks_df", None)
            st.session_state.pop("boundary_xml_name", None)
            st.session_state.pop("boundary_source", None)
            return _empty_boundary()

        st.session_state["boundary_xml_name"] = getattr(xml_file, "name", None)
        st.session_state["boundary_source"] = "upload_xml"

        xml_bytes = xml_file.getvalue()

        try:
            boundary_df = parse_course_limit_xml(xml_bytes)
        except Exception as e:
            st.error(f"Erreur parsing Boundary XML : {e}")
            boundary_df = _empty_boundary()

        try:
            marks_df = parse_marks_xml(xml_bytes)
        except Exception as e:
            st.error(f"Erreur parsing marks XML : {e}")
            marks_df = _empty_marks()

        st.session_state["marks_df"] = marks_df

        if marks_df is not None and not marks_df.empty:
            st.caption(
                "Marques : "
                + ", ".join(marks_df["mark"].astype(str).tolist())
            )

        return boundary_df

    # ==============================================================
    # MODE LAST XML
    # ==============================================================
    st.caption("Source : xml.sailgp.tech/latest/json")

    try:
        payload = _fetch_latest_xml_json()
        boundary_df, marks_df = _parse_last_xml_json(payload)
        meta = _latest_metadata(payload)

        st.session_state["marks_df"] = marks_df
        st.session_state["boundary_xml_name"] = "Last XML"
        st.session_state["boundary_source"] = "last_xml"

        race_id = meta.get("RaceID")
        race_start = meta.get("RaceStartTime")
        creation_time = meta.get("CreationTimeDate")
        endpoint_time = meta.get("timestamp")

        info = []
        if race_id:
            info.append(f"Race {race_id}")
        if race_start:
            info.append(f"Start {race_start}")
        if creation_time:
            info.append(f"Créé {creation_time}")
        elif endpoint_time:
            info.append(f"Updated {endpoint_time}")

        if info:
            st.caption(" • ".join(info))

        if boundary_df.empty:
            st.warning("Last XML : Boundary non trouvée.")
        else:
            st.success(f"Last XML : Boundary chargée ({len(boundary_df)} points)")

        if marks_df.empty:
            st.warning("Last XML : aucune marque reconnue.")
        else:
            st.caption(
                "Marques : "
                + ", ".join(marks_df["mark"].astype(str).tolist())
            )

        return boundary_df

    except Exception as e:
        st.session_state["marks_df"] = _empty_marks()
        st.session_state["boundary_xml_name"] = "Last XML"
        st.session_state["boundary_source"] = "last_xml"

        st.error(f"Impossible de charger Last XML : {e}")
        return _empty_boundary()
