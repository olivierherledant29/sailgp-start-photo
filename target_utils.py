from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_TARGETS_PATH = Path("targets/Targets_S6_updatebeforeNY.xlsx")

TARGET_COL_STATE_IDX = 0    # A
TARGET_COL_DB_IDX = 1       # B
TARGET_COL_RUDDER_IDX = 2   # C
TARGET_COL_WING_IDX = 3     # D
TARGET_COL_CONFIG_IDX = 4   # E
TARGET_COL_TWS_IDX = 5      # F
TARGET_COL_TWA_IDX = 8      # I

CH_WING_CONFIG = "WING_CONFIG_unk"
CH_DB_SEL = "MD4_SEL_DB_unk"
CH_RUD_SEL = "MD4_SEL_RUD_unk"

WING_CONFIG_MAP = {9: "HAW", 11: "APW", 15: "LAW", 143: "LAW2"}
WING_TARGET_NUM_MAP = {9: 18.0, 11: 24.0, 15: 28.0, 143: 27.5}

DB_CONFIG_MAP = {1: "LAB", 4: "LAB2", 2: "HSB", 3: "HSB2"}

RUD_CONFIG_MAP = {
    1: "LARW",
    4: "LARW2",
    2: "HSRW",
    3: "HSRW2",
}

TARGET_STATE_COLORS = {
    "Foiling": "#ff69b4",   # rose
    "H1": "#c49a6c",        # marron clair
}

DEFAULT_TARGET_COLOR = "#000000"


def target_config_channels() -> list[str]:
    return [CH_WING_CONFIG, CH_DB_SEL, CH_RUD_SEL]


@st.cache_data(show_spinner=False)
def load_target_workbook_from_path(path_str: str) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path_str, engine="openpyxl")
    return {
        sheet: pd.read_excel(path_str, sheet_name=sheet, engine="openpyxl")
        for sheet in xls.sheet_names
    }


@st.cache_data(show_spinner=False)
def load_target_workbook_from_bytes(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    bio = BytesIO(file_bytes)
    xls = pd.ExcelFile(bio, engine="openpyxl")

    out = {}
    for sheet in xls.sheet_names:
        bio.seek(0)
        out[sheet] = pd.read_excel(bio, sheet_name=sheet, engine="openpyxl")
    return out


def load_target_workbook(
    target_source: str,
    uploaded_file: Any | None,
    default_path: Path = DEFAULT_TARGETS_PATH,
) -> dict[str, pd.DataFrame] | None:
    if target_source == "Default file":
        if default_path.exists():
            return load_target_workbook_from_path(str(default_path))
        st.warning(f"Fichier targets par défaut introuvable : {default_path}")
        return None

    if uploaded_file is not None:
        return load_target_workbook_from_bytes(uploaded_file.getvalue())

    return None


def pick_sheet_for_mode(sheet_names: list[str], mode_name: str) -> str | None:
    if not sheet_names:
        return None

    mode_lower = mode_name.lower()

    exact = [s for s in sheet_names if s.lower() == mode_lower]
    if exact:
        return exact[0]

    contains = [s for s in sheet_names if mode_lower in s.lower()]
    if contains:
        return contains[0]

    return sheet_names[0]


def _norm(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper().replace(",", ".").replace(" ", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _state_color(state: str | None) -> str:
    if state is None:
        return DEFAULT_TARGET_COLOR

    for key, color in TARGET_STATE_COLORS.items():
        if str(state).strip().lower() == key.lower():
            return color

    return DEFAULT_TARGET_COLOR


def _first_valid_value(df: pd.DataFrame, boat: str, col: str) -> float:
    d = df[df["boat"].astype(str) == str(boat)].copy()
    if d.empty or col not in d.columns:
        return np.nan

    d = d.sort_values("time_utc")
    s = pd.to_numeric(d[col], errors="coerce").dropna()
    if s.empty:
        return np.nan

    return float(s.iloc[0])


def decode_auto_config_inputs(df_raw: pd.DataFrame, ref_boat: str = "FRA") -> dict:
    wing_code = _first_valid_value(df_raw, ref_boat, CH_WING_CONFIG)
    db_code = _first_valid_value(df_raw, ref_boat, CH_DB_SEL)
    rud_code = _first_valid_value(df_raw, ref_boat, CH_RUD_SEL)

    wing_code_int = int(round(wing_code)) if np.isfinite(wing_code) else None
    db_code_int = int(round(db_code)) if np.isfinite(db_code) else None
    rud_code_int = int(round(rud_code)) if np.isfinite(rud_code) else None

    return {
        "wing_code": wing_code,
        "db_code": db_code,
        "rud_code": rud_code,
        "wing": WING_CONFIG_MAP.get(wing_code_int),
        "wing_target_num": WING_TARGET_NUM_MAP.get(wing_code_int),
        "db": DB_CONFIG_MAP.get(db_code_int),
        "rudder": RUD_CONFIG_MAP.get(rud_code_int),
    }


def target_sheet_to_clean_df(
    df: pd.DataFrame,
    target_columns: dict[str, int],
) -> pd.DataFrame:
    # Force TWA_target to be available for VMG recommendation.
    effective_target_columns = dict(target_columns)
    if "TWA_target" not in effective_target_columns:
        effective_target_columns["TWA_target"] = TARGET_COL_TWA_IDX

    max_idx = max(
        [
            TARGET_COL_STATE_IDX,
            TARGET_COL_DB_IDX,
            TARGET_COL_RUDDER_IDX,
            TARGET_COL_WING_IDX,
            TARGET_COL_CONFIG_IDX,
            TARGET_COL_TWS_IDX,
            *effective_target_columns.values(),
        ]
    )

    base_cols = [
        "state",
        "db",
        "rudder",
        "wing_num",
        "config",
        "TWS",
        *effective_target_columns.keys(),
    ]

    if df is None or df.empty or df.shape[1] <= max_idx:
        return pd.DataFrame(columns=base_cols)

    out = pd.DataFrame(
        {
            "state": df.iloc[:, TARGET_COL_STATE_IDX].astype(str).str.strip(),
            "db": df.iloc[:, TARGET_COL_DB_IDX].astype(str).str.strip(),
            "rudder": df.iloc[:, TARGET_COL_RUDDER_IDX].astype(str).str.strip(),
            "wing_num": pd.to_numeric(
                df.iloc[:, TARGET_COL_WING_IDX]
                .astype(str)
                .str.replace(",", ".", regex=False),
                errors="coerce",
            ),
            "config": df.iloc[:, TARGET_COL_CONFIG_IDX].astype(str).str.strip(),
            "TWS": pd.to_numeric(df.iloc[:, TARGET_COL_TWS_IDX], errors="coerce"),
        }
    )

    for name, idx in effective_target_columns.items():
        out[name] = pd.to_numeric(df.iloc[:, idx], errors="coerce")
        if name.lower() in {"ca1_target", "twist_target"}:
            out[name] = out[name].abs()

    out = out.dropna(subset=["config", "TWS"])
    out = out[out["config"].str.lower().ne("nan")]
    out = out[out["state"].str.lower().ne("nan")]
    out = out.sort_values(["state", "config", "TWS"]).reset_index(drop=True)
    return out


def get_available_states(clean_df: pd.DataFrame) -> list[str]:
    if clean_df is None or clean_df.empty or "state" not in clean_df.columns:
        return []

    states = (
        clean_df["state"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return states


def filter_target_state(clean_df: pd.DataFrame, state: str | None) -> pd.DataFrame:
    if clean_df is None or clean_df.empty or "state" not in clean_df.columns or not state:
        return clean_df

    return clean_df[
        clean_df["state"].astype(str).str.strip().str.lower()
        == str(state).strip().lower()
    ].reset_index(drop=True)


def find_default_config_from_targets(
    clean_df: pd.DataFrame,
    auto: dict,
) -> tuple[str | None, str]:
    if clean_df is None or clean_df.empty:
        return None, "error"

    if auto.get("wing_target_num") is None or auto.get("db") is None:
        return None, "error"

    needed = {"db", "rudder", "wing_num", "config"}
    if not needed.issubset(clean_df.columns):
        return None, "error"

    d = clean_df.copy()
    wing_target = float(auto["wing_target_num"])
    wing_match = np.isclose(
        pd.to_numeric(d["wing_num"], errors="coerce"),
        wing_target,
        atol=0.01,
    )

    if auto.get("rudder") is not None:
        exact_mask = (
            d["db"].apply(_norm).eq(_norm(auto["db"]))
            & d["rudder"].apply(_norm).eq(_norm(auto["rudder"]))
            & wing_match
        )

        exact_hit = d[exact_mask]
        if not exact_hit.empty:
            return str(exact_hit.iloc[0]["config"]), "exact"

    fallback_mask = d["db"].apply(_norm).eq(_norm(auto["db"])) & wing_match
    fallback_hit = d[fallback_mask]

    if not fallback_hit.empty:
        return str(fallback_hit.iloc[0]["config"]), "fallback"

    return None, "error"


def interp_target(
    clean_df: pd.DataFrame,
    config: str,
    tws_mean: float,
    target_names: list[str],
) -> dict:
    names = list(dict.fromkeys([*target_names, "TWA_target"]))

    d = clean_df[clean_df["config"].astype(str) == str(config)].copy()
    d = d.dropna(subset=["TWS"]).sort_values("TWS")

    out = {name: np.nan for name in names}
    out.update({"TWS_min": np.nan, "TWS_max": np.nan})

    if d.empty or not np.isfinite(tws_mean):
        return out

    x_all = d["TWS"].astype(float).to_numpy()
    out["TWS_min"] = float(np.nanmin(x_all))
    out["TWS_max"] = float(np.nanmax(x_all))

    for col in names:
        if col not in d.columns:
            out[col] = np.nan
            continue

        ydf = d[["TWS", col]].dropna()
        if len(ydf) == 0:
            out[col] = np.nan
            continue

        xx = ydf["TWS"].astype(float).to_numpy()
        yy = ydf[col].astype(float).to_numpy()

        if len(xx) == 1:
            out[col] = float(yy[0])
        else:
            out[col] = float(np.interp(float(tws_mean), xx, yy))

    return out


def _vmg_from_target(target: dict) -> float:
    bsp = target.get("BSP_target", np.nan)
    twa = target.get("TWA_target", np.nan)

    if not np.isfinite(bsp) or not np.isfinite(twa):
        return np.nan

    return float(bsp * np.cos(np.deg2rad(twa)))


def _is_tws_inside_target_range(clean_df_state_config: pd.DataFrame, tws_mean: float) -> bool:
    if clean_df_state_config is None or clean_df_state_config.empty:
        return False

    tws_values = pd.to_numeric(clean_df_state_config["TWS"], errors="coerce").dropna()
    if tws_values.empty or not np.isfinite(tws_mean):
        return False

    return float(tws_values.min()) <= float(tws_mean) <= float(tws_values.max())


def recommend_best_vmg_mode(
    clean_df_all_states: pd.DataFrame,
    auto_inputs: dict,
    tws_mean: float,
    sailing_mode: str,
    target_names: list[str],
) -> dict:
    """
    Recommends the target mode that gives the best target VMG for a given sheet/mode.

    Eligibility:
    - target mode must have rows matching DB / rudder / wing, or fallback DB / wing;
    - TWS mean must be inside the available TWS range for that target mode/config;
    - target must contain BSP_target and TWA_target.

    Selection:
    - UW: maximum positive VMG = BSP_target * cos(TWA_target)
    - DW: most negative VMG
    - Reaching/other: maximum absolute VMG by fallback convention.
    """
    out = {
        "mode": None,
        "config": None,
        "vmg": np.nan,
        "target": None,
        "status": "error",
        "eligible": [],
    }

    if clean_df_all_states is None or clean_df_all_states.empty:
        return out

    available_modes = get_available_states(clean_df_all_states)
    if not available_modes:
        return out

    candidates = []

    for target_mode in available_modes:
        df_state = filter_target_state(clean_df_all_states, target_mode)
        if df_state.empty:
            continue

        config, config_status = find_default_config_from_targets(df_state, auto_inputs)
        if not config:
            continue

        df_state_config = df_state[df_state["config"].astype(str) == str(config)].copy()
        if not _is_tws_inside_target_range(df_state_config, tws_mean):
            continue

        target = interp_target(df_state, config, float(tws_mean), target_names)
        vmg = _vmg_from_target(target)

        if not np.isfinite(vmg):
            continue

        candidates.append(
            {
                "mode": target_mode,
                "config": config,
                "config_status": config_status,
                "vmg": float(vmg),
                "target": target,
                "TWS_min": target.get("TWS_min", np.nan),
                "TWS_max": target.get("TWS_max", np.nan),
            }
        )

    out["eligible"] = candidates

    if not candidates:
        return out

    mode_upper = str(sailing_mode).upper()

    if mode_upper == "UW":
        positives = [c for c in candidates if c["vmg"] > 0]
        pool = positives if positives else candidates
        best = max(pool, key=lambda c: c["vmg"])
    elif mode_upper == "DW":
        negatives = [c for c in candidates if c["vmg"] < 0]
        pool = negatives if negatives else candidates
        best = min(pool, key=lambda c: c["vmg"])
    else:
        best = max(candidates, key=lambda c: abs(c["vmg"]))

    out.update(
        {
            "mode": best["mode"],
            "config": best["config"],
            "vmg": best["vmg"],
            "target": best["target"],
            "status": "ok",
        }
    )
    return out


def build_single_target_overlay(
    clean_df_all_states: pd.DataFrame,
    selected_mode: str | None,
    selected_config: str | None,
    tws_mean: float,
    target_names: list[str],
) -> list[dict]:
    if not selected_mode or not selected_config:
        return []

    df_mode = filter_target_state(clean_df_all_states, selected_mode)
    if df_mode is None or df_mode.empty:
        return []

    target = interp_target(
        df_mode,
        selected_config,
        float(tws_mean),
        target_names,
    )

    return [
        {
            "state": selected_mode,
            "mode": selected_mode,
            "config": selected_config,
            "color": _state_color(selected_mode),
            "target": target,
        }
    ]


def build_targets_for_modes(
    *,
    df_raw: pd.DataFrame,
    ref_boat: str,
    target_dict: dict[str, pd.DataFrame] | None,
    target_columns: dict[str, int],
    target_names: list[str],
    tws_mean: float,
    page_key: str,
    modes: list[str] = ["UW", "DW"],
) -> dict:
    result = {
        "target_by_mode": {m: None for m in modes},
        "target_overlays_by_mode": {m: [] for m in modes},
        "target_clean_by_mode": {m: pd.DataFrame() for m in modes},
        "selected_sheet_by_mode": {m: None for m in modes},
        "recommended_mode_by_mode": {m: None for m in modes},
        "recommended_config_by_mode": {m: None for m in modes},
        "recommended_vmg_by_mode": {m: np.nan for m in modes},
        "selected_config": None,
        "selected_state": None,
        "selected_mode": None,
        "displayed_target_states": [],
        "displayed_target_modes": [],
        "auto_config": None,
        "auto_status": "error",
        "auto_inputs": decode_auto_config_inputs(df_raw, ref_boat),
    }

    if not target_dict:
        return result

    available_sheets = list(target_dict.keys())
    if not available_sheets:
        return result

    config_sheet = pick_sheet_for_mode(available_sheets, "UW") or available_sheets[0]
    config_clean_df_all_modes = target_sheet_to_clean_df(
        target_dict[config_sheet],
        target_columns,
    )

    available_target_modes = get_available_states(config_clean_df_all_modes)
    default_target_mode = (
        "Foiling"
        if "Foiling" in available_target_modes
        else (available_target_modes[0] if available_target_modes else None)
    )

    with st.sidebar:
        selected_target_mode = (
            st.selectbox(
                "Mode target",
                available_target_modes,
                index=available_target_modes.index(default_target_mode)
                if default_target_mode in available_target_modes
                else 0,
                key=f"{page_key}_target_mode",
            )
            if available_target_modes
            else None
        )

    result["selected_state"] = selected_target_mode  # backward compatibility
    result["selected_mode"] = selected_target_mode
    result["displayed_target_states"] = [selected_target_mode] if selected_target_mode else []
    result["displayed_target_modes"] = [selected_target_mode] if selected_target_mode else []

    config_clean_df = filter_target_state(config_clean_df_all_modes, selected_target_mode)

    configs = sorted(config_clean_df["config"].dropna().astype(str).unique().tolist())
    auto_config, auto_status = find_default_config_from_targets(
        config_clean_df,
        result["auto_inputs"],
    )

    result["auto_config"] = auto_config
    result["auto_status"] = auto_status

    with st.sidebar:
        if configs:
            config_mode = st.radio(
                "Mode config target",
                ["Détection auto config", "Forçage manuel config"],
                index=0,
                key=f"{page_key}_config_mode",
            )

            if config_mode == "Détection auto config":
                selected_config = auto_config if auto_config in configs else configs[0]
                st.selectbox(
                    "Config target détectée",
                    configs,
                    index=configs.index(selected_config),
                    key=f"{page_key}_target_config_auto_display",
                    disabled=True,
                )
            else:
                default_idx = configs.index(auto_config) if auto_config in configs else 0
                selected_config = st.selectbox(
                    "Config target",
                    configs,
                    index=default_idx,
                    key=f"{page_key}_target_config_manual",
                )

            if auto_status == "exact":
                st.caption(
                    f"Auto config DB : wing={result['auto_inputs']['wing']} "
                    f"({result['auto_inputs']['wing_target_num']}) / "
                    f"DB={result['auto_inputs']['db']} / "
                    f"rudder={result['auto_inputs']['rudder']} → {auto_config}"
                )
            elif auto_status == "fallback":
                st.warning(
                    "Config hybride non présente dans les targets, config proche sélectionnée : "
                    f"wing={result['auto_inputs']['wing']} "
                    f"({result['auto_inputs']['wing_target_num']}) / "
                    f"DB={result['auto_inputs']['db']} / "
                    f"rudder={result['auto_inputs']['rudder']} → {auto_config}"
                )
            else:
                st.error(
                    "Erreur détection auto de la config : "
                    f"wing={result['auto_inputs']['wing']} "
                    f"({result['auto_inputs']['wing_target_num']}) / "
                    f"DB={result['auto_inputs']['db']} / "
                    f"rudder={result['auto_inputs']['rudder']}"
                )

            if selected_target_mode:
                st.caption(f"Target affichée : {selected_target_mode}")
        else:
            selected_config = None
            st.warning("Aucune config trouvée dans le fichier targets pour ce Mode.")

    result["selected_config"] = selected_config

    for mode in modes:
        sheet = pick_sheet_for_mode(available_sheets, mode)
        if sheet is None:
            continue

        clean_df_all_modes = target_sheet_to_clean_df(target_dict[sheet], target_columns)
        clean_df = filter_target_state(clean_df_all_modes, selected_target_mode)

        result["selected_sheet_by_mode"][mode] = sheet
        result["target_clean_by_mode"][mode] = clean_df

        recommendation = recommend_best_vmg_mode(
            clean_df_all_modes,
            result["auto_inputs"],
            float(tws_mean),
            mode,
            target_names,
        )

        if recommendation.get("status") == "ok":
            result["recommended_mode_by_mode"][mode] = recommendation.get("mode")
            result["recommended_config_by_mode"][mode] = recommendation.get("config")
            result["recommended_vmg_by_mode"][mode] = recommendation.get("vmg")

        if selected_config and not clean_df.empty:
            target = interp_target(
                clean_df,
                selected_config,
                float(tws_mean),
                target_names,
            )

            result["target_by_mode"][mode] = target

            result["target_overlays_by_mode"][mode] = build_single_target_overlay(
                clean_df_all_modes,
                selected_target_mode,
                selected_config,
                float(tws_mean),
                target_names,
            )

    with st.sidebar:
        warnings = []
        for mode in modes:
            recommended_mode = result["recommended_mode_by_mode"].get(mode)
            recommended_vmg = result["recommended_vmg_by_mode"].get(mode)

            if recommended_mode and _norm(recommended_mode) != _norm("Foiling"):
                vmg_txt = ""
                if np.isfinite(recommended_vmg):
                    vmg_txt = f" — VMG target {recommended_vmg:.2f}"
                warnings.append(
                    f"{mode} : Foiling n'est pas le meilleur mode VMG. "
                    f"Mode conseillé : {recommended_mode}{vmg_txt}."
                )

        if warnings:
            st.warning("\n\n".join(warnings))

    return result
