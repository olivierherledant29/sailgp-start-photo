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
    max_idx = max(
        [
            TARGET_COL_STATE_IDX,
            TARGET_COL_DB_IDX,
            TARGET_COL_RUDDER_IDX,
            TARGET_COL_WING_IDX,
            TARGET_COL_CONFIG_IDX,
            TARGET_COL_TWS_IDX,
            *target_columns.values(),
        ]
    )

    base_cols = [
        "state",
        "db",
        "rudder",
        "wing_num",
        "config",
        "TWS",
        *target_columns.keys(),
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

    for name, idx in target_columns.items():
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
    d = clean_df[clean_df["config"].astype(str) == str(config)].copy()
    d = d.dropna(subset=["TWS"]).sort_values("TWS")

    out = {name: np.nan for name in target_names}
    out.update({"TWS_min": np.nan, "TWS_max": np.nan})

    if d.empty or not np.isfinite(tws_mean):
        return out

    x_all = d["TWS"].astype(float).to_numpy()
    out["TWS_min"] = float(np.nanmin(x_all))
    out["TWS_max"] = float(np.nanmax(x_all))

    for col in target_names:
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
        "target_clean_by_mode": {m: pd.DataFrame() for m in modes},
        "selected_sheet_by_mode": {m: None for m in modes},
        "selected_config": None,
        "selected_state": None,
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
    config_clean_df_all_states = target_sheet_to_clean_df(
        target_dict[config_sheet],
        target_columns,
    )

    available_states = get_available_states(config_clean_df_all_states)
    default_state = (
        "Foiling"
        if "Foiling" in available_states
        else (available_states[0] if available_states else None)
    )

    with st.sidebar:
        selected_state = (
            st.selectbox(
                "State target",
                available_states,
                index=available_states.index(default_state)
                if default_state in available_states
                else 0,
                key=f"{page_key}_target_state",
            )
            if available_states
            else None
        )

        if (
            np.isfinite(tws_mean)
            and tws_mean < 18
            and selected_state
            and selected_state.strip().lower() == "foiling"
        ):
            st.error(
                "TWS moyen < 18 : les targets sont en Foiling par défaut, "
                "mais tu peux changer le State target."
            )

    result["selected_state"] = selected_state

    config_clean_df = filter_target_state(config_clean_df_all_states, selected_state)

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
        else:
            selected_config = None
            st.warning("Aucune config trouvée dans le fichier targets pour ce State.")

    result["selected_config"] = selected_config

    for mode in modes:
        sheet = pick_sheet_for_mode(available_sheets, mode)
        if sheet is None:
            continue

        clean_df = target_sheet_to_clean_df(target_dict[sheet], target_columns)
        clean_df = filter_target_state(clean_df, selected_state)

        result["selected_sheet_by_mode"][mode] = sheet
        result["target_clean_by_mode"][mode] = clean_df

        if selected_config and not clean_df.empty:
            result["target_by_mode"][mode] = interp_target(
                clean_df,
                selected_config,
                float(tws_mean),
                target_names,
            )

    return result