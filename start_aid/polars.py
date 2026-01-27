from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ----------------------------
# Polar file discovery
# ----------------------------
def _default_polars_dir() -> Path:
    """
    Convention: store polars under start_aid/polars/ (repo-root relative).
    This module lives in start_aid/, so its parent is the package dir.
    """
    return Path(__file__).resolve().parent / "polars"


def list_polar_files(polars_dir: str | Path | None = None, exts: Sequence[str] = (".csv",)) -> list[str]:
    """
    Returns relative filenames (no directory) found in polars_dir.
    Used to populate the Streamlit selectbox.
    """
    d = Path(polars_dir) if polars_dir is not None else _default_polars_dir()
    if not d.exists() or not d.is_dir():
        return []
    out: list[str] = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in set(e.lower() for e in exts):
            out.append(p.name)
    return out


# ----------------------------
# Polar loading / interpolation
# ----------------------------
def load_polars_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Reads a polar CSV:
      - First row: TWS values (km/h) starting from column 2
      - First col:  TWA values (deg) starting from row 2
      - Interior:   BSP (km/h)

    Returns a clean table indexed by TWA with float TWS columns.
    """
    p = Path(csv_path)
    df_raw = pd.read_csv(p, header=None)

    if df_raw.shape[0] < 3 or df_raw.shape[1] < 3:
        raise ValueError(f"Polar file too small: {p}")

    # First row contains TWS headers starting at col 1
    tws = pd.to_numeric(df_raw.iloc[0, 1:], errors="coerce").to_numpy(float)
    # First col contains TWA values starting at row 1
    twa = pd.to_numeric(df_raw.iloc[1:, 0], errors="coerce").to_numpy(float)

    # BSP grid
    grid = df_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    # Drop any fully-NaN rows/cols
    valid_twa = np.isfinite(twa)
    valid_tws = np.isfinite(tws)
    twa = twa[valid_twa]
    tws = tws[valid_tws]
    grid = grid[valid_twa, :][:, valid_tws]

    table = pd.DataFrame(grid, index=twa, columns=tws)
    table.index.name = "TWA_deg"
    table.columns.name = "TWS_kmh"
    return table.sort_index(axis=0).sort_index(axis=1)


@dataclass(frozen=True)
class PolarInterpolator:
    """
    Pickle-friendly polar interpolator (for Streamlit cache_data).
    Bilinear interpolation over (TWS, TWA) grid; clamps to bounds.
    """
    tws_grid: tuple[float, ...]
    twa_grid: tuple[float, ...]
    bsp_grid: tuple[tuple[float, ...], ...]  # row-major [twa_i][tws_j]

    def __call__(self, tws_kmh: float, twa_deg: float) -> float:
        tws = float(tws_kmh)
        twa = float(twa_deg)

        tws_arr = np.asarray(self.tws_grid, dtype=float)
        twa_arr = np.asarray(self.twa_grid, dtype=float)
        bsp = np.asarray(self.bsp_grid, dtype=float)

        if tws_arr.size < 2 or twa_arr.size < 2:
            return float("nan")

        # Clamp
        tws = float(np.clip(tws, tws_arr.min(), tws_arr.max()))
        twa = float(np.clip(twa, twa_arr.min(), twa_arr.max()))

        # Find surrounding indices
        j1 = int(np.searchsorted(tws_arr, tws, side="right"))
        i1 = int(np.searchsorted(twa_arr, twa, side="right"))
        j0 = max(0, min(j1 - 1, tws_arr.size - 2))
        i0 = max(0, min(i1 - 1, twa_arr.size - 2))
        j1 = j0 + 1
        i1 = i0 + 1

        x0, x1 = tws_arr[j0], tws_arr[j1]
        y0, y1 = twa_arr[i0], twa_arr[i1]

        # Protect against duplicate axes
        tx = 0.0 if abs(x1 - x0) < 1e-9 else (tws - x0) / (x1 - x0)
        ty = 0.0 if abs(y1 - y0) < 1e-9 else (twa - y0) / (y1 - y0)

        z00 = bsp[i0, j0]
        z10 = bsp[i0, j1]
        z01 = bsp[i1, j0]
        z11 = bsp[i1, j1]

        # If any corner is NaN, fall back to nearest-neighbor among finite corners
        corners = np.array([z00, z10, z01, z11], dtype=float)
        if not np.isfinite(corners).all():
            # nearest in (tx, ty) space
            pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
            finite = np.isfinite(corners)
            if not finite.any():
                return float("nan")
            d = np.sum((pts[finite] - np.array([tx, ty])) ** 2, axis=1)
            return float(corners[finite][int(np.argmin(d))])

        # Bilinear
        z0 = z00 * (1 - tx) + z10 * tx
        z1 = z01 * (1 - tx) + z11 * tx
        return float(z0 * (1 - ty) + z1 * ty)


def make_polar_interpolator(table: pd.DataFrame) -> PolarInterpolator:
    tws = tuple(float(x) for x in table.columns.to_numpy(float))
    twa = tuple(float(x) for x in table.index.to_numpy(float))
    bsp = tuple(tuple(float(v) for v in row) for row in table.to_numpy(float))
    return PolarInterpolator(tws_grid=tws, twa_grid=twa, bsp_grid=bsp)


def load_polar_interpolator(csv_path: str | Path) -> PolarInterpolator:
    table = load_polars_csv(csv_path)
    return make_polar_interpolator(table)


# Backward-compatible alias (some modules used this earlier name)
load_polar_interpolator = load_polar_interpolator  # explicit export


def polar_bsp_kmh(table_or_interp, tws_kmh: float, twa_deg: float) -> float:
    """
    Convenience helper: accepts either a PolarInterpolator or a DataFrame.
    """
    if isinstance(table_or_interp, PolarInterpolator):
        return float(table_or_interp(tws_kmh, twa_deg))
    if isinstance(table_or_interp, pd.DataFrame):
        return float(make_polar_interpolator(table_or_interp)(tws_kmh, twa_deg))
    return float("nan")
