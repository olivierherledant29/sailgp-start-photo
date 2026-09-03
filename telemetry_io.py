from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

import influx_io
import timescale_io


# Keep the same boat list exposed by influx_io so existing pages can import
# ALL_BOATS from telemetry_io without any other code change.
ALL_BOATS = influx_io.ALL_BOATS

_VALID_BACKENDS = {"influx", "timescale"}


def get_backend() -> str:
    """
    Backend selected in .env:

        TELEMETRY_BACKEND=influx
    or
        TELEMETRY_BACKEND=timescale

    Default = influx, so existing deployments remain safe until explicitly
    switched to Timescale.
    """
    backend = os.getenv("TELEMETRY_BACKEND", "influx").strip().lower()

    if backend not in _VALID_BACKENDS:
        raise RuntimeError(
            f"TELEMETRY_BACKEND invalide : {backend!r}. "
            "Valeurs autorisées : influx, timescale."
        )

    return backend


def get_cfg():
    """
    Return the configuration object corresponding to the selected backend.
    """
    if get_backend() == "timescale":
        return timescale_io.get_cfg()
    return influx_io.get_cfg()


def load_channels_timeseries(
    cfg,
    boats: list[str],
    channels: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    every: str = "1s",
    level_expr: str = "strm|mdss|mdss_fast|raw",
    agg_fn: str = "mean",
) -> pd.DataFrame:
    """
    Backend-independent wrapper.

    Output contract is the same for both backends:
        time_utc, boat, <channel1>, <channel2>, ...

    This allows pages to switch from Influx to Timescale without changing
    their plotting/filtering logic.
    """
    if get_backend() == "timescale":
        return timescale_io.load_channels_timeseries(
            cfg=cfg,
            boats=boats,
            channels=channels,
            start_utc=start_utc,
            stop_utc=stop_utc,
            every=every,
            level_expr=level_expr,
            agg_fn=agg_fn,
        )

    return influx_io.load_channels_timeseries(
        cfg=cfg,
        boats=boats,
        channels=channels,
        start_utc=start_utc,
        stop_utc=stop_utc,
        every=every,
        level_expr=level_expr,
        agg_fn=agg_fn,
    )


def query_mean_by_boat(
    cfg,
    measurement: str,
    boats: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    level_expr: str,
) -> pd.DataFrame:
    if get_backend() == "timescale":
        return timescale_io.query_mean_by_boat(
            cfg=cfg,
            measurement=measurement,
            boats=boats,
            start_utc=start_utc,
            stop_utc=stop_utc,
            level_expr=level_expr,
        )

    return influx_io.query_mean_by_boat(
        cfg=cfg,
        measurement=measurement,
        boats=boats,
        start_utc=start_utc,
        stop_utc=stop_utc,
        level_expr=level_expr,
    )


def query_last_by_boat(
    cfg,
    measurement: str,
    boats: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    level_expr: str,
) -> pd.DataFrame:
    if get_backend() == "timescale":
        return timescale_io.query_last_by_boat(
            cfg=cfg,
            measurement=measurement,
            boats=boats,
            start_utc=start_utc,
            stop_utc=stop_utc,
            level_expr=level_expr,
        )

    return influx_io.query_last_by_boat(
        cfg=cfg,
        measurement=measurement,
        boats=boats,
        start_utc=start_utc,
        stop_utc=stop_utc,
        level_expr=level_expr,
    )
