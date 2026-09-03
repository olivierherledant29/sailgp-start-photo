from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ALL_BOATS = [
    "AUS", "BRA", "CAN", "DEN", "ESP", "FRA", "GBR",
    "GER", "ITA", "JPN", "NZL", "SUI", "SWE", "USA",
]

DEFAULT_TS_HOST = "tsdb.sailgp.tech"
DEFAULT_TS_PORT = 5432
DEFAULT_TS_DB = "sailgp"
DEFAULT_TS_USER = "sailgp_team_fra"


@dataclass(frozen=True)
class TimescaleCfg:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslcert: str
    sslkey: str
    sslrootcert: str


def _resolve_path(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return str(p.resolve())


def _secret_value(name: str, default=None):
    """
    Read a value from environment first, then Streamlit secrets.
    This keeps local .env behavior unchanged while supporting Streamlit Cloud.
    """
    env_val = os.getenv(name)
    if env_val not in (None, ""):
        return env_val

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return default


def _looks_like_pem(value: str) -> bool:
    s = str(value or "")
    return "-----BEGIN " in s and "-----END " in s


def _materialize_pem(value: str, filename: str) -> str:
    """
    Write PEM content to a private temp file for libpq/psycopg2.
    PostgreSQL expects sslcert/sslkey as file paths, not raw PEM strings.
    """
    tmp_dir = Path("/tmp/sailgp_timescale_ssl")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    path = tmp_dir / filename
    content = str(value).replace("\\n", "\n").strip() + "\n"
    path.write_text(content, encoding="utf-8")

    # Private key must not be group/world-readable for libpq.
    if filename.endswith(".key"):
        try:
            path.chmod(0o600)
        except Exception:
            pass

    return str(path)


def _resolve_ssl_material(
    path_or_pem_name: str,
    pem_name: str,
    default_path: str,
    temp_filename: str,
) -> str:
    """
    Resolution order:
      1) TIMESCALE_SSL_*_PEM from env / st.secrets
      2) TIMESCALE_SSL_* from env / st.secrets:
         - PEM content -> /tmp file
         - existing path -> use path
      3) local default certs/client.* path
    """
    pem_value = _secret_value(pem_name)
    if pem_value:
        return _materialize_pem(str(pem_value), temp_filename)

    value = _secret_value(path_or_pem_name, default_path)
    if value and _looks_like_pem(str(value)):
        return _materialize_pem(str(value), temp_filename)

    resolved = _resolve_path(str(value or default_path))
    if Path(resolved).exists():
        return resolved

    raise FileNotFoundError(
        f"Matériel SSL Timescale introuvable pour {path_or_pem_name}. "
        f"Chemin testé : {resolved}. "
        f"Sur Streamlit Cloud, ajoute {pem_name} dans st.secrets."
    )


def get_cfg() -> TimescaleCfg:
    password = _secret_value("TIMESCALE_PASSWORD")
    if not password:
        raise RuntimeError(
            "TIMESCALE_PASSWORD manquant dans .env / Streamlit secrets"
        )

    sslcert = _resolve_ssl_material(
        path_or_pem_name="TIMESCALE_SSL_CERT",
        pem_name="TIMESCALE_SSL_CERT_PEM",
        default_path="certs/client.crt",
        temp_filename="client.crt",
    )
    sslkey = _resolve_ssl_material(
        path_or_pem_name="TIMESCALE_SSL_KEY",
        pem_name="TIMESCALE_SSL_KEY_PEM",
        default_path="certs/client.key",
        temp_filename="client.key",
    )

    return TimescaleCfg(
        host=str(_secret_value("TIMESCALE_HOST", DEFAULT_TS_HOST)),
        port=int(_secret_value("TIMESCALE_PORT", str(DEFAULT_TS_PORT))),
        dbname=str(_secret_value("TIMESCALE_DB", DEFAULT_TS_DB)),
        user=str(_secret_value("TIMESCALE_USER", DEFAULT_TS_USER)),
        password=str(password),
        sslcert=sslcert,
        sslkey=sslkey,
        sslrootcert=certifi.where(),
    )


def get_connection(cfg: TimescaleCfg):
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        sslmode="verify-full",
        sslcert=cfg.sslcert,
        sslkey=cfg.sslkey,
        sslrootcert=cfg.sslrootcert,
        connect_timeout=15,
    )


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_level(level_expr: str) -> str:
    s = str(level_expr or "").lower()
    if "strm" in s:
        return "strm"
    if "mdss" in s:
        return "mdss"
    return "strm"


def _parse_every(every: str) -> str:
    s = str(every or "1s").strip().lower()
    m = re.fullmatch(r"(\d+)\s*(ms|s|m|h)", s)
    if not m:
        return "1 second"

    n = int(m.group(1))
    unit = m.group(2)
    names = {
        "ms": "millisecond",
        "s": "second",
        "m": "minute",
        "h": "hour",
    }
    name = names[unit]
    if n != 1:
        name += "s"
    return f"{n} {name}"


def _aggregate_expr(agg_fn: str) -> str:
    agg = str(agg_fn or "mean").lower()
    if agg == "last":
        return "last(value, time)"
    if agg == "max":
        return "max(value)"
    if agg == "min":
        return "min(value)"
    return "avg(value)"


def _execute_df(cfg: TimescaleCfg, sql: str, params: tuple) -> pd.DataFrame:
    conn = None
    try:
        conn = get_connection(cfg)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [d.name for d in cur.description] if cur.description else []
        return pd.DataFrame(rows, columns=cols)
    except psycopg2.errors.QueryCanceled as e:
        raise RuntimeError(
            "Requête Timescale annulée par le serveur (timeout 5 min). "
            "Réduis la fenêtre ou vérifie les filtres time/level/boat/channel."
        ) from e
    finally:
        if conn is not None:
            conn.close()


@st.cache_data(show_spinner=False, ttl=60)
def load_channels_timeseries(
    cfg: TimescaleCfg,
    boats: list[str],
    channels: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    every: str = "1s",
    level_expr: str = "strm|mdss|mdss_fast|raw",
    agg_fn: str = "mean",
) -> pd.DataFrame:
    """
    Remplacement Timescale de influx_io.load_channels_timeseries().

    Sortie :
        time_utc, boat, <channel1>, <channel2>, ...
    """
    if not boats or not channels:
        return pd.DataFrame()

    level = _normalize_level(level_expr)
    bucket = _parse_every(every)
    value_expr = _aggregate_expr(agg_fn)

    sql = f"""
        SELECT
            time_bucket(%s::interval, time) AS time_utc,
            boat,
            channel,
            {value_expr} AS value
        FROM sgp_telemetry
        WHERE level = %s
          AND boat = ANY(%s)
          AND channel = ANY(%s)
          AND time >= %s
          AND time < %s
        GROUP BY time_utc, boat, channel
        ORDER BY time_utc, boat, channel
    """

    long_df = _execute_df(
        cfg,
        sql,
        (
            bucket,
            level,
            [str(x) for x in boats],
            [str(x) for x in channels],
            _utc(start_utc),
            _utc(stop_utc),
        ),
    )

    if long_df.empty:
        return pd.DataFrame(columns=["time_utc", "boat", *channels])

    long_df["time_utc"] = pd.to_datetime(long_df["time_utc"], utc=True, errors="coerce")
    long_df["boat"] = long_df["boat"].astype(str)
    long_df["channel"] = long_df["channel"].astype(str)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    wide = (
        long_df
        .pivot_table(
            index=["time_utc", "boat"],
            columns="channel",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    for ch in channels:
        if ch not in wide.columns:
            wide[ch] = np.nan

    return (
        wide[["time_utc", "boat", *channels]]
        .dropna(subset=["time_utc", "boat"])
        .sort_values(["boat", "time_utc"])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False, ttl=300)
def query_mean_by_boat(
    cfg: TimescaleCfg,
    measurement: str,
    boats: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    level_expr: str,
) -> pd.DataFrame:
    if not boats:
        return pd.DataFrame(columns=["boat", "value"])

    sql = """
        SELECT boat, avg(value) AS value
        FROM sgp_telemetry
        WHERE level = %s
          AND boat = ANY(%s)
          AND channel = %s
          AND time >= %s
          AND time < %s
        GROUP BY boat
        ORDER BY boat
    """

    df = _execute_df(
        cfg,
        sql,
        (
            _normalize_level(level_expr),
            [str(x) for x in boats],
            str(measurement),
            _utc(start_utc),
            _utc(stop_utc),
        ),
    )

    if df.empty:
        return pd.DataFrame(columns=["boat", "value"])

    df["boat"] = df["boat"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["boat", "value"]].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=300)
def query_last_by_boat(
    cfg: TimescaleCfg,
    measurement: str,
    boats: list[str],
    start_utc: datetime,
    stop_utc: datetime,
    level_expr: str,
) -> pd.DataFrame:
    if not boats:
        return pd.DataFrame(columns=["boat", "value"])

    sql = """
        SELECT boat, last(value, time) AS value
        FROM sgp_telemetry
        WHERE level = %s
          AND boat = ANY(%s)
          AND channel = %s
          AND time >= %s
          AND time < %s
        GROUP BY boat
        ORDER BY boat
    """

    df = _execute_df(
        cfg,
        sql,
        (
            _normalize_level(level_expr),
            [str(x) for x in boats],
            str(measurement),
            _utc(start_utc),
            _utc(stop_utc),
        ),
    )

    if df.empty:
        return pd.DataFrame(columns=["boat", "value"])

    df["boat"] = df["boat"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["boat", "value"]].reset_index(drop=True)
