#URL = "https://data.sailgp.tech"
#ORG = "0c2a130d50b8facc"
#TOKEN = "2vTlG__z6bc7bibptc1FE_gXRwK6761dmxW_sasiAC1qsNqwbAbAj0PJD9yRIQPR0bfwdl_4-S_5gIecgkfz_Q=="
#BUCKET = "sailgp"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
from influxdb_client import InfluxDBClient

# Optionnel: couper le warning "MissingPivotFunction"
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)


def _parse_utc_datetime(s: str) -> datetime:
    """
    Accepte:
      - "2025-11-30T11:00:00Z"
      - "2025-11-30 11:00:00"
      - "2025-11-30T11:00:00"
    Retourne un datetime timezone-aware en UTC.
    """
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    s = s.replace("T", " ")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _load_influx_config():
    """
    Priorité:
      1) Variables d'environnement: URL, ORG, TOKEN, BUCKET
      2) Fichier .streamlit/secrets.toml (format Streamlit)
    """
    url = "https://data.sailgp.tech"
    org = "0c2a130d50b8facc"
    token = "2vTlG__z6bc7bibptc1FE_gXRwK6761dmxW_sasiAC1qsNqwbAbAj0PJD9yRIQPR0bfwdl_4-S_5gIecgkfz_Q=="
    bucket = "sailgp"


    if all([url, org, token, bucket]):
        return url, org, token, bucket

    # Fallback secrets Streamlit
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib  # py>=3.11
        except Exception:
            import tomli as tomllib  # pip install tomli si besoin

        with open(secrets_path, "rb") as f:
            sec = tomllib.load(f)

        url = url or sec.get("URL")
        org = org or sec.get("ORG")
        token = token or sec.get("TOKEN")
        bucket = bucket or sec.get("BUCKET")

    missing = [k for k, v in [("URL", url), ("ORG", org), ("TOKEN", token), ("BUCKET", bucket)] if not v]
    if missing:
        raise RuntimeError(
            "Configuration InfluxDB incomplète. Fournis URL/ORG/TOKEN/BUCKET via "
            "variables d'environnement ou via .streamlit/secrets.toml. Manquant: "
            + ", ".join(missing)
        )

    return url, org, token, bucket


def list_channels_at_time(
    url: str,
    org: str,
    token: str,
    bucket: str,
    center_utc: datetime,
    window_seconds: int = 300,
    level_regex: str = "strm",
) -> pd.DataFrame:
    """
    Retourne un DataFrame avec une colonne:
      - measurement (channel)

    Implémentation optimisée:
      - range sur une petite fenêtre
      - filtre _field == "value" + level regex
      - distinct sur _measurement (évite group+count lourd)
    """
    start = (center_utc - timedelta(seconds=window_seconds)).isoformat().replace("+00:00", "Z")
    stop = (center_utc + timedelta(seconds=window_seconds)).isoformat().replace("+00:00", "Z")

    flux = f'''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r.level =~ /{level_regex}/)
  |> keep(columns: ["_measurement"])
  |> distinct(column: "_measurement")
  |> sort(columns: ["_measurement"])
'''

    client = InfluxDBClient(
        url=url,
        token=token,
        org=org,
        verify_ssl=False,
        timeout=30_000,  # 30s (ms) - augmente la tolérance réseau
    )

    try:
        df = client.query_api().query_data_frame(org=org, query=flux)
    finally:
        client.close()

    # query_data_frame peut renvoyer une liste de DataFrame
    if isinstance(df, list):
        if len(df) == 0:
            return pd.DataFrame(columns=["measurement"])
        df = pd.concat(df, ignore_index=True)

    if df is None or df.empty:
        return pd.DataFrame(columns=["measurement"])

    # Normalisation colonne
    if "_measurement" in df.columns:
        df = df.rename(columns={"_measurement": "measurement"})
    elif "measurement" not in df.columns:
        # fallback ultra défensif
        cols = [c for c in df.columns if "measurement" in c.lower()]
        if cols:
            df = df.rename(columns={cols[0]: "measurement"})
        else:
            return pd.DataFrame(columns=["measurement"])

    df = df[["measurement"]].dropna().drop_duplicates().sort_values("measurement").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time",
        default="2025-11-30T11:00:00Z",
        help="Timestamp UTC (ex: 2025-11-30T11:00:00Z)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=300,
        help="Fenêtre +/- en secondes autour du timestamp (défaut: 300 = 5 min)",
    )
    parser.add_argument(
        "--level-regex",
        default="strm",
        help='Regex sur le tag "level" (défaut: strm). Ex: "strm|mdss|raw"',
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Chemin CSV de sortie (défaut: auto)",
    )
    args = parser.parse_args()

    center = _parse_utc_datetime(args.time)
    url, org, token, bucket = _load_influx_config()

    df = list_channels_at_time(
        url=url,
        org=org,
        token=token,
        bucket=bucket,
        center_utc=center,
        window_seconds=args.window_seconds,
        level_regex=args.level_regex,
    )

    stamp = center.strftime("%Y-%m-%dT%H%M%SZ")
    out = args.out or f"channels_{stamp}.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"[OK] {len(df)} channels trouvés sur bucket='{bucket}' autour de {center.isoformat()} -> {out}")


if __name__ == "__main__":
    main()
