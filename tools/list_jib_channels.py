from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

URL = os.getenv("URL", "https://data.sailgp.tech")
ORG = os.getenv("ORG", "0c2a130d50b8facc")
BUCKET = os.getenv("BUCKET", "sailgp")
TOKEN = os.getenv("SailGP_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "Token manquant : vérifie SailGP_TOKEN dans ton fichier .env"
    )

client = InfluxDBClient(
    url=URL,
    token=TOKEN,
    org=ORG,
    verify_ssl=False,
)

query = f'''
import "influxdata/influxdb/schema"

schema.measurements(bucket: "{BUCKET}")
  |> filter(fn: (r) => r._value =~ /(?i)jib/)
  |> sort(columns: ["_value"])
'''

df = client.query_api().query_data_frame(
    org=ORG,
    query=query,
)

if df is None:
    print("Aucun résultat.")
    raise SystemExit

if isinstance(df, list):
    df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()

if df.empty:
    print("Aucun channel contenant 'jib' ou 'Jib' trouvé.")
    raise SystemExit

channels = (
    df["_value"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print("\nChannels contenant 'jib' ou 'Jib' :\n")

for ch in channels:
    print(ch)

print(f"\nTotal : {len(channels)} channels")