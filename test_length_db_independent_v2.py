from __future__ import annotations

from datetime import datetime, timedelta
import re
import pandas as pd
from dotenv import load_dotenv

from influx_io import get_cfg, _query_data_frame_safe


CHANNELS = ["LENGTH_DB_H_P_mm", "LENGTH_DB_H_S_mm"]


def build_query(bucket: str, ch: str, boat: str, start_time: datetime, end_time: datetime) -> str:
    boats_regex = re.escape(boat)
    return f"""
from(bucket: "{bucket}")
  |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
  |> filter(fn: (r) => r._measurement == "{ch}")
  |> filter(fn: (r) => r._field == "value" and r.level == "strm")
  |> filter(fn: (r) => r.boat =~ /^{boats_regex}/)
  |> keep(columns: ["_time", "_value", "boat"])
  |> rename(columns: {{_time: "time", _value: "{ch}"}})
"""


def main():
    load_dotenv()
    cfg = get_cfg()

    boat = "FRA"
    start_time = datetime(2026, 4, 10, 17, 15, 0)
    end_time = start_time + timedelta(minutes=1)

    print(f"Boat       : {boat}")
    print(f"Window UTC : {start_time.isoformat()}Z -> {end_time.isoformat()}Z")
    print(f"Bucket     : {cfg.bucket}\n")

    dfs = []
    for ch in CHANNELS:
        print(f"=== CHANNEL: {ch} ===")
        query = build_query(cfg.bucket, ch, boat, start_time, end_time)
        try:
            df = _query_data_frame_safe(cfg, query)
        except Exception as e:
            print(f"ERROR: {e}\n")
            continue

        if df is None or df.empty:
            print("No data\n")
            continue

        df = df.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        print(df.head(20).to_string(index=False))
        print()
        dfs.append(df)

    if not dfs:
        print("No LENGTH_DB data found with this query structure.")
        return

    merged = None
    for df in dfs:
        cols = [c for c in df.columns if c != "boat"]
        cur = df[cols].drop_duplicates()
        if merged is None:
            merged = cur
        else:
            merged = pd.merge(merged, cur, on="time", how="outer")

    merged = merged.sort_values("time").reset_index(drop=True)
    print("=== MERGED PREVIEW ===")
    print(merged.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
