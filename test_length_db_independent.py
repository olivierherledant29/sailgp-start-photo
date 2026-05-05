from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
from dotenv import load_dotenv

from influx_io import get_cfg, _query_data_frame_safe, iso_z


CHANNELS = ["LENGTH_DB_H_P_mm", "LENGTH_DB_H_S_mm"]
LEVELS_TO_TRY = ["strm", "mdss", "mdss_fast", "raw"]


def query_length_db(start_utc: datetime, stop_utc: datetime, boat: str, level: str) -> pd.DataFrame:
    cfg = get_cfg()

    measurement_filters = " or ".join([f'r["_measurement"] == "{ch}"' for ch in CHANNELS])

    flux = f"""
from(bucket: "{cfg.bucket}")
  |> range(start: {iso_z(start_utc)}, stop: {iso_z(stop_utc)})
  |> filter(fn: (r) => ({measurement_filters}))
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["level"] == "{level}")
  |> filter(fn: (r) => r["boat"] == "{boat}")
  |> keep(columns: ["_time", "_measurement", "_value", "boat", "level"])
"""
    df = _query_data_frame_safe(cfg, flux)
    if df is None or df.empty:
        return pd.DataFrame()

    if "_time" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["_time"] = pd.to_datetime(out["_time"], utc=True, errors="coerce")
    out = out.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)
    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test independent LENGTH_DB reads from SailGP InfluxDB.")
    parser.add_argument("--boat", default="FRA", help="Boat code, default FRA")
    parser.add_argument("--start", default="2026-04-10T17:15:00Z", help="UTC start, default 2026-04-10T17:15:00Z")
    parser.add_argument("--duration-s", type=int, default=60, help="Duration in seconds, default 60")
    args = parser.parse_args()

    start_utc = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    stop_utc = start_utc + timedelta(seconds=args.duration_s)

    print(f"\nBoat        : {args.boat}")
    print(f"Window UTC  : {start_utc.isoformat()} -> {stop_utc.isoformat()} ({args.duration_s}s)")
    print(f"Channels    : {CHANNELS}\n")

    any_data = False

    for level in LEVELS_TO_TRY:
        print(f"=== LEVEL: {level} ===")
        df = query_length_db(start_utc, stop_utc, args.boat, level)

        if df.empty:
            print("No data\n")
            continue

        any_data = True
        print(f"Rows: {len(df)}")
        print("Measurements count:")
        counts = df["_measurement"].value_counts()
        for meas, cnt in counts.items():
            print(f"  - {meas}: {cnt}")

        print("\nFirst rows:")
        print(df[["_time", "_measurement", "_value", "boat", "level"]].head(20).to_string(index=False))
        print()

        piv = (
            df.pivot_table(index="_time", columns="_measurement", values="_value", aggfunc="last")
            .reset_index()
            .sort_values("_time")
        )
        print("Pivot preview:")
        print(piv.head(20).to_string(index=False))
        print()

    if not any_data:
        print("No LENGTH_DB data found on any tested level.")
        print("Possible causes:")
        print("  1) Wrong boat for that time window")
        print("  2) Different channel names than LENGTH_DB_H_P_mm / LENGTH_DB_H_S_mm")
        print("  3) Data exists on another level/tag structure")
        print("  4) No Influx data for that moment")


if __name__ == "__main__":
    main()
