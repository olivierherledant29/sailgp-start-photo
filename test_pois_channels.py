import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter

import requests


def parse_utc_iso(ts: str) -> datetime:
    """
    Parse ISO 8601 with timezone offset, e.g. 2024-11-23T10:35:38.404000+00:00
    Returns timezone-aware datetime in UTC.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_pois(base_url: str, api_key: str, race_id: str, boat: str) -> list[dict]:
    url = f"{base_url.rstrip('/')}/v1/races/{race_id}/boats/{boat}/pois"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response type: {type(data)}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Explore POI 'channels' (type) around or after a given UTC time."
    )
    parser.add_argument("--race", required=True, help="Race ID, e.g. 24112301")
    parser.add_argument("--boat", required=True, help="Boat code, e.g. AUS")
    parser.add_argument(
        "--t",
        default="2026-01-17T05:30:00+00:00",
        help='Reference UTC time ISO (default: 2026-01-17T05:30:00+00:00)',
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--window",
        type=int,
        help="Half-window in seconds around t (mode +/-window). Example: 180 for +/-180s.",
    )
    group.add_argument(
        "--after-minutes",
        type=int,
        default=10,
        help="Minutes AFTER t to search (mode [t ; t+minutes]). Default: 10",
    )

    parser.add_argument(
        "--base-url",
        default="https://api.f50.sailgp.tech",
        help="Base URL (default: https://api.f50.sailgp.tech)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("SAILGP_POI_TOKEN", ""),
        help="Bearer token (default: env var SAILGP_POI_TOKEN)",
    )

    args = parser.parse_args()

    if not args.token:
        print("ERROR: No token provided. Use --token or set SAILGP_POI_TOKEN.", file=sys.stderr)
        sys.exit(1)

    t0 = parse_utc_iso(args.t)

    if args.window is not None:
        start = t0 - timedelta(seconds=args.window)
        end = t0 + timedelta(seconds=args.window)
        mode_str = f"+/- {args.window}s"
    else:
        start = t0
        end = t0 + timedelta(minutes=args.after_minutes)
        mode_str = f"[t ; t+{args.after_minutes}min]"

    print(f"\nReference UTC : {t0.isoformat()}")
    print(f"Mode          : {mode_str}")
    print(f"Window        : [{start.isoformat()}  ->  {end.isoformat()}]\n")

    pois = fetch_pois(args.base_url, args.token, args.race, args.boat)

    # Parse + filter
    in_window = []
    skipped = 0
    for p in pois:
        sdt = p.get("start_datetime")
        if not sdt:
            skipped += 1
            continue
        try:
            dt = parse_utc_iso(sdt)
        except Exception:
            skipped += 1
            continue

        if start <= dt <= end:
            in_window.append((dt, p))

    in_window.sort(key=lambda x: x[0])

    print(f"Total POIs fetched : {len(pois)}")
    print(f"POIs w/ bad/missing time skipped: {skipped}")
    print(f"POIs in window     : {len(in_window)}\n")

    if not in_window:
        print("No POIs found in the selected time window.")
        return

    # Count types (channels)
    type_counts = Counter((p.get("type") or "UNKNOWN") for _, p in in_window)
    print("Types (channels) found in window (count):")
    for t, c in type_counts.most_common():
        print(f"  - {t}: {c}")
    print()

    # Timeline view
    print("Timeline (UTC):")
    for dt, p in in_window:
        typ = p.get("type", "UNKNOWN")
        name = p.get("display_name", "")
        poi_id = p.get("poi_id", "")
        print(f"  {dt.isoformat()}  |  {typ:20}  |  {name}  |  {poi_id}")


if __name__ == "__main__":
    main()

