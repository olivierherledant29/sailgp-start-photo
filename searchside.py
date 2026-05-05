import requests

TOKEN = "R5YnGV2dyN6sePo3Cif3o4hTKcN9S8M-"
BASE = "https://api.f50.sailgp.tech"

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

FROM = "2026-03-01T06:55:00Z"
TO = "2026-03-01T07:00:00Z"
BOAT = "ESP"


def get_pois():
    r = requests.get(
        f"{BASE}/v1/pois",
        headers=HEADERS,
        params={
            "from": FROM,
            "to": TO,
            "boat": BOAT,
            "poi_type": ["boarddrop", "boardraise"]
        },
        timeout=30
    )
    r.raise_for_status()
    return r.json()


def get_side(poi_id):
    r = requests.get(
        f"{BASE}/v1/pois/{poi_id}/analysis",
        headers=HEADERS,
        timeout=30
    )
    if r.status_code != 200:
        return None

    data = r.json()
    return data.get("scalars", {}).get("board_side")


def main():
    pois = get_pois()

    babord = 0
    tribord = 0

    for p in pois:
        poi_id = p["poi_id"]
        side = get_side(poi_id)

        print(poi_id, "→", side)

        if side == "port":
            babord += 1
        elif side == "starboard":
            tribord += 1

    print("\nRESULTAT")
    print("Babord :", babord)
    print("Tribord:", tribord)


if __name__ == "__main__":
    main()