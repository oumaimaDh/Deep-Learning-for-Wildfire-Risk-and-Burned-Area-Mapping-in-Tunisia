import os
import requests

API = "https://www.geoboundaries.org/api/current/gbOpen/TUN/ADM1/"
OUT = "data/tunisia_governorates.geojson"

os.makedirs("data", exist_ok=True)

meta = requests.get(API, timeout=60)
meta.raise_for_status()
geojson_url = meta.json()["gjDownloadURL"]

geo = requests.get(geojson_url, timeout=120)
geo.raise_for_status()

with open(OUT, "wb") as f:
    f.write(geo.content)

print(f"Saved {OUT} ({len(geo.content) / 1024:.0f} KB)")