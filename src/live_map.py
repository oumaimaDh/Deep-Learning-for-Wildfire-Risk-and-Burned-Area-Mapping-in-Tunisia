import os
from datetime import datetime
from io import StringIO

import folium
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
MAP_KEY = os.getenv("FIRMS_MAP_KEY")

BBOX = "8.0,30.0,12.0,38.0"  # west, south, east, north
DAYS = 5                      # NRT endpoint allows 1-5
SOURCE = "VIIRS_SNPP_NRT"

CSV_PATH = "data/tunisia_fires_latest.csv"
MAP_PATH = "outputs/tunisia_fires_live.html"


def fetch_fires():
    if not MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY not found. Check your .env file.")

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{SOURCE}/{BBOX}/{DAYS}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def build_map(df):
    m = folium.Map(
        location=[34.0, 9.5],
        zoom_start=7,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    )

    folium.GeoJson(
        "data/tunisia_governorates.geojson",
        name="Governorates",
        style_function=lambda _: {
            "color": "#444",
            "weight": 1,
            "fillOpacity": 0,
        },
        tooltip=folium.GeoJsonTooltip(fields=["shapeName"], aliases=["Governorate:"]),
    ).add_to(m)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color="red",
            fill=True,
            fill_color="orange",
            fill_opacity=0.7,
            popup=(
                f"Date: {row['acq_date']}<br>"
                f"Brightness: {row['bright_ti4']}<br>"
                f"FRP: {row['frp']}"
            ),
        ).add_to(m)

    title = (
        f"<div style='position:fixed;top:10px;left:50px;z-index:9999;"
        f"background:white;padding:6px 10px;border-radius:4px;"
        f"font-family:sans-serif;font-size:13px;'>"
        f"<b>Tunisia fires, last {DAYS} days</b><br>"
        f"{len(df)} detections | updated {datetime.now():%Y-%m-%d %H:%M}"
        f"</div>"
    )
    m.get_root().html.add_child(folium.Element(title))

    os.makedirs("outputs", exist_ok=True)
    m.save(MAP_PATH)


if __name__ == "__main__":
    df = fetch_fires()
    print(f"Fetched {len(df)} detections")
    build_map(df)
    print(f"Map saved to {MAP_PATH}")