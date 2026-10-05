import pandas as pd
import folium

df = pd.read_csv("data/tunisia_fires_latest.csv")

# Center the map on Tunisia
m = folium.Map(
    location=[34.0, 9.5],
    zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri"
)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=4,
        color="red",
        fill=True,
        fill_color="orange",
        fill_opacity=0.7,
        popup=f"Date: {row['acq_date']}<br>Brightness: {row['bright_ti4']}<br>FRP: {row['frp']}"
    ).add_to(m)

m.save("outputs/tunisia_fires_live.html")
print(f"Map saved with {len(df)} fire points -> outputs/tunisia_fires_live.html")