import os

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import HeatMap

SRC = "data/fires_tunisia_viirs.csv"
CLEAN = "data/fires_tunisia_viirs_clean.csv"
GEOJSON = "data/tunisia_governorates.geojson"
GRID = 0.005  # about 500 m, used to match recent detections against known static sources
DROP_LOW_CONFIDENCE = True
REMOVED = "data/removed_fixed_sources.csv"
STATIC_MIN_MONTHS = 6        # a cell active in 6+ different calendar months is not seasonal
STATIC_MIN_DETECTIONS = 40   # a real fire rarely hits one ~500 m cell this many times

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(SRC, parse_dates=["acq_date"])
print(f"Loaded: {len(df):,} detections")

# Archive rows have a 'type'; the recent near-real-time rows have none (NaN)
archive = df[df["type"].notna()]
recent = df[df["type"].isna()]
print(f"Archive rows: {len(archive):,} | recent rows without a type: {len(recent):,}")

# 1. Keep only presumed vegetation fires from the archive (type 0)
veg = archive[archive["type"] == 0]
print(f"Archive vegetation fires (type 0): {len(veg):,}")

# 2. Recent rows can't be classified, so drop the ones that sit on a known static source
static = archive[archive["type"] == 2]
static_cells = set(zip((static["latitude"] / GRID).round().astype(int),
                       (static["longitude"] / GRID).round().astype(int)))
recent_cells = list(zip((recent["latitude"] / GRID).round().astype(int),
                        (recent["longitude"] / GRID).round().astype(int)))
on_static = pd.Series([c in static_cells for c in recent_cells], index=recent.index)
recent_kept = recent[~on_static]
print(f"Recent rows on known static sources removed: {int(on_static.sum()):,} | kept: {len(recent_kept):,}")

clean = pd.concat([veg, recent_kept], ignore_index=True)

# 3. Optionally drop low-confidence detections
if DROP_LOW_CONFIDENCE:
    n = len(clean)
    clean = clean[clean["confidence"] != "l"]
    print(f"Low-confidence detections removed: {n - len(clean):,}")

# 4. Remove persistent fixed sources (flares, kilns, plants) that the 'type' flag missed.
# A real vegetation fire is seasonal and rarely repeats many times in one ~500 m cell;
# a fixed source shows up in many different months and/or hundreds of times.
clean["cell_lat"] = (clean["latitude"] / GRID).round().astype(int)
clean["cell_lon"] = (clean["longitude"] / GRID).round().astype(int)
cell_stats = clean.groupby(["cell_lat", "cell_lon"]).agg(
    n=("acq_date", "size"),
    months=("acq_date", lambda s: s.dt.month.nunique()),
)
flagged = cell_stats[
    (cell_stats["months"] >= STATIC_MIN_MONTHS) | (cell_stats["n"] >= STATIC_MIN_DETECTIONS)
].index
# also block the 8 neighbouring cells, because one source often straddles several cells
blocked = {(a + da, b + db) for a, b in flagged for da in (-1, 0, 1) for db in (-1, 0, 1)}
is_static = pd.Series(
    [(a, b) in blocked for a, b in zip(clean["cell_lat"], clean["cell_lon"])], index=clean.index
)
print(f"Fixed-source cells flagged: {len(flagged):,} | detections removed (with ~500 m buffer): "
      f"{int(is_static.sum()):,} ({is_static.mean():.0%})")
clean[is_static].to_csv(REMOVED, index=False)
clean = clean[~is_static].drop(columns=["cell_lat", "cell_lon"])

clean = clean.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)
clean.to_csv(CLEAN, index=False)
print(f"\nFinal clean dataset: {len(clean):,} detections -> {CLEAN}\n")

clean["year"] = clean["acq_date"].dt.year
clean["month"] = clean["acq_date"].dt.month

# Chart 1: detections per year (first and last years are partial)
per_year = clean["year"].value_counts().sort_index()
partial = {per_year.index.min(), per_year.index.max()}
colors = ["#f4a582" if y in partial else "#b2182b" for y in per_year.index]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(per_year.index.astype(str), per_year.values, color=colors)
ax.set_title("Vegetation fire detections in Tunisia per year (VIIRS)")
ax.set_ylabel("Detections")
ax.tick_params(axis="x", rotation=45)
fig.text(0.01, 0.01, "Lighter bars = partial years", fontsize=8, color="#555")
fig.tight_layout()
fig.savefig("outputs/fires_per_year.png", dpi=150)
plt.close(fig)

# Chart 2: detections per month, all years combined
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
per_month = clean["month"].value_counts().reindex(range(1, 13), fill_value=0)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(months, per_month.values, color="#b2182b")
ax.set_title("Seasonality of vegetation fire detections in Tunisia (all years)")
ax.set_ylabel("Detections")
fig.tight_layout()
fig.savefig("outputs/fires_per_month.png", dpi=150)
plt.close(fig)

# Map: density of all historical detections
m = folium.Map(
    location=[34.0, 9.5],
    zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
)
if os.path.exists(GEOJSON):
    folium.GeoJson(
        GEOJSON,
        style_function=lambda _: {"color": "#444", "weight": 1, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(fields=["shapeName"], aliases=["Governorate:"]),
    ).add_to(m)
HeatMap(clean[["latitude", "longitude"]].values.tolist(), radius=8, blur=10, min_opacity=0.3).add_to(m)
m.save("outputs/tunisia_fire_density.html")

print("Saved: outputs/fires_per_year.png, outputs/fires_per_month.png, outputs/tunisia_fire_density.html")