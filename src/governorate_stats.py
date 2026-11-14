import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path

GEOJSON = "data/tunisia_governorates.geojson"
VIIRS = "data/fires_tunisia_viirs_clean.csv"
MODIS = "data/fires_tunisia_modis_clean.csv"
OUT_CSV = "outputs/governorate_stats.csv"
OUT_PNG = "outputs/governorate_stats.png"
TOP = 12

os.makedirs("outputs", exist_ok=True)


def ring_area_km2(ring):
    """Approximate area of a lon/lat ring (flat-earth projection, fine for comparing governorates)."""
    pts = np.asarray(ring)
    x = pts[:, 0] * 111.32 * math.cos(math.radians(pts[:, 1].mean()))
    y = pts[:, 1] * 110.57
    return abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def load_governorates():
    with open(GEOJSON, encoding="utf-8") as f:
        data = json.load(f)
    gov = []
    for ft in data["features"]:
        name = ft["properties"].get("shapeName")
        g = ft["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        exteriors = [np.asarray(p[0])[:, :2] for p in polys]
        area = sum(ring_area_km2(e) for e in exteriors)
        gov.append((name, [Path(e) for e in exteriors], area))
    return gov


def assign(df, gov):
    """Return the governorate name for each detection (None if it falls outside every polygon)."""
    pts = df[["longitude", "latitude"]].to_numpy()
    result = pd.Series([None] * len(df), index=df.index, dtype=object)
    for name, paths, _ in gov:
        inside = np.zeros(len(df), dtype=bool)
        for p in paths:
            inside |= p.contains_points(pts)
        result[inside & result.isna().to_numpy()] = name
    return result


if not os.path.exists(GEOJSON):
    raise SystemExit(f"{GEOJSON} not found. Run python src/boundaries.py first.")

gov = load_governorates()
print(f"Loaded {len(gov)} governorates")

v = pd.read_csv(VIIRS)
m = pd.read_csv(MODIS)
v["gov"] = assign(v, gov)
m["gov"] = assign(m, gov)
print(f"VIIRS detections outside every governorate (coast and border pixels): {int(v['gov'].isna().sum()):,} of {len(v):,}")
print(f"MODIS detections outside every governorate: {int(m['gov'].isna().sum()):,} of {len(m):,}\n")

table = pd.DataFrame({"VIIRS": v["gov"].value_counts(), "MODIS": m["gov"].value_counts()})
table = table.reindex([g[0] for g in gov]).fillna(0).astype(int)  # list every governorate, even with zero detections
table["VIIRS_%"] = 100 * table["VIIRS"] / table["VIIRS"].sum()
table["MODIS_%"] = 100 * table["MODIS"] / table["MODIS"].sum()
areas = {name: area for name, _, area in gov}
table["area_km2"] = table.index.map(areas)
table["VIIRS_per_100km2"] = 100 * table["VIIRS"] / table["area_km2"]
table = table.sort_values("VIIRS", ascending=False)
table.round(2).to_csv(OUT_CSV, index_label="governorate")

pd.set_option("display.width", 140)
print("Detections per governorate (cleaned data), sorted by VIIRS count:")
print(table.round({"VIIRS_%": 1, "MODIS_%": 1, "area_km2": 0, "VIIRS_per_100km2": 2}).to_string(), "\n")

print("Ranking by detections per 100 km2 (VIIRS), top 8:")
print(table.sort_values("VIIRS_per_100km2", ascending=False)["VIIRS_per_100km2"].head(8).round(2).to_string(), "\n")

# Chart: share of detections by governorate (both sensors) and density
top_share = table.head(TOP).iloc[::-1]
top_density = table.sort_values("VIIRS_per_100km2", ascending=False).head(TOP).iloc[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
y = np.arange(len(top_share))
ax1.barh(y + 0.2, top_share["VIIRS_%"], height=0.4, color="#b2182b", label="VIIRS")
ax1.barh(y - 0.2, top_share["MODIS_%"], height=0.4, color="#2166ac", label="MODIS")
ax1.set_yticks(y)
ax1.set_yticklabels(top_share.index)
ax1.set_xlabel("Share of the sensor's detections (%)")
ax1.set_title(f"Where fires are detected: top {TOP} governorates")
ax1.legend(loc="lower right")

ax2.barh(top_density.index, top_density["VIIRS_per_100km2"], color="#b2182b")
ax2.set_xlabel("VIIRS detections per 100 km2 (2012-2026)")
ax2.set_title("Fire density, adjusted for governorate size")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)
print(f"Saved: {OUT_CSV} and {OUT_PNG}")