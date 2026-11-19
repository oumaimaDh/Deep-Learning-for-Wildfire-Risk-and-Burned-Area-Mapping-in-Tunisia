"""Burned area per year inside Tunisia (MODIS MCD64A1, June-October) next to the VIIRS fire counts.

An independent check of the cleaned VIIRS data: if both products rank the years the same way,
the cleaning kept real wildfire signal. Only pixels and detections inside Tunisia are counted.
"""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.stats import spearmanr

from fetch_mcd64a1 import PIX, crop_window, pixel_to_latlon

GEOJSON, VIIRS, BURN_DIR = "data/tunisia_governorates.geojson", "data/fires_tunisia_viirs_clean.csv", "data/burned_area"
OUT = "outputs/burned_area_by_year.csv"
YEARS = range(2013, 2026)                      # full years covered by both products
PIX_KM2 = (PIX / 1000.0) ** 2


def polygons():
    out = []
    for ft in json.load(open(GEOJSON, encoding="utf-8"))["features"]:
        g = ft["geometry"]
        out += [Path(np.asarray(p[0])[:, :2]) for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"])]
    return out


def inside(points, polys):
    m = np.zeros(len(points), dtype=bool)
    for p in polys:
        m |= p.contains_points(points)
    return m


polys = polygons()
r0, r1, c0, c1 = crop_window()
lat, lon = pixel_to_latlon(*np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij"))
tun = inside(np.column_stack([lon.ravel(), lat.ravel()]), polys).reshape(lat.shape)

v = pd.read_csv(VIIRS, parse_dates=["acq_date"])
v = v[v["acq_date"].dt.month.between(6, 10)]
v = v[inside(v[["longitude", "latitude"]].to_numpy(), polys)]
viirs = v.groupby(v["acq_date"].dt.year).size()

rows = []
for y in YEARS:
    total, all_box = 0, 0
    for m in range(6, 11):
        p = os.path.join(BURN_DIR, f"burn_{y}{m:02d}.npz")
        if os.path.exists(p):
            b = np.load(p)["burn_date"] > 0
            total += int((b & tun).sum())
            all_box += int(b.sum())
    rows.append({"year": y, "burned_km2_tunisia": total * PIX_KM2, "burned_km2_whole_box": all_box * PIX_KM2,
                 "viirs_detections": int(viirs.get(y, 0))})
t = pd.DataFrame(rows)
t.round(1).to_csv(OUT, index=False)
print(t.round(1).to_string(index=False))
rho, p = spearmanr(t["burned_km2_tunisia"], t["viirs_detections"])
print(f"\nSpearman rank correlation, burned area (Tunisia) vs VIIRS detections, {YEARS[0]}-{YEARS[-1]}: {rho:.2f} (p = {p:.3f})")
print("Largest years by burned area:", ", ".join(str(int(y)) for y in t.nlargest(3, "burned_km2_tunisia")["year"]))
print("Smallest years by burned area:", ", ".join(str(int(y)) for y in t.nsmallest(3, "burned_km2_tunisia")["year"]))
print("Largest years by VIIRS:", ", ".join(str(int(y)) for y in t.nlargest(3, "viirs_detections")["year"]))
print("Smallest years by VIIRS:", ", ".join(str(int(y)) for y in t.nsmallest(3, "viirs_detections")["year"]))
print(f"\nSaved: {OUT}")