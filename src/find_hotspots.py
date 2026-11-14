import json
import math
import os

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from sklearn.cluster import DBSCAN

GEOJSON = "data/tunisia_governorates.geojson"
VIIRS = "data/fires_tunisia_viirs_clean.csv"
MODIS = "data/fires_tunisia_modis_clean.csv"
OUT_CSV = "outputs/hotspot_clusters.csv"
OUT_PNG = "outputs/hotspots_map.png"
OUT_HTML = "outputs/tunisia_hotspots.html"

EPS_KM = 1.0            # two detections closer than this are neighbours
MIN_SAMPLES = 10        # a cluster needs at least this many detections in a neighbourhood
PERSISTENT_YEARS = 5    # a cluster active in this many different years counts as persistent
MODIS_RADIUS_KM = 3.0   # minimum search radius for MODIS detections around a cluster
MIN_MODIS_SUPPORT = 3   # fewer MODIS detections nearby means the cluster is seen by VIIRS only
TOP_ON_MAP = 80

os.makedirs("outputs", exist_ok=True)


def to_km(df):
    """Lon/lat to approximate kilometres, so DBSCAN's distance is in km."""
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def load_governorates():
    with open(GEOJSON, encoding="utf-8") as f:
        data = json.load(f)
    gov = []
    for ft in data["features"]:
        g = ft["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        exteriors = [np.asarray(p[0])[:, :2] for p in polys]
        gov.append((ft["properties"].get("shapeName"), exteriors))
    return gov


def governorate_of(lon, lat, gov_paths):
    for name, paths in gov_paths:
        if any(p.contains_point((lon, lat)) for p in paths):
            return name
    return "outside"


gov = load_governorates() if os.path.exists(GEOJSON) else []
gov_paths = [(name, [Path(e) for e in ex]) for name, ex in gov]

v = pd.read_csv(VIIRS, parse_dates=["acq_date"])
m = pd.read_csv(MODIS)
xy = to_km(v)
mxy = to_km(m)

print(f"Clustering {len(v):,} VIIRS detections (eps = {EPS_KM} km, min_samples = {MIN_SAMPLES}) ...")
v["cluster"] = DBSCAN(eps=EPS_KM, min_samples=MIN_SAMPLES).fit_predict(xy)
n_clusters = int(v["cluster"].max()) + 1
if n_clusters == 0:
    raise SystemExit("No clusters found. Try a larger EPS_KM or a smaller MIN_SAMPLES.")

clustered = v[v["cluster"] >= 0]
print(f"Clusters found: {n_clusters:,}")
print(f"Detections in a cluster: {len(clustered):,} ({len(clustered) / len(v):.0%}); isolated (noise): {len(v) - len(clustered):,}")

rows = []
for cid, g in clustered.groupby("cluster"):
    pts = xy[g.index]
    cx, cy = pts.mean(axis=0)
    extent = math.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    radius = max(MODIS_RADIUS_KM, extent / 2 + 1)
    modis_n = int((np.hypot(mxy[:, 0] - cx, mxy[:, 1] - cy) <= radius).sum())
    lat, lon = g["latitude"].mean(), g["longitude"].mean()
    rows.append({
        "cluster": cid,
        "governorate": governorate_of(lon, lat, gov_paths),
        "detections": len(g),
        "years_active": g["acq_date"].dt.year.nunique(),
        "first_year": g["acq_date"].dt.year.min(),
        "last_year": g["acq_date"].dt.year.max(),
        "peak_month": int(g["acq_date"].dt.month.mode().iloc[0]),
        "extent_km": round(extent, 1),
        "modis_nearby": modis_n,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
    })

cl = pd.DataFrame(rows).sort_values(["years_active", "detections"], ascending=False).reset_index(drop=True)
cl["persistent"] = cl["years_active"] >= PERSISTENT_YEARS
cl["modis_supported"] = cl["modis_nearby"] >= MIN_MODIS_SUPPORT
cl["confirmed_persistent"] = cl["persistent"] & cl["modis_supported"]
cl.to_csv(OUT_CSV, index=False)

pd.set_option("display.width", 160)
print("\nTop 15 clusters by number of years active:")
print(cl.head(15).drop(columns=["lat", "lon", "persistent"]).to_string(index=False))

pers = cl[cl["persistent"]]
conf = cl[cl["confirmed_persistent"]]
weak = pers[~pers["modis_supported"]]
print(f"\nPersistent clusters (active in {PERSISTENT_YEARS}+ different years): {len(pers):,}")
print(f"  seen by MODIS too ({MIN_MODIS_SUPPORT}+ MODIS detections nearby): {len(conf):,}, "
      f"holding {conf['detections'].sum() / len(clustered):.0%} of clustered detections")
print(f"  seen by VIIRS only (set aside): {len(weak):,}, {int(weak['detections'].sum()):,} detections")
print(f"Largest cluster: {int(cl['detections'].max()):,} detections, {cl['extent_km'].max():.1f} km across")
print("\nConfirmed persistent clusters per governorate:")
print(conf["governorate"].value_counts().head(10).to_string())
nw = int(conf["governorate"].isin(["Bizerte", "Béja", "Jendouba"]).sum())
print(f"\nIn Bizerte, Béja and Jendouba: {nw} of {len(conf)} confirmed persistent clusters ({nw / len(conf):.0%})")

# Static map
fig, ax = plt.subplots(figsize=(6.5, 9))
for _, exteriors in gov:
    for e in exteriors:
        ax.plot(e[:, 0], e[:, 1], color="#888", lw=0.6)
noise = v[v["cluster"] < 0]
ax.scatter(noise["longitude"], noise["latitude"], s=0.6, color="#cccccc", label="isolated detections")
years = cl.set_index("cluster")["years_active"]
supported = clustered["cluster"].map(cl.set_index("cluster")["modis_supported"]).astype(bool)
weak_pts = clustered[~supported]
cdata = clustered[supported].assign(years=clustered.loc[supported, "cluster"].map(years))
ax.scatter(weak_pts["longitude"], weak_pts["latitude"], s=3, color="#4c78a8", label="clusters seen only by VIIRS")
sc = ax.scatter(cdata["longitude"], cdata["latitude"], c=cdata["years"], s=3, cmap="YlOrRd", vmin=1)
fig.colorbar(sc, ax=ax, shrink=0.6, label="Years active (clusters also seen by MODIS)")
ax.legend(loc="lower left", markerscale=4, fontsize=8)
ax.set_aspect(1 / math.cos(math.radians(34)))
ax.set_title("Persistent fire hotspots in Tunisia (VIIRS, DBSCAN clusters)")
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)

# Light interactive map: only the top clusters, no embedded boundaries
fmap = folium.Map(
    location=[34.0, 9.5],
    zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
)
for _, r in cl.head(TOP_ON_MAP).iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=min(25, 3 + math.sqrt(r["detections"])),
        color="#b2182b" if r["confirmed_persistent"] else ("#4c78a8" if r["persistent"] else "#f28e2b"),
        fill=True,
        fill_opacity=0.6,
        popup=(f"{r['governorate']}<br>{int(r['detections'])} detections<br>"
               f"active in {int(r['years_active'])} years ({int(r['first_year'])}-{int(r['last_year'])})<br>"
               f"peak month: {int(r['peak_month'])}<br>MODIS detections nearby: {int(r['modis_nearby'])}"),
    ).add_to(fmap)
fmap.save(OUT_HTML)

print(f"\nSaved: {OUT_CSV}, {OUT_PNG}, {OUT_HTML}")