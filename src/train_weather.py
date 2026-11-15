import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

GEOJSON = "data/tunisia_governorates.geojson"
VIIRS = "data/fires_tunisia_viirs_clean.csv"
MERRA = "data/merra2_tunisia_monthly.csv"
OUT_METRICS = "outputs/weather_metrics.csv"
OUT_IMPORTANCE = "outputs/weather_importance.png"

REGION = ["Bizerte", "Béja", "Jendouba", "El Kef", "Siliana", "Kasserine",
          "Nabeul", "Zaghouan", "Manouba", "Ariana", "Ben Arous", "Tunis"]
CELL = 0.05
FIRST_YEAR, LAST_YEAR = 2012, 2025
TRAIN_END = 2021
SEEDS = [0, 1, 2]

BASE = ["month_sin", "month_cos", "lat", "lon", "hist_years_cell", "hist_same_month", "hist_neighbours"]
WEATHER = ["temp_c", "humidity", "wind", "rain_mm_day", "rain_prev3m", "soil_root", "soil_top", "lai",
           "temp_anom", "soil_anom"]

os.makedirs("outputs", exist_ok=True)

# ---- 1. Region and grid (same as train_baseline.py) --------------------------------------
with open(GEOJSON, encoding="utf-8") as f:
    gj = json.load(f)
polys = []
for ft in gj["features"]:
    if ft["properties"].get("shapeName") in REGION:
        g = ft["geometry"]
        parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        polys += [np.asarray(p[0])[:, :2] for p in parts]
if not polys:
    raise SystemExit("Region names not found in the boundaries file.")

allpts = np.vstack(polys)
lon0, lon1 = allpts[:, 0].min(), allpts[:, 0].max()
lat0, lat1 = allpts[:, 1].min(), allpts[:, 1].max()
nx, ny = int(np.ceil((lon1 - lon0) / CELL)), int(np.ceil((lat1 - lat0) / CELL))
cx = lon0 + (np.arange(nx) + 0.5) * CELL
cy = lat0 + (np.arange(ny) + 0.5) * CELL
gx, gy = np.meshgrid(cx, cy)
centres = np.column_stack([gx.ravel(), gy.ravel()])
mask = np.zeros(len(centres), dtype=bool)
for p in polys:
    mask |= Path(p).contains_points(centres)
mask = mask.reshape(ny, nx)
iy_sel, ix_sel = np.nonzero(mask)
n_cells = len(iy_sel)
print(f"Region: {n_cells:,} cells")

# ---- 2. Fire per cell, year, month -------------------------------------------------------
NY = LAST_YEAR - FIRST_YEAR + 1
v = pd.read_csv(VIIRS, parse_dates=["acq_date"])
v = v[(v["acq_date"].dt.year >= FIRST_YEAR) & (v["acq_date"].dt.year <= LAST_YEAR)]
ix = np.floor((v["longitude"].to_numpy() - lon0) / CELL).astype(int)
iy = np.floor((v["latitude"].to_numpy() - lat0) / CELL).astype(int)
ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
fire = np.zeros((ny, nx, NY, 12), dtype=np.uint8)
fire[iy[ok], ix[ok], (v["acq_date"].dt.year.to_numpy() - FIRST_YEAR)[ok], (v["acq_date"].dt.month.to_numpy() - 1)[ok]] = 1

any_year = fire.max(axis=3).astype(int)
cum_before = np.cumsum(any_year, axis=2) - any_year
cum_month_before = np.cumsum(fire.astype(int), axis=2) - fire


def neighbour_sum(a):
    p = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2))
    out = np.zeros(a.shape, dtype=float)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out += p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
    return out


nb_before = neighbour_sum(cum_before)

shape = (n_cells, NY, 12)
years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
months = np.arange(1, 13)
table = pd.DataFrame({
    "year": np.broadcast_to(years[None, :, None], shape).ravel(),
    "month": np.broadcast_to(months[None, None, :], shape).ravel(),
    "lat": np.broadcast_to(cy[iy_sel][:, None, None], shape).ravel(),
    "lon": np.broadcast_to(cx[ix_sel][:, None, None], shape).ravel(),
    "hist_years_cell": np.broadcast_to(cum_before[iy_sel, ix_sel][:, :, None], shape).ravel(),
    "hist_same_month": cum_month_before[iy_sel, ix_sel].ravel(),
    "hist_neighbours": np.broadcast_to(nb_before[iy_sel, ix_sel][:, :, None], shape).ravel(),
    "fire": fire[iy_sel, ix_sel].ravel(),
})
table["month_sin"] = np.sin(2 * np.pi * table["month"] / 12)
table["month_cos"] = np.cos(2 * np.pi * table["month"] / 12)

# ---- 3. Weather: nearest MERRA-2 grid point for every cell -------------------------------
w = pd.read_csv(MERRA).sort_values(["lat", "lon", "year", "month"]).reset_index(drop=True)
w["temp_c"] = w["T2M"] - 273.15
w["humidity"] = w["QV2M"] * 1000                       # g/kg
w["wind"] = np.hypot(w["U10M"], w["V10M"])
w["rain_mm_day"] = w["PRECTOTCORR"] * 86400            # kg m-2 s-1 -> mm/day
w["rain_prev3m"] = (w.groupby(["lat", "lon"])["rain_mm_day"]
                    .transform(lambda s: s.shift(1).rolling(3).sum()))   # the 3 months before
w["soil_root"] = w["GWETROOT"]
w["soil_top"] = w["GWETTOP"]
w["lai"] = w["LAI"]
clim = w[w["year"].between(FIRST_YEAR, TRAIN_END)].groupby(["lat", "lon", "month"])[["temp_c", "soil_root"]].mean()
clim.columns = ["temp_clim", "soil_clim"]                # climatology from training years only
w = w.join(clim, on=["lat", "lon", "month"])
w["temp_anom"] = w["temp_c"] - w["temp_clim"]
w["soil_anom"] = w["soil_root"] - w["soil_clim"]

pts = w[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
cell_lat = cy[iy_sel]
cell_lon = cx[ix_sel]
d = (cell_lat[:, None] - pts["lat"].to_numpy()[None, :]) ** 2 + \
    ((cell_lon[:, None] - pts["lon"].to_numpy()[None, :]) * np.cos(np.radians(36))) ** 2
nearest = d.argmin(axis=1)
cell_key = pd.DataFrame({"lat": cell_lat, "lon": cell_lon,
                         "m_lat": pts["lat"].to_numpy()[nearest], "m_lon": pts["lon"].to_numpy()[nearest]})
table = table.merge(cell_key, on=["lat", "lon"], how="left")
wcols = ["lat", "lon", "year", "month"] + WEATHER
table = table.merge(w[wcols].rename(columns={"lat": "m_lat", "lon": "m_lon"}),
                    on=["m_lat", "m_lon", "year", "month"], how="left")
missing = int(table[WEATHER].isna().any(axis=1).sum())
print(f"Rows: {len(table):,} | rows missing weather: {missing:,} | MERRA-2 points used: {table[['m_lat', 'm_lon']].drop_duplicates().shape[0]}")
table[WEATHER] = table[WEATHER].fillna(table[WEATHER].median())

train = table[table["year"] <= TRAIN_END]
test = table[table["year"] > TRAIN_END]
print(f"train {len(train):,} | test {len(test):,}\n")

# ---- 4. Compare models over a few random seeds --------------------------------------------
sets = {
    "Baseline (month + place + fire history)": BASE,
    "Weather only (month + weather, no place or history)": ["month_sin", "month_cos"] + WEATHER,
    "Baseline + weather": BASE + WEATHER,
}
rows = []
last_rf = None
for name, feats in sets.items():
    roc, pr = [], []
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=s)
        rf.fit(train[feats], train["fire"])
        p = rf.predict_proba(test[feats])[:, 1]
        roc.append(roc_auc_score(test["fire"], p))
        pr.append(average_precision_score(test["fire"], p))
    rows.append({"model": name, "ROC_AUC": np.mean(roc), "PR_AUC": np.mean(pr), "PR_AUC_std": np.std(pr)})
    print(f"{name}: ROC {np.mean(roc):.3f}  PR {np.mean(pr):.3f} (+/- {np.std(pr):.3f} over {len(SEEDS)} seeds)")
    last_rf = rf
metrics = pd.DataFrame(rows)
metrics.round(3).to_csv(OUT_METRICS, index=False)
print(f"\nRandom guessing would give PR_AUC {test['fire'].mean():.3f}")
print(metrics.round(3).to_string(index=False))

# ---- 5. Importance of the combined model ---------------------------------------------------
imp = pd.Series(last_rf.feature_importances_, index=BASE + WEATHER).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp.index, imp.values, color=["#2166ac" if i in WEATHER else "#b2182b" for i in imp.index])
ax.set_xlabel("Feature importance")
ax.set_title("Baseline + weather (blue = weather, red = place/history)")
fig.tight_layout()
fig.savefig(OUT_IMPORTANCE, dpi=150)
plt.close(fig)
print("\nFeature importance:")
print(imp.sort_values(ascending=False).round(3).to_string())
print(f"\nSaved: {OUT_METRICS}, {OUT_IMPORTANCE}")