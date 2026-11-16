"""Fire-risk model, part 3: adds NDVI/SWIR vegetation features and tests on unseen places (spatial cross-validation).

Uses the MODIS MOD13A1 composites from data/vegetation/ (already downloaded for the U-Net).
For each month the vegetation features come from the last composite that ended BEFORE the month began,
so nothing from the month being predicted leaks in. This gives features for June-November only,
so every model below is trained and tested on those six months (about 90% of all detections).
"""
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
from sklearn.model_selection import GroupKFold

from fetch_mcd64a1 import pixel_to_latlon

GEOJSON = "data/tunisia_governorates.geojson"
VIIRS = "data/fires_tunisia_viirs_clean.csv"
MERRA = "data/merra2_tunisia_monthly.csv"
VEG_DIR = "data/vegetation"
OUT_TIME, OUT_SPATIAL, OUT_IMP = "outputs/risk_v2_time_split.csv", "outputs/risk_v2_spatial_cv.csv", "outputs/risk_v2_importance.png"

REGION = ["Bizerte", "Béja", "Jendouba", "El Kef", "Siliana", "Kasserine",
          "Nabeul", "Zaghouan", "Manouba", "Ariana", "Ben Arous", "Tunis"]
CELL = 0.05
FIRST_YEAR, LAST_YEAR, TRAIN_END = 2012, 2025, 2021
MONTHS = [6, 7, 8, 9, 10, 11]
PRE_COMPOSITE = {6: 129, 7: 161, 8: 193, 9: 225, 10: 257, 11: 289}     # composite finished before the month starts
BLOCK_DEG = 0.5                                                          # spatial blocks of about 55 km
SEEDS = [0, 1, 2]

BASE = ["month_sin", "month_cos", "lat", "lon", "hist_years_cell", "hist_same_month", "hist_neighbours"]
WEATHER = ["temp_c", "humidity", "wind", "rain_mm_day", "rain_prev3m", "soil_root", "soil_top", "lai", "temp_anom", "soil_anom"]
VEG = ["ndvi_pre", "ndvi_anom", "mir_pre", "nbr_pre"]
os.makedirs("outputs", exist_ok=True)

# ---- 1. Region, grid, fire table (same as train_weather.py) --------------------------------
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
lon0, lon1, lat0, lat1 = allpts[:, 0].min(), allpts[:, 0].max(), allpts[:, 1].min(), allpts[:, 1].max()
nx, ny = int(np.ceil((lon1 - lon0) / CELL)), int(np.ceil((lat1 - lat0) / CELL))
cx, cy = lon0 + (np.arange(nx) + 0.5) * CELL, lat0 + (np.arange(ny) + 0.5) * CELL
gx, gy = np.meshgrid(cx, cy)
centres = np.column_stack([gx.ravel(), gy.ravel()])
mask = np.zeros(len(centres), dtype=bool)
for p in polys:
    mask |= Path(p).contains_points(centres)
iy_sel, ix_sel = np.nonzero(mask.reshape(ny, nx))
n_cells = len(iy_sel)

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
table = pd.DataFrame({
    "year": np.broadcast_to(years[None, :, None], shape).ravel(),
    "month": np.broadcast_to(np.arange(1, 13)[None, None, :], shape).ravel(),
    "lat": np.broadcast_to(cy[iy_sel][:, None, None], shape).ravel(),
    "lon": np.broadcast_to(cx[ix_sel][:, None, None], shape).ravel(),
    "hist_years_cell": np.broadcast_to(cum_before[iy_sel, ix_sel][:, :, None], shape).ravel(),
    "hist_same_month": cum_month_before[iy_sel, ix_sel].ravel(),
    "hist_neighbours": np.broadcast_to(nb_before[iy_sel, ix_sel][:, :, None], shape).ravel(),
    "fire": fire[iy_sel, ix_sel].ravel(),
})
table["month_sin"] = np.sin(2 * np.pi * table["month"] / 12)
table["month_cos"] = np.cos(2 * np.pi * table["month"] / 12)

# ---- 2. MERRA-2 weather (same as train_weather.py) -----------------------------------------
w = pd.read_csv(MERRA).sort_values(["lat", "lon", "year", "month"]).reset_index(drop=True)
w["temp_c"] = w["T2M"] - 273.15
w["humidity"] = w["QV2M"] * 1000
w["wind"] = np.hypot(w["U10M"], w["V10M"])
w["rain_mm_day"] = w["PRECTOTCORR"] * 86400
w["rain_prev3m"] = w.groupby(["lat", "lon"])["rain_mm_day"].transform(lambda s: s.shift(1).rolling(3).sum())
w["soil_root"], w["soil_top"], w["lai"] = w["GWETROOT"], w["GWETTOP"], w["LAI"]
clim = w[w["year"].between(FIRST_YEAR, TRAIN_END)].groupby(["lat", "lon", "month"])[["temp_c", "soil_root"]].mean()
clim.columns = ["temp_clim", "soil_clim"]
w = w.join(clim, on=["lat", "lon", "month"])
w["temp_anom"], w["soil_anom"] = w["temp_c"] - w["temp_clim"], w["soil_root"] - w["soil_clim"]
pts = w[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
d = (cy[iy_sel][:, None] - pts["lat"].to_numpy()[None, :]) ** 2 + \
    ((cx[ix_sel][:, None] - pts["lon"].to_numpy()[None, :]) * np.cos(np.radians(36))) ** 2
nearest = d.argmin(axis=1)
cell_key = pd.DataFrame({"lat": cy[iy_sel], "lon": cx[ix_sel],
                         "m_lat": pts["lat"].to_numpy()[nearest], "m_lon": pts["lon"].to_numpy()[nearest]})
table = table.merge(cell_key, on=["lat", "lon"], how="left")
table = table.merge(w[["lat", "lon", "year", "month"] + WEATHER].rename(columns={"lat": "m_lat", "lon": "m_lon"}),
                    on=["m_lat", "m_lon", "year", "month"], how="left")
table[WEATHER] = table[WEATHER].fillna(table[WEATHER].median())

# ---- 3. Vegetation features: average the 500 m pixels inside every 5 km cell ---------------
first = np.load(os.path.join(VEG_DIR, sorted(f for f in os.listdir(VEG_DIR) if f.startswith("veg_"))[0]))
r0, r1, c0, c1 = [int(a) for a in first["window"]]
plat, plon = pixel_to_latlon(*np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij"))
pix_ix = np.floor((plon - lon0) / CELL).astype(int)
pix_iy = np.floor((plat - lat0) / CELL).astype(int)
inside = (pix_ix >= 0) & (pix_ix < nx) & (pix_iy >= 0) & (pix_iy < ny)
flat = np.where(inside, pix_iy * nx + pix_ix, 0)
sel_flat = iy_sel * nx + ix_sel


def cell_mean(values, valid):
    m = valid & inside
    num = np.bincount(flat[m], weights=values[m], minlength=nx * ny)
    cnt = np.bincount(flat[m], minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt >= 5, num / cnt, np.nan)
    return out[sel_flat]


ndvi_a = np.full(shape, np.nan, np.float32)
mir_a = np.full(shape, np.nan, np.float32)
nbr_a = np.full(shape, np.nan, np.float32)
n_found = 0
for yi, y in enumerate(years):
    for m, doy in PRE_COMPOSITE.items():
        p = os.path.join(VEG_DIR, f"veg_{y}{doy:03d}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        good = (z["ndvi"] > -3000) & (z["red"] > -1000) & (z["nir"] > -1000) & (z["mir"] > -1000)
        nir, mir = z["nir"] * 1e-4, z["mir"] * 1e-4
        ndvi_a[:, yi, m - 1] = cell_mean(z["ndvi"] * 1e-4, good)
        mir_a[:, yi, m - 1] = cell_mean(mir, good)
        nbr_a[:, yi, m - 1] = cell_mean((nir - mir) / np.maximum(nir + mir, 1e-6), good)
        n_found += 1
print(f"Region: {n_cells:,} cells | vegetation composites used: {n_found} of {len(years) * len(MONTHS)}")
tr_idx = (years <= TRAIN_END)
ndvi_clim = np.nanmean(ndvi_a[:, tr_idx, :], axis=1, keepdims=True)           # training years only
table["ndvi_pre"] = ndvi_a.ravel()
table["ndvi_anom"] = (ndvi_a - ndvi_clim).ravel()
table["mir_pre"] = mir_a.ravel()
table["nbr_pre"] = nbr_a.ravel()

table = table[table["month"].isin(MONTHS)].dropna(subset=VEG).reset_index(drop=True)
train, test = table[table["year"] <= TRAIN_END], table[table["year"] > TRAIN_END]
print(f"Rows (June-November, with vegetation data): {len(table):,} | train {len(train):,} ({train['fire'].mean():.1%} fire) "
      f"| test {len(test):,} ({test['fire'].mean():.1%} fire)\n")


def rf(seed, n=200):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=5, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)


# ---- 4. Time split: does vegetation add anything? ---------------------------------------------
time_sets = {
    "Baseline (month + place + history)": BASE,
    "Baseline + weather": BASE + WEATHER,
    "Baseline + weather + vegetation": BASE + WEATHER + VEG,
    "Baseline + vegetation": BASE + VEG,
}
rows, last = [], None
for name, feats in time_sets.items():
    roc, pr = [], []
    for s in SEEDS:
        m = rf(s).fit(train[feats], train["fire"])
        p = m.predict_proba(test[feats])[:, 1]
        roc.append(roc_auc_score(test["fire"], p)); pr.append(average_precision_score(test["fire"], p))
    rows.append({"model": name, "ROC_AUC": np.mean(roc), "PR_AUC": np.mean(pr), "PR_AUC_std": np.std(pr)})
    print(f"[time split {TRAIN_END + 1}-{LAST_YEAR}] {name}: ROC {np.mean(roc):.3f}  PR {np.mean(pr):.3f} (+/- {np.std(pr):.3f})")
    if name == "Baseline + weather + vegetation":
        last = m
pd.DataFrame(rows).round(3).to_csv(OUT_TIME, index=False)
print(f"Random guessing: PR_AUC {test['fire'].mean():.3f}\n")

# ---- 5. Spatial cross-validation: predict places the model has never seen ---------------------
# No coordinates and no fire history, so the model must rely on season, weather and vegetation alone.
table["block"] = (np.floor(table["lat"] / BLOCK_DEG) * 1000 + np.floor(table["lon"] / BLOCK_DEG)).astype(int)
n_blocks = table["block"].nunique()
spatial_sets = {
    "Season only": ["month_sin", "month_cos"],
    "Season + weather": ["month_sin", "month_cos"] + WEATHER,
    "Season + weather + vegetation": ["month_sin", "month_cos"] + WEATHER + VEG,
    "Season + vegetation": ["month_sin", "month_cos"] + VEG,
}
gkf = GroupKFold(n_splits=5)
rows = []
for name, feats in spatial_sets.items():
    roc, pr = [], []
    for tr_i, te_i in gkf.split(table, table["fire"], table["block"]):
        a, b = table.iloc[tr_i], table.iloc[te_i]
        if b["fire"].sum() == 0:
            continue
        p = rf(0, 150).fit(a[feats], a["fire"]).predict_proba(b[feats])[:, 1]
        roc.append(roc_auc_score(b["fire"], p)); pr.append(average_precision_score(b["fire"], p))
    rows.append({"model": name, "ROC_AUC": np.mean(roc), "PR_AUC": np.mean(pr), "PR_AUC_std_over_folds": np.std(pr)})
    print(f"[spatial CV, {n_blocks} blocks, 5 folds] {name}: ROC {np.mean(roc):.3f}  PR {np.mean(pr):.3f} (+/- {np.std(pr):.3f} across folds)")
pd.DataFrame(rows).round(3).to_csv(OUT_SPATIAL, index=False)
print(f"Random guessing: PR_AUC {table['fire'].mean():.3f}\n")

# ---- 6. Importance ------------------------------------------------------------------------
cols = BASE + WEATHER + VEG
imp = pd.Series(last.feature_importances_, index=cols).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(imp.index, imp.values, color=["#1b7837" if i in VEG else "#2166ac" if i in WEATHER else "#b2182b" for i in imp.index])
ax.set_xlabel("Feature importance")
ax.set_title("Baseline + weather + vegetation\n(green = vegetation, blue = weather, red = place/history)")
fig.tight_layout(); fig.savefig(OUT_IMP, dpi=150); plt.close(fig)
print("Feature importance (top 10):")
print(imp.sort_values(ascending=False).head(10).round(3).to_string())
print(f"\nSaved: {OUT_TIME}, {OUT_SPATIAL}, {OUT_IMP}")