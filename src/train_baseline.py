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
OUT_METRICS = "outputs/baseline_metrics.csv"
OUT_IMPORTANCE = "outputs/baseline_importance.png"
OUT_MAP = "outputs/baseline_risk_map.png"

# Northern governorates, where almost all wildfires happen
REGION = ["Bizerte", "Béja", "Jendouba", "El Kef", "Siliana", "Kasserine",
          "Nabeul", "Zaghouan", "Manouba", "Ariana", "Ben Arous", "Tunis"]
CELL = 0.05                       # grid cell size in degrees, about 5 km
FIRST_YEAR, LAST_YEAR = 2012, 2025
TRAIN_END = 2021                  # train on years up to here, test on the later years
FEATURES = ["month_sin", "month_cos", "lat", "lon", "hist_years_cell", "hist_same_month", "hist_neighbours"]

os.makedirs("outputs", exist_ok=True)

# ---- 1. Region and grid ------------------------------------------------------------------
with open(GEOJSON, encoding="utf-8") as f:
    gj = json.load(f)
polys = []
available = []
for ft in gj["features"]:
    name = ft["properties"].get("shapeName")
    available.append(name)
    if name in REGION:
        g = ft["geometry"]
        parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        polys += [np.asarray(p[0])[:, :2] for p in parts]
if not polys:
    raise SystemExit(f"None of the region names were found. Names in the file: {available}")

allpts = np.vstack(polys)
lon0, lon1 = allpts[:, 0].min(), allpts[:, 0].max()
lat0, lat1 = allpts[:, 1].min(), allpts[:, 1].max()
nx, ny = int(np.ceil((lon1 - lon0) / CELL)), int(np.ceil((lat1 - lat0) / CELL))
cx = lon0 + (np.arange(nx) + 0.5) * CELL
cy = lat0 + (np.arange(ny) + 0.5) * CELL
gx, gy = np.meshgrid(cx, cy)                      # shape (ny, nx)
centres = np.column_stack([gx.ravel(), gy.ravel()])
mask = np.zeros(len(centres), dtype=bool)
for p in polys:
    mask |= Path(p).contains_points(centres)
mask = mask.reshape(ny, nx)
iy_sel, ix_sel = np.nonzero(mask)
n_cells = len(iy_sel)
print(f"Region: {n_cells:,} cells of {CELL} degrees (about 5 km) in {len(set(REGION) & set(available))} governorates")

# ---- 2. Which cell and month had a fire --------------------------------------------------
NY = LAST_YEAR - FIRST_YEAR + 1
v = pd.read_csv(VIIRS, parse_dates=["acq_date"])
v = v[(v["acq_date"].dt.year >= FIRST_YEAR) & (v["acq_date"].dt.year <= LAST_YEAR)]
ix = np.floor((v["longitude"].to_numpy() - lon0) / CELL).astype(int)
iy = np.floor((v["latitude"].to_numpy() - lat0) / CELL).astype(int)
ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
fire = np.zeros((ny, nx, NY, 12), dtype=np.uint8)
fire[iy[ok], ix[ok], (v["acq_date"].dt.year.to_numpy() - FIRST_YEAR)[ok], (v["acq_date"].dt.month.to_numpy() - 1)[ok]] = 1

# ---- 3. History features (only from strictly earlier years, so nothing leaks) -------------
any_year = fire.max(axis=3).astype(int)                      # fire in the cell during the year
cum_before = np.cumsum(any_year, axis=2) - any_year          # earlier years with a fire in the cell
cum_month_before = np.cumsum(fire.astype(int), axis=2) - fire  # earlier years with a fire in this calendar month


def neighbour_sum(a):
    """Sum over the 8 neighbouring cells."""
    p = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2))
    out = np.zeros(a.shape, dtype=float)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out += p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
    return out


nb_before = neighbour_sum(cum_before)

# ---- 4. Table: one row per cell and month ------------------------------------------------
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

train = table[table["year"] <= TRAIN_END]
test = table[table["year"] > TRAIN_END]
print(f"Rows: {len(table):,} | train {FIRST_YEAR}-{TRAIN_END}: {len(train):,} ({train['fire'].mean():.1%} with fire) "
      f"| test {TRAIN_END + 1}-{LAST_YEAR}: {len(test):,} ({test['fire'].mean():.1%} with fire)\n")

# ---- 5. Models ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=0)
rf.fit(train[FEATURES], train["fire"])
test = test.assign(rf=rf.predict_proba(test[FEATURES])[:, 1])

month_rate = train.groupby("month")["fire"].mean()
scores = {
    "Month only (seasonal average)": test["month"].map(month_rate),
    "Cell history only": test["hist_years_cell"],
    "Random Forest (month + place + history)": test["rf"],
}
rows = []
for name, s in scores.items():
    rows.append({"model": name,
                 "ROC_AUC": roc_auc_score(test["fire"], s),
                 "PR_AUC": average_precision_score(test["fire"], s)})
metrics = pd.DataFrame(rows)
metrics.round(3).to_csv(OUT_METRICS, index=False)
print(f"Test years {TRAIN_END + 1}-{LAST_YEAR}. Random guessing would give ROC_AUC 0.5 and PR_AUC {test['fire'].mean():.3f}:")
print(metrics.round(3).to_string(index=False), "\n")

# ---- 6. What the forest relies on --------------------------------------------------------
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color="#b2182b")
ax.set_xlabel("Feature importance")
ax.set_title("Random Forest baseline: what drives the prediction")
fig.tight_layout()
fig.savefig(OUT_IMPORTANCE, dpi=150)
plt.close(fig)
print("Feature importance:")
print(imp.sort_values(ascending=False).round(3).to_string(), "\n")

# ---- 7. Risk map for August of the last year (not seen in training) ----------------------
sel = test[(test["year"] == LAST_YEAR) & (test["month"] == 8)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.5), sharex=True, sharey=True)
for a in (a1, a2):
    for p in polys:
        a.plot(p[:, 0], p[:, 1], color="#888", lw=0.5)
    a.set_aspect(1 / np.cos(np.radians(36)))
    a.set_xticks([])
    a.set_yticks([])
sc = a1.scatter(sel["lon"], sel["lat"], c=sel["rf"], s=14, marker="s", cmap="YlOrRd")
fig.colorbar(sc, ax=a1, shrink=0.6, label="Predicted fire probability")
a1.set_title(f"Predicted risk, August {LAST_YEAR}")
a2.scatter(sel["lon"], sel["lat"], s=14, marker="s", color="#e8e8e8")
hit = sel[sel["fire"] == 1]
a2.scatter(hit["lon"], hit["lat"], s=14, marker="s", color="#b2182b")
a2.set_title(f"Where VIIRS actually detected fire, August {LAST_YEAR}")
fig.tight_layout()
fig.savefig(OUT_MAP, dpi=150)
plt.close(fig)

print(f"Saved: {OUT_METRICS}, {OUT_IMPORTANCE}, {OUT_MAP}")