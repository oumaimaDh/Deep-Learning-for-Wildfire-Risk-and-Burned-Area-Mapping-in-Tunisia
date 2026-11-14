import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = "data/fires_tunisia_modis.csv"
CLEAN = "data/fires_tunisia_modis_clean.csv"
REMOVED = "data/removed_fixed_sources_modis.csv"
GRID = 0.01                  # about 1 km, matching the MODIS pixel size
MIN_CONFIDENCE = 30          # MODIS confidence is 0-100; below 30 is "low"
STATIC_MIN_MONTHS = 6        # a cell active in 6+ different calendar months is not seasonal
STATIC_MIN_DETECTIONS = 20   # a real fire rarely hits one ~1 km cell this many times

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(SRC, parse_dates=["acq_date"])
print(f"Loaded: {len(df):,} detections")

# Archive rows have a 'type'; the recent near-real-time rows don't (NaN)
archive = df[df["type"].notna()]
recent = df[df["type"].isna()]
print(f"Archive rows: {len(archive):,} | recent rows without a type: {len(recent):,}")

# 1. Keep only presumed vegetation fires from the archive (type 0)
veg = archive[archive["type"] == 0]
print(f"Archive vegetation fires (type 0): {len(veg):,}")

# 2. Recent rows can't be classified, so drop those sitting on a known static source
static = archive[archive["type"] == 2]
static_cells = set(zip((static["latitude"] / GRID).round().astype(int),
                       (static["longitude"] / GRID).round().astype(int)))
recent_cells = list(zip((recent["latitude"] / GRID).round().astype(int),
                        (recent["longitude"] / GRID).round().astype(int)))
on_static = pd.Series([c in static_cells for c in recent_cells], index=recent.index)
recent_kept = recent[~on_static]
print(f"Recent rows on known static sources removed: {int(on_static.sum()):,} | kept: {len(recent_kept):,}")

clean = pd.concat([veg, recent_kept], ignore_index=True)

# 3. Drop low-confidence detections
conf = pd.to_numeric(clean["confidence"], errors="coerce")
n = len(clean)
clean = clean[conf >= MIN_CONFIDENCE].copy()
print(f"Low-confidence detections removed (confidence < {MIN_CONFIDENCE}): {n - len(clean):,}")

# 4. Remove persistent fixed sources the type flag missed (same idea as for VIIRS)
clean["cell_lat"] = (clean["latitude"] / GRID).round().astype(int)
clean["cell_lon"] = (clean["longitude"] / GRID).round().astype(int)
cell_stats = clean.groupby(["cell_lat", "cell_lon"]).agg(
    n=("acq_date", "size"),
    months=("acq_date", lambda s: s.dt.month.nunique()),
)
flagged = cell_stats[
    (cell_stats["months"] >= STATIC_MIN_MONTHS) | (cell_stats["n"] >= STATIC_MIN_DETECTIONS)
].index
blocked = {(a + da, b + db) for a, b in flagged for da in (-1, 0, 1) for db in (-1, 0, 1)}
is_static = pd.Series(
    [(a, b) in blocked for a, b in zip(clean["cell_lat"], clean["cell_lon"])], index=clean.index
)
print(f"Fixed-source cells flagged: {len(flagged):,} | detections removed (with ~1 km buffer): "
      f"{int(is_static.sum()):,} ({is_static.mean():.0%})")

# Sanity check: removed detections should look like fixed sources (night, all year round),
# kept detections like wildfires (day, summer)
removed = clean[is_static]
removed.to_csv(REMOVED, index=False)
clean = clean[~is_static].drop(columns=["cell_lat", "cell_lon"])
if len(removed):
    print("Sanity check (removed / kept):")
    print(f"  night share:   {(removed['daynight'] == 'N').mean():.2f} / {(clean['daynight'] == 'N').mean():.2f}")
    print(f"  Jun-Sep share: {removed['acq_date'].dt.month.between(6, 9).mean():.2f} / "
          f"{clean['acq_date'].dt.month.between(6, 9).mean():.2f}")

clean = clean.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)
clean.to_csv(CLEAN, index=False)
print(f"\nFinal clean dataset: {len(clean):,} detections -> {CLEAN}\n")

clean["year"] = clean["acq_date"].dt.year
clean["month"] = clean["acq_date"].dt.month

# Chart 1: detections per year. Aqua only launched in mid-2002, so 2001-2002 have one satellite;
# the last year is incomplete.
per_year = clean["year"].value_counts().sort_index()
partial = {per_year.index.min(), per_year.index.min() + 1, per_year.index.max()}
colors = ["#f4a582" if y in partial else "#b2182b" for y in per_year.index]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(per_year.index.astype(str), per_year.values, color=colors)
ax.set_title("Vegetation fire detections in Tunisia per year (MODIS)")
ax.set_ylabel("Detections")
ax.tick_params(axis="x", rotation=60)
fig.text(0.01, 0.01, "Lighter bars = partial coverage (first two years: Terra only; last year: incomplete)",
         fontsize=8, color="#555")
fig.tight_layout()
fig.savefig("outputs/modis_fires_per_year.png", dpi=150)
plt.close(fig)

# Chart 2: seasonality
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
per_month = clean["month"].value_counts().reindex(range(1, 13), fill_value=0)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(months, per_month.values, color="#b2182b")
ax.set_title("Seasonality of vegetation fire detections in Tunisia (MODIS, all years)")
ax.set_ylabel("Detections")
fig.tight_layout()
fig.savefig("outputs/modis_fires_per_month.png", dpi=150)
plt.close(fig)

print("Saved: outputs/modis_fires_per_year.png, outputs/modis_fires_per_month.png")