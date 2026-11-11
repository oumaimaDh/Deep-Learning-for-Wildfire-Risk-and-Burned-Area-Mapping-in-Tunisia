import glob
import os

import pandas as pd

RAW_DIR = "data/viirs_raw"
MERGED = "data/fires_tunisia_viirs.csv"

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
if not files:
    raise SystemExit(f"No CSV files found in {RAW_DIR}. Unzip the FIRMS download into that folder.")

# 1. Load every CSV from the download and stack them
parts = []
for f in files:
    d = pd.read_csv(f)
    print(f"{os.path.basename(f)}: {len(d):,} rows")
    parts.append(d)

df = pd.concat(parts, ignore_index=True)
df["acq_date"] = pd.to_datetime(df["acq_date"])

# 2. Remove duplicates in case the files overlap
before = len(df)
df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
print(f"\nMerged: {len(df):,} rows ({before - len(df):,} duplicates removed)")

df = df.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)
df.to_csv(MERGED, index=False)
print(f"Saved to {MERGED}\n")

# 3. Quick look at the data
print("Columns:", list(df.columns), "\n")
print(f"Date range: {df['acq_date'].min().date()} to {df['acq_date'].max().date()}\n")

print("Detections per year:")
print(df["acq_date"].dt.year.value_counts().sort_index(), "\n")

print("Detections per month (all years combined):")
print(df["acq_date"].dt.month.value_counts().sort_index(), "\n")

print("Confidence levels:")
print(df["confidence"].value_counts(), "\n")

print("Day vs night:")
print(df["daynight"].value_counts(), "\n")

# FIRMS archive files include a 'type' column: 0 = presumed vegetation fire,
# 1 = active volcano, 2 = other static land source (industry, gas flares), 3 = offshore
if "type" in df.columns:
    print("Detection type (0 = vegetation fire, 2 = static land source, 3 = offshore):")
    print(df["type"].value_counts(), "\n")
else:
    print("No 'type' column in this download.\n")