import pandas as pd

SRC = "data/fires_tunisia_viirs_clean.csv"
OUT = "data/top_hotspot_cells.csv"
GRID = 0.005  # about 500 m
TOP = 20

df = pd.read_csv(SRC, parse_dates=["acq_date"])
df["cell_lat"] = (df["latitude"] / GRID).round().astype(int)
df["cell_lon"] = (df["longitude"] / GRID).round().astype(int)
df["year"] = df["acq_date"].dt.year
df["month"] = df["acq_date"].dt.month

cells = (
    df.groupby(["cell_lat", "cell_lon"])
    .agg(
        detections=("acq_date", "size"),
        years=("year", "nunique"),
        months=("month", "nunique"),
        night_share=("daynight", lambda s: (s == "N").mean()),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
    )
    .sort_values("detections", ascending=False)
    .reset_index(drop=True)
)
cells[["lat", "lon", "night_share"]] = cells[["lat", "lon", "night_share"]].round(3)

print(f"Total detections: {len(df):,} in {len(cells):,} distinct 500 m cells\n")
print(f"Top {TOP} most-repeated cells:")
print(cells.head(TOP)[["lat", "lon", "detections", "years", "months", "night_share"]].to_string(index=False))

# Information only, nothing is removed here
persistent = cells[(cells["years"] >= 5) & (cells["months"] >= 6)]
print(f"\nCells active in 5+ different years AND 6+ different months: {len(persistent):,}")
print(f"Detections inside those cells: {int(persistent['detections'].sum()):,} "
      f"({persistent['detections'].sum() / len(df):.0%} of all detections)")

cells.head(200).to_csv(OUT, index=False)
print(f"\nTop 200 cells saved to {OUT}")