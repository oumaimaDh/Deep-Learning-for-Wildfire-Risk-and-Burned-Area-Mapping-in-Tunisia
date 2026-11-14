import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VIIRS = "data/fires_tunisia_viirs_clean.csv"
MODIS = "data/fires_tunisia_modis_clean.csv"
OUT = "outputs/sensor_comparison.png"
# Full years covered by both sensors: VIIRS starts in January 2012, MODIS has both satellites from 2003
FIRST_YEAR, LAST_YEAR = 2013, 2025

os.makedirs("outputs", exist_ok=True)

v = pd.read_csv(VIIRS, parse_dates=["acq_date"])
m = pd.read_csv(MODIS, parse_dates=["acq_date"])


def yearly(df):
    s = df["acq_date"].dt.year.value_counts().sort_index()
    return s.reindex(range(FIRST_YEAR, LAST_YEAR + 1), fill_value=0)


def monthly_share(df):
    s = df["acq_date"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    return 100 * s / s.sum()


yv, ym = yearly(v), yearly(m)
print(f"Cleaned detections per year, {FIRST_YEAR} to {LAST_YEAR}:")
print(pd.DataFrame({"VIIRS": yv, "MODIS": ym}).to_string(), "\n")

pearson = yv.corr(ym)
spearman = yv.rank().corr(ym.rank())  # rank correlation: do the two sensors order the years the same way?
print(f"Pearson correlation of yearly counts:  {pearson:.2f}")
print(f"Spearman rank correlation:             {spearman:.2f}")
print(f"(based on {len(yv)} years, so treat it as an indication, not proof)\n")

# Each sensor relative to its own average, because MODIS (1 km) detects far fewer fires than VIIRS (375 m)
iv, im = yv / yv.mean(), ym / ym.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax1.plot(iv.index, iv.values, marker="o", color="#b2182b", label="VIIRS (375 m)")
ax1.plot(im.index, im.values, marker="s", color="#2166ac", label="MODIS (1 km)")
ax1.axhline(1, color="#999", lw=0.8, ls="--")
ax1.set_xticks(list(iv.index))
ax1.tick_params(axis="x", rotation=60)
ax1.set_ylabel("Detections relative to the sensor's average")
ax1.set_title(f"Year-to-year pattern ({FIRST_YEAR}-{LAST_YEAR})\nSpearman rank correlation = {spearman:.2f}")
ax1.legend()

mv, mm = monthly_share(v), monthly_share(m)
x = range(1, 13)
w = 0.4
ax2.bar([i - w / 2 for i in x], mv.values, width=w, color="#b2182b", label="VIIRS")
ax2.bar([i + w / 2 for i in x], mm.values, width=w, color="#2166ac", label="MODIS")
ax2.set_xticks(list(x))
ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax2.set_ylabel("Share of each sensor's detections (%)")
ax2.set_title("Seasonality (all years)")
ax2.legend()

fig.suptitle("VIIRS vs MODIS: cleaned vegetation-fire detections in Tunisia", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print(f"Saved: {OUT}")