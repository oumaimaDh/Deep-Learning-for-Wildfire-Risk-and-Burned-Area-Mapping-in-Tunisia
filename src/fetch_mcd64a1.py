"""Download MODIS MCD64A1 monthly burned-area tiles for northern Tunisia and cut out the region.

Step 1: downloads the HDF file of tile h18v05 (which covers Tunisia) for the fire-season months.
Step 2: reads each file, crops the northern region and saves a small .npz in data/burned_area/.

Safe to stop and restart: files already downloaded or converted are skipped.
Needs: pip install earthaccess pyhdf
"""
import datetime as dt
import glob
import math
import os
import re

import numpy as np

RAW_DIR = "data/mcd64a1_raw"
OUT_DIR = "data/burned_area"
TILE = "h18v05"
YEARS = range(2012, 2026)
MONTHS = range(6, 11)                    # June to October: the fire season (about 70 files in total)
BOX = {"lat0": 34.5, "lat1": 37.6, "lon0": 7.5, "lon1": 11.5}   # northern Tunisia

# MODIS sinusoidal grid
R = 6371007.181
TILE_M = 1111950.5197
PIX = TILE_M / 2400
H, V = int(TILE[1:3]), int(TILE[4:6])
X0 = -20015109.354 + H * TILE_M          # left edge of the tile
Y0 = 10007554.677 - V * TILE_M           # top edge of the tile


def to_pixel(lat, lon):
    x = R * math.radians(lon) * math.cos(math.radians(lat))
    y = R * math.radians(lat)
    return (Y0 - y) / PIX, (x - X0) / PIX      # row, col


def pixel_to_latlon(rows, cols):
    y = Y0 - (np.asarray(rows) + 0.5) * PIX
    x = X0 + (np.asarray(cols) + 0.5) * PIX
    lat = y / R
    lon = x / (R * np.cos(lat))
    return np.degrees(lat), np.degrees(lon)


def crop_window():
    corners = [to_pixel(la, lo) for la in (BOX["lat0"], BOX["lat1"]) for lo in (BOX["lon0"], BOX["lon1"])]
    rows = [c[0] for c in corners]
    cols = [c[1] for c in corners]
    r0, r1 = int(math.floor(min(rows))), int(math.ceil(max(rows)))
    c0, c1 = int(math.floor(min(cols))), int(math.ceil(max(cols)))
    if r0 < 0 or c0 < 0 or r1 > 2400 or c1 > 2400:
        raise SystemExit(f"The region is not fully inside tile {TILE} (rows {r0}-{r1}, cols {c0}-{c1}).")
    return r0, r1, c0, c1


def file_month(name):
    m = re.search(r"\.A(\d{4})(\d{3})\.", name)
    d = dt.date(int(m.group(1)), 1, 1) + dt.timedelta(days=int(m.group(2)) - 1)
    return d.year, d.month


def convert(path, window):
    from pyhdf.SD import SD, SDC

    r0, r1, c0, c1 = window
    year, month = file_month(os.path.basename(path))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"burn_{year}{month:02d}.npz")
    if os.path.exists(out):
        return out
    sd = SD(path, SDC.READ)
    arr = np.asarray(sd.select("Burn Date")[r0:r1, c0:c1]).astype(np.int16)
    sd.end()
    np.savez_compressed(out, burn_date=arr, window=np.array(window), year=year, month=month)
    return out


def main():
    import earthaccess

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    window = crop_window()
    print(f"Tile {TILE}, crop rows {window[0]}-{window[1]}, cols {window[2]}-{window[3]} "
          f"({window[1] - window[0]} x {window[3] - window[2]} pixels of ~463 m)")

    auth = earthaccess.login(strategy="all", persist=True)
    if not auth.authenticated:
        raise SystemExit("Earthdata login failed.")

    results = earthaccess.search_data(short_name="MCD64A1", version="061",
                                      temporal=(f"{YEARS[0]}-06-01", f"{YEARS[-1]}-10-31"),
                                      bounding_box=(8.5, 35.5, 10.5, 36.5), count=1000)
    wanted = []
    for g in results:
        links = [u for u in g.data_links() if u.endswith(".hdf")]
        if not links or TILE not in links[0]:
            continue
        y, m = file_month(os.path.basename(links[0]))
        if y in YEARS and m in MONTHS:
            wanted.append((g, links[0]))
    print(f"{len(wanted)} monthly files for tile {TILE} (expected {len(YEARS) * len(MONTHS)})")
    if not wanted:
        raise SystemExit("No files found. Paste this message to me.")

    todo = [g for g, u in wanted if not os.path.exists(os.path.join(RAW_DIR, os.path.basename(u)))]
    print(f"{len(todo)} still to download (about 5-15 MB each)")
    if todo:
        earthaccess.download(todo, RAW_DIR)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.hdf")))
    for i, f in enumerate(files, 1):
        try:
            convert(f, window)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read {os.path.basename(f)}: {type(exc).__name__}: {exc}")
    print(f"\nConverted {len(glob.glob(os.path.join(OUT_DIR, 'burn_*.npz')))} months to {OUT_DIR}/")

    # quick summary: burned area per year
    print("\nBurned area per year (June-October, northern box):")
    for y in YEARS:
        total = 0
        for m in MONTHS:
            p = os.path.join(OUT_DIR, f"burn_{y}{m:02d}.npz")
            if os.path.exists(p):
                a = np.load(p)["burn_date"]
                total += int((a > 0).sum())
        print(f"  {y}: {total * PIX * PIX / 1e6:8.1f} km2")


if __name__ == "__main__":
    main()