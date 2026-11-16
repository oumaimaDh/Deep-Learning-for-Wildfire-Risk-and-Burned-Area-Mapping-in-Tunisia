"""Download MODIS MOD13A1 (16-day, 500 m reflectance and NDVI) for northern Tunisia: the U-Net's input images.

Same tile and same 500 m grid as the burned-area labels (MCD64A1), so pixels line up exactly.
For each year 2012-2025 it takes six composites (start days 129, 161, 193, 225, 257, 289, about
one per month from May to October), crops the northern region and keeps red, NIR, MIR (SWIR) and NDVI.
The big HDF file is deleted after conversion to save disk space.

Safe to stop and restart. Needs: pip install earthaccess pyhdf (run fetch_mcd64a1.py first).
"""
import glob
import os
import re

import numpy as np

from fetch_mcd64a1 import TILE, YEARS, crop_window

RAW_DIR = "data/mod13a1_raw"
OUT_DIR = "data/vegetation"
DOYS = {129, 161, 193, 225, 257, 289}
BANDS = {"ndvi": "500m 16 days NDVI", "red": "500m 16 days red reflectance",
         "nir": "500m 16 days NIR reflectance", "mir": "500m 16 days MIR reflectance"}


def file_date(name):
    m = re.search(r"\.A(\d{4})(\d{3})\.", name)
    return int(m.group(1)), int(m.group(2))


def convert(path, window):
    from pyhdf.SD import SD, SDC

    r0, r1, c0, c1 = window
    year, doy = file_date(os.path.basename(path))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"veg_{year}{doy:03d}.npz")
    if os.path.exists(out):
        return out
    sd = SD(path, SDC.READ)
    data = {k: np.asarray(sd.select(name)[r0:r1, c0:c1]).astype(np.int16) for k, name in BANDS.items()}
    sd.end()
    np.savez_compressed(out, window=np.array(window), year=year, doy=doy, **data)
    return out


def main():
    import earthaccess

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    window = crop_window()

    auth = earthaccess.login(strategy="all", persist=True)
    if not auth.authenticated:
        raise SystemExit("Earthdata login failed.")

    results = earthaccess.search_data(short_name="MOD13A1", version="061",
                                      temporal=(f"{YEARS[0]}-05-01", f"{YEARS[-1]}-10-31"),
                                      bounding_box=(8.5, 35.5, 10.5, 36.5), count=2000)
    wanted = []
    for g in results:
        links = [u for u in g.data_links() if u.endswith(".hdf")]
        if not links or TILE not in links[0]:
            continue
        y, d = file_date(os.path.basename(links[0]))
        if y in YEARS and d in DOYS:
            wanted.append((g, y, d, links[0]))
    print(f"{len(wanted)} composites found (expected {len(YEARS) * len(DOYS)})")
    if not wanted:
        raise SystemExit("No files found. Paste this message to me.")

    # download and convert in small batches so that disk use stays low and progress is saved
    todo = [w for w in wanted if not os.path.exists(os.path.join(OUT_DIR, f"veg_{w[1]}{w[2]:03d}.npz"))]
    print(f"{len(todo)} still to download and convert")
    for i in range(0, len(todo), 6):
        batch = todo[i:i + 6]
        earthaccess.download([w[0] for w in batch], RAW_DIR)
        for _, y, d, url in batch:
            f = os.path.join(RAW_DIR, os.path.basename(url))
            try:
                convert(f, window)
                os.remove(f)
            except Exception as exc:  # noqa: BLE001
                print(f"  could not convert {os.path.basename(f)}: {type(exc).__name__}: {exc}")
        print(f"  {min(i + 6, len(todo))}/{len(todo)} done")

    n = len(glob.glob(os.path.join(OUT_DIR, "veg_*.npz")))
    print(f"\nConverted {n} composites in {OUT_DIR}/")
    sample = sorted(glob.glob(os.path.join(OUT_DIR, "veg_*.npz")))[0]
    d = np.load(sample)
    ndvi = d["ndvi"][d["ndvi"] > -3000] * 0.0001
    print(f"Sample {os.path.basename(sample)}: NDVI range {ndvi.min():.2f} to {ndvi.max():.2f}, mean {ndvi.mean():.2f}")


if __name__ == "__main__":
    main()