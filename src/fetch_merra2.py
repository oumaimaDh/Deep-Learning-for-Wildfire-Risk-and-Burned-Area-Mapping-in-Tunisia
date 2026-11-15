"""Download monthly MERRA-2 weather for northern Tunisia from NASA's OPeNDAP server.

Only the needed variables and a small box around Tunisia are requested, so each file is tiny.
Every month is saved to data/merra2/ as soon as it arrives, so you can stop and restart freely.
"""
import datetime as dt
import glob
import os
import re
import sys
import time

import pandas as pd

OUT_DIR = "data/merra2"
OUT_CSV = "data/merra2_tunisia_monthly.csv"
LAT = (33.0, 38.0)
LON = (7.0, 12.0)
START, END = dt.datetime(2011, 10, 1), dt.datetime(2025, 12, 31)  # a few months before 2012 for lagged rainfall
VERSION = "5.12.4"
# key: (NASA short name, variables)
COLLECTIONS = {
    "slv": ("M2TMNXSLV", ["T2M", "QV2M", "U10M", "V10M"]),      # 2 m temperature, humidity, 10 m wind
    "flx": ("M2TMNXFLX", ["PRECTOTCORR"]),                      # precipitation
    "lnd": ("M2TMNXLND", ["GWETROOT", "GWETTOP", "LAI"]),       # soil wetness, leaf area index
}


def year_month(url):
    m = re.search(r"\.(\d{6})\.nc4", url)
    return m.group(1) if m else None


def to_table(ds, variables, ym):
    """Turn one monthly dataset into rows of (year, month, lat, lon, variables...) for the Tunisia box."""
    sub = ds[variables].sel(lat=slice(*LAT), lon=slice(*LON))
    if "time" in sub.dims:
        sub = sub.isel(time=0)
    sub = sub.drop_vars("time", errors="ignore").load()
    df = sub.to_dataframe().reset_index()
    for v in variables:
        df[v] = df[v].where(df[v].abs() < 1e14)  # MERRA-2 marks missing values with 1e15
    df.insert(0, "month", int(ym[4:]))
    df.insert(0, "year", int(ym[:4]))
    return df[["year", "month", "lat", "lon"] + variables]


def combine():
    merged = None
    for key in COLLECTIONS:
        files = sorted(glob.glob(os.path.join(OUT_DIR, f"{key}_*.csv")))
        if not files:
            raise SystemExit(f"No files for '{key}' in {OUT_DIR}")
        part = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        merged = part if merged is None else merged.merge(part, on=["year", "month", "lat", "lon"], how="outer")
    merged = merged.sort_values(["year", "month", "lat", "lon"]).reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False)
    return merged


def main():
    import earthaccess
    import xarray as xr
    from pydap.client import get_cmr_urls
    from pydap.net import create_session

    os.makedirs(OUT_DIR, exist_ok=True)

    auth = earthaccess.login(strategy="all", persist=True)
    if not auth.authenticated:
        raise SystemExit("Earthdata login failed. Check your username and password at urs.earthdata.nasa.gov.")
    
    # ✅ Use this specific method to get a session with the correct auth headers for pydap
    _session = earthaccess.get_requests_https_session()
    session = create_session(session=_session)

    failed = []
    for key, (short_name, variables) in COLLECTIONS.items():
        print(f"\n=== {short_name}: {', '.join(variables)} ===")
        urls = get_cmr_urls(short_name=short_name, version=VERSION, time_range=[START, END], limit=250)
        items = sorted({year_month(u): u for u in urls if year_month(u)
                        and "201110" <= year_month(u) <= "202512"}.items())
        if not items:
            raise SystemExit(f"NASA's catalog returned no monthly files for {short_name}. Paste this message to me.")
        print(f"{len(items)} monthly files found ({items[0][0]} to {items[-1][0]})")

        ce = "?dap4.ce=/" + ";/".join(["lon", "lat", "time"] + variables)
        for i, (ym, url) in enumerate(items, 1):
            path = os.path.join(OUT_DIR, f"{key}_{ym}.csv")
            if os.path.exists(path):
                continue
            t0 = time.time()
            for attempt in range(3):
                try:
                    ds = xr.open_dataset(url.replace("https", "dap4", 1) + ce, engine="pydap", session=session)
                    to_table(ds, variables, ym).to_csv(path, index=False)
                    print(f"  {ym} done ({i}/{len(items)}, {time.time() - t0:.0f}s)")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  {ym} attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc)[:200]}")
                    if i == 1 and attempt == 2:
                        raise SystemExit(f"\nThe very first {short_name} file failed 3 times. Paste the messages above to me.")
                    time.sleep(5)
            else:
                failed.append((key, ym))

    if failed:
        print(f"\n{len(failed)} files failed: {failed[:10]} ... Run the script again to retry only these.")
        sys.exit(1)

    merged = combine()
    print(f"\nSaved {OUT_CSV}: {len(merged):,} rows, {merged[['year', 'month']].drop_duplicates().shape[0]} months, "
          f"{merged[['lat', 'lon']].drop_duplicates().shape[0]} grid points")
    print(merged.describe().loc[["min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()