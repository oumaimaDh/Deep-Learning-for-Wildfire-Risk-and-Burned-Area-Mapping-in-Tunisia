"""Download SRTM elevation (90 m) for northern Tunisia and summarise it on a 1 km grid.

Downloads 20 small tiles (about 3 MB each), computes elevation, slope and how south-facing the ground is,
averages them over 0.01 degree blocks and saves data/terrain_1km.npz. Safe to stop and restart.
Needs: pip install earthaccess   (same Earthdata login as before)
"""
import glob
import os
import zipfile

import numpy as np

RAW_DIR = "data/srtm_raw"
OUT = "data/terrain_1km.npz"
LATS, LONS = range(34, 38), range(7, 12)          # tiles N34..N37, E007..E011 cover the northern region
N = 1201                                          # samples per tile edge (3 arc-seconds)
BLOCK = 12                                        # 12 samples = 0.01 degrees (about 1 km)


def tile_name(lat, lon):
    return f"N{lat:02d}E{lon:03d}"


def read_tile(path):
    with zipfile.ZipFile(path) as z:
        member = [n for n in z.namelist() if n.lower().endswith(".hgt")][0]
        raw = z.read(member)
    z = np.frombuffer(raw, dtype=">i2").reshape(N, N).astype(np.float32)
    z[z <= -32000] = np.nan                       # voids
    return z


def block_mean(a):
    a = a[:N - 1, :N - 1]                         # drop the shared last row and column
    return np.nanmean(a.reshape((N - 1) // BLOCK, BLOCK, (N - 1) // BLOCK, BLOCK), axis=(1, 3))


def summarise_tile(z, lat):
    step = 1.0 / (N - 1)
    dy = 110570.0 * step
    dx = 111320.0 * np.cos(np.radians(lat + 1 - (np.arange(N) + 0.5) * step))[:, None] * step
    dz_east = np.gradient(z, axis=1) / dx
    dz_north = -np.gradient(z, axis=0) / dy       # rows run from north to south
    grad = np.hypot(dz_east, dz_north)
    slope = np.degrees(np.arctan(grad))
    return {"elev": block_mean(z), "elev_sq": block_mean(z * z), "slope": block_mean(slope),
            "dz_north": block_mean(dz_north), "grad": block_mean(grad)}


def main():
    import earthaccess

    os.makedirs(RAW_DIR, exist_ok=True)
    auth = earthaccess.login(strategy="all", persist=True)
    if not auth.authenticated:
        raise SystemExit("Earthdata login failed.")

    results = earthaccess.search_data(short_name="SRTMGL3", version="003",
                                      bounding_box=(7.0, 34.0, 12.0, 38.0), count=200)
    wanted = {tile_name(la, lo) for la in LATS for lo in LONS}
    todo = []
    for g in results:
        links = [u for u in g.data_links() if u.endswith(".hgt.zip")]
        if links and os.path.basename(links[0]).split(".")[0] in wanted:
            if not os.path.exists(os.path.join(RAW_DIR, os.path.basename(links[0]))):
                todo.append(g)
    print(f"{len(results)} tiles in the search, {len(todo)} still to download (about 3 MB each)")
    if todo:
        earthaccess.download(todo, RAW_DIR)

    keys = ["elev", "elev_sq", "slope", "dz_north", "grad"]
    nb = (N - 1) // BLOCK                          # 100 blocks per tile edge
    grids = {k: np.full((len(LATS) * nb, len(LONS) * nb), np.nan, np.float32) for k in keys}
    found = 0
    for la in LATS:
        for lo in LONS:
            files = glob.glob(os.path.join(RAW_DIR, f"{tile_name(la, lo)}.*.hgt.zip"))
            if not files:
                continue
            res = summarise_tile(read_tile(files[0]), la)
            r = (LATS[-1] - la) * nb               # the northernmost tile goes on top
            c = (lo - LONS[0]) * nb
            for k in keys:
                grids[k][r:r + nb, c:c + nb] = res[k]
            found += 1
    print(f"Tiles processed: {found} of {len(LATS) * len(LONS)}")
    if found == 0:
        raise SystemExit("No tiles found. Paste the messages above to me.")
    np.savez_compressed(OUT, lat_top=LATS[-1] + 1.0, lon_left=float(LONS[0]), step=BLOCK / (N - 1), **grids)

    elev = grids["elev"]
    print(f"Saved {OUT}: {elev.shape[0]} x {elev.shape[1]} blocks of 0.01 degrees")
    print(f"Elevation: min {np.nanmin(elev):.0f} m, max {np.nanmax(elev):.0f} m, mean {np.nanmean(elev):.0f} m")
    print(f"Mean slope: {np.nanmean(grids['slope']):.1f} degrees")


if __name__ == "__main__":
    main()