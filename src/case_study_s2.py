"""Sentinel-2 burn-severity case study: an independent check on the MODIS burned-area maps and the U-Net.

1. Finds the largest burned cluster of the chosen year in the MODIS MCD64A1 data (default 2022) and its burn dates.
2. Reads a small window of two Sentinel-2 images (20 m) from the public AWS archive: one before, one after the fire.
3. Computes dNBR = NBR(before) - NBR(after) and classes it with the USGS severity thresholds.
4. Compares the Sentinel-2 burned area with MCD64A1 and with the U-Net at 500 m.

No account is needed; only the window around the fire is downloaded (tens of MB at most).
Needs: pip install pystac-client rasterio scipy
Run:   python src/case_study_s2.py        (YEAR=2021 python ... picks another year)
"""
import datetime as dt
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
from scipy import ndimage

from fetch_mcd64a1 import PIX, R, X0, Y0, crop_window, pixel_to_latlon

YEAR = int(os.environ.get("YEAR", 2022))
BURN_DIR, UNET_PRED = "data/burned_area", "data/unet_predictions.npz"
GEOJSON = "data/tunisia_governorates.geojson"
MAX_WINDOW_DEG = 0.25
STAC = "https://earth-search.aws.element84.com/v1"
OUT_FIG, OUT_CSV = "outputs/case_study_dnbr.png", "outputs/case_study_summary.csv"
SEVERITY_EDGES = [0.10, 0.27, 0.44, 0.66]            # USGS dNBR classes
SEVERITY_NAMES = ["Unburned", "Low", "Moderate-low", "Moderate-high", "High"]
SEVERITY_COLORS = ["#d9d9d9", "#fee08b", "#fdae61", "#d73027", "#67001f"]
BURNED_DNBR = 0.27                                    # moderate-low or worse counts as burned
BAD_SCL = [0, 1, 3, 8, 9, 10]                         # no data, saturated, cloud shadow, cloud, cirrus
S2_PIXEL_KM2 = 0.02 * 0.02
PIX_KM2 = (PIX / 1000.0) ** 2                          # one 500 m MODIS pixel in km2
os.makedirs("outputs", exist_ok=True)


# ---- 1. Find the fire in the MODIS burned-area data ---------------------------------------
def find_event(year):
    r0, r1, c0, c1 = crop_window()
    doy = None
    for m in range(6, 11):
        p = os.path.join(BURN_DIR, f"burn_{year}{m:02d}.npz")
        if os.path.exists(p):
            a = np.load(p)["burn_date"]
            doy = np.where(a > 0, a, 0) if doy is None else np.where(a > 0, a, doy)
    if doy is None or not (doy > 0).any():
        raise SystemExit(f"No MCD64A1 burned pixels found for {year}.")
    burned = doy > 0
    if os.path.exists(GEOJSON):                           # the crop box includes part of Algeria: keep Tunisia only
        import json
        plat, plon = pixel_to_latlon(*np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij"))
        pts = np.column_stack([plon.ravel(), plat.ravel()])
        tun = np.zeros(len(pts), dtype=bool)
        for ft in json.load(open(GEOJSON, encoding="utf-8"))["features"]:
            g = ft["geometry"]
            for part in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                ring = np.asarray(part[0])[:, :2]
                near = np.nonzero((pts[:, 0] >= ring[:, 0].min()) & (pts[:, 0] <= ring[:, 0].max()) &
                                  (pts[:, 1] >= ring[:, 1].min()) & (pts[:, 1] <= ring[:, 1].max()))[0]
                if len(near):
                    tun[near] |= Path(ring).contains_points(pts[near])
        burned = burned & tun.reshape(burned.shape)
        doy = np.where(burned, doy, 0)
        if not burned.any():
            raise SystemExit(f"No MCD64A1 burned pixels inside Tunisia for {year}.")
    lab, n = ndimage.label(ndimage.binary_dilation(burned, structure=np.ones((3, 3)), iterations=5), structure=np.ones((3, 3)))
    counts = np.bincount(lab[burned], minlength=n + 1)
    counts[0] = 0
    cluster = burned & (lab == counts.argmax())
    rows, cols = np.nonzero(cluster)
    lat, lon = pixel_to_latlon(rows + r0, cols + c0)
    pad = 0.03
    bbox = [float(lon.min() - pad), float(lat.min() - pad), float(lon.max() + pad), float(lat.max() + pad)]
    for lo, hi in ((0, 2), (1, 3)):                       # keep the window small enough for one Sentinel-2 tile
        if bbox[hi] - bbox[lo] > MAX_WINDOW_DEG:
            mid = (bbox[hi] + bbox[lo]) / 2
            bbox[lo], bbox[hi] = mid - MAX_WINDOW_DEG / 2, mid + MAX_WINDOW_DEG / 2
    d1, d2 = np.percentile(doy[cluster], [2, 98])
    day = lambda d: dt.date(year, 1, 1) + dt.timedelta(days=int(d) - 1)
    return {"year": year, "doy_map": doy, "bbox": bbox, "start": day(d1), "end": day(d2),
            "area_km2": float(cluster.sum() * PIX_KM2), "centre": (float(lat.mean()), float(lon.mean()))}


# ---- 2. Read two Sentinel-2 images --------------------------------------------------------
def href(item, key):
    return item.assets[key].href.replace("s3://sentinel-cogs/", "https://sentinel-cogs.s3.us-west-2.amazonaws.com/")


def grid_of(item):
    return item.properties.get("grid:code") or item.id.split("_")[1]


class Skip(Exception):
    pass


def read_item(item, bbox):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(href(item, "nir08")) as src:
            win = from_bounds(*transform_bounds("EPSG:4326", src.crs, *bbox), transform=src.transform)
            win = win.round_offsets().round_lengths()
            if win.col_off < 0 or win.row_off < 0 or win.col_off + win.width > src.width or win.row_off + win.height > src.height:
                raise Skip("image does not cover the whole window")
            transform, crs = src.window_transform(win), src.crs
            nir = src.read(1, window=win).astype(np.float32)
        with rasterio.open(href(item, "swir22")) as src:
            swir = src.read(1, window=win).astype(np.float32)
        with rasterio.open(href(item, "scl")) as src:
            scl = src.read(1, window=win)
    if not (nir.shape == swir.shape == scl.shape):
        raise Skip("band shapes differ")
    offset = -1000.0 if float(item.properties.get("s2:processing_baseline", "04.00")) >= 4.0 else 0.0
    valid = ~np.isin(scl, BAD_SCL) & (nir > 0) & (swir > 0)
    nir, swir = (nir + offset) / 1e4, (swir + offset) / 1e4
    if valid.mean() < 0.7:
        raise Skip(f"too cloudy over the window (clean pixels {valid.mean():.0%})")
    return {"nir": nir, "swir": swir, "valid": valid, "transform": transform, "crs": crs,
            "date": item.datetime.date(), "id": item.id, "cloud": item.properties.get("eo:cloud_cover")}


def find_pair(event):
    from pystac_client import Client

    cat = Client.open(STAC)
    bbox = event["bbox"]
    search = lambda a, b: sorted(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{a}/{b}",
                                            query={"eo:cloud_cover": {"lt": 50}}, max_items=100).item_collection(),
                                 key=lambda it: it.properties.get("eo:cloud_cover", 100))
    pre_items = search(event["start"] - dt.timedelta(days=40), event["start"] - dt.timedelta(days=1))
    post_items = search(event["end"] + dt.timedelta(days=2), event["end"] + dt.timedelta(days=40))
    print(f"Fire window {event['start']} to {event['end']}: {len(pre_items)} candidate images before, {len(post_items)} after")
    why, cache = {}, {}

    def get(it, label):
        if it.id not in cache:
            try:
                cache[it.id] = read_item(it, bbox)
            except Skip as e:
                cache[it.id] = None
                why[f"{label}: {e}"] = why.get(f"{label}: {e}", 0) + 1
        return cache[it.id]

    tried = 0
    for pre_it in pre_items[:10]:
        pre = get(pre_it, "before")
        if pre is None:
            continue
        print(f"  before candidate {pre['id']} is usable, looking for an 'after' image from the same tile ...")
        for post_it in post_items[:15]:
            if grid_of(post_it) != grid_of(pre_it):
                continue                                  # same tile, so the pixels line up exactly
            post = get(post_it, "after")
            if post is not None and post["nir"].shape == pre["nir"].shape:
                return pre, post
        tried += 1
    print("Images that were rejected:")
    for k, v in sorted(why.items()):
        print(f"  {v} x {k}")
    raise SystemExit("Could not find a clean before/after pair. Paste this output to me.")


# ---- 3. dNBR and comparison ---------------------------------------------------------------
def dice(a, b):
    tp = int((a & b).sum())
    return 2 * tp / max(int(a.sum()) + int(b.sum()), 1)


def analyse(pre, post, event):
    import rasterio.warp

    nbr = lambda d: (d["nir"] - d["swir"]) / np.maximum(d["nir"] + d["swir"], 1e-6)
    dnbr = nbr(pre) - nbr(post)
    valid = pre["valid"] & post["valid"]
    sev = np.digitize(dnbr, SEVERITY_EDGES)
    burned = valid & (dnbr >= BURNED_DNBR)
    rows = {"fire_start": str(event["start"]), "fire_end": str(event["end"]),
            "before_image": str(pre["date"]), "after_image": str(post["date"]),
            "s2_burned_km2": burned.sum() * S2_PIXEL_KM2}
    for k, name in enumerate(SEVERITY_NAMES[1:], start=1):
        rows[f"s2_{name.lower().replace('-', '_')}_km2"] = (valid & (sev == k)).sum() * S2_PIXEL_KM2

    # map every Sentinel-2 pixel onto the 500 m MODIS grid
    h, w = dnbr.shape
    cc, rr = np.meshgrid(np.arange(w), np.arange(h))
    tr = post["transform"]
    xs, ys = tr.c + (cc + 0.5) * tr.a, tr.f + (rr + 0.5) * tr.e
    lon, lat = rasterio.warp.transform(post["crs"], "EPSG:4326", xs.ravel().tolist(), ys.ravel().tolist())
    lon, lat = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
    r0, r1, c0, c1 = crop_window()
    mrow = np.floor((Y0 - R * lat) / PIX).astype(int) - r0
    mcol = np.floor((R * lon * np.cos(lat) - X0) / PIX).astype(int) - c0
    H, W = r1 - r0, c1 - c0
    inside = (mrow >= 0) & (mrow < H) & (mcol >= 0) & (mcol < W)
    idx = np.where(inside, mrow * W + mcol, 0)
    nvalid = np.bincount(idx[inside & valid.ravel()], minlength=H * W)
    nburn = np.bincount(idx[inside & burned.ravel()], minlength=H * W)
    ok = nvalid >= 200                                      # MODIS pixels well covered by clean Sentinel-2 pixels
    ref = (nburn >= 0.5 * np.maximum(nvalid, 1)) & ok       # Sentinel-2 says "burned" for at least half of the pixel

    d0, d1 = pre["date"].timetuple().tm_yday, post["date"].timetuple().tm_yday
    mcd = ((event["doy_map"] >= d0) & (event["doy_map"] <= d1)).ravel()
    rows["mcd64a1_km2_in_window"] = int((mcd & ok).sum()) * PIX_KM2
    rows["s2_burned_km2_on_modis_pixels"] = int(ref.sum()) * PIX_KM2
    rows["dice_mcd64a1_vs_s2"] = dice(mcd[ok], ref[ok])
    unet = None
    if os.path.exists(UNET_PRED):
        z = np.load(UNET_PRED)
        parts = [z[k] for k in z.files if k.startswith(f"{event['year']}_") and int(k.split("_")[1]) + 48 >= d0 and int(k.split("_")[1]) + 16 <= d1]
        if parts:
            unet = np.any(np.stack(parts), axis=0).ravel().astype(bool)
            rows["unet_km2_in_window"] = int((unet & ok).sum()) * PIX_KM2
            rows["dice_unet_vs_s2"] = dice(unet[ok], ref[ok])
    # figure
    fig, axs = plt.subplots(1, 4 if unet is not None else 3, figsize=(17 if unet is not None else 13, 5))
    cmap = ListedColormap(SEVERITY_COLORS)
    show = np.where(valid, sev, np.nan)
    axs[0].imshow(show, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axs[0].set_title(f"Sentinel-2 dNBR severity\n{pre['date']} to {post['date']}")
    for ax, mask, title in ((axs[1], mcd, "MODIS MCD64A1 burned area"), (axs[2], unet, "U-Net prediction")):
        if mask is None:
            continue
        img = np.where(inside, mask[idx], False).reshape(h, w)
        ax.imshow(np.where(valid, sev, np.nan), cmap=cmap, vmin=0, vmax=4, interpolation="nearest", alpha=0.35)
        ax.contour(img.astype(float), levels=[0.5], colors="k", linewidths=0.8)
        ax.set_title(title + " (outline)")
    last = axs[-1]
    areas = [(valid & (sev == k)).sum() * S2_PIXEL_KM2 for k in range(1, 5)]
    last.bar(SEVERITY_NAMES[1:], areas, color=SEVERITY_COLORS[1:])
    last.set_ylabel("km2"); last.set_title("Sentinel-2 burned area by severity")
    last.tick_params(axis="x", rotation=25)
    for ax in axs[:-1]:
        ax.axis("off")
    if unet is None:
        axs[2].axis("off")
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=150); plt.close(fig)
    pd.DataFrame([rows]).round(3).T.rename(columns={0: "value"}).to_csv(OUT_CSV)
    return rows


def main():
    ev = find_event(YEAR)
    print(f"{YEAR}: largest MODIS burn cluster {ev['area_km2']:.0f} km2 near lat {ev['centre'][0]:.2f}, lon {ev['centre'][1]:.2f}, "
          f"burn dates {ev['start']} to {ev['end']}")
    pre, post = find_pair(ev)
    print(f"Using {pre['id']} (before, {pre['cloud']:.0f}% cloud) and {post['id']} (after, {post['cloud']:.0f}% cloud)")
    rows = analyse(pre, post, ev)
    print("\nCase study results:")
    for k, v in rows.items():
        print(f"  {k}: {v if isinstance(v, str) else round(v, 3)}")
    print(f"\nSaved: {OUT_FIG}, {OUT_CSV}")


if __name__ == "__main__":
    main()