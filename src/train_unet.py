"""Burned-area segmentation: PyTorch U-Net vs a per-pixel Random Forest vs a simple dNBR threshold.

Input : two MODIS MOD13A1 composites 32 days apart (red, NIR, SWIR, NDVI for each, plus dNBR and dNDVI).
Label : MODIS MCD64A1 burned pixels whose burn date falls between the two composites.
Split : train 2012-2020, validation 2021 (picks the thresholds), test 2022-2025 (never used for tuning).
Run   : python src/train_unet.py     (set EPOCHS lower with the environment variable EPOCHS to try it quickly)
"""
import datetime as dt
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.path import Path
from sklearn.ensemble import RandomForestClassifier

from fetch_mcd64a1 import crop_window, pixel_to_latlon

BURN_DIR, VEG_DIR = "data/burned_area", "data/vegetation"
GEOJSON = "data/tunisia_governorates.geojson"
OUT_METRICS, OUT_YEARS = "outputs/unet_metrics.csv", "outputs/unet_metrics_by_year.csv"
OUT_FIG, OUT_CURVE, OUT_MODEL = "outputs/unet_examples.png", "outputs/unet_training.png", "data/unet.pt"
YEARS = range(2012, 2026)
TRAIN_Y, VAL_Y, TEST_Y = range(2012, 2021), [2021], range(2022, 2026)
DOYS = [129, 161, 193, 225, 257, 289]
EPOCHS = int(os.environ.get("EPOCHS", 30))
STEPS, BATCH, PATCH = 60, 16, 128
SEED = 0

os.makedirs("outputs", exist_ok=True)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---- 1. Build samples ---------------------------------------------------------------------
def month_range(y, m):
    a = (dt.date(y, m, 1) - dt.date(y, 1, 1)).days + 1
    b = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.date(y, 1, 1)).days + 1
    return a, b                                    # day-of-year range [a, b)


def nbr(nir, mir):
    return (nir - mir) / np.maximum(nir + mir, 1e-6)


def load_veg(y, d):
    z = np.load(os.path.join(VEG_DIR, f"veg_{y}{d:03d}.npz"))
    ok = (z["ndvi"] > -3000) & (z["red"] > -1000) & (z["nir"] > -1000) & (z["mir"] > -1000)
    red, nir, mir, ndvi = [np.where(ok, z[k] * 1e-4, 0.0).astype(np.float32) for k in ("red", "nir", "mir", "ndvi")]
    return red, nir, mir, ndvi, ok


samples = []
for y in YEARS:
    burn = {m: np.load(os.path.join(BURN_DIR, f"burn_{y}{m:02d}.npz"))["burn_date"] for m in range(6, 11)
            if os.path.exists(os.path.join(BURN_DIR, f"burn_{y}{m:02d}.npz"))}
    for d in DOYS[:-1]:
        d2 = d + 32
        pa, pb = os.path.join(VEG_DIR, f"veg_{y}{d:03d}.npz"), os.path.join(VEG_DIR, f"veg_{y}{d2:03d}.npz")
        if not (os.path.exists(pa) and os.path.exists(pb)) or not burn:
            continue
        a, b = max(d + 16, 152), min(d + 48, 305)           # burn days between the two composites
        label, valid = None, None
        for m, arr in burn.items():
            ma, mb = month_range(y, m)
            if mb <= a or ma >= b:
                continue
            hit = (arr >= max(a, ma)) & (arr < min(b, mb))
            bad = (arr == -1) | (arr == -2)
            label = hit if label is None else (label | hit)
            valid = ~bad if valid is None else (valid & ~bad)
        ra, na, ma_, nda, oka = load_veg(y, d)
        rb, nb, mb_, ndb, okb = load_veg(y, d2)
        dnbr = nbr(na, ma_) - nbr(nb, mb_)
        dndvi = nda - ndb
        x = np.stack([ra, na, ma_, nda, rb, nb, mb_, ndb, dnbr, dndvi]).astype(np.float16)
        samples.append({"year": y, "doy": d, "x": x, "y": label.astype(np.uint8),
                        "valid": valid & oka & okb})
if not samples:
    raise SystemExit("No samples built. Check data/burned_area and data/vegetation.")
H, W = samples[0]["y"].shape
tr = [s for s in samples if s["year"] in TRAIN_Y]
va = [s for s in samples if s["year"] in VAL_Y]
te = [s for s in samples if s["year"] in TEST_Y]

# The crop box also contains a strip of Algeria. Validation and test scores count Tunisian pixels only.
if os.path.exists(GEOJSON):
    import json
    r0, r1, c0, c1 = crop_window()
    plat, plon = pixel_to_latlon(*np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij"))
    pts = np.column_stack([plon.ravel(), plat.ravel()])
    tun = np.zeros(len(pts), dtype=bool)
    for ft in json.load(open(GEOJSON, encoding="utf-8"))["features"]:
        g = ft["geometry"]
        for part in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            tun |= Path(np.asarray(part[0])[:, :2]).contains_points(pts)
    tun = tun.reshape(H, W)
    for s_ in va + te:
        s_["valid"] = s_["valid"] & tun
    print(f"Evaluation restricted to Tunisia ({tun.mean():.0%} of the crop box)")
print(f"{len(samples)} image pairs of {H}x{W} pixels | train {len(tr)}, validation {len(va)}, test {len(te)}")
for name, grp in (("train", tr), ("validation", va), ("test", te)):
    pos = sum(int((s["y"] & s["valid"]).sum()) for s in grp)
    tot = sum(int(s["valid"].sum()) for s in grp)
    print(f"  {name}: {pos:,} burned pixels of {tot:,} valid ({pos / tot:.3%})")
if sum(int(s["y"].sum()) for s in te) == 0:
    raise SystemExit("No burned pixels in the test years.")


# ---- 2. U-Net -----------------------------------------------------------------------------
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, cin=10, c=16):
        super().__init__()
        self.e1, self.e2, self.e3, self.mid = block(cin, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)[:, 0]


def batch_of(group, n):
    xs, ys, vs = [], [], []
    for _ in range(n):
        s = group[rng.integers(len(group))]
        if rng.random() < 0.7 and s["y"].any():              # centre half of the patches on a burn
            rr, cc = np.nonzero(s["y"])
            k = rng.integers(len(rr))
            r0 = int(np.clip(rr[k] - PATCH // 2 + rng.integers(-30, 31), 0, H - PATCH))
            c0 = int(np.clip(cc[k] - PATCH // 2 + rng.integers(-30, 31), 0, W - PATCH))
        else:
            r0, c0 = rng.integers(0, H - PATCH + 1), rng.integers(0, W - PATCH + 1)
        sl = (slice(r0, r0 + PATCH), slice(c0, c0 + PATCH))
        x, y, v = s["x"][:, sl[0], sl[1]].astype(np.float32), s["y"][sl], s["valid"][sl]
        if rng.random() < 0.5:
            x, y, v = x[:, :, ::-1], y[:, ::-1], v[:, ::-1]
        if rng.random() < 0.5:
            x, y, v = x[:, ::-1], y[::-1], v[::-1]
        xs.append(x.copy()); ys.append(y.copy()); vs.append(v.copy())
    t = lambda a, dtp: torch.from_numpy(np.stack(a)).to(device, dtp)
    return t(xs, torch.float32), t(ys, torch.float32), t(vs, torch.float32)


def loss_fn(logit, y, v):
    bce = (F.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=torch.tensor(5.0, device=device)) * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logit) * v
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * v).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict(model, s):
    model.eval()
    x = torch.from_numpy(s["x"].astype(np.float32))[None].to(device)
    ph, pw = (-H) % 8, (-W) % 8
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return torch.sigmoid(model(x))[0, :H, :W].cpu().numpy()


model = UNet().to(device)
LOADED = bool(os.environ.get("LOAD")) and os.path.exists(OUT_MODEL)   # LOAD=1 reuses the saved model, no retraining
if LOADED:
    model.load_state_dict(torch.load(OUT_MODEL, map_location=device))
    print(f"\nLoaded {OUT_MODEL}, skipping training")
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(EPOCHS, 1))
if not LOADED:
    print(f"\nTraining the U-Net on {device} for {EPOCHS} epochs ({STEPS} steps of {BATCH} patches each) ...")
curve, t0 = [], time.time()
for ep in range(0 if LOADED else EPOCHS):
    model.train()
    tot = 0.0
    for _ in range(STEPS):
        x, y, v = batch_of(tr, BATCH)
        opt.zero_grad()
        l = loss_fn(model(x), y, v)
        l.backward()
        opt.step()
        tot += l.item()
    sched.step()
    curve.append(tot / STEPS)
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f"  epoch {ep + 1}/{EPOCHS}: loss {curve[-1]:.3f} ({time.time() - t0:.0f}s)")
if not LOADED:
    torch.save(model.state_dict(), OUT_MODEL)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(1, EPOCHS + 1), curve, color="#b2182b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); fig.tight_layout(); fig.savefig(OUT_CURVE, dpi=150); plt.close(fig)


# ---- 3. Scores and thresholds (chosen on validation only) ---------------------------------
def counts(score, group, thr):
    tp = fp = fn = 0
    for s, sc in zip(group, score):
        pred, y, v = sc >= thr, s["y"].astype(bool), s["valid"]
        tp += int((pred & y & v).sum()); fp += int((pred & ~y & v).sum()); fn += int((~pred & y & v).sum())
    return tp, fp, fn


def dice_iou(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def best_threshold(score, group, grid):
    return max(grid, key=lambda t: dice_iou(*counts(score, group, t))[0])


THR = np.r_[np.linspace(0.1, 0.9, 17), 0.95, 0.97, 0.98, 0.99, 0.995]   # fine steps near 1: the first run hit the 0.95 edge
results = {}
# a) dNBR threshold
dn_va = [s["x"][8].astype(np.float32) for s in va]
dn_te = [s["x"][8].astype(np.float32) for s in te]
t = best_threshold(dn_va, va, np.linspace(0.02, 0.6, 30))
results["Simple dNBR threshold"] = (dn_te, t)

# b) per-pixel Random Forest
Xs, ys = [], []
for s in tr:
    v = s["valid"]
    pos = np.argwhere(v & (s["y"] == 1))
    neg = np.argwhere(v & (s["y"] == 0))
    neg = neg[rng.choice(len(neg), min(len(neg), 4000), replace=False)]
    idx = np.vstack([pos[:3000], neg]) if len(pos) else neg
    Xs.append(s["x"][:, idx[:, 0], idx[:, 1]].T.astype(np.float32)); ys.append(s["y"][idx[:, 0], idx[:, 1]])
rf = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=SEED).fit(np.vstack(Xs), np.concatenate(ys))


def rf_scores(group):
    out = []
    for s in group:
        sc = np.zeros((H, W), np.float32)
        m = s["valid"]
        sc[m] = rf.predict_proba(s["x"][:, m].T.astype(np.float32))[:, 1]
        out.append(sc)
    return out


rf_va, rf_te = rf_scores(va), rf_scores(te)
results["Random Forest (per pixel)"] = (rf_te, best_threshold(rf_va, va, THR))

# c) U-Net
un_va = [predict(model, s) for s in va]
un_te = [predict(model, s) for s in te]
results["U-Net"] = (un_te, best_threshold(un_va, va, THR))

rows = []
for name, (score, thr) in results.items():
    d, i, p, r = dice_iou(*counts(score, te, thr))
    rows.append({"model": name, "Dice": d, "IoU": i, "precision": p, "recall": r, "threshold": thr})
metrics = pd.DataFrame(rows)
metrics.round(3).to_csv(OUT_METRICS, index=False)
print(f"\nTest years {TEST_Y[0]}-{TEST_Y[-1]} (thresholds picked on {VAL_Y[0]}):")
print(metrics.round(3).to_string(index=False))

by_year = []
for yr in TEST_Y:
    idx = [k for k, s in enumerate(te) if s["year"] == yr]
    row = {"year": yr, "burned_km2": sum(int((te[k]["y"] & te[k]["valid"]).sum()) for k in idx) * 0.463 ** 2}
    for name, (score, thr) in results.items():
        row[name] = dice_iou(*counts([score[k] for k in idx], [te[k] for k in idx], thr))[0]
    by_year.append(row)
by_year = pd.DataFrame(by_year)
by_year.round(3).to_csv(OUT_YEARS, index=False)
print("\nDice per test year:")
print(by_year.round(2).to_string(index=False))

# predictions of the test pairs, used by src/case_study_s2.py
np.savez_compressed("data/unet_predictions.npz", **{f"{sm['year']}_{sm['doy']}": (un_te[i] >= results["U-Net"][1]).astype(np.uint8)
                                                    for i, sm in enumerate(te)})

# ---- 4. Example figure: the test pair with the most burned area ---------------------------
k = int(np.argmax([int((s["y"] & s["valid"]).sum()) for s in te]))
s, thr = te[k], results["U-Net"][1]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
rgb = np.clip(np.stack([s["x"][5], s["x"][4], s["x"][4] * 0 + s["x"][7] * 0.5]).transpose(1, 2, 0).astype(np.float32) * 2.5, 0, 1)
axs[0].imshow(np.clip(s["x"][[6, 5, 4]].transpose(1, 2, 0).astype(np.float32) * 2.5, 0, 1))
axs[0].set_title(f"After (SWIR-NIR-red), {s['year']} day {s['doy'] + 32}")
axs[1].imshow(s["y"], cmap="gray_r"); axs[1].set_title("MODIS MCD64A1 burned area (reference)")
axs[2].imshow(un_te[k] >= thr, cmap="gray_r"); axs[2].set_title("U-Net prediction")
for a in axs:
    a.axis("off")
fig.tight_layout(); fig.savefig(OUT_FIG, dpi=150); plt.close(fig)
print(f"\nSaved: {OUT_METRICS}, {OUT_YEARS}, {OUT_FIG}, {OUT_CURVE}, {OUT_MODEL}")