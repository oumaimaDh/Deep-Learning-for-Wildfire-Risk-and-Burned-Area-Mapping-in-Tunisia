# Wildfire Patterns, Risk and Burned-Area Mapping in Tunisia: Technical Report

Oumaima Dhaouadi, November 2026. Code and figures: this repository. All numbers come from the scripts in `src/` and the tables in `outputs/`.

## Summary

This project analyzes 25 years of NASA satellite fire data for Tunisia, models where and when fires occur, and tests a deep-learning model for mapping burned areas. The main findings:

1. **Cleaning matters.** 66% of raw VIIRS detections (57,879 of them) turned out to be likely non-wildfire (gas flares, industry, low confidence). After cleaning, VIIRS and MODIS agree on which years were active (rank correlation 0.97), and cleaned VIIRS counts also agree with MODIS burned area (0.96).
2. **Fires concentrate in the northwest.** Jendouba, Béja and Bizerte cover about 7% of the country but hold 41% of cleaned VIIRS and 51% of MODIS detections. 102 persistent hotspots were confirmed by both sensors, 73% of them in these three governorates.
3. **Fire risk is predictable from season, place and history** (PR-AUC 0.13 against 0.04 for a seasonal average, on unseen years). Satellite vegetation features help at places with no fire history; coarse weather and terrain add little.
4. **Burned-area mapping at 500 m works for large fires but not for the small fires typical of Tunisia.** A U-Net beats baselines (Tunisia-only Dice 0.15 against 0.05 and 0.03) but is far from the 0.80 target. A Sentinel-2 check on one large fire shows the extents agree (Dice 0.94 for MODIS and 0.88 for the U-Net against 20 m dNBR).

Everything here is exploratory; the sample sizes and caveats are stated with each result.

## 1. Data

| Source | Use | Coverage |
|---|---|---|
| NASA FIRMS VIIRS S-NPP (375 m) | Active fires | 2012 to November 2026 |
| NASA FIRMS MODIS Terra/Aqua (1 km) | Active fires | 2001 to 2026 |
| MODIS MCD64A1 (500 m) | Burned-area labels | June to October, 2012 to 2025 |
| MODIS MOD13A1 (500 m, 16-day) | U-Net inputs, vegetation features | 6 composites per year, 2012 to 2025 |
| MERRA-2 monthly reanalysis | Temperature, humidity, wind, rain, soil wetness, LAI | 2011 to 2025 |
| SRTM 3 arc-second | Elevation, slope, aspect | 20 tiles |
| Sentinel-2 L2A (20 m) | Case study | July and August 2022 |
| geoBoundaries | Governorate polygons | |

## 2. Cleaning the fire detections

Applied in `src/analyze_fires.py` and `src/analyze_modis.py`: keep presumed vegetation fires (type 0), remove near-real-time detections near known static sources, drop low-confidence detections, and remove grid cells that are active in 6 or more calendar months or have too many detections (VIIRS: 40 per ~500 m cell; MODIS: 20 per ~1 km cell) with their neighbours.

- VIIRS: 57,879 raw to 19,753 clean. 46% of the archive was already flagged as static land sources, against 8% for MODIS.
- MODIS: 7,488 raw to 6,437 clean.
- Sanity check on the persistence filter: the removed detections are 91% night-time (VIIRS) and spread across the year, while the kept ones are mostly daytime and concentrated in June to September (83% for VIIRS and 89% for MODIS).

Thresholds are heuristics set after inspecting the most-repeated cells. VIIRS counts in the south and east probably still include small industrial sources (VIIRS records 8 to 19 times more detections than MODIS there, against 2 to 4 times in the north); the northern results hold for both sensors.

## 3. Results

### 3.1 Fire patterns and sensor agreement

- Fires are strongly seasonal: June to September holds 83% of cleaned VIIRS and 89% of MODIS detections, peaking in August.
- Over the 13 full years covered by both sensors (2013 to 2025), yearly counts correlate at 0.97 (Spearman). Both rank 2021, 2017 and 2014 as the most active years and 2018, 2024 and 2023 as the quietest.
- Independent check: MODIS burned area inside Tunisia (June to October) correlates at 0.96 with cleaned VIIRS counts inside Tunisia over the same years. Both give 2024, 2018 and 2023 as the three quietest years (burned area 8, 11 and 50 km²) and place 2021 and 2014 in the top two (256 and 213 km²). `outputs/burned_area_by_year.csv`.
- Detections per 100 km² of VIIRS: Bizerte 93, Béja 68, Jendouba 64 (first to third).

### 3.2 Persistent hotspots

DBSCAN (1 km distance, at least 10 detections) on cleaned VIIRS detections gives 255 clusters. 128 are persistent (active in 5 or more years); 102 of these are also seen by MODIS (3 or more detections nearby) and contain 73% of all clustered detections. 74 of the 102 are in Bizerte (41), Béja (25) and Jendouba (8). Twenty-six VIIRS-only persistent clusters were set aside as probably not forest fires. Persistence cannot separate forest fires from recurring agricultural burning without land-cover data, and the largest cluster (31 km across, El Kef) probably merges several fire areas.

### 3.3 Fire-risk model

Setup: 5 km grid cells in 12 northern governorates, one row per cell and month, target = at least one VIIRS detection. History features use only earlier years. Random Forest, 300 trees (200 in the comparisons), balanced class weights. Metric: PR-AUC, the better metric for rare events.

**Baseline, all months, trained 2012 to 2021, tested 2022 to 2025:**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Seasonal average | 0.749 | 0.038 |
| Cell history only | 0.758 | 0.047 |
| Random Forest (season, place, history) | 0.830 | 0.128 |

Random guessing gives a PR-AUC of 0.017.

**Adding drivers (June to November, where vegetation composites exist).** Time split as above, where the model also knows place and history:

| Features | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline | 0.787 | 0.138 |
| + weather | 0.799 | 0.138 |
| + weather + vegetation | 0.802 | 0.142 |
| + vegetation | 0.800 | 0.150 |

Spatial cross-validation, 5 folds in which whole MERRA-2 grid cells are held out, with no coordinates and no fire history, so the model must predict places it has never seen:

| Features | ROC-AUC | PR-AUC |
|---|---|---|
| Season only | 0.665 | 0.059 |
| + weather | 0.701 | 0.073 |
| + vegetation | 0.730 | 0.096 |
| + weather + vegetation | 0.757 | 0.103 |
| + vegetation + terrain | 0.743 | 0.105 |
| + weather + vegetation + terrain | 0.763 | 0.106 |

Random guessing gives 0.037; fold-to-fold spread is about 0.03 to 0.04.

Interpretation: where fire history is known, drivers add little (vegetation: +0.012 PR-AUC on four test years, tentative). At unseen places, satellite vegetation features carry most of the improvement (0.059 to about 0.10). Terrain adds a gain within the spread. Weather from MERRA-2 (cells about 50 km wide) adds little: in a first spatial test with 55 km blocks weather looked stronger (PR-AUC 0.122), but that shrank to 0.073 once no weather cell was shared between training and test places, i.e. part of the gain was weather acting as a location code.

### 3.4 Burned-area segmentation

Task: label each 500 m pixel as burned or not between two MODIS composites 32 days apart. Inputs: red, near-infrared, shortwave infrared and NDVI of both composites, plus dNBR and dNDVI (10 channels). Labels: MCD64A1 pixels with a burn date between the two composites. Split by year: train 2012 to 2020, validation 2021 (thresholds chosen here), test 2022 to 2025. Scores count pixels inside Tunisia only. Models: a small U-Net (4 levels, about 0.5 M parameters, Dice plus weighted cross-entropy loss), a per-pixel Random Forest, and a dNBR threshold.

| Test 2022 to 2025, Tunisia | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| dNBR threshold | 0.025 | 0.013 | 0.015 | 0.092 |
| Random Forest (per pixel) | 0.053 | 0.027 | 0.033 | 0.137 |
| **U-Net** | **0.152** | 0.082 | 0.142 | 0.163 |

The U-Net Dice per year is 0.24 (2022, 85 km² of burn), 0.08 (2023, 50 km²) and 0.02 (2025, 63 km²); 2024 (0.59) rests on 4 km² and is not meaningful. The test set has only 942 burned pixels in total.

Correction recorded for transparency: the first evaluation also counted the Algerian strip inside the crop box, where fires are larger, and gave a Dice of 0.48 for the U-Net. That figure does not describe Tunisia and is not used.

Why the score is low: Tunisian fires are mostly a few 500 m pixels wide, and 16-day composites blur the timing. The model picks up large burns well but misses small ones and raises false alarms. A further limit is circularity: MCD64A1 is derived from MODIS reflectance, so the model partly learns to reproduce that algorithm. The target Dice of 0.80 was not reached.

### 3.5 Sentinel-2 case study

Event: the largest MODIS burn cluster inside Tunisia in 2022 (16 km², 24 to 28 July, near 36.02°N, 9.64°E). dNBR from Sentinel-2 L2A (20 m), 19 July (before) and 13 August (after), both with 0% cloud, burned where dNBR is at least 0.27.

| Source | Burned area | Dice against Sentinel-2 (500 m) |
|---|---|---|
| Sentinel-2 dNBR | 14.2 km² | reference |
| MODIS MCD64A1 | 15.9 km² | 0.94 |
| U-Net | 18.9 km² | 0.88 |

Severity (USGS dNBR classes): 9.2 km² high, 2.9 km² moderate-high, 2.1 km² moderate-low, 1.4 km² low. The fire was chosen from the MODIS data, so this shows extent agreement for a large isolated fire and says nothing about the small fires that both MODIS and the U-Net miss. See `outputs/case_study_dnbr.png`.

## 4. Limitations

- **Resolution.** 500 m pixels are too coarse for most Tunisian fires; this is the main reason for the low U-Net score.
- **Small samples.** The Tunisian test set has 942 burned pixels; the risk-model comparisons rest on four test years, and spatial cross-validation on five folds. Most differences between feature sets are within the spread.
- **Labels and targets.** The fire-risk target is VIIRS detections, which include some agricultural and small industrial burning. Burned-area labels come from the same family of MODIS data as the inputs.
- **Heuristics.** The fixed-source filter and clustering parameters were set by hand.
- **Single runs.** The U-Net was trained once; no confidence intervals were computed.
- **Region.** The risk model covers 12 northern governorates; the burned-area crop box (34.5 to 37.6°N, 7.5 to 11.5°E) covers northern Tunisia only.

## 5. Next steps

Train and evaluate the burned-area model on 10 to 20 m Sentinel-2 imagery, add land cover to separate forest fires from agricultural burning, split large hotspot clusters (smaller distance or HDBSCAN), and report uncertainty with repeated training runs.

## 6. Reproducing

See the Getting started section of `README.md`. Data downloads need a free NASA FIRMS key and a NASA Earthdata account; the Sentinel-2 case study needs no account.