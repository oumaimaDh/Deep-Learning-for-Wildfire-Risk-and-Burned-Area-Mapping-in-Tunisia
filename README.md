# Deep Learning for Wildfire Risk and Burned-Area Mapping in Tunisia

An independent research project analyzing 20+ years of NASA Earth observation data to characterize wildfire patterns in Tunisia, model fire risk, and automatically map burned areas with deep learning.

**Status:** In progress. Live fire map done; historical data collection under way. 🚧

## Goals

1. Characterize wildfire patterns in Tunisia over two decades (NASA FIRMS, MODIS and VIIRS)
2. Identify the climatic and vegetation drivers of fire risk (MERRA-2, NDVI)
3. Build a Random Forest baseline for fire risk
4. Train a PyTorch U-Net for burned-area segmentation, using MODIS MCD64A1 as reference labels, and compare it to the baseline (target Dice > 0.80)
5. Produce burned-area maps and a technical report

Focus area: the Kroumirie-Mogods region in the northwest, which is Tunisia's most fire-affected zone.

## Why

Tunisia has little public research on wildfire patterns, while NASA provides free, open data going back more than 20 years.

## Data sources

| Source | Used for |
|---|---|
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (MODIS 2000+, VIIRS 2012+) | Active fire detections |
| MODIS MCD64A1 | Monthly burned-area product, used as segmentation labels |
| MERRA-2 | Meteorological reanalysis (temperature, humidity, wind, rainfall) |
| [Copernicus Data Space](https://dataspace.copernicus.eu/) (Sentinel-2) | NDVI and before/after imagery |
| [geoBoundaries](https://www.geoboundaries.org/) | Tunisia governorate boundaries |

## Roadmap

**Phase 1: Live detection**
- [x] Project setup and Python environment
- [x] Near-real-time VIIRS fire data from the FIRMS API
- [x] Interactive map with governorate boundaries (`outputs/tunisia_fires_live.html`)
- [x] Single fetch-and-plot pipeline (`src/live_map.py`)

**Phase 2: Historical data and exploration**
- [x] Request historical VIIRS (2012 to 2025) and MODIS (2000 to 2025) archives
- [ ] Explore trends: yearly counts, seasonality, confidence filtering
- [ ] Spatial clustering of persistent hotspots (DBSCAN/HDBSCAN)

**Phase 3: Drivers and baseline model**
- [ ] Download MERRA-2 weather variables and compute NDVI
- [ ] Add terrain and distance-to-road/settlement features
- [ ] Random Forest fire-risk baseline with time-aware and spatial validation

**Phase 4: Burned-area segmentation**
- [ ] Prepare MCD64A1 burned-area labels
- [ ] PyTorch U-Net for burned-area segmentation
- [ ] Compare against the Random Forest baseline (Dice)
- [ ] Burn severity (NBR) case study on one recent fire

**Phase 5: Write-up**
- [ ] Burned-area maps, technical report, demo

## Project structure

```
.
├── src/
│   ├── live_map.py          # Fetches recent VIIRS detections and builds the live map
│   ├── boundaries.py        # Downloads Tunisia governorate boundaries
│   └── explore_history.py   # Summary statistics on the historical archive
├── data/                    # Downloaded data (not tracked by git)
├── outputs/                 # Generated maps
├── notebooks/               # Exploration notebooks
├── requirements.txt
└── README.md
```

## Getting started

Requires Python 3 and a free [NASA FIRMS MAP_KEY](https://firms.modaps.eosdis.nasa.gov/api/).

```powershell
git clone https://github.com/oumaimaDh/Deep-Learning-for-Wildfire-Risk-and-Burned-Area-Mapping-in-Tunisia.git
cd Deep-Learning-for-Wildfire-Risk-and-Burned-Area-Mapping-in-Tunisia

python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is gitignored):

```
FIRMS_MAP_KEY=your_key_here
```

Then run:

```powershell
python src/boundaries.py    # one-time: downloads governorate boundaries
python src/live_map.py      # saves data/tunisia_fires_latest.csv and outputs/tunisia_fires_live.html
```

Notes:
- The near-real-time endpoint only returns the last 1 to 5 days, and the map can be empty outside the summer fire season.
- Historical archives are downloaded from the [FIRMS download page](https://firms.modaps.eosdis.nasa.gov/download/) and placed in `data/` (gitignored).

## Author

Oumaima Dhaouadi, 3rd year Computer Science Engineering student at ESPRIT (AI specialization). [GitHub](https://github.com/oumaimaDh)