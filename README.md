# Wildfire Risk and Burned-Area Mapping in Tunisia

An independent research project investigating wildfire patterns in Tunisia using free NASA and Copernicus satellite data, with a focus on the Kroumirie-Mogods region in the northwest.

**Status:** Part 1 (live fire detection) complete. Part 2 (risk modeling and burn severity) in planning. 🚧

## What is this?

Tunisia has a documented wildfire problem, especially in the northwest, but little public research on fire patterns. This project combines NASA FIRMS active fire detections with Sentinel-2 imagery to:

1. Detect and visualize recent fires in near real time
2. Find where fires persistently occur and what drives them
3. Map fire risk across the region
4. Measure burn severity for a real fire event

## Project structure

```
.
├── src/
│   ├── fetch_fires.py      # Pulls recent VIIRS fire detections from NASA FIRMS
│   └── plot_fires.py       # Builds the interactive fire map
├── data/                   # Downloaded data (not tracked by git)
├── outputs/                # Generated maps
├── requirements.txt
└── README.md
```

## Roadmap

**Part 1: Active fire detection and visualization**
- [x] Set up project and Python environment
- [x] Pull near-real-time VIIRS fire data for Tunisia (NASA FIRMS API)
- [x] Plot detections on an interactive map (`outputs/tunisia_fires_live.html`)
- [x] Combine fetch and plot into a single pipeline
- [x] Add governorate boundaries to the map

**Part 2: Fire risk mapping and burn area analysis**
- [ ] Download historical FIRMS data for Tunisia (2012 to 2025)
- [ ] Clean and explore the data
- [ ] Spatial clustering (DBSCAN/HDBSCAN) to find persistent hotspots
- [ ] Add vegetation (NDVI), slope, and distance to roads/settlements
- [ ] Train a Random Forest fire-risk classifier with spatial cross-validation
- [ ] Burn severity (NBR) from Sentinel-2 before/after imagery for one 2025 fire
- [ ] Final report and demo

**Possible extension:** deep learning (for example ConvLSTM) for short-term fire prediction or burned-area mapping.

## Data sources

| Source | Used for |
|---|---|
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Active fire detections (VIIRS 375 m) |
| [Copernicus Data Space](https://dataspace.copernicus.eu/) | Sentinel-2 imagery |
| [Humanitarian Data Exchange](https://data.humdata.org/) | Historical Tunisia fire data |
| [Tunisia Open Data](https://catalog.data.gov.tn/) | Fire counts by governorate |

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
python src/fetch_fires.py   # saves data/tunisia_fires_latest.csv
python src/plot_fires.py    # saves outputs/tunisia_fires_live.html
```

Open `outputs/tunisia_fires_live.html` in a browser to see the map.

Note: the near-real-time FIRMS endpoint only returns the last 1 to 5 days. Long-term analysis uses the historical archive instead.

## Author

Oumaima Dhaouadi, 3rd year AI student at ESPRIT. [GitHub](https://github.com/oumaimaDh)
