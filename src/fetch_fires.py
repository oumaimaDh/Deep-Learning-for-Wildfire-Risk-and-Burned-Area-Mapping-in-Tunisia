import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
MAP_KEY = os.getenv("FIRMS_MAP_KEY")

# Tunisia bounding box: west, south, east, north
BBOX = "8.0,30.0,12.0,38.0"
DAYS = 5

url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{BBOX}/{DAYS}"

df = pd.read_csv(url)
print(df.head())
print(f"Fires in Tunisia last {DAYS} days: {len(df)}")

df.to_csv("data/tunisia_fires_latest.csv", index=False)
print("Saved to data/tunisia_fires_latest.csv")