"""
ingest.py — Pull earthquake data from the USGS Earthquake Catalog API
            and load it into the SQLite database.

Two modes:
  1. Live fetch  — recent events via API (used by Streamlit app on load)
  2. Bulk load   — historical CSV export from USGS ComCat (one-time setup)

USGS API docs: https://earthquake.usgs.gov/fdsnws/event/1/
ComCat CSV:    https://earthquake.usgs.gov/earthquakes/search/
               (Filter: M4.0+, worldwide, download in yearly chunks)

Run directly for a quick 30-day fetch:
    python ingest.py
"""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from database import get_connection, init_db

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MAX_PER_REQUEST = 20_000   # USGS hard limit per call


#  Parsing 

def _parse_geojson(data: dict) -> list[dict]:
    """Convert USGS GeoJSON feature list to flat dicts ready for INSERT."""
    rows = []
    for feature in data.get("features", []):
        props  = feature["properties"]
        coords = feature["geometry"]["coordinates"]  # [lon, lat, depth]
        rows.append({
            "quake_id":     feature["id"],
            "event_time":   datetime.utcfromtimestamp(
                                props["time"] / 1000).strftime("%Y-%m-%dT%H:%M:%S"),
            "longitude":    coords[0],
            "latitude":     coords[1],
            "depth_km":     coords[2],
            "magnitude":    props.get("mag"),
            "mag_type":     props.get("magType"),
            "place":        props.get("place"),
            "tsunami_flag": 1 if props.get("tsunami") == 1 else 0,
            "alert_level":  props.get("alert"),
            "source":       "api",
        })
    return rows


#  Single-window API fetch 

def fetch_usgs_window(start: str, end: str,
                      min_mag: float = 4.0) -> list[dict]:
    """
    Fetch one time window from the USGS API.
    start / end: 'YYYY-MM-DD' strings.
    """
    params = {
        "format":       "geojson",
        "starttime":    start,
        "endtime":      end,
        "minmagnitude": min_mag,
        "limit":        MAX_PER_REQUEST,
        "orderby":      "time-asc",
    }
    resp = requests.get(USGS_BASE, params=params, timeout=90)
    resp.raise_for_status()
    rows = _parse_geojson(resp.json())
    print(f"  {start} → {end}: {len(rows)} events")
    return rows


# Multi-year fetch (chunked to avoid the 20 k limit) 

def fetch_usgs_range(start_year: int, end_year: int,
                     min_mag: float = 4.0,
                     pause_sec: float = 1.5) -> list[dict]:
    """
    Fetch M4+ events year-by-year from start_year to end_year (inclusive).
    Sleeps between requests to stay within USGS rate limits.
    """
    all_rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        start = f"{year}-01-01"
        end   = f"{year}-12-31"
        rows  = fetch_usgs_window(start, end, min_mag)
        all_rows.extend(rows)
        time.sleep(pause_sec)
    return all_rows


#  Live fetch (Streamlit uses this) 

def fetch_live_events(days: int = 7, min_mag: float = 4.0) -> list[dict]:
    """
    Return M4+ events from the last `days` days.
    Called by app.py on every Streamlit page load.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=days)
    return fetch_usgs_window(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        min_mag=min_mag,
    )


# Insert helper 

_INSERT_SQL = """
INSERT OR IGNORE INTO earthquakes
    (quake_id, event_time, latitude, longitude, depth_km,
     magnitude, mag_type, place, tsunami_flag, alert_level, source)
VALUES
    (:quake_id, :event_time, :latitude, :longitude, :depth_km,
     :magnitude, :mag_type, :place, :tsunami_flag, :alert_level, :source)
"""

def insert_events(rows: list[dict], conn: sqlite3.Connection) -> int:
    """Bulk-insert rows, skipping duplicates. Returns count inserted."""
    if not rows:
        print("  No events to insert.")
        return 0
    before = conn.execute("SELECT COUNT(*) FROM earthquakes").fetchone()[0]
    with conn:
        conn.executemany(_INSERT_SQL, rows)
    after  = conn.execute("SELECT COUNT(*) FROM earthquakes").fetchone()[0]
    added  = after - before
    print(f"  Inserted {added} new rows ({len(rows) - added} duplicates skipped).")
    return added


# Bulk CSV loader (run once after downloading ComCat CSV) 

def bulk_load_csv(csv_path: str) -> None:
    """
    Load a USGS ComCat CSV export into the database.

    Download from: https://earthquake.usgs.gov/earthquakes/search/
    Settings: M4.0+, worldwide, select all years, CSV format.
    Download in yearly chunks if the file is >500 MB.

    Expected CSV columns (USGS standard):
        time, latitude, longitude, depth, mag, magType, id, place, ...
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names to schema
    df = df.rename(columns={
        "time":    "event_time",
        "depth":   "depth_km",
        "mag":     "magnitude",
        "magType": "mag_type",
        "id":      "quake_id",
    })

    df["tsunami_flag"] = 0
    df["alert_level"]  = None
    df["source"]       = "historical"

    needed = ["quake_id", "event_time", "latitude", "longitude",
              "depth_km", "magnitude", "mag_type", "place",
              "tsunami_flag", "alert_level", "source"]
    df = df[needed].dropna(subset=["quake_id", "magnitude", "latitude", "longitude"])

    conn = get_connection()
    added = insert_events(df.to_dict("records"), conn)
    conn.close()
    print(f"Bulk load complete — {added} new events added.")



if __name__ == "__main__":
    init_db()
    conn = get_connection()

    print("Fetching last 30 days from USGS API (M4.0+)...")
    end   = datetime.utcnow()
    start = end - timedelta(days=30)
    rows  = fetch_usgs_window(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        min_mag=4.0,
    )
    insert_events(rows, conn)
    conn.close()
    print("Done. Run aftershocks.py next to build sequence table.")