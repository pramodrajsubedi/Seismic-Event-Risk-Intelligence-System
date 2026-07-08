"""
aftershocks.py — Build the aftershock_sequences table.

Logic:
  For each M5+ mainshock, find all M2+ events within
  200 km and 7 days.  Flag each sequence with
  had_damaging_after = 1  if any aftershock reached M4+ within 48 h.

This two-step spatial filter (bounding-box → haversine) keeps the
script fast enough to run on ~500 k rows in a few minutes.

Run after ingest.py:
    python aftershocks.py
"""

import math
import sqlite3
from typing import Iterator

import pandas as pd

from database import get_connection

# Tuneable parameters
MIN_MAINSHOCK_MAG   = 5.0    # Only M5+ events act as mainshocks
SEARCH_RADIUS_KM    = 200    # Maximum distance for a candidate aftershock
SEARCH_WINDOW_HOURS = 168    # 7-day aftershock window
DAMAGE_MAG          = 4.0    # Threshold for had_damaging_after flag
DAMAGE_WINDOW_HOURS = 48     # How soon a damaging aftershock must occur


# Haversine 

def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6_371.0
    la1, lo1, la2, lo2 = (math.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Bounding-box pre-filter (fast, approximate) 

def _lat_delta(km: float) -> float:
    return km / 111.0

def _lon_delta(km: float, lat: float) -> float:
    cos_lat = math.cos(math.radians(lat))
    return km / (111.0 * cos_lat) if cos_lat > 0.001 else 360.0


# Main builder 

def build_sequences(batch_size: int = 200, verbose: bool = True) -> None:
    """
    Populate aftershock_sequences for every M5+ mainshock not yet processed.
    Uses a bounding-box pre-filter then exact haversine check to keep it fast.
    """
    conn = get_connection()

    # Load all events into memory once (sorted by time)
    if verbose:
        print("Loading earthquake catalogue into memory...")
    df = pd.read_sql_query(
        """SELECT quake_id, event_time, latitude, longitude, magnitude
           FROM   earthquakes
           ORDER  BY event_time""",
        conn,
        parse_dates=["event_time"],
    )
    if verbose:
        print(f"  {len(df):,} events loaded.")

    # Identify mainshocks not yet in the sequences table
    processed = pd.read_sql_query(
        "SELECT DISTINCT mainshock_id FROM aftershock_sequences", conn
    )["mainshock_id"].tolist()

    mainshocks = df[
        (df["magnitude"] >= MIN_MAINSHOCK_MAG) &
        (~df["quake_id"].isin(processed))
    ].copy()

    if verbose:
        print(f"  {len(mainshocks):,} unprocessed mainshocks to analyse.")

    records: list[dict] = []
    total_seqs = 0

    for i, (_, main) in enumerate(mainshocks.iterrows()):
        window_end = main["event_time"] + pd.Timedelta(hours=SEARCH_WINDOW_HOURS)

        # 1 time filter
        time_mask = (
            (df["event_time"] > main["event_time"]) &
            (df["event_time"] <= window_end) &
            (df["quake_id"]   != main["quake_id"])
        )
        candidates = df[time_mask]

        if candidates.empty:
            continue

        # 2 bounding-box pre-filter
        dlat = _lat_delta(SEARCH_RADIUS_KM)
        dlon = _lon_delta(SEARCH_RADIUS_KM, main["latitude"])
        bbox = candidates[
            (candidates["latitude"]  >= main["latitude"]  - dlat) &
            (candidates["latitude"]  <= main["latitude"]  + dlat) &
            (candidates["longitude"] >= main["longitude"] - dlon) &
            (candidates["longitude"] <= main["longitude"] + dlon)
        ]

        if bbox.empty:
            continue

        # 3 exact haversine filter
        bbox = bbox.copy()
        bbox["dist_km"] = bbox.apply(
            lambda r: _haversine_km(main["latitude"],  main["longitude"],
                                    r["latitude"],     r["longitude"]),
            axis=1,
        )
        nearby = bbox[bbox["dist_km"] <= SEARCH_RADIUS_KM]

        # 4 determine had_damaging_after flag (once per mainshock)
        had_damaging = int(
            nearby[
                (nearby["magnitude"] >= DAMAGE_MAG) &
                (((nearby["event_time"] - main["event_time"])
                    .dt.total_seconds() / 3600) <= DAMAGE_WINDOW_HOURS)
            ].shape[0] > 0
        )

        for _, cand in nearby.iterrows():
            delta_h = (cand["event_time"] - main["event_time"]).total_seconds() / 3600
            records.append({
                "mainshock_id":      main["quake_id"],
                "aftershock_id":     cand["quake_id"],
                "delta_hours":       round(delta_h, 3),
                "mag_ratio":         round(cand["magnitude"] / main["magnitude"], 4)
                                     if main["magnitude"] else None,
                "had_damaging_after": had_damaging,
            })
            total_seqs += 1

        # Flush in batches to avoid huge memory build-up
        if len(records) >= batch_size * 50:
            _insert_batch(records, conn)
            if verbose:
                print(f"  [{i+1}/{len(mainshocks)}] flushed {len(records)} records...")
            records = []

    # Final flush
    if records:
        _insert_batch(records, conn)

    conn.close()
    if verbose:
        print(f"Done — {total_seqs:,} aftershock sequence records built.")


def _insert_batch(records: list[dict], conn: sqlite3.Connection) -> None:
    sql = """INSERT OR IGNORE INTO aftershock_sequences
             (mainshock_id, aftershock_id, delta_hours, mag_ratio, had_damaging_after)
             VALUES
             (:mainshock_id, :aftershock_id, :delta_hours, :mag_ratio, :had_damaging_after)"""
    with conn:
        conn.executemany(sql, records)


if __name__ == "__main__":
    build_sequences()