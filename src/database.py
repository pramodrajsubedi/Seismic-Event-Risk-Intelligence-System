"""
database.py — SQLite schema, connection factory, and custom math functions.
Run directly to initialise a fresh database: python database.py
"""

import math
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "seismic.db"

#  Schema 

DDL_REGIONS = """
CREATE TABLE IF NOT EXISTS regions (
    region_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL,
    tectonic_setting     TEXT,                     -- e.g. 'Subduction zone', 'Transform fault'
    plate_boundary_type  TEXT,                     -- e.g. 'Convergent', 'Divergent', 'Transform'
    risk_tier            INTEGER CHECK(risk_tier BETWEEN 1 AND 4),
    avg_depth_km         REAL,
    historical_max_mag   REAL
);
"""

DDL_EARTHQUAKES = """
CREATE TABLE IF NOT EXISTS earthquakes (
    quake_id      TEXT    PRIMARY KEY,            -- USGS event ID  e.g. 'us7000abc1'
    event_time    DATETIME NOT NULL,
    latitude      REAL    NOT NULL,
    longitude     REAL    NOT NULL,
    depth_km      REAL,
    magnitude     REAL    NOT NULL,
    mag_type      TEXT,                           -- e.g. 'mw', 'mb', 'ml'
    place         TEXT,
    region_id     INTEGER REFERENCES regions(region_id),
    tsunami_flag  INTEGER DEFAULT 0,
    alert_level   TEXT,                           -- green / yellow / orange / red
    source        TEXT    DEFAULT 'api'           -- 'api' or 'historical'
);
"""

DDL_AFTERSHOCK_SEQUENCES = """
CREATE TABLE IF NOT EXISTS aftershock_sequences (
    seq_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    mainshock_id       TEXT    NOT NULL REFERENCES earthquakes(quake_id),
    aftershock_id      TEXT    NOT NULL REFERENCES earthquakes(quake_id),
    delta_hours        REAL    NOT NULL,          -- time from mainshock to this event
    mag_ratio          REAL,                      -- aftershock_mag / mainshock_mag
    had_damaging_after INTEGER DEFAULT 0          -- 1 if any M4+ within 48 h of mainshock
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_eq_time    ON earthquakes(event_time);",
    "CREATE INDEX IF NOT EXISTS idx_eq_mag     ON earthquakes(magnitude);",
    "CREATE INDEX IF NOT EXISTS idx_eq_region  ON earthquakes(region_id);",
    "CREATE INDEX IF NOT EXISTS idx_eq_latlon  ON earthquakes(latitude, longitude);",
    "CREATE INDEX IF NOT EXISTS idx_after_main ON aftershock_sequences(mainshock_id);",
]

# Connection factory 

def get_connection() -> sqlite3.Connection:
    """
    Return a configured SQLite connection.
    Registers LOG10() and POWER() so they work in plain SQL queries
    (SQLite has neither built-in).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")

    # Custom math functions ─ available in every query on this connection
    conn.create_function("LOG10", 1,
        lambda x: math.log10(x) if x and x > 0 else None)
    conn.create_function("LOG",   1,
        lambda x: math.log(x)   if x and x > 0 else None)
    conn.create_function("POWER", 2,
        lambda x, y: x ** y      if x is not None and y is not None else None)
    conn.create_function("SQRT",  1,
        lambda x: math.sqrt(x)   if x is not None and x >= 0 else None)

    return conn

#  Init 

def init_db(verbose: bool = True) -> None:
    """Create all tables and indexes if they do not already exist."""
    conn = get_connection()
    with conn:
        conn.execute(DDL_REGIONS)
        conn.execute(DDL_EARTHQUAKES)
        conn.execute(DDL_AFTERSHOCK_SEQUENCES)
        for idx in INDEXES:
            conn.execute(idx)
    if verbose:
        print(f"Database ready → {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    init_db()