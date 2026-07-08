"""
seed_regions.py — Populate the regions table with major tectonic zones
                  and assign region_id to every earthquake in the database.

Run once after ingest.py:
    python seed_regions.py

Uses simple lat/lon bounding boxes.  Order matters: more specific regions
are listed first so they take priority over the catch-all at the end.
"""

import sqlite3
from database import get_connection, init_db

# Region definitions 
# Each tuple: (name, tectonic_setting, plate_boundary_type, risk_tier,
#              lat_min, lat_max, lon_min, lon_max)

REGIONS = [
    # High-activity subduction zones 
    ("Japan / Kuril Islands",
     "Subduction zone", "Convergent", 4,
     28.0,  50.0, 128.0,  150.0),

    ("Indonesia / Philippines",
     "Subduction zone", "Convergent", 4,
    -10.0,  20.0,  95.0,  142.0),

    ("Alaska / Aleutian Islands",
     "Subduction zone", "Convergent", 4,
     48.0,  65.0, -180.0, -130.0),

    ("South America – Andes",
     "Subduction zone", "Convergent", 4,
    -55.0,  12.0, -82.0,  -60.0),

    ("Central America",
     "Subduction zone", "Convergent", 3,
      5.0,  22.0, -100.0,  -75.0),

    ("New Zealand / Tonga / Vanuatu",
     "Subduction zone", "Convergent", 4,
    -50.0,  -8.0, 163.0,  180.0),

    ("Cascadia / Pacific Northwest",
     "Subduction zone", "Convergent", 3,
     38.0,  52.0, -130.0, -116.0),

    ("Caribbean",
     "Subduction zone", "Convergent", 3,
     10.0,  24.0,  -85.0,  -58.0),

    # Collision / fold-and-thrust belts 
    ("Himalaya / Hindu Kush / Karakoram",
     "Collision zone", "Convergent", 3,
     25.0,  42.0,  60.0,  100.0),

    ("Mediterranean / Aegean / Turkey",
     "Collision zone", "Convergent", 3,
     28.0,  45.0, -10.0,   45.0),

    ("Iran / Zagros",
     "Collision zone", "Convergent", 3,
     24.0,  40.0,  43.0,   65.0),

    # Spreading ridges 
    ("Mid-Atlantic Ridge",
     "Spreading ridge", "Divergent", 2,
    -60.0,  70.0, -45.0,  -10.0),

    ("East Pacific Rise",
     "Spreading ridge", "Divergent", 2,
    -60.0,  30.0, -115.0,  -95.0),

    ("Indian Ocean Ridge",
     "Spreading ridge", "Divergent", 1,
    -55.0,  10.0,  55.0,   90.0),

    # Transform faults 
    ("San Andreas / California",
     "Transform fault", "Transform", 3,
     32.0,  42.0, -124.0, -114.0),

    ("New Zealand – Alpine Fault",
     "Transform fault", "Transform", 3,
    -47.0, -34.0,  166.0,  174.0),

    # Rift zones 
    ("East African Rift",
     "Continental rift", "Divergent", 2,
    -15.0,  15.0,  25.0,   42.0),

    ("Iceland / Jan Mayen",
     "Spreading ridge", "Divergent", 2,
     62.0,  68.0, -25.0,  -10.0),

    # Intraplate / catch-all 
    ("Global – Intraplate",
     "Intraplate", "Intraplate", 1,
    -90.0,  90.0, -180.0,  180.0),
]


# Insert regions 

def seed_regions(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert regions and return name → region_id mapping."""
    sql = """
    INSERT OR IGNORE INTO regions
        (name, tectonic_setting, plate_boundary_type, risk_tier)
    VALUES (?, ?, ?, ?)
    """
    with conn:
        for row in REGIONS:
            conn.execute(sql, (row[0], row[1], row[2], row[3]))

    rows = conn.execute("SELECT region_id, name FROM regions").fetchall()
    return {r["name"]: r["region_id"] for r in rows}


# Assign region_id to earthquakes 

def assign_regions(conn: sqlite3.Connection) -> int:
    """
    For each earthquake, set region_id to the first matching bounding box.
    Regions are checked in REGIONS order (specific → general), so the
    catch-all 'Global – Intraplate' only fires when nothing else matches.
    """
    id_map = {r["name"]: r["region_id"]
              for r in conn.execute("SELECT region_id, name FROM regions").fetchall()}

    earthquakes = conn.execute(
        "SELECT quake_id, latitude, longitude FROM earthquakes WHERE region_id IS NULL"
    ).fetchall()

    updates: list[tuple[int, str]] = []
    for eq in earthquakes:
        lat, lon = eq["latitude"], eq["longitude"]
        assigned = None
        for reg in REGIONS:
            name, _, _, _, lat_min, lat_max, lon_min, lon_max = reg
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                assigned = id_map.get(name)
                break
        if assigned:
            updates.append((assigned, eq["quake_id"]))

    if updates:
        with conn:
            conn.executemany(
                "UPDATE earthquakes SET region_id = ? WHERE quake_id = ?", updates
            )

    return len(updates)


# avg_depth + historical_max_mag 

def update_region_stats(conn: sqlite3.Connection) -> None:
    """Back-fill avg_depth_km and historical_max_mag from the loaded data."""
    with conn:
        conn.execute("""
            UPDATE regions
            SET    avg_depth_km       = (SELECT AVG(depth_km)  FROM earthquakes e WHERE e.region_id = regions.region_id),
                   historical_max_mag = (SELECT MAX(magnitude) FROM earthquakes e WHERE e.region_id = regions.region_id)
            WHERE  EXISTS (SELECT 1 FROM earthquakes e WHERE e.region_id = regions.region_id)
        """)


# Entry-point 

if __name__ == "__main__":
    init_db(verbose=False)
    conn = get_connection()

    print("Seeding regions table...")
    seed_regions(conn)
    total_regions = conn.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
    print(f"  {total_regions} regions ready.")

    print("Assigning region_id to earthquakes...")
    updated = assign_regions(conn)
    print(f"  {updated} earthquakes assigned.")

    print("Updating region stats...")
    update_region_stats(conn)

    # Verify
    print("\nRegion summary:")
    rows = conn.execute("""
        SELECT r.name, r.tectonic_setting, r.risk_tier,
               COUNT(e.quake_id)     AS event_count,
               ROUND(MAX(e.magnitude),1) AS max_mag
        FROM   regions r
        LEFT JOIN earthquakes e ON e.region_id = r.region_id
        GROUP BY r.region_id
        ORDER BY event_count DESC
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(f"  [{r['risk_tier']}] {r['name']:<35} events={r['event_count']:>5}  max_mag={r['max_mag']}")

    conn.close()
    print("\nDone. Now run: python aftershocks.py")