-- ─────────────────────────────────────────────────────────────────────────────
-- Query 2: Top 10 most seismically active regions (rolling 30-day window)
-- ─────────────────────────────────────────────────────────────────────────────
-- Uses a JOIN between earthquakes and regions, plus a window function
-- for cumulative totals — shows both GROUP BY and OVER() in one query.
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    r.name                                                        AS region,
    r.tectonic_setting,
    COUNT(*)                                                      AS quake_count,
    ROUND(AVG(e.magnitude), 2)                                    AS avg_magnitude,
    MAX(e.magnitude)                                              AS max_magnitude,
    SUM(COUNT(*)) OVER (ORDER BY COUNT(*) DESC
                        ROWS UNBOUNDED PRECEDING)                 AS cumulative_total
FROM   earthquakes e
JOIN   regions     r  ON e.region_id = r.region_id
WHERE  e.event_time >= DATE('now', '-30 days')
GROUP  BY r.name, r.tectonic_setting
ORDER  BY quake_count DESC
LIMIT  10;