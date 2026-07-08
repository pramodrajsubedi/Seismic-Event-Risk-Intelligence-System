-- ─────────────────────────────────────────────────────────────────────────────
-- Query 3: Aftershock damage rate within 48 h, by tectonic setting
-- ─────────────────────────────────────────────────────────────────────────────
-- Three-table JOIN: aftershock_sequences → earthquakes → regions.
-- Shows that subduction zones generate more damaging aftershock sequences
-- than intraplate events — a physically meaningful result.
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    r.tectonic_setting,
    COUNT(DISTINCT a.mainshock_id)                                    AS mainshock_count,
    COUNT(a.seq_id)                                                   AS total_aftershocks,
    ROUND(
        AVG(CASE WHEN a.had_damaging_after = 1 THEN 1.0 ELSE 0.0 END) * 100,
        1)                                                            AS pct_with_damaging_after,
    ROUND(AVG(e.magnitude), 2)                                        AS avg_mainshock_mag
FROM   aftershock_sequences a
JOIN   earthquakes           e  ON a.mainshock_id = e.quake_id
JOIN   regions               r  ON e.region_id    = r.region_id
WHERE  a.delta_hours <= 48
GROUP  BY r.tectonic_setting
ORDER  BY pct_with_damaging_after DESC;