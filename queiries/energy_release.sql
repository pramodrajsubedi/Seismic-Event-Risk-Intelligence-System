-- ─────────────────────────────────────────────────────────────────────────────
-- Query 4: Monthly seismic energy release (Richter-Gutenberg energy formula)
-- ─────────────────────────────────────────────────────────────────────────────
-- Energy in joules: E = 10^(1.5·M + 4.8)
-- Reported in petajoules (÷ 10^15) to keep numbers readable.
-- The cumulative column uses a window function with ROWS UNBOUNDED PRECEDING.
--
-- Interview talking point: "A single M8 releases ~10,000× the energy of an M6.
-- I encoded that directly in the SQL using the physical formula."
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    strftime('%Y-%m', event_time)                              AS month,
    COUNT(*)                                                   AS event_count,
    ROUND(
        SUM(POWER(10.0, 1.5 * magnitude + 4.8)) / 1e15,
        4)                                                     AS energy_petajoules,
    ROUND(
        SUM(SUM(POWER(10.0, 1.5 * magnitude + 4.8)) / 1e15)
            OVER (ORDER BY strftime('%Y-%m', event_time)
                  ROWS UNBOUNDED PRECEDING),
        4)                                                     AS cumulative_energy_PJ
FROM   earthquakes
WHERE  magnitude IS NOT NULL
GROUP  BY month
ORDER  BY month;