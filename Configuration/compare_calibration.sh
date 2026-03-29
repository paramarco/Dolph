#!/bin/bash
# Compare calibration params with example TP/SL/profit for a simulated long trade
# Usage: ./compare_calibration.sh

PGPASSWORD=dolph_password psql -h 127.0.0.1 -p 4713 -U dolph_user -d dolph_db -c "
WITH last_prices AS (
    SELECT q.security_id,
           q.close::numeric AS last_close
    FROM quote q
    INNER JOIN (
        SELECT security_id, MAX(date_time) AS max_dt
        FROM quote
        GROUP BY security_id
    ) latest ON q.security_id = latest.security_id AND q.date_time = latest.max_dt
),
params AS (
    SELECT s.code, s.id, s.timezone, s.currency,
           (s.alg_parameters->>'TP_MULT')::numeric AS tp_mult,
           (s.alg_parameters->>'SL_RR')::numeric AS sl_rr,
           (s.alg_parameters->>'CALIBRATION_GAUSS_MU')::int AS mu,
           (s.alg_parameters->>'CALIBRATION_GAUSS_SIGMA')::int AS sigma,
           (s.alg_parameters->>'calibration_score')::numeric AS score
    FROM security s
    WHERE s.alg_parameters IS NOT NULL
),
atr_data AS (
    SELECT q.security_id,
           AVG(q.high - q.low)::numeric AS avg_range
    FROM quote q
    JOIN security s ON q.security_id = s.id
    WHERE s.alg_parameters IS NOT NULL
      AND q.date_time >= NOW() - INTERVAL '5 days'
      AND q.vol > 0
    GROUP BY q.security_id
)
SELECT p.code,
       ROUND(lp.last_close, 2) AS price,
       ROUND(p.tp_mult, 2) AS tp_m,
       p.mu,
       p.sigma AS sig,
       ROUND(p.tp_mult * a.avg_range / lp.last_close * 100, 2) AS mrg_pct,
       ROUND(lp.last_close + p.tp_mult * a.avg_range, 2) AS tp_long,
       ROUND(lp.last_close - p.sl_rr * p.tp_mult * a.avg_range, 2) AS sl_long,
       ROUND(5200.0 / lp.last_close)::int AS qty,
       ROUND(
           ROUND(5200.0 / lp.last_close) * p.tp_mult * a.avg_range
           - 2 * GREATEST(1.0, ROUND(5200.0 / lp.last_close) * 0.005),
       2) AS profit,
       ROUND(p.score, 1) AS score
FROM params p
JOIN last_prices lp ON lp.security_id = p.id
JOIN atr_data a ON a.security_id = p.id
ORDER BY p.timezone, p.code;
"
