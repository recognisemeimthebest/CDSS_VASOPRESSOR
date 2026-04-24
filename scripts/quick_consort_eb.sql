WITH base AS (
    SELECT i.stay_id, i.subject_id, i.intime, i.outtime
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_derived.age a USING (hadm_id)
    WHERE a.age >= 18 AND i.los >= 1.0
      AND EXISTS (SELECT 1 FROM mimiciv_derived.sepsis3 s WHERE s.stay_id = i.stay_id)
),
first_stay AS (
    SELECT * FROM (
        SELECT b.*, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS sn
        FROM base b
    ) x WHERE sn = 1
),
ne_per_stay AS (
    SELECT n.stay_id,
           MIN(n.starttime) AS first_ne_time,
           SUM(EXTRACT(EPOCH FROM (n.endtime - n.starttime)))/3600.0 AS cum_hours
    FROM mimiciv_derived.norepinephrine_equivalent_dose n
    WHERE n.norepinephrine_equivalent_dose > 0
    GROUP BY n.stay_id
),
joined AS (
    SELECT f.*, ne.first_ne_time, ne.cum_hours,
           EXTRACT(EPOCH FROM (ne.first_ne_time - f.intime))/60.0 AS minutes_to_first_ne
    FROM first_stay f
    LEFT JOIN ne_per_stay ne USING (stay_id)
)
SELECT 'A_first_stay_pool' AS step, count(*) AS n_stays FROM first_stay
UNION ALL SELECT 'B_got_ne_ever', count(*) FROM joined WHERE first_ne_time IS NOT NULL
UNION ALL SELECT 'C_ne_start_after_intime_0min', count(*) FROM joined WHERE minutes_to_first_ne > 0
UNION ALL SELECT 'D_ne_start_after_intime_60min', count(*) FROM joined WHERE minutes_to_first_ne > 60
UNION ALL SELECT 'E_ne_cum_hours_ge_1', count(*) FROM joined WHERE cum_hours >= 1.0
UNION ALL SELECT 'F_E0min_AND_cum1h', count(*) FROM joined WHERE minutes_to_first_ne > 0 AND cum_hours >= 1.0
UNION ALL SELECT 'G_E60min_AND_cum1h', count(*) FROM joined WHERE minutes_to_first_ne > 60 AND cum_hours >= 1.0
ORDER BY step;
