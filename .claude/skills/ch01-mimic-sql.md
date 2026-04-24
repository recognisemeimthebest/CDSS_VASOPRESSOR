# ch01 — MIMIC-IV SQL & 연결

## 핵심 규칙

1. **자격증명 절대 하드코딩 금지**. `src/db.py`의 `get_conn()` / `get_engine()` 만 사용. `.env`로 주입.
2. **f-string SQL 금지**. parameterized query만 사용 (`%s` for psycopg2, `text(...)` + `bindparams` for SQLAlchemy).
3. **스키마 명시**: `mimiciv_hosp.patients` 처럼 항상 스키마 prefix. 검색 경로 의존 금지.
4. **대용량 테이블 (`chartevents`, `labevents`, `inputevents`) 풀스캔 금지**: 항상 `subject_id` 또는 `stay_id` 필터를 동반.
5. **시간 필터**: ICU 입실 후 윈도우만 추출 (`icustays.intime` 기준 BETWEEN).

## 스키마 빠른 참조

```sql
-- patient/admission
mimiciv_hosp.patients(subject_id, gender, anchor_age, anchor_year, dod)
mimiciv_hosp.admissions(hadm_id, subject_id, admittime, dischtime, deathtime, hospital_expire_flag)
mimiciv_icu.icustays(stay_id, subject_id, hadm_id, intime, outtime, los)

-- measurements
mimiciv_icu.chartevents(stay_id, charttime, itemid, value, valuenum)         -- vitals (시간 단위 매우 많음)
mimiciv_hosp.labevents(subject_id, hadm_id, charttime, itemid, value, valuenum)
mimiciv_icu.inputevents(stay_id, starttime, endtime, itemid, amount, rate, ordercategoryname)  -- 약물/수액
mimiciv_icu.outputevents(stay_id, charttime, itemid, value)

-- itemid 정의는 mimiciv_icu.d_items, mimiciv_hosp.d_labitems
mimiciv_icu.d_items(itemid, label, category, unitname)
```

## derived 스키마 우선 사용

`mimiciv_derived` 가 빌드되어 있으면 raw 테이블 직접 파싱 대신 항상 derived 테이블을 우선 쿼리:

| 필요한 것 | derived 테이블 | raw 대체 (권장 X) |
|---|---|---|
| 환자 인구학 | `age`, `weight_durations`, `height` | patients + chartevents |
| sepsis-3 코호트 | `sepsis3` | sofa + suspicion_of_infection 직접 계산 |
| SOFA | `sofa` (시간별), `first_day_sofa` | 직접 6개 시스템 계산 |
| vitals 시간별 | `vitalsign` | chartevents 풀파싱 |
| 검사 시간별 | `bg`, `chemistry`, `coagulation`, `cbc` | labevents 풀파싱 |
| vasopressor 통합 | `vasoactive_agent`, **`norepinephrine_equivalent_dose`** | 약물별 inputevents 합산 |
| 환기 | `ventilation` | chartevents 모드 추적 |

## 연결 예시

```python
from src.db import get_engine
import pandas as pd

engine = get_engine()
df = pd.read_sql(
    """
    SELECT stay_id, charttime, ne_equivalent_dose
    FROM mimiciv_derived.norepinephrine_equivalent_dose
    WHERE stay_id = ANY(%(stay_ids)s)
      AND charttime BETWEEN %(t0)s AND %(t1)s
    """,
    engine,
    params={"stay_ids": stay_ids, "t0": t0, "t1": t1},
)
```

## 디버깅

- 쿼리 느릴 때: `EXPLAIN (ANALYZE, BUFFERS) ...` → 인덱스 사용 확인
- 메모리 부족: `LIMIT 100` 으로 먼저 검증
- 결과 없음: derived 테이블 마지막 빌드 시점 확인 (`SELECT pg_relation_size(...)`)
- 스키마 확인: `\dn` (psql) 또는 `information_schema.schemata`
