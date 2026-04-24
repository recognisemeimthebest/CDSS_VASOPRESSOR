# ch02 — Sepsis-3 코호트 정의

## 코호트 기준 (확정 — PROJECT_PLAN §2)

1. `mimiciv_derived.sepsis3` 의 stay_id (sepsis-3 만족)
2. ICU 입실 시 성인 (`age >= 18` from `mimiciv_derived.age`)
3. ICU 첫 stay만 (한 환자 다수 stay 시 가장 이른 것)
4. ICU 길이 ≥ 24h (`icustays.los >= 1.0` days)
5. **vasopressor 한 번이라도 받은 stay 만 학습 코호트** (추천 모델은 vasopressor 의사결정 학습)

## 윈도우

- t0: ICU 입실 (`intime`)
- t1: min(intime + 72h, outtime, deathtime)
- bin: 4 hours
- 따라서 stay 당 최대 18개 timestep

## 표준 SQL 스니펫

```sql
WITH adult_first_icu AS (
    SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime,
           a.age,
           ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS stay_num
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_derived.age a USING (hadm_id)
    WHERE a.age >= 18 AND i.los >= 1.0
),
sepsis_first AS (
    SELECT s.stay_id
    FROM mimiciv_derived.sepsis3 s
    JOIN adult_first_icu f ON f.stay_id = s.stay_id
    WHERE f.stay_num = 1
),
got_vasopressor AS (
    SELECT DISTINCT v.stay_id
    FROM mimiciv_derived.norepinephrine_equivalent_dose v
    JOIN sepsis_first sf USING (stay_id)
)
SELECT * FROM got_vasopressor;
```

## 제외 사유 정직히 기록

코호트 산출 시 단계별 환자/stay 수를 표(consort flow)로 출력하고 `notebooks/01_cohort_eda.ipynb` 마지막에 저장. 이게 없으면 plan-auditor 에이전트가 reject.

## 흔한 함정

- `sepsis3` 테이블의 `sepsis3` 컬럼이 boolean 인지 확인. 일부 빌드는 onset 시간만 들어있음.
- vasopressor 정의가 norepinephrine만 잡는지, 카테고리 전체인지 확인 — 우리는 `norepinephrine_equivalent_dose` (카테고리 통합).
- `los`는 days 단위. 24h = 1.0.
- 첫 ICU stay 한정 시 환자가 transfer 된 케이스 (icustays 두 행 같은 hadm_id) 주의.
