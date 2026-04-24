# ch03 — 시계열 피처 엔지니어링

## 표준 (PROJECT_PLAN §2 확정)

- **bin**: 4시간
- **윈도우**: ICU intime ~ intime+72h
- **결측 처리**: forward-fill within stay → fill 안 되면 코호트 평균
- **정규화**: train fold 통계로 z-score (test에 fit 금지)

## 피처 카테고리

| 카테고리 | 출처 | 컬럼 예시 |
|---|---|---|
| Vitals | `mimiciv_derived.vitalsign` | hr, sbp, dbp, mbp, rr, spo2, temperature |
| Blood gas | `mimiciv_derived.bg` | ph, po2, pco2, lactate, base_excess |
| Chemistry | `mimiciv_derived.chemistry` | sodium, potassium, creatinine, bun, glucose |
| Coagulation | `mimiciv_derived.coagulation` | inr, ptt, platelet |
| CBC | `mimiciv_derived.complete_blood_count` | wbc, hematocrit, hemoglobin |
| Severity | `mimiciv_derived.sofa` | sofa_24hours, respiration, coagulation_score, ... |
| Demographics (static) | `age`, `weight_durations`, `patients.gender` | age, weight, gender |
| Treatment context | `ventilation`, `urine_output_rate` | vent_flag, urine_rate |

## 액션 (타깃)

- 출처: `mimiciv_derived.norepinephrine_equivalent_dose`
- 4h bin 내 max NE-equiv mcg/kg/min 또는 mean
- 5-bin 이산화: `0`, `(0, 8.4]`, `(8.4, 20.28]`, `(20.28, 50]`, `>50` (NE-equiv mcg/min)
  - mcg/min vs mcg/kg/min 단위 변환 시 weight_durations 필수
- 회귀용은 연속값 (mcg/kg/min) 그대로

## DataFrame 스키마

```python
# (stay_id, t_bin, ...)  long format
columns = [
    "stay_id", "t_bin",                          # int
    "hr_mean", "hr_min", "hr_max",               # mean/min/max per bin
    "sbp_mean", ..., "lactate_mean", ...,
    "sofa_24h", "vent_flag",
    "age", "weight", "gender_M",                 # static (반복)
    "ne_equiv_dose",                             # 다음 bin의 액션 = 타깃
    "ne_action_bin",                             # 5-bin 이산
]
```

## 함정

- chartevents 의 단위가 mcg/kg/min vs mcg/min 섞임 → 항상 unitname 확인 후 통일.
- forward-fill 시 stay 간 누수 금지. **반드시 groupby('stay_id')** 후 ffill.
- z-score 정규화는 train fold에서만 mean/std 계산. valid/test 는 transform만.
- sparse한 lab 변수(예: lactate)는 marker variable (있는지 여부 0/1) 추가.
- 시간 정렬: `chartevents.charttime` UTC 가정, 모든 join에 timezone 일치 확인.

## RL용 추가 변수

- reward proxy: SOFA 변화량, 90-day mortality (terminal)
- done flag: 사망 또는 ICU 퇴실 시 True
- terminal reward: alive=+15, dead=-15 (AI Clinician 기본)
