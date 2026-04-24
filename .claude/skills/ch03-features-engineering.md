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

## Encoder Ablation (필수 — 모델 선택보다 중요)

Killian et al. 2020 (NeurIPS ML4H *An Empirical Study of Representation Learning for RL in Healthcare*):
> "상태 표현(encoder) 선택이 OPE 결과에 정책(알고리즘) 선택보다 더 큰 영향을 미친다."

→ 우리 코드 구조는 **피처 추출과 인코더를 분리**:

```
src/
├── features.py              # 4h bin DataFrame 생성 (이 챕터 본문)
└── models/
    └── encoders/
        ├── mlp.py           # 평균/최근값 피처를 그대로 MLP 입력
        ├── tcn.py           # 1D causal conv + dilation, 시퀀스 처리 (Bai 2018, arXiv:1803.01271)
        ├── lstm.py          # 사용 안 함 — TCN 으로 대체 (병렬학습 빠르고 안정적)
        └── transformer.py   # 사용 안 함 — 18-step 시퀀스에 비해 모델 큼
```

각 인코더는 (flat 입력, 또는 reshape 후 시퀀스) → `(B, hidden)` 표현으로 통일 → 같은 head/policy 에 꽂아 비교.

**ablation 대상 (확정)**:
1. **MLP_last** — 현재 bin만 (B, F=80), 샘플 수 ~165k
2. **MLP_flat** — 모든 18 bin 평탄화 (B, T*F=1440), 샘플 수 ~9k stays
3. **TCN** — 진짜 시퀀스 인코더 (B, T, F → 1D dilated conv → mean-pool), ~9k stays

LSTM/Transformer 제외 이유:
- LSTM = TCN 과 거의 같은 효과 (둘 다 시퀀스), TCN이 GPU 병렬 더 빠름
- Transformer = 18 timestep × 80 dim 입력에 4-layer 모델 과다, dilated conv가 충분

각 인코더 동일 task (회귀 + 분류) Optuna 15 trials → 결과 표 + best 선정 → Phase 4 RL의 Q-network/policy backbone에 사용.

**TCN 하이퍼파라미터** (Optuna 탐색 공간):
- n_blocks: 2-4
- channels per block: {32, 64, 96, 128}
- kernel_size: {3, 5}
- dilation: 자동 (2^i)
- dropout: 0.05-0.5
- head_hidden: {0, 64, 128}
