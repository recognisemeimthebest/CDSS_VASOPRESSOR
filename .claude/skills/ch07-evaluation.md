# ch07 — 평가 (지표 + OPE + Fairness)

## 보고할 지표

### Supervised (회귀)
- MAE, RMSE, R²
- 잔차 vs 예측값 산점도
- 5-bin 변환 후 분류 지표 (아래) 도 추가 보고

### Supervised (분류 5-bin)
- accuracy, macro-F1, weighted-F1
- top-2 accuracy ("권고와 1단계 차이까지 허용")
- confusion matrix (5x5)
- per-class precision/recall

### Calibration
- reliability diagram, ECE
- Brier score (multi-class adapted)

### Offline RL (필수 OPE)

단일 점추정 보고 금지. 항상 다중 OPE + 신뢰구간 + null-policy 비교 (Gottesman 2018/19; Tang & Wiens 2021).

- **WIS** (Weighted Importance Sampling): trajectory IS weight × reward
- **FQE** (Fitted Q Evaluation): 별도 critic 학습으로 V^π 추정. WIS 보완.
- **WIS와 FQE 둘 다 보고 필수** (한 가지만 보고 → reject)
- **Bootstrap CI**: 1000회, 95% percentile interval
- **ESS** (Effective Sample Size): IS weight 분포의 유효 샘플 수. ESS < 5% of N 이면 그 정책 비교 결과 무효 처리.
- **Null-policy baseline**: zero-action / uniform-random / constant-dose (예: 항상 NE-equiv 0.1) 정책의 WIS/FQE를 학습 정책과 같은 표에 출력 (Jeter 2019)
- **clinician matching rate**: 학습 정책 추천 = 의사 실제 액션 비율
- **OOD 비율**: behavior policy 분포 밖 액션 추천 빈도. 너무 높으면 (>10%) BCQ threshold 또는 CQL alpha 조정.

### 정직성 가드 (Honesty Guard)

- **Mortality 감소 주장 절대 금지** — prospective RCT 없으면 OPE 결과를 "환자 살림"으로 해석 금지 (Festor 2022, Wu 2023, Roggeveen 2021).
- 보고서 결론은 "정책 X는 의사 정책 대비 OPE 상 ___의 expected return을 보였으나, 임상 효과는 prospective 검증 전까지 알 수 없다" 식으로.
- README/논문 disclaimer 필수 (PROJECT_PLAN §8 참조).

## Fairness / Subgroup 분석 (필수)

다음 subgroup 별로 위 지표 분리 보고:

| Subgroup | 분리 기준 |
|---|---|
| 성별 | gender |
| 연령대 | age 18-39, 40-64, 65-79, 80+ |
| 인종 | admissions.race (group: WHITE, BLACK, ASIAN, HISPANIC, OTHER) |
| 입실 유형 | first_careunit (MICU, SICU, CCU, ...) |
| 중증도 | first_day_sofa 4-quartiles |

성능 격차가 절대 차이 ≥ 5%p 이면 보고서에 명시 + 원인 검토 (데이터 부족? 모델 편향?).

## 결과 저장 형식

```
artifacts/
└── runs/
    └── {date}_{model}_{seed}/
        ├── config.json           # 하이퍼파라미터, seed
        ├── metrics.json          # 위 지표 (subgroup 포함)
        ├── predictions.parquet   # stay_id, t_bin, true, pred
        ├── plots/
        │   ├── confusion.png
        │   ├── calibration.png
        │   └── subgroup_metrics.png
        └── model.pt              # 체크포인트
```

`predictions.parquet` 가 있어야 사후 평가/시각화/감사 가능. 무조건 저장.

## eval-agent 가 요구하는 산출물

1. metrics.json (전체 + subgroup)
2. predictions.parquet
3. confusion / calibration / subgroup plot
4. (RL 시) OPE 결과 + clinician matching rate

이게 빠지면 PASS 안 줌.
