# CDSS Vasopressor — Presentation Summary
Generated: 2026-04-27  |  Cohort: MIMIC-IV Sepsis-3

## 1. 코호트
| 항목 | 값 |
|---|---|
| ICU stays | 13,071 |
| 총 timesteps | 235,278 (4h bin × 18 bins) |
| 입원 사망률 | 23.1% |
| Train / Val / Test | 9,149 / 1,961 / 1,961 stays |

**액션 분포 (class imbalance)**
| Bin | 범위 | 비율 |
|---|---|---|
| 0 | No NE | 57.9% |
| 1 | ≤8.4 mcg/min | 24.6% |
| 2 | 8.4–20 mcg/min | 10.4% |
| 3 | 20–50 mcg/min | 5.7% |
| 4 | >50 mcg/min | 1.4% |

## 2. 지도학습 결과 (Test Fold)
| 모델 | Macro-F1 | Accuracy | Top-2 Acc |
|---|---|---|---|
| LR | 0.455 | 0.598 | 0.814 |
| **GBM** | **0.529** | **0.676** | **0.875** |
| MLP | 0.504 | 0.642 | 0.858 |
| TCN* | 0.433 | 0.734 | 0.893 |

*TCN은 stay당 1 sample (비교 불공정, caveat 있음)

**Per-class F1 (GBM)**
| Class | F1 |
|---|---|
| 0 (No NE) | 0.837 |
| 1 (≤8.4) | 0.550 |
| **2 (8.4-20)** | **0.356 (최약점)** |
| 3 (20-50) | 0.444 |
| 4 (>50) | 0.460 |

## 3. Class 2 개선 실험 결과
| 시도 | Class 2 F1 | Macro-F1 | 결론 |
|---|---|---|---|
| Baseline GBM | 0.356 | 0.529 | 기준선 |
| Cost-Sensitive GBM (C[i,j]=|i-j|^2) | 0.299 | 0.387 | 음성 결과 |
| Soft Labels + MLP (Gaussian σ=0.5) | 0.347 | 0.479 | 음성 결과 |
| **GBM Threshold (cls2×1.5)** | **0.395** | 0.519 | **양성 +0.039** |
| **GBM Threshold (cls3×0.8, Strategy C)** | **0.371** | **0.527** | **양성 +0.015 (균형)** |

- 음성 결과 공통 원인: Optuna 파라미터가 다른 loss 함수에서 최적화됨
- Threshold 조정은 precision-recall tradeoff이며 재학습 불필요

## 4. Off-Policy Evaluation (OPE)
| Policy | WIS [95% CI] | FQE | ESS% |
|---|---|---|---|
| BC | 6.642 [6.150, 7.157] | 0.017 | 100.0 |
| dBCQ | 5.138 | 0.121 | 15.0 |
| CQL | 2.810 | 0.109 | 17.6 |
| Clinician | 6.004 [5.453, 6.550] | 0.055 | 87.0 |

- BC: WIS 신뢰도 최고 (ESS=100%, clinician 모방에 가까움)
- dBCQ: FQE 기준 clinician 상회, ESS 15% (통계적 신뢰도 제한적)
- CQL: conservative penalty로 action 분포 변화 → WIS 낮음

## 5. 주요 결론
1. **GBM이 지도학습 최강** (macro-F1=0.529, top-2=87.5%)
2. **Class 2 경계 구간은 임상적 모호성** — 라벨 노이즈, loss 교체로 개선 어려움
3. **Threshold 조정으로 class 2 recall 개선 가능** (0.40→0.61) — 재학습 불필요
4. **dBCQ가 RL 최강** (FQE > clinician), 단 ESS 낮아 통계적 주장 제한
5. Top-2 accuracy 87.5% → "한 칸 이내" 정확도는 충분히 높음

## 파일 목록
```
figures/
  01_cohort_overview.png        코호트 통계
  02_supervised_comparison.png  지도학습 모델 비교
  03_per_class_f1_heatmap.png   F1 히트맵
  04_ope_results.png            OPE 결과
  05_threshold_tradeoff.png     Threshold sweep
  06_rl_action_dist.png         RL action 분포
  07_confusion_matrix_gbm.png   GBM confusion matrix
data/
  summary_supervised.csv
  summary_rl.csv
  summary_threshold.csv
```
