# CDSS Vasopressor — 팀 브리핑

> 본 문서는 팀원/교수 대상 **프로젝트 현재 상태 요약**. 기술 상세는 각 파일/챕터 링크 참조.

---

## 1. 한 문장 요약

방사선치료 용량 추천 CDSS의 본 목적을 위해, MIMIC-IV 패혈증 환자의 vasopressor(NE 등가 용량) 결정을 AI Clinician(Komorowski 2018) 스타일 **지도학습 + 오프라인 강화학습**으로 학습 중. 동일 파이프라인을 방사선 데이터로 이식 예정.

## 2. 왜 이 프로젝트

- **본 목적**: 시계열 방사선치료 데이터 입력 → `totaldose / radiationcnt / technique` 추천 (암환자 5,019명, PDF 스키마 있음)
- **현 연습 단계**: MIMIC-IV로 동일 구조 ("환자 시계열 상태 → 치료 용량 결정") 먼저 완성
- **입력 어댑터만 교체**하면 방사선 데이터로 이전 가능한 구조 유지

## 3. 코호트 (확정 — 13,071 환자)

[`.claude/PROJECT_PLAN.md`](../.claude/PROJECT_PLAN.md) §2.1 기준

**Inclusion** (7 단계 — Consort flow 원본: `data/cohort_v1_consort.csv`):

| 단계 | n_stays |
|---|---:|
| sepsis-3 모든 stay | 41,296 |
| + 성인 (age ≥ 18) | 41,296 |
| + ICU LOS ≥ 24h | 37,143 |
| + 첫 ICU stay만 | 29,320 |
| + NE-equivalent 한 번이라도 | 14,506 |
| + NE 시작 > ICU intime (외부 transfer 배제) | 13,964 |
| **+ 누적 NE ≥ 1h (bolus 배제)** | **13,071** |

**임상적 의미**: 성인 패혈증 ICU 환자 중 ICU 입실 후 vasopressor 사용을 결정하고 1시간 이상 지속한 환자 ≈ **septic shock 환자**.

**윈도우**: ICU 입실 ~ +72h, **4시간 bin** (stay당 최대 18 timestep).

## 4. 데이터 규모

- `data/features_v1.parquet`: **235,278 rows × 86 cols** (13,071 stays × 시간별 × 피처)
- 피처 카테고리: vitals 8종(HR/BP/RR/SpO2/Temp/Glucose) × 4 통계, bg/chem/coag/cbc lab, SOFA 6 components, ventilation, urine, 인구학, weight
- Action: `next_ne_dose` (mcg/kg/min 연속) + `next_ne_action_bin` (5-bin: 0 / (0, 8.4] / (8.4, 20.28] / (20.28, 50] / >50 mcg/min, AI Clinician 표준)
- **Class 불균형**: bin 0 (NE 없음) 57.9%, bin 4 (>50) 1.4%

## 5. 모델 결과 (Phase 3 + 3.5, Test fold)

### Phase 3 — Supervised 베이스라인 (per-bin 예측)

| 모델 | 회귀 MAE | 회귀 R² | 분류 acc | 분류 macroF1 | 분류 top-2 | 분류 ECE |
|---|---:|---:|---:|---:|---:|---:|
| LR (class_weight balanced) | 0.065 | 0.318 | 0.614 | 0.455 | 0.814 | 0.049 |
| **GBM** (Optuna 50 trials) | 0.044 | **0.532** | 0.676 | **0.529** | 0.875 | **0.030** |
| MLP (Optuna 25 trials, dropout + AdamW + early stop) | 0.041 | 0.477 | 0.642 | 0.504 | 0.858 | 0.088 |

→ **GBM 우승** (R² + macro-F1 + calibration 최고)

### Phase 3.5 — Encoder ablation

| 모델 | 회귀 MAE | 분류 acc | 분류 top-2 |
|---|---:|---:|---:|
| TCN (15 trials, causal dilated conv) | **0.027** | **0.734** | **0.893** |

⚠️ TCN은 **stay-level 예측** (마지막 bin 기준) — per-bin 모델과 직접 비교 불공정. Phase 4 RL Q-network 에서 공정 비교.

## 6. 기술 결정 (문헌 근거)

[`.claude/skills/ch05-offline-rl.md`](../.claude/skills/ch05-offline-rl.md), [`ch07-evaluation.md`](../.claude/skills/ch07-evaluation.md), 메모리 `research_grounded_decisions.md`

**채택**:
- AI Clinician 뼈대 (Komorowski 2018 Nat Med) — sepsis-3 + 4h bin + NE-equiv
- Tabular 건너뛰고 continuous state 직진 (Jeter 2019 재현성 의문)
- Offline RL: BC → Dueling DDQN (Raghu 2017) → dBCQ → CQL (Killian 2023)
- **Dual OPE 필수**: WIS + FQE + ESS + Bootstrap CI + null-policy baseline (Gottesman 2018/19)
- Encoder ablation (Killian 2020 — encoder > algo 영향)
- Dead-end head 옵션 (Fatemi 2021) — "하지 말 액션" 경고

**금지**:
- ❌ Mortality 감소 주장 (Festor 2022, Wu 2023)
- ❌ 750-state tabular
- ❌ Random row split (subject 누수)
- ❌ PHI 외부 송신
- ❌ SMOTE/oversampling (시계열 누수)

## 7. 진행 Phase

- ✅ **Phase 0**: 환경 + DB + HARNESS
- ✅ **Phase 1**: 코호트 정의 + EDA (13,071)
- ✅ **Phase 2**: 시계열 피처 + train/val/test 분할 (leakage 0)
- ✅ **Phase 3**: LR/GBM/MLP baseline + Optuna + class weights
- ✅ **Phase 3.5**: TCN encoder ablation
- 🔄 **Phase 4**: Offline RL (BC → DDQN → dBCQ → CQL) + OPE — **진행 중**
- ⬜ **Phase 5**: Streamlit 데모
- ⬜ **Phase 6**: 방사선 데이터 어댑터

## 8. 재현 방법

```bash
conda env create -f environment.yml
conda activate cdss_vasopressor   # 또는 scripts\run.bat 런처 사용 (Windows OMP 워크어라운드)

# 코호트 + 피처 빌드
scripts\run.bat -m src.cohort          # 13,071 stays 확인
scripts\run.bat -m src.features        # 235,278 rows
scripts\run.bat -m src.splits          # train/val/test

# Supervised baseline
scripts\run.bat -m src.training.run_baseline --model gbm --task cls --tune --n-trials 50

# 결과 시각화
scripts\run.bat -m jupyter nbconvert --to notebook --execute notebooks\03_supervised_baseline.ipynb --inplace
```

## 9. 주요 파일

| 경로 | 내용 |
|---|---|
| [`.claude/PROJECT_PLAN.md`](../.claude/PROJECT_PLAN.md) | 읽기 전용 기획서 |
| [`.claude/workspace/context-note.md`](../.claude/workspace/context-note.md) | 결정 근거 누적 |
| [`.claude/workspace/checklist.md`](../.claude/workspace/checklist.md) | 진행 체크리스트 |
| [`.claude/skills/`](../.claude/skills/) | 도메인 가이드 10 챕터 |
| [`.claude/agents/`](../.claude/agents/) | 도메인 에이전트 8개 |
| [`src/cohort.py`](../src/cohort.py) | 코호트 SQL → DataFrame |
| [`src/features.py`](../src/features.py) | 4h bin 피처 추출 |
| [`src/models/_gbm.py`](../src/models/_gbm.py) | LightGBM with Optuna |
| [`src/models/_mlp.py`](../src/models/_mlp.py) | MLP + dropout + focal loss |
| [`src/models/_tcn.py`](../src/models/_tcn.py) | TCN encoder-based |
| [`notebooks/`](../notebooks/) | EDA 노트북 (코호트/피처/baseline) |
| [`artifacts/runs/`](../artifacts/runs/) | 학습 결과 (config/metrics/predictions/model) |

## 10. 팀원 질문 받을 법한 것

**Q: 왜 MIMIC으로 연습해?**  
A: 방사선 데이터가 아직 없고, 임상 변수 훨씬 풍부해서 파이프라인 검증에 유리. 같은 추상화 ("시계열 → 용량") 라서 어댑터 교체로 이전 가능.

**Q: 왜 vasopressor 카테고리 통합(NE 등가)?**  
A: AI Clinician 표준. 약물 선택 정보는 잃지만 의사결정의 핵심 (총 vasopressor 강도) 유지. Phase 6에서 약물별로 확장 가능.

**Q: GBM 이 더 좋으면 GBM 배포하면 되나?**  
A: GBM은 의사 결정 "모방"만 학습. RL은 장기 결과 최적화. 다른 문제. Phase 4가 진짜 답.

**Q: mortality 줄어들 수 있어?**  
A: prospective RCT 없이 주장 금지. 본 시스템은 학습 프로토타입. 임상 사용 금지 (README 디스클레이머).

**Q: 어떤 환자에 쓸거야?**  
A: 성인 패혈증 ICU 환자 중 첫 72h 동안 vasopressor 결정이 필요한 환자 (septic shock). 소아, 비-sepsis 쇼크, 외래는 범위 밖.
