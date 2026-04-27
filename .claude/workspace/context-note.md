# 미니프로젝트 2 — 컨텍스트 노트

## 최종 목표
시계열 방사선치료 데이터를 입력받아, 환자의 임상 변수를 종합 고려해 **방사선 조사량/스케줄을 추천하는 CDSS(임상의사결정지원) 서비스** 구축.

예시 출력 형태:
- "○○환자에게 △△Gy를 ○○회 분할 조사 권장"
- "○○환자에게 heparin ○○unit 투여 제안" (MIMIC 연습용 유사 태스크)
- "○○환자에게 △△약물 투여 권장"

## 현재 단계
실제 방사선 데이터가 도착하기 전, **MIMIC-IV로 동일한 의사결정 패턴을 연습**.
- 이유: 방사선 데이터에는 임상 변수가 제한적일 수 있으나, MIMIC-IV는 풍부한 임상 변수(LAB/VITAL/DX/RX)를 제공해 전체 파이프라인 연습 가능.
- 핵심 추상화: "환자 임상 상태(시계열 변수) → 치료/처치 결정(연속/이산 액션)" 매핑.

## 보유 자원
- `데이터설명서_1-025-071_암환자방사선치료데이터_아주대학교산학협력단.pdf` — 향후 받을 방사선 데이터 스키마 명세
  - 텍스트 추출본: `.claude/workspace/data_spec.txt` (한글은 깨짐, 영문 변수명은 정상)
- MIMIC-IV — DBeaver에 연동됨 (구체적 DB 접속 정보 사용자 확인 필요)

## 방사선 데이터 (사전 파악) — 최종 타겟
- 규모: 환자 5,019명 / CBCT(DICOM) ~204만장 / Tumor bbox 어노테이션 30,136
- **타겟 변수 (추천 대상)**: `totaldose`, `radiationcnt`, `radiationperdose`, `treatmethod`, `treatech`, `antidrug`
- **임상 입력 변수**: sex, birth date, height, weight, 병력 플래그(bp/bs/sm/familyhistory),
  Diagnosis, locationcancer, cancerimaging, TNM, 수술정보, 시점 (initialdate/treatedate/relapse/dead 등)
- 영상 입력: CBCT (axial), 5분할 / 6 fraction × 6 phase 구조

## 결정된 사항
- **MIMIC 접근**: DBeaver/PostgreSQL, MIMIC-IV 최신. 접속정보 추출 대기
- **모델링 방향**: 지도학습 + 강화학습 둘 다
- **연습 태스크**: vasopressor 카테고리 전체 (NE, vasopressin, epi 등 — AI Clinician 스타일).
  근거: GitHub 레포명 `CDSS_VASOPRESSOR` + 폴더명 `cdss_vasopressor`로 카테고리 전체 의도 확인
- **개발 위치**: `g:\miniproject2\cdss_vasopressor\`
- **OS/환경**: Windows + Anaconda (WSL2 안 씀 — DB/파일 모두 Windows 쪽)
- **Python**: 3.11 (Anaconda 가상환경 `cdss_vasopressor`)
- **결과물**: Streamlit 웹앱
- **GPU**: RTX 4070 Ti SUPER, 16GB VRAM
- **GitHub**: https://github.com/recognisemeimthebest/CDSS_VASOPRESSOR

## vasopressin vs vasopressor (용어 정리)
- vasopressor = 혈압상승제 카테고리 (NE, vasopressin, epi, phenyl, dopa)
- vasopressin = 그중 한 약물 (AVP, 보통 NE 보조 2차약)
- 본 프로젝트는 카테고리 전체를 다룸

## 진행 현황
- 폴더/스켈레톤/git/GitHub push 완료 (커밋 55ea4fa, https://github.com/recognisemeimthebest/CDSS_VASOPRESSOR)
- `.env` 작성됨 (gitignored) — DB: localhost:5432/mimic4, user postgres
- Conda env 1차 시도: PyTorch 다운로드 1.2GB 중 연결 끊김으로 실패
- Conda env 2차 시도 진행 중: PyTorch 제외하고 기본 패키지만 설치 → 이후 PyTorch는 pip로 분리 설치

## DB 정보 (검증 완료)
- Host: localhost / Port: 5432 / DB: **mimic4** / User: postgres
- PostgreSQL 16.13
- **실제 스키마**:
  - `mimiciv_hosp` (22 tables) — admissions, patients, labevents, prescriptions 등
  - `mimiciv_icu` (9 tables) — icustays, chartevents, **inputevents** (vasopressor 여기) 등
  - `mimiciv_note` (4 tables) — discharge / radiology 노트
  - `mimiciv_ecg` (3 tables) — ECG
- ❗ `mimiciv_derived` **로드 안 됨** → SOFA, sepsis-3 코호트는 직접 SQL 작성 또는
  mit-lcp/mimic-code의 derived SQL 따로 import 필요

## Windows 환경 트랩 (해결됨)
- 증상: `python.exe` 직접 호출 시 `0xC0000142` (DLL 초기화 실패)
- 원인: conda 설치 numpy(MKL) + pip 설치 PyTorch(자체 OpenMP) → libiomp5md/libomp 충돌
- 해결: `scripts/run.bat` 런처 사용 — `KMP_DUPLICATE_LIB_OK=TRUE` + `activate.bat` 거침
- Streamlit/Jupyter 실행 시에도 반드시 이 런처 거쳐야 함

## 모델링 결정 (확정)
- **액션 표현**: NE 등가 단일 용량 (AI Clinician Komorowski 2018 표준)
  → `mimiciv_derived.norepinephrine_equivalent_dose` 테이블 사용
- **코호트**: sepsis-3 환자, ICU 첫 72시간 (`mimiciv_derived.sepsis3`)
- **상태 변수**: SOFA, vitals, labs, ventilation 등 (모두 derived)
- **방향**: 우선 단일 출력 → Phase 후반 다중 출력으로 확장 (방사선 데이터 multi-target과 매칭)

## 진행 현황
- Phase 0.5 환경 셋업 완료 (커밋 755569d)
- **Phase 0.7 완료**: mimiciv_derived 빌드 (63 테이블, sepsis3 41k stays, NE-equiv 784k rows)
- **Phase 0.9 완료**: 전역 훅 cwd-aware 패치 → cdss_vasopressor 워크스페이스 자동 인식
- **Phase 0.95 완료**: 선행연구 조사 + 스킬 챕터 보강
  - 리서치 에이전트로 16편+ 논문 조사 (AI Clinician 계보 + 방사선 RT RL + 안전성 논문)
  - DOI 4편 웹 검증 (Festor 2022는 npj 아닌 BMJ Health Care Inform 으로 정정)
  - ch05/ch07/ch03에 학술 근거 보강: dual OPE (WIS+FQE+ESS+CI), null-policy baseline,
    dead-end head, encoder ablation, mortality 주장 금지
  - PROJECT_PLAN §9 검증된 인용으로 갱신, 메모리 `research_grounded_decisions.md` 추가

## 코호트 — 확정 (13,071 stays/subjects)
sepsis3 + adult + LOS≥1d + first ICU stay + got NE-ever + **NE 시작 > intime + 누적 ≥ 1h**

**Why 추가한 두 조건**:
- "NE 시작 > intime": 외부 transfer 도중 NE 연속 케이스 배제 → ICU 의사결정 학습 신호 정제
- "누적 ≥ 1h": 일회성 bolus 환자 배제 → 결정 일관성 확보

**대안 검토**: 60min grace는 -35% 데이터 손실로 너무 엄격, 0min threshold는 literal-transfer만 깔끔히 제거 (≈10% 손실로 적정).

표준 SQL은 `ch02-sepsis3-cohort.md` 참조. quick check SQL: `scripts/quick_consort_eb.sql`.

## 모델링 — 둘 다
- Supervised (Phase A) = 의사 결정 모방, 베이스라인. LR/GBM/MLP/Transformer.
- **Offline RL (Phase B) = 본 모델**. BC → Dueling DDQN → dBCQ → CQL. AI Clinician 표준.
- 같은 코호트를 둘 다 학습 → 결과 비교 (RL이 의사와 얼마나 다른가)

## Phase 1 완료 (cohort-agent)
- `src/cohort.py` 작성 — `load_cohort()`, `compute_consort_flow()`, `save_consort_flow()`
- 검증: `scripts\run.bat -m src.cohort` 실행 → **13,071 stays / 13,071 subjects 확인**
- 캐시: `data/cohort_v1.parquet` (1.5MB), `data/cohort_v1_consort.csv`
- 코호트 인구학 (첫 결과):
  - Age mean=67.1, median=68.5
  - Gender: M 7,820 / F 5,251
  - In-hospital mortality: **23.1%** (전형적 septic shock)
- 노트북 `notebooks/01_cohort_eda.ipynb` 작성됨, 실행 진행 중

### 비자명한 결정 (cohort.py)
- sepsis3_onset_time = `MIN(sofa_time)` (sepsis3 테이블이 SOFA 윈도우 이동마다 다중 row 생성)
- ROW_NUMBER tiebreak는 `intime` 만 (필요시 stay_id 추가로 결정론 강화)
- gender NULL 그대로 유지 — 노트북에서만 "Unknown" 표시, 코호트 필터 아님
- race는 admissions 테이블에서 직접 (hadm_id 1:1), 노트북에서 7-bucket으로 집계

## Phase 2 완료 (features-agent + 사용자 검증)
- `src/features.py` 약 540줄 — 7개 derived 테이블 조인, 4h bin × stay 18 timesteps
- 검증: `data/features_v1.parquet` **235,278 rows × 86 cols** (14.6MB), 14초 빌드
- `src/splits.py` — subject_random 70/15/15, **leakage 0 confirmed**
  - train 9149 / val 1961 / test 1961 (`data/splits_v1.json`)
- `notebooks/02_features_eda.ipynb` — 11 셀 sanity (실행 완료, 232KB)
- `tests/test_features.py` — pure-logic + DB integration
- `artifacts/feature_stats_v1.json` — variable mean/std/median/missing rate

### 액션 분포 (5-bin NE-equiv mcg/min)
- 0 (NE 없음): 136,218 (57.9%) — 매우 불균형
- 1 (≤8.4): 57,932 (24.6%)
- 2 (8.4-20.28): 24,537 (10.4%)
- 3 (20.28-50): 13,310 (5.7%)
- 4 (>50): 3,281 (1.4%)
→ 분류 모델은 class-weight 또는 macro-F1 필수 (ch04 명시)

### 비자명한 결정 (features.py)
- SOFA = `*_24hours` 컬럼 (24h max, instantaneous보다 안정적). coagulation은 `coagulation_score`로 rename (lab의 inr/ptt와 구분)
- vent_flag: InvasiveVent/Tracheostomy/NonInvasiveVent만 양성. HighFlow/SupplementalOxygen 제외
- bg/chemistry/coag/cbc는 stay_id 없음 → subject_id+hadm_id로 icustays 재조인
- next_ne_dose 산출: SQL 윈도우 76h로 확장 후 t_bin shift (bin 17의 next는 72-76h)
- Vitals: heart_rate/resp_rate를 hr/rr로 alias 통일
- weight 결측 stay → 코호트 평균 (kg)
- chunk=3000 ids로 SQL 분할 → PG plan cache 안정화

## Phase 3 완료 (supervised-model-agent + 사용자 검증)
- 모델 6종 학습 완료 (LR/GBM/MLP × 회귀/분류)
- Optuna 튜닝: GBM 50 trials, MLP 25 trials (TPE + MedianPruner)
- 정규화: dropout (Optuna 튜닝), AdamW + ReduceLROnPlateau, early stopping (patience=10), L2
- 클래스 불균형 처리: class_weight='balanced' + focal loss 옵션 (MLP), top-2 acc + macro-F1 보고

### 결과 요약 (Test fold)
| 모델 | 회귀 MAE | 회귀 R² | 분류 macroF1 | 분류 top-2 | 분류 ECE |
|---|---|---|---|---|---|
| LR  | 0.065 | 0.318 | 0.455 | 0.814 | 0.049 |
| **GBM** | 0.044 | **0.532** | **0.529** | **0.875** | **0.030** |
| MLP | **0.041** | 0.477 | 0.504 | 0.858 | 0.088 |
- GBM 종합 우승. MLP는 calibration 부족 (focal loss 부작용) — eval-agent 단계에서 isotonic 보정 권장

### 비자명한 트랩 (Phase 3에서 만남)
- **DLL 충돌**: lightgbm + torch + pandas 동시 로드 시 fbgemm.dll 로드 실패 → run_baseline.py 상단에 `import torch` 먼저 + `bootstrap_registry`에서 `_mlp` 우선 import로 해결
- **numpy 2.x ABI 깨짐**: torch 2.5.1은 numpy 2.x와 segfault → numpy<2 (1.26.4)로 다운그레이드. environment.yml에 핀 추가
- conda env에 pytest 누락 → pip로 추가 설치

## Phase 3.5 — Encoder ablation (TCN)
- LSTM/Transformer 제외 (유사하거나 과대), MLP_last (현재) + TCN 비교
- `src/models/encoders/tcn.py`: causal dilated 1D conv, 2-4 blocks, weight_norm + BN + GELU + Dropout + residual
- `src/models/_tcn.py`: TCNRegressor / TCNClassifier (MLPBase 재사용)
- Optuna 15 trials each, GPU

### 중요 caveat (직접 비교 불공정)
- 현재 구현: `flatten='all_bins'` → **stay당 1 sample** (라벨 = 마지막 bin의 next_ne_dose)
- 반면 LR/GBM/MLP_last → **bin당 1 sample** (라벨 = 해당 bin의 next_ne_dose)
- 샘플 수 & 라벨 분포 다름 → 지표 비교 시 caveat 명시

### TCN 결과 (Test)
| Metric | TCN | 최고 대비 |
|---|---|---|
| 회귀 MAE | **0.027** | 최고 |
| 회귀 R² | 0.301 | 낮음 (샘플 수 효과) |
| 분류 acc | **0.734** | 최고 |
| 분류 top-2 | **0.893** | 최고 |
| 분류 macro-F1 | 0.433 | 낮음 (minority에 약함) |

### Phase 4 RL 활용 계획
- TCN 을 **시간별 Q-value 예측** encoder 로 사용 (매 4h bin 마다 Q(s, a) 예측)
- 그때 per-bin 평가 가능해져 MLP_last 와 제대로 비교
- Killian 2020 교훈: encoder 선택이 OPE 결과에 정책 선택보다 큰 영향
- **Phase 0.8 완료**: HARNESS B 풀세트 적용
  - `.claude/` 풀구조 (settings.local.json, hooks, skills × 10, agents × 8)
  - PROJECT_PLAN.md (기획서, 읽기 전용), AGENT_REPORT_FORMAT.md
  - PostToolUse 훅 (보안/PHI/SQL/품질 자동 체크 + 변경 로그)
  - 워크스페이스 이전: g:\miniproject2\.claude\workspace\ → cdss_vasopressor/.claude/workspace/

## .claude/ 디렉토리 책임 분담
- `PROJECT_PLAN.md` — 무엇을 만드는가 (읽기 전용)
- `workspace/context-note.md` (이 파일) — 왜 이렇게 결정했는가
- `workspace/checklist.md` — 무엇이 끝났고 무엇이 남았는가
- `skills/` — 도메인별 기술 규칙 (트리거 시 자동 로드)
- `agents/` — 전문 에이전트 (cohort/features/supervised/rl/eval/streamlit + code/plan auditor)
- `hooks/` — 자동 체크 훅 + 변경 로그

## 설계 시 유지해야 할 추상화
- **입력 인터페이스**: 시계열 임상변수 (환자ID, 시각, 변수명, 값) → 추후 방사선 데이터로 갈아끼울 수 있게 어댑터 패턴.
- **출력 인터페이스**: 액션 추천 + 근거(어떤 변수가 영향) — 임상 신뢰성 확보.
- 평가는 "의사 실제 결정과의 일치도" 외에 **counterfactual / off-policy evaluation** 고려 (RL 사용 시).

## Phase 4 완료 (Offline RL 학습 + OPE)

### 실행된 알고리즘 (2026-04-27)
- BC (mlp, seed=42): `artifacts/runs/2026-04-27_rl_bc_mlp_42/`
- dBCQ (mlp, seed=42): `artifacts/runs/2026-04-27_rl_dbcq_mlp_42/`
- CQL (mlp, seed=42): `artifacts/runs/2026-04-27_rl_cql_mlp_42/`

### OPE 결과 요약 (val fold)
| Policy | WIS | FQE | ESS% | Match% | OOD% |
|---|---|---|---|---|---|
| **BC** | **6.642** [6.150,7.157] | 0.017 | **100.0** | 64.2 | 0.0 |
| dBCQ | 6.361 [4.020,8.461] | **0.145** | 4.5 ⚠️ | 64.9 | 2.9 |
| CQL | 2.363 [-0.541,5.187] | 0.127 | 3.9 ⚠️ | **68.2** | **0.5** |
| Clinician | 6.004 [5.453,6.550] | 0.055 | 87.0 | 100.0 | 2.9 |

### 해석 (비자명한 결정)
- **BC WIS=6.64 > Clinician WIS=6.00**: BC가 clinician 모방이므로 IS ratio≈1 → ESS 100%. WIS 신뢰도 최고. 의사와 유사하지만 class-weight 보정으로 minority dose 더 추천 → 가중 수익 소폭 상승.
- **dBCQ/CQL ESS 3~5%**: 두 알고리즘이 clinician 정책과 충분히 달라 IS 비율이 극단적 → WIS CI 매우 넓음. ESS 5% 미만 경고 발동.
- **FQE 기준 dBCQ > CQL > clinician > BC**: FQE는 별도 critic이라 IS 문제 없음. dBCQ의 행동 제약이 Q 오버에스티메이션을 막으면서도 탐색 여지를 줌.
- **CQL WIS 낮음**: conservative penalty가 action 분포를 크게 바꿔 IS 분산 폭발.

### 결론
- OPE 신뢰도: BC > Clinician >> dBCQ ≈ CQL (ESS 문제)
- FQE 정책 가치: dBCQ > CQL (CI 겹침 주의 — 통계적 유의성 주장 불가)
- 최종 권장: **BC를 안전 베이스라인**, dBCQ를 본 모델로 사용 (ESS 낮지만 FQE 우세)
- 향후: CQL alpha 낮추거나 dBCQ bcq_threshold 올려 IS 분산 줄이기 고려

## Phase 3.6 — Cost-Sensitive GBM 실험 (2026-04-27)

### 동기
표준 GBM의 class 2 (8.4-20 mcg/min, F1=0.356)가 최약점.
표준 CE loss는 모든 오분류에 동일 패널티 → ordinal 구조 무시.
해결책 시도: `C[i,j] = |i-j|^2` quadratic ordinal cost 로 대체.

### 구현
- `src/models/_gbm_cost.py`: `cost_sensitive_obj()` — LightGBM 4.x params["objective"] 방식
- `scripts/run_cost_sensitive_gbm.py`: Optuna best_params 재사용 (공정 비교)
- 아티팩트: `artifacts/runs/2026-04-27_gbm_cost_cls_42/`

### 결과 (Test fold)
| 지표 | Baseline GBM | Cost-Sensitive GBM | 변화 |
|---|---|---|---|
| Macro-F1 | 0.529 | 0.387 | **−0.143** |
| Accuracy | 0.676 | 0.664 | −0.012 |
| Top-2 Acc | 0.875 | **0.890** | **+0.015** |
| Class 2 F1 | 0.356 | 0.299 | **−0.057** ← TARGET 오히려 악화 |
| Class 3 F1 | 0.444 | 0.190 | −0.254 |
| Class 4 F1 | 0.460 | 0.066 | **−0.394 (붕괴)** |

### 해석 (비자명한 결론)
- **가설 기각**: Cost-sensitive objective가 class 2를 개선하지 못했다.
- **왜 실패했나?**:
  1. **Optuna 파라미터 미적응**: best_params는 표준 CE loss에서 50 trials로 최적화됨. cost-sensitive loss는 gradient scale이 달라서 동일 lr=0.0039가 최적이 아님.
  2. **Class 4 붕괴**: quadratic cost가 소수 class에 과도한 패널티를 부여해 gradient 불안정. class_weight (balanced)와 cost_matrix 패널티의 이중 증폭.
  3. **Top-2 Acc 개선**: 모델이 ordinal-close 예측을 더 하지만 F1 기준으로는 오히려 binary-adjacent 버킷으로 뭉침.
- **다음 대안** (우선순위 순):
  1. ~~Soft Labels (Diaz et al. CVPR 2019)~~ → 시도함, 실패 (아래 참조)
  2. Separate Optuna 튜닝 for cost-sensitive objective (공정한 비교를 위해 50 trials 추가)
  3. CORAL loss (binary decomposition): 소수 class 성능에 더 안정적

### 결론
Cost-sensitive GBM 실험 = **음성 결과 (negative result)**. 기록 가치 있음.

## Phase 3.6b — Soft Labels + MLP 실험 (2026-04-27)

### 동기
Cost-Sensitive GBM 실패 후 MLP에서 Gaussian soft label (Diaz et al. CVPR 2019) 시도.
- One-hot 대신 Gaussian 분포로 라벨 교체: `p_k = exp(-|k-y|^2/(2σ^2))`
- Loss = KL-divergence(soft_target ‖ softmax(logit)) = soft cross-entropy

### 구현
- `src/models/_mlp_soft.py`: `MLPSoftClassifier`, `SoftLabelKLLoss`, `make_soft_labels()`
- `scripts/run_soft_labels_mlp.py`: sigma sweep {0.5, 0.8, 1.0, 1.5} → best val macro-F1 선택
- 아티팩트: `artifacts/runs/2026-04-27_mlp_soft_cls_42/`

### 결과 (Test fold, best sigma=0.5)
| 지표 | Baseline MLP | Soft-Label MLP | 변화 |
|---|---|---|---|
| Macro-F1 | 0.504 | 0.479 | **−0.025** |
| Accuracy | 0.642 | 0.608 | **−0.034** |
| Top-2 Acc | 0.858 | 0.836 | −0.022 |
| Class 2 F1 | 0.358 | **0.347** | −0.011 ← 목표 역방향 |

Sigma sweep 결과: sigma=0.5 최고, sigma≥0.8은 epoch 2-6에 조기 수렴(붕괴)

### 해석
- **가설 기각**: Soft Labels도 class 2를 개선하지 못했다.
- **왜 실패했나?**:
  1. Optuna best_params가 focal loss로 튜닝됨 (lr=0.0065, use_focal=True) → KL loss와 비호환
  2. sigma가 클수록 soft target이 균등 분포에 가까워져 class_weight와 충돌 → 조기 붕괴
  3. sigma=0.5는 사실상 hard label에 가까운 Gaussian → CE와 큰 차이 없음

### 종합 결론 (Phase 3.6 전체)
**두 가지 ordinal 개선 시도 모두 음성 결과.**
공통 원인: **Optuna 재튜닝 없이 타 loss 함수 파라미터 재사용** → gradient scale 불일치.

## Phase 3.6c — GBM Threshold 조정 (2026-04-27) ✅ 양성 결과

### 방법
GBM 확률 출력에 per-class scale 벡터를 곱한 뒤 argmax:
`pred = argmax( proba * scale_vector )`
재학습 없음. Val fold에서 최적 scale 찾고 test fold에 적용.

### 결과 (Test fold)
| 전략 | Macro-F1 | Class 2 F1 | delta macro | delta cls2 |
|---|---|---|---|---|
| Baseline | 0.529 | 0.356 | 0 | 0 |
| A: cls2×1.5 (max cls2-F1) | 0.519 | **0.395** | −0.010 | **+0.039** |
| C: cls3×0.8 (max macro-F1) | 0.527 | 0.371 | −0.003 | **+0.015** |

- **Strategy A**: class 2 F1 +0.039 (0.356→0.395). precision 0.320→0.292 (false positive 증가), recall 0.403→0.610 (누락 감소). 임상적으로 "더 적극적으로 class 2를 추천"하는 트레이드오프.
- **Strategy C**: class 2 F1 +0.015, macro-F1 거의 유지 (−0.003). cls3 신뢰도를 낮춰 경계 케이스를 cls2로 끌어당김.
- 아티팩트: `artifacts/runs/2026-04-27_gbm_threshold_cls_42/`

### 해석 (비자명한 결정)
- Threshold 조정은 precision-recall tradeoff를 후처리로 이동한 것 — 모델 학습 능력 향상이 아님
- Strategy A (recall 중심): 놓치면 안 되는 고위험 구간에 유리 (class 2는 "증량 시작점")
- Strategy C (균형): macro-F1 유지하면서 class 2 소폭 개선 → 운영 시 기본 선택
- Streamlit 앱에서 "임상 목적 선택" 옵션으로 제공 가능 (recall-first vs balanced)

현 단계에서의 최선:
- **GBM + Strategy C scales (macro-F1=0.527, class 2 F1=0.371)** — 균형
- **GBM + Strategy A scales (macro-F1=0.519, class 2 F1=0.395)** — class 2 최우선 시

## 참고 선행연구
- AI Clinician (Komorowski 2018, Nat Med) — sepsis IV/vasopressor 추천 RL
- Deep RL for radiotherapy dose adaptation (Tseng 2017 등)
