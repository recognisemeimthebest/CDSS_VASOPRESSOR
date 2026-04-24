# 기획서 — CDSS Vasopressor Recommendation (MIMIC-IV 연습)

> 본 문서는 **읽기 전용 기획서**다. 사람 또는 plan-auditor 에이전트만 수정한다.
> 구현 결정의 근거 변동은 `workspace/context-note.md`에, 진행 상태는 `workspace/checklist.md`에 기록한다.

## 1. 목적

향후 도착할 **암환자 방사선치료 시계열 데이터**를 입력받아 환자의 임상 변수를 종합 고려해 **방사선 조사량/스케줄(`totaldose`, `radiationcnt`, `radiationperdose`, `treatmethod`, `treatech`)** 을 추천하는 임상의사결정지원 서비스(CDSS)의 사전 연습.

방사선 데이터가 제한적 변수만 제공하는 것에 비해 MIMIC-IV는 풍부한 임상 변수를 제공하므로 **동일한 의사결정 구조 — "환자 상태(시계열) → 치료 용량 결정" — 을 vasopressor 추천 태스크로 먼저 학습**한다. 입력 어댑터만 갈아끼우면 방사선 데이터로 이전 가능하도록 설계한다.

## 2. 연습 태스크 정의

**입력**: sepsis-3 ICU 환자의 시계열 임상 상태 (활력, 검사, SOFA, 인공호흡 등)
**출력**: norepinephrine-equivalent dose 추천 (단일 연속값 또는 5-bin 이산값)
**평가**: 의사 실제 결정과의 일치도, off-policy evaluation (RL 시), 임상 결과 보정

### 코호트 (확정 — 13,071 stays/subjects)
- `mimiciv_derived.sepsis3` 기준 patient-stay
- 성인 (≥18세)
- ICU 첫 stay만, ICU LOS ≥ 24h
- vasopressor (NE-equiv > 0) 한 번이라도 받음
- **NE 첫 시작 시점 > intime** (외부 transfer NE 연속 케이스 제외)
- **누적 NE 사용 시간 ≥ 1h** (일회성 bolus 제외)

Consort flow: 29,320 first-stay pool → 14,506 (got NE) → 13,964 (start>intime) → **13,071 (cum≥1h)**.

### 분석 윈도우
- ICU 입실 후 첫 72시간, 4시간 bin (stay당 최대 18 timestep)

### 액션 표현 (확정)
- `mimiciv_derived.norepinephrine_equivalent_dose` 사용
- AI Clinician (Komorowski 2018) 표준: 모든 vasopressor를 NE 등가 mcg/kg/min 으로 통합
- Bin 정의: `0`, `(0, 8.4]`, `(8.4, 20.28]`, `(20.28, 50]`, `>50` mcg/min (NE-equiv)
- 시간 단위: 4시간 bin

### 상태 표현 (확정)
- 시간 단위: 4시간 bin, 윈도우 [0, 72]h
- 변수: vitals (`mimiciv_derived.vitalsign`), bg/lab 패널, SOFA components, 인구학(`age`, `weight_durations`), ventilation flag
- 결측: forward-fill 후 코호트 평균
- 정규화: train fold 통계로 z-score

## 3. 모델링

### Phase A — Supervised Baseline
- 회귀: 다음 4h NE-equiv 용량 예측 (현재 상태 → 다음 액션)
- 분류: 5-bin 다음 액션
- 모델: LR, GBM, MLP (베이스라인); 시계열 트랜스포머 (확장)

### Phase B — Offline RL
- 알고리즘 후보: Behavior Cloning, dBCQ, CQL (offline-safe)
- 평가: WIS / FQE / clinician matching rate
- 안전성: 학습 정책 vs 의사 결정 분포 차이 시각화 필수

### Phase C — Multi-output 확장 (방사선 데이터 매핑)
- NE-equiv → 약물별 용량으로 출력 head 확장
- 방사선 데이터의 (totaldose, radiationcnt, technique) multi-target 구조와 매칭 검증

## 4. 결과물

- **웹 데모**: Streamlit (`app/main.py`) — 환자 ID 입력 → 시계열 상태 시각화 → 추천 + 근거 차트
- **재현성**: notebooks (EDA, 학습) + 모델 체크포인트 + `environment.yml`
- **GitHub**: https://github.com/recognisemeimthebest/CDSS_VASOPRESSOR

## 5. 기술 스택

- **OS**: Windows 11
- **Python**: 3.11 (Anaconda env `cdss_vasopressor`, prefix `G:\anaconda_envs\cdss_vasopressor`)
- **DB**: PostgreSQL 16 (`localhost:5432/mimic4`), 스키마 `mimiciv_hosp/icu/note/ecg/derived`
- **GPU**: RTX 4070 Ti SUPER (16GB VRAM), CUDA 12.1
- **핵심 라이브러리**: PyTorch 2.5+cu121, scikit-learn, pandas, psycopg2/SQLAlchemy, streamlit, plotly
- **외부 의존성**: mit-lcp/mimic-code (concepts_postgres) — `g:\miniproject2\mimic-code\`

## 6. 디렉토리 구조

```
cdss_vasopressor/
├── .claude/                  # 오케스트레이션 (HARNESS)
├── data/                     # gitignored
├── notebooks/                # EDA
├── src/
│   ├── db.py                 # PG 연결
│   ├── cohort.py             # sepsis-3 코호트
│   ├── features.py           # 4h bin, vitals/lab 집계
│   ├── models/               # supervised, RL
│   └── eval/
├── app/main.py               # Streamlit
├── tests/
├── scripts/
│   ├── run.bat / run.ps1     # conda env 런처
│   ├── verify_env.py
│   └── build_derived.py
└── README.md, environment.yml, .env.example, .gitignore
```

## 7. 마일스톤

| Phase | 내용 | 산출물 |
|---|---|---|
| ✅ 0 | 요구사항 + 환경 셋업 | conda env, GitHub repo, DB 연결 |
| 🔄 0.7 | mimiciv_derived 빌드 | derived 스키마 + AI Clinician 핵심 테이블 |
| ⬜ 1 | 코호트 정의 + EDA | `notebooks/01_cohort_eda.ipynb`, `src/cohort.py` |
| ⬜ 2 | 시계열 피처 | `src/features.py`, train/val/test 분할 |
| ⬜ 3 | Supervised 베이스라인 | `src/models/supervised.py`, 평가 리포트 |
| ⬜ 4 | Offline RL | `src/models/rl.py`, OPE 결과 |
| ⬜ 5 | Streamlit 데모 | `app/main.py` 동작, 추천 + 근거 시각화 |
| ⬜ 6 | 방사선 데이터 어댑터 | 입력 스키마 매핑 layer |

## 8. 제약사항 / 비목표

- **임상 사용 금지**: 본 시스템은 학습/연구 프로토타입. 실제 환자 진료에 사용하면 안 됨.
- **MIMIC 데이터 외부 유출 금지**: PhysioNet 약관 준수, 학습 결과/시각화에 환자 정보 노출 금지.
- **외부 API 호출 없음**: 모델은 로컬에서만 학습/추론. PHI(환자 식별 정보)가 외부로 나가는 코드 작성 금지.
- 본 단계에서는 영상(CBCT) 데이터 미사용. 방사선 데이터 본 단계에서 영상 추가 검토.

## 9. 참고 선행연구 (검증된 인용)

핵심 (반드시 따라야 함):
- **Komorowski et al. 2018** — *AI Clinician*, Nat Med 24:1716-1720, doi:10.1038/s41591-018-0213-5 — sepsis vasopressor RL 원조
- **Raghu et al. 2017** — *Continuous State-Space Models for Optimal Sepsis Treatment*, MLHC 2017, arXiv:1705.08422 — continuous state + Dueling DDQN
- **Tseng et al. 2017** — *Deep RL for automated radiation adaptation in lung cancer*, Med Phys 44(12):6690-6705, doi:10.1002/mp.12625 — RT RL 원조 (방사선 phase)

평가/안전성 (필수 가드):
- **Gottesman et al. 2018/19** — *Evaluating RL algorithms in observational health settings*, Nat Med correspondence, OPE 5대 함정
- **Tang & Wiens 2021** — *Model Selection for Offline RL*, CHIL — WIS vs FQE bound 비교
- **Killian et al. 2020** — *Empirical Study of Representation Learning for RL in Healthcare*, NeurIPS ML4H — encoder ablation 필수
- **Fatemi et al. 2021** — *Medical Dead-ends*, NeurIPS — contraindication head 권장
- **Festor et al. 2022** — *Assuring the safety of AI-based CDSS*, BMJ Health Care Inform 29:e100549 — safety envelope
- **Wu et al. 2023** — multi-cohort 재현 실패, Lancet Digital Health
- **Roggeveen et al. 2021** — transatlantic transfer 실패, Comput Biol Med
- **Jeter et al. 2019** — Komorowski 재현 의문, arXiv:1902.03271

자세한 적용 방침: `.claude/skills/ch05-offline-rl.md`, `ch07-evaluation.md`, `ch03-features-engineering.md`. 메모리: `research_grounded_decisions.md`.
