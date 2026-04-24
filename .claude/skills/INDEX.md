# 스킬 매뉴얼 INDEX

도메인별 기술 가이드. 전부 로드하지 않고 트리거 키워드/파일경로/코드 패턴 감지 시 해당 챕터만 자동 로드한다.

## 활성화 표

| 챕터 | 트리거 키워드 | 파일 경로 | 코드 패턴 |
|---|---|---|---|
| [ch01-mimic-sql.md](ch01-mimic-sql.md) | mimic, postgres, schema, query, chartevents, inputevents, labevents | `src/db.py`, `src/cohort.py`, `*.sql` | `psycopg2`, `read_sql`, `mimiciv_` |
| [ch02-sepsis3-cohort.md](ch02-sepsis3-cohort.md) | 코호트, cohort, sepsis, sofa, inclusion, exclusion | `src/cohort.py`, `notebooks/01_*` | `sepsis3`, `icustays`, `suspicion_of_infection` |
| [ch03-features-engineering.md](ch03-features-engineering.md) | 피처, feature, binning, vitals, lab, 시계열 | `src/features.py`, `notebooks/02_*` | `groupby('charttime')`, `resample`, `rolling` |
| [ch04-supervised-baseline.md](ch04-supervised-baseline.md) | 지도학습, supervised, regression, classification, baseline, gbm, mlp | `src/models/supervised*.py` | `LogisticRegression`, `LightGBM`, `nn.Module` |
| [ch05-offline-rl.md](ch05-offline-rl.md) | RL, 강화학습, offline, dBCQ, CQL, BC, FQE, OPE, off-policy | `src/models/rl*.py` | `replay_buffer`, `q_network`, `policy` |
| [ch06-streamlit-cdss.md](ch06-streamlit-cdss.md) | streamlit, 웹, 데모, UI, 추천 화면 | `app/*.py` | `st.sidebar`, `st.plotly_chart`, `@st.cache` |
| [ch07-evaluation.md](ch07-evaluation.md) | 평가, evaluation, metric, fairness, calibration, OPE | `src/eval/*.py` | `roc_auc_score`, `brier_score`, `WIS`, `FQE` |
| [ch08-python-quality.md](ch08-python-quality.md) | (모든 코드 작업 시 자동 로드) | `*.py` | — |
| [ch09-security-privacy.md](ch09-security-privacy.md) | 보안, security, PHI, deidentification, .env, secret | `.env*`, `src/db.py`, `app/*.py` | `os.environ`, `password`, `subject_id` |
| [ch10-git-workflow.md](ch10-git-workflow.md) | git, commit, push, PR, merge, 커밋 | — | — |

## 작업 패턴 → 챕터 자동 추가

- "만들어", "구현해", "코드 작성" → ch08 (python quality) 자동 추가
- DB 쿼리 작성 → ch01 + ch09 (보안)
- 모델 학습 코드 → ch04 또는 ch05 + ch07 (평가)
- UI 작업 → ch06 + ch09 (PHI 노출 주의)

## 챕터 작성 규칙

- 각 챕터는 **이 프로젝트 한정 규칙**만 담는다 (일반 Python 가이드는 ch08만).
- 상위 결정사항은 `PROJECT_PLAN.md` 참고, 규칙은 챕터에서 강제.
- 변경 시 INDEX의 트리거 키워드도 동기화.
