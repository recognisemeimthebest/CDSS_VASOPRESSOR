# 미니프로젝트 2 — 체크리스트

## Phase 0: 요구사항/환경 확정
- [x] MIMIC-IV 접근 방식 — DBeaver/PostgreSQL, MIMIC-IV 최신
- [x] 첫 연습 태스크 — vasopressor 카테고리 전체 (AI Clinician 스타일)
- [x] 모델링 방향 — 지도학습 + RL 둘 다
- [x] 방사선 데이터설명서 PDF 변수 추출 → context-note.md
- [x] 결과물 형태 — Streamlit 웹
- [x] GPU 가용성 — RTX 4070 Ti SUPER 16GB
- [x] OS — Windows + Anaconda (WSL2 안 씀)
- [x] Python 버전 — 3.11 (conda env)
- [x] 프로젝트 폴더명 — cdss_vasopressor
- [x] GitHub remote — recognisemeimthebest/CDSS_VASOPRESSOR
- [ ] **사용자 승인 대기**: 폴더/env/git 셋업 진행
- [ ] MIMIC DB 접속 정보 추출 (DBeaver Edit Connection)

## Phase 0.5: 환경 셋업 ✅ 완료
- [x] 폴더 + 스켈레톤 + 설정 파일
- [x] git init + GitHub push (커밋 55ea4fa, 755569d)
- [x] `.env` 작성 + 실제 스키마 반영
- [x] Conda base env (Python 3.11 + 데이터/ML 기본)
- [x] PyTorch CUDA 12.1 (pip 분리 설치)
- [x] Streamlit + plotly
- [x] 패키지 + GPU 검증 (RTX 4070 Ti SUPER 인식)
- [x] DB 연결 + 스키마 탐색 (mimiciv_hosp/icu/note/ecg)
- [x] OMP 충돌 해결 (`scripts/run.bat` 런처)
- [x] README 업데이트 + 커밋

## Phase 1: MIMIC-IV 데이터 탐색
- [ ] 대상 환자 코호트 정의 (예: ICU 입실 + heparin 처방)
- [ ] 사용할 임상 변수 목록 (LAB, VITAL, DX, demographics)
- [ ] 시계열 윈도우 정의 (시간 단위, 룩백 길이)
- [ ] 결측/이상치 처리 정책
- [ ] EDA 노트북 작성

## Phase 2: 베이스라인 모델
- [ ] 데이터셋 구성 (train/valid/test 분할)
- [ ] 베이스라인 모델 학습 (처음엔 단순 모델)
- [ ] 평가지표 정의 (의사 결정 일치도, MAE 등)
- [ ] 결과 분석

## Phase 3: 본 모델 / 추천 시스템
- [ ] 시계열 모델 또는 RL 정책 학습
- [ ] 근거 설명 기능 (feature importance / attention)
- [ ] 평가 (counterfactual 평가 포함 여부)

## Phase 4: 서비스 형태로 wrap
- [ ] 추천 API/CLI 인터페이스 설계
- [ ] 방사선 데이터 어댑터 자리 마련 (스키마만)
- [ ] 데모 시나리오

## Phase 0.7: mimiciv_derived 빌드
- [x] psql 위치 확인 (`C:\Program Files\PostgreSQL\16\bin\psql.exe`)
- [x] mit-lcp/mimic-code clone (`g:\miniproject2\mimic-code\`)
- [x] postgres-make-concepts.sql 구조 확인 (norepinephrine_equivalent_dose 포함됨)
- [x] 빌드 스크립트 (`scripts/build_derived.py`)
- [x] PG 튜닝 (parallel workers 2→4)
- [~] 빌드 실행 중 (sofa 단계, ~85%)
- [ ] 핵심 테이블 검증
- [ ] `.env`에 MIMIC_DERIVED_SCHEMA 추가
- [ ] `verify_env.py` 재실행
- [ ] 커밋

## Phase 0.8: HARNESS B 풀세트 ✅ 완료
- [x] `.claude/` 디렉토리 구조
- [x] 워크스페이스 이전 → cdss_vasopressor/.claude/workspace/
- [x] PROJECT_PLAN.md (기획서)
- [x] AGENT_REPORT_FORMAT.md (보고서 표준)
- [x] settings.local.json (PostToolUse 훅 등록)
- [x] PostToolUse 훅 (자동 코드체크 + 변경 로그)
- [x] 스킬 매뉴얼 INDEX + 10 챕터 (mimic-sql, sepsis3, features, supervised, rl, streamlit, eval, py-quality, security, git)
- [x] 에이전트 8개 (cohort/features/supervised/rl/eval/streamlit + code-auditor/plan-auditor)
- [x] gitignore 갱신 (.claude/hooks/shared/change-log.md 제외)
- [ ] 커밋 + push

## 다음 할 일 (Phase 1) — cohort-agent 호출
1. EDA 노트북 (`notebooks/01_cohort_eda.ipynb`)
2. 코호트 정의 (`src/cohort.py`) + consort flow
이후 features-agent → supervised → rl → eval → streamlit, 각 단계 후 code-auditor + plan-auditor.
