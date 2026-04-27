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

## Phase 0.7: mimiciv_derived 빌드 ✅ 완료
- [x] psql 위치 확인
- [x] mit-lcp/mimic-code clone
- [x] postgres-make-concepts.sql 구조 확인
- [x] 빌드 스크립트 (`scripts/build_derived.py`)
- [x] PG 튜닝 (parallel workers 2→4)
- [x] 빌드 실행 (63 테이블 생성, 핵심 8개 모두 정상)
  - sepsis3: 41,296 rows / norepinephrine_equivalent_dose: 783,613 / sofa: 8.2M / vitalsign: 13.5M
- [x] `.env` + `.env.example` 에 MIMIC_DERIVED_SCHEMA 추가
- [x] `verify_env.py` 재실행 — 5 schemas 인식
- [ ] 커밋 (다음 단계)

## Phase 0.9: 전역 훅 cwd-aware 수정 ✅
- [x] g:/Mimicmultimodal/.claude/hooks/analyze_prompt.py 에 `_resolve_workspace_root()` 추가
- [x] stop_checklist.py 에도 동일 로직 추가
- [x] 동작 검증 — `g:/miniproject2/cdss_vasopressor/.claude/workspace/` 가리키도록 자동 해석
- [x] 옛 워크스페이스 (`g:/miniproject2/.claude/`) 제거 + data_spec.txt 이전
- [ ] (사용자) Mimicmultimodal repo에 훅 변경 커밋

## Phase 0.95: 선행연구 조사 + 스킬 보강 ✅
- [x] 리서치 에이전트로 16편+ 논문 정리
- [x] 핵심 4편 DOI 웹 검증 (Festor 2022 정정)
- [x] ch05-offline-rl.md 보강 (dual OPE, null baseline, dead-end head, mortality 가드)
- [x] ch07-evaluation.md 보강 (WIS+FQE+ESS+CI 표준, 정직성 가드)
- [x] ch03-features-engineering.md 보강 (encoder ablation 섹션)
- [x] PROJECT_PLAN §9 검증된 인용으로 갱신
- [x] 메모리 `research_grounded_decisions.md` 추가
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

## Phase 4: Offline RL 학습 ✅ 완료
- [x] BC (mlp, seed=42) — WIS 6.642, ESS 100%, OPE 통과
- [x] dBCQ (mlp, seed=42) — FQE 0.145, ESS 4.5% (낮음, 경고)
- [x] CQL (mlp, seed=42) — FQE 0.127, ESS 3.9% (낮음, 경고)
- [x] OPE (WIS + FQE + Bootstrap CI + Null baselines) — 모든 알고리즘 완료
- [x] artifacts/runs/ 에 결과물 저장 (config.json, metrics.json, predictions.parquet, model.pt, ope_report.md)

## 다음 할 일 (Phase 5 → 6)
1. **Phase 5**: ~~eval-agent~~ 종합 리포트 → 완료 (텍스트 리포트로 대체)
2. **Phase 6**: Streamlit 웹앱 구현 ✅
   - [x] app/utils.py — 공통 데이터 로드 (캐시, 액션 레이블)
   - [x] app/main.py — 메인 대시보드 (환자 선택 + GBM vs dBCQ 추천 + vitals 타임라인)
   - [x] app/pages/01_ope_report.py — OPE 리포트 페이지
   - [x] 데이터 로딩 검증 (1961 test stays, 18 bins/stay)
   - [x] RL 버그 수정 (policy_proba one-hot→softmax, _q_monitor Q→match rate)
   - [x] dBCQ/CQL 재실행 (ESS 4.5%→15%, 17.6%)
   - [x] Streamlit 앱 실행 및 UI 확인 (HTTP 200, 데이터 로딩 정상)
## Phase 3.6: GBM Cost-Sensitive 실험 ✅ 완료 (음성 결과)
- [x] src/models/_gbm_cost.py — cost-sensitive custom objective 구현 (LightGBM 4.x API)
- [x] scripts/run_cost_sensitive_gbm.py — 실험 스크립트
- [x] 실행 및 기존 GBM 비교 — 결과: class 2 F1 0.356→0.299 (오히려 악화)
  - Macro-F1: 0.529→0.387 (-0.143), Top-2 Acc: 0.875→0.890 (+0.015)
  - 실패 원인: Optuna 파라미터 미적응 + class 4 gradient 붕괴
- [x] context-note.md 결과 기록 (음성 결과 포함)
- [x] (선택) Soft Labels + MLP 재학습 — 음성 결과 (class 2 F1 0.358→0.347, 더 낮음)
  - sigma sweep {0.5, 0.8, 1.0, 1.5}: best sigma=0.5, 큰 sigma는 조기 붕괴
  - 원인: Optuna focal-loss params와 KL-loss 비호환

- [x] GBM Threshold 조정 (per-class scale sweep) — **양성 결과**
  - Strategy A (cls2×1.5): class 2 F1 0.356→0.395 (+0.039), macro -0.010
  - Strategy C (cls3×0.8): class 2 F1 0.356→0.371 (+0.015), macro -0.003
  - 재학습 없음, 확률 스케일링만으로 달성

## Phase 3.6 최종 판정 ✅
- cost-sensitive GBM, soft labels MLP: 음성 결과
- **GBM threshold 조정 (Strategy C): 양성 결과** — macro-F1=0.527, class 2 F1=0.371
- 아티팩트: artifacts/runs/2026-04-27_gbm_threshold_cls_42/

- [x] presentation/ 폴더 생성 + 발표자료용 그림/데이터/요약 생성
  - figures/ : 7개 PNG (코호트, 모델비교, F1 히트맵, OPE, threshold, RL action dist, confusion matrix)
  - data/ : summary_supervised.csv, summary_rl.csv
  - SUMMARY.md : 전체 결과 마크다운 요약

## 다음 할 일
- [ ] 전체 코드 + 결과물 GitHub push

3. **커밋**: 전체 코드 + 결과물 GitHub push
