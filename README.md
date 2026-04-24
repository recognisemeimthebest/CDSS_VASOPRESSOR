# CDSS — Vasopressor Recommendation (MIMIC-IV practice)

방사선치료 데이터 도착 전, 동일한 의사결정 구조("환자 임상 시계열 → 치료 용량 결정")를
**MIMIC-IV의 vasopressor 추천 태스크**로 연습하는 프로젝트.

## 목표

패혈성 쇼크 환자의 시계열 임상 변수(활력징후, 검사수치, 인공호흡 설정 등)를 보고
**vasopressor 종류와 용량을 추천**하는 임상 의사결정 지원 시스템(CDSS)의 미니 프로토타입.
지도학습과 강화학습 두 방식 모두 비교한다.

## 최종 목적

향후 도착 예정인 암환자 방사선치료 데이터(NIA/AI Hub, 5,019명)에서
`totaldose`, `radiationcnt`, `radiationperdose`, `treatmethod`, `treatech` 등을
추천하는 본 서비스의 사전 연습. 입력 어댑터만 갈아끼울 수 있게 설계한다.

## 폴더 구조

```
cdss_vasopressor/
├── data/          # gitignored, 쿼리 결과 캐시
├── notebooks/     # EDA
├── src/
│   ├── db.py      # PostgreSQL 연결
│   ├── cohort.py  # 코호트 정의
│   ├── features.py
│   ├── models/    # supervised, RL
│   └── eval/
├── app/
│   └── main.py    # Streamlit
├── tests/
├── .env.example
├── .gitignore
├── environment.yml
└── README.md
```

## 환경

- OS: Windows
- Python 3.11 (Anaconda env: `cdss_vasopressor`)
- GPU: CUDA (RTX 4070 Ti SUPER)
- DB: MIMIC-IV (PostgreSQL)
- Web: Streamlit

## 셋업

```bash
# 기본 패키지 (PyTorch는 OMP 충돌 회피 위해 분리 설치)
conda create --prefix G:/anaconda_envs/cdss_vasopressor -c conda-forge -y \
    python=3.11 psycopg2 sqlalchemy pandas numpy scipy scikit-learn \
    matplotlib seaborn jupyterlab tqdm python-dotenv pip

# PyTorch CUDA 12.1 + Streamlit (pip)
conda activate G:/anaconda_envs/cdss_vasopressor
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install streamlit plotly

cp .env.example .env   # DB 접속정보 채우기
```

## 실행 (Windows)

`python.exe`를 직접 호출하면 conda MKL과 PyTorch OpenMP 충돌로 DLL 에러가 남.
런처 스크립트로 실행:

```bat
scripts\run.bat scripts\verify_env.py        :: 환경/DB 검증
scripts\run.bat -m streamlit run app\main.py :: Streamlit
scripts\run.bat -m jupyter lab               :: JupyterLab
```

PowerShell이면 `.\scripts\run.ps1 ...` 사용.

## 실제 MIMIC-IV 스키마 (이 환경 기준)

| 스키마 | 테이블 수 | 비고 |
|---|---|---|
| `mimiciv_hosp` | 22 | admissions, patients, labevents, prescriptions 등 |
| `mimiciv_icu` | 9 | icustays, chartevents, inputevents 등 |
| `mimiciv_note` | 4 | discharge, radiology 노트 |
| `mimiciv_ecg` | 3 | ECG 메타/파형 |

`mimiciv_derived`는 로드되어 있지 않음 → SOFA/sepsis-3 등은 직접 계산하거나
[mit-lcp/mimic-code](https://github.com/MIT-LCP/mimic-code) SQL을 적용해야 함.

## 참고 선행연구

- Komorowski et al., *AI Clinician* (Nat Med 2018) — sepsis IV/vasopressor RL
- Tseng et al. — radiotherapy dose adaptation (RL)
