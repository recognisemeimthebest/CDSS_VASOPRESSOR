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
conda env create -f environment.yml
conda activate cdss_vasopressor
cp .env.example .env   # DB 접속정보 채우기
```

## 참고 선행연구

- Komorowski et al., *AI Clinician* (Nat Med 2018) — sepsis IV/vasopressor RL
- Tseng et al. — radiotherapy dose adaptation (RL)
