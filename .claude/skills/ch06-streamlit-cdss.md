# ch06 — Streamlit CDSS 데모

## 페이지 구조

```
app/
├── main.py                # 진입점 (router)
├── pages/
│   ├── 1_patient_overview.py    # 환자 시계열 요약
│   ├── 2_recommendation.py      # 모델 추천 + 근거
│   └── 3_cohort_explorer.py     # (옵션) 코호트 탐색
└── components/
    ├── timeseries_plot.py
    └── recommendation_card.py
```

## 핵심 UX

1. 환자 ID(stay_id) 선택 → 코호트 내 stay 만 (random patient demo 버튼 포함)
2. **시계열 시각화**: vitals, SOFA, 실제 NE-equiv 용량 plotly chart
3. **추천 카드**: 모델이 권하는 다음 4h bin NE-equiv (mcg/kg/min + 5-bin 라벨)
4. **근거**: feature importance (LightGBM SHAP) 또는 attention weights
5. **의사 결정 비교**: 실제 의사가 선택한 용량 vs 모델 권고

## PHI 노출 금지

- subject_id / hadm_id 화면에 절대 표시 금지 (UI에선 가짜 별명 또는 stay_id 만)
- 환자명, 생년월일, 성별 단독 표시 금지 → 집계로만
- 차트 export 시 metadata 제거

## 캐싱

```python
@st.cache_data(ttl=3600)
def load_patient_timeseries(stay_id: int) -> pd.DataFrame: ...

@st.cache_resource
def load_model() -> Model: ...
```

## 실행

`scripts\run.bat -m streamlit run app\main.py` (Windows OMP 워크어라운드 필수, 직접 streamlit 실행 금지)

## 학생 또는 교수 데모 시 디스클레이머

페이지 footer 에 항상:
> 본 데모는 학습 프로토타입입니다. 실제 환자 진료에 사용 금지.
