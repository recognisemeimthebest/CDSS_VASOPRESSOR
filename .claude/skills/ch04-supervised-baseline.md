# ch04 — Supervised 베이스라인

## 목적

RL 본 모델 전, **(상태) → (다음 4h NE-equiv 용량)** 매핑을 단순 ML로 학습해 베이스라인 확보.

## 두 가지 prediction 형태

| 형태 | 출력 | 손실 | 평가 |
|---|---|---|---|
| **회귀** | NE-equiv mcg/kg/min (연속) | MSE / Huber | MAE, R², 잔차 분포 |
| **분류** | 5-bin (0/1/2/3/4) | CrossEntropy | accuracy, macro-F1, 혼동행렬, top-2 acc |

## 분할

- **시간 분할 우선**: 환자 입실년 기준 train < year < test (data leakage 방지)
- 시간 분할이 어려우면 stay_id 기준 80/10/10 (환자 단위로 누수 방지 — subject_id 기준)
- 절대 random row split 금지 (같은 stay의 rows가 섞임 = leakage)

## 모델 후보 (작은 → 큰 순)

1. **LogisticRegression / Ridge** — sanity check
2. **LightGBM / XGBoost** — 강력한 베이스라인, feature importance 무료
3. **MLP** — torch nn.Module, BatchNorm + Dropout
4. **시계열 트랜스포머** — 윈도우 입력, attention pooling (확장 옵션)

## 코드 구조

```
src/models/
├── supervised.py           # 인터페이스: fit/predict/predict_proba
├── _gbm.py                 # LightGBM 구현
├── _mlp.py                 # MLP (torch)
└── _transformer.py         # 시계열 트랜스포머
```

각 모델은 동일 인터페이스:
```python
class Model:
    def fit(self, X_train, y_train, X_val, y_val) -> dict: ...   # returns metrics
    def predict(self, X) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Model": ...
```

## 흔한 함정

- **클래스 불균형**: 5-bin에서 0(=무처치)이 절반 이상. macro-F1 또는 class-weight 사용.
- **지속/시작 구분**: "용량 유지"와 "신규 시작" 의 의사결정 맥락이 다름. 별도 컬럼/모델 고려.
- **Causal time direction**: t bin에서 t+1 액션 예측. 절대 미래 정보(post-action vitals) 사용 금지.
- **Calibration**: 분류 모델이면 Platt scaling 또는 isotonic 계산해서 reliability diagram 그릴 것.
- **인구학적 fairness**: 성별/인종 subgroup별 metric 분리 보고 (ch07).
