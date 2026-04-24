# ch05 — Offline RL (Vasopressor 정책 학습)

## 왜 offline RL인가

병원 데이터는 fixed historical batch. 환경과 상호작용 불가. **online RL 알고리즘 (DQN, PPO 등) 그대로 쓰면 distribution shift 로 망가짐**. Offline-safe 알고리즘만 사용.

## 알고리즘 후보

| 알고리즘 | 특징 | 우리 프로젝트 적합도 |
|---|---|---|
| **Behavior Cloning (BC)** | 의사 정책을 supervised로 모방. RL 베이스라인 | ⭐⭐⭐ 베이스라인 |
| **dBCQ** (discrete BCQ) | Q-learning + behavior policy support 제약 | ⭐⭐⭐⭐⭐ AI Clinician 후속 |
| **CQL** (Conservative Q-Learning) | OOD 액션의 Q값 페널티 | ⭐⭐⭐⭐ 안정적 |
| **AI Clinician 원형 (FQI)** | tabular fitted Q | ⭐⭐⭐ 재현용 |
| Online RL (DQN, PPO) | ❌ 사용 금지 | offline 환경 부적합 |

## 표준 셋업 (AI Clinician 따라)

- 상태: ch03 의 4h bin 피처 벡터
- 액션: 5-bin NE-equiv (이산)
- 보상:
  - intermediate: SOFA 감소량 × c1 + 락테이트 감소량 × c2
  - terminal: alive +15, dead -15 (90일 사망 기준)
- discount: γ = 0.99
- 에피소드 = 1 ICU stay

## 안전 평가 (필수)

오프라인 RL 모델은 **반드시** off-policy evaluation을 통과해야 함:

- **WIS** (Weighted Importance Sampling)
- **FQE** (Fitted Q Evaluation) — 별도 critic 학습으로 정책 가치 추정
- **Doubly Robust** (선택)

평가 결과 + 의사 정책과 비교 시각화는 ch07 참조.

## 학습 정책 vs 의사 정책 차이 시각화 (필수)

```
- 액션 분포 히스토그램 (의사 vs 학습 정책)
- 상태별 액션 추천 차이 (예: SOFA 별 권고 분포)
- "OOD" 비율 — 학습 정책이 의사가 한 번도 안 한 액션을 추천하는 빈도
```

OOD 비율이 너무 높으면 (>10%) BCQ 의 임계값 조정 또는 CQL 알파 키울 것.

## 흔한 함정

- **Reward hacking**: SOFA 감소를 보상으로 쓰면 모델이 "환자 살림"이 아닌 "SOFA 측정 회피"를 학습할 수 있음. terminal reward를 충분히 크게.
- **Behavior policy estimation**: BCQ/CQL 모두 behavior π_b 추정이 필요. supervised classifier 로 추정 (ch04).
- **Action space mismatch**: 학습 코호트의 액션 분포와 평가 코호트가 다르면 IS 가중치가 폭발. clip 처리 필수.
- **Reproducibility**: seed 고정 + GPU determinism 설정 + 결과 csv 저장.
