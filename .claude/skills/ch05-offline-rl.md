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

오프라인 RL 모델은 **반드시** off-policy evaluation을 통과해야 함. 단일 OPE 점추정만 보고 정책 결정 금지 (Gottesman 2018/19, Tang & Wiens 2021):

- **WIS** (Weighted Importance Sampling) — variance 큼, ESS 같이 보고
- **FQE** (Fitted Q Evaluation) — 별도 critic 학습으로 정책 가치 추정. WIS 보완.
- **WIS + FQE 둘 다 필수** (한 가지만 보고 금지)
- **Bootstrap CI 1000회** + **ESS (Effective Sample Size)** 명시
- **Null-policy baseline 비교**: zero-action / random / constant-dose 정책의 OPE 값과 같이 표 출력 (Jeter 2019: 일부 RL 정책이 random보다 낫지 않음에도 좋아 보일 수 있음)
- **Doubly Robust** (선택, variance 추가 감소)

평가 결과 + 의사 정책과 비교 시각화는 ch07 참조.

## Dead-end head (선택, 안전성 강화)

Fatemi et al. 2021 (NeurIPS *Medical Dead-ends and Learning to Identify High-Risk States and Treatments*):
- 단일 최적 액션 추천보다 "**해서는 안 되는 액션**" 식별이 임상 안전성에서 더 가치 있음
- 우리 출력 head 두 개로 확장 고려:
  - (1) 권장 NE-equiv dose (기존)
  - (2) Contraindication flag — "이 상태에서 dose ↑ 하면 사망 확률 급증" 같은 경고
- Streamlit UI에 "추천 카드" + "경고 카드" 두 영역 (ch06)

## 정직성 가드 — 임상 결과 주장 금지

- **Mortality 감소 주장 절대 금지** (prospective RCT 없음). OPE는 예측일 뿐.
- AI Clinician 후속 비판 다수 (Festor 2022 BMJ Health Care Inform; Wu 2023 multi-cohort 재현 실패; Roggeveen 2021 transatlantic transfer 실패).
- 보고서/논문에 "본 연구는 prospective validation 미수행, 임상 사용 금지" 명시 (PROJECT_PLAN §8 + ch09).

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
