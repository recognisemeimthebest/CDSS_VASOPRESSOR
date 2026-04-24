# ch09 — 보안 & 프라이버시 (PHI 보호)

## 절대 금지 (위반 시 plan-auditor 자동 reject)

1. **자격증명 하드코딩** — DB 비밀번호, API 키 등 코드에 직접 적기 금지. `.env` + `python-dotenv`.
2. **`.env` 커밋** — `.gitignore` 의 `*.env` 패턴이 차단 중. 추가 파일 만들 때 패턴 확인.
3. **PHI 외부 송신** — MIMIC 데이터의 어떤 부분도 외부 API/서비스/로그 수집기에 보내지 말 것.
4. **환자 식별자 print/log** — `subject_id`, `hadm_id`, `stay_id` 단독 print/log 금지. 디버깅 시 hash 또는 집계만.
5. **f-string SQL** — SQL injection 방지. parameterized query 만 사용.

## .env 점검

- `.env.example` 만 커밋. 실제 `.env` 는 gitignored 확인 (`git check-ignore -v .env`).
- DB 비밀번호 변경 시 `.env` 만 수정, 코드 변경 없음.

## 로깅

```python
# 나쁨 ❌
logger.info(f"Loaded patient {subject_id} with {n} rows")

# 좋음 ✅
logger.info("Loaded patient (hash=%s) with %d rows", hash(subject_id) % 10000, n)
```

## Streamlit UI

- ch06 참조. PHI 화면 노출 금지.
- chart 의 hover tooltip 에 환자식별자 들어가지 않게.
- 다운로드 버튼에서 raw 데이터 export 금지 (집계만).

## 모델 저장

- 모델 가중치에 PHI 가 직접 포함되진 않지만, training data hash + cohort definition 을 함께 기록 (재현/감사용).
- 외부 호스팅 (HF Hub 등) 업로드 금지. 본 단계에서는 로컬 저장만.

## DB 접근 분리

- 가능하면 `read_only` PG role 만들어서 .env 의 user 를 그걸로 사용. (선택, 학습 단계 OK 미시행)

## 인시던트 대응

- 코드에 비밀번호가 들어간 채 commit/push 됐으면:
  1. `git filter-repo` 또는 BFG 로 history 에서 제거
  2. 즉시 비밀번호 변경 (PG `ALTER USER ... PASSWORD ...`)
  3. plan-auditor 호출해서 영향 범위 평가
