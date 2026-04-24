# ch08 — Python 코드 품질 (이 프로젝트 한정 규칙)

## 필수 규칙

1. **타입 힌트** — public 함수는 모두 type hints. internal helper는 권장.
2. **`from __future__ import annotations`** — 모든 새 파일 상단.
3. **Path 객체** — 파일 경로는 `pathlib.Path`. 문자열 경로 금지.
4. **f-string 로깅 금지** — `logging` 사용 시 `logger.info("x=%s", x)` (`logger.info(f"x={x}")` 금지).
5. **bare except 금지** — 구체 예외 클래스. `except Exception:` 도 가능한 좁힐 것.
6. **하드코딩된 상수** — 매직 넘버는 모듈 상단 상수 또는 config dict로.
7. **데이터프레임 mutation 명시** — `df.copy()` 또는 `df.assign(...)` 사용. inplace=True 가능한 회피.

## 의존성 관리

- 새 패키지 추가 시 `environment.yml` 업데이트 + 커밋.
- pip 추가는 `environment.yml` 의 `- pip:` 섹션에.
- 절대 system Python 사용 금지. `scripts\run.bat` 런처 거치기.

## 테스트

- `pytest` 사용. 테스트 파일은 `tests/test_*.py`.
- DB 연결이 필요한 테스트는 `@pytest.mark.db` 마커 + skip if DB unreachable.
- 회귀/추론 결과 deterministic 보장 위해 seed 고정 fixture.

## 노트북

- EDA 외 본 코드는 노트북 금지. 노트북에서 만든 함수는 `src/`로 이전.
- 노트북 commit 전 `Cell > Clear All Outputs` (대용량 output 회피).
- `.ipynb_checkpoints/` 는 .gitignore 처리됨 (확인).

## 임포트 순서

```python
# 표준 라이브러리
import os
from pathlib import Path

# 서드파티
import numpy as np
import pandas as pd
import torch

# 로컬
from src.db import get_engine
from src.cohort import load_cohort
```

## 함수 길이

- 50줄 넘으면 분리 고려. 100줄 넘으면 무조건 분리.
- 단, 시각화 함수는 예외 (matplotlib boilerplate 길어짐).

## 주석

- **WHY 만 쓰기**. WHAT 은 코드가 말함.
- 비자명한 임계값 (`THRESH = 8.4` 같은 magic number) 옆에 출처 코멘트 (예: `# AI Clinician bin edge`).
- Docstring 은 public 함수만, 한두 줄로 짧게.

## 금지

- `time.sleep` 폴링 — 비동기 또는 callback.
- `os.system`, `subprocess.call` (deprecated) — `subprocess.run` 사용.
- `pickle.load` 외부 데이터 — 신뢰된 경로만.
