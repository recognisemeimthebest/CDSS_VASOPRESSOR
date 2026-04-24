# ch10 — Git 워크플로

## 브랜치 / 커밋

- `main` 직접 푸시 OK (개인 프로젝트). 대규모 변경은 feature 브랜치 권장.
- 커밋 단위는 작게 (논리적 1개 변화 = 1 커밋).
- 커밋 메시지: 영문 제목 (50자 이내), 본문은 "왜" 위주 (한국어 OK).
- Co-Authored-By 라인은 Claude 가 추가하는 작업 시 포함.

## 커밋 전 자동 체크 (수동으로 verify)

```bash
git status                # 의도치 않은 파일?
git diff --staged         # 변경 내용 확인
grep -r "tlsghktk6\|password\s*=\s*\"" --include="*.py" src/ app/  # 비번 하드코딩?
```

## 푸시 전

- `.env` 가 staged 안 됐는지 확인 (`git diff --staged --name-only | grep -E "\.env$"`)
- 대용량 파일 staged 안 됐는지 (`git diff --staged --stat` 줄 보기)

## .gitignore 확인 패턴

다음은 절대 커밋되면 안 됨:
- `.env`, `*.env` (제외: `.env.example`)
- `data/`, `*.csv`, `*.parquet`, `*.pkl`, `*.h5` 등 데이터 파일
- `artifacts/`, `checkpoints/`, `*.pt`, `*.pth`, `mlruns/`, `wandb/`
- `__pycache__/`, `.ipynb_checkpoints/`

## PR (사용 시)

- 제목은 70자 이내
- 본문에 Summary + Test plan
- 머지 전 plan-auditor 와 code-auditor 둘 다 통과

## 위험 명령 (사용자 확인 후)

- `git push --force` (특히 main)
- `git reset --hard`
- `git clean -fd`
- 위는 Claude 가 임의로 실행 금지. 사용자가 명시적으로 요청한 경우만.

## GitHub remote

- `origin` = https://github.com/recognisemeimthebest/CDSS_VASOPRESSOR
- gh CLI 로 인증됨 (계정 recognisemeimthebest)
