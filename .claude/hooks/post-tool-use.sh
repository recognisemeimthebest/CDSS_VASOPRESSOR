#!/usr/bin/env bash
# PostToolUse hook for cdss_vasopressor.
# Fires after Write/Edit. Logs the change and runs lightweight code checks.
# Output goes back to Claude as a system reminder via stdout JSON.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/.claude/hooks/shared"
mkdir -p "$LOG_DIR"
CHANGE_LOG="${LOG_DIR}/change-log.md"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

# Read tool-use payload from stdin (Claude Code passes JSON)
PAYLOAD="$(cat || true)"
TOOL=$(printf '%s' "$PAYLOAD" | python -c "import sys,json; d=json.loads(sys.stdin.read() or '{}'); print(d.get('tool_name',''))" 2>/dev/null || echo "")
FILE=$(printf '%s' "$PAYLOAD" | python -c "import sys,json; d=json.loads(sys.stdin.read() or '{}'); ti=d.get('tool_input',{}); print(ti.get('file_path',''))" 2>/dev/null || echo "")

# Append to change log
echo "- ${TS} | ${TOOL} | ${FILE}" >> "$CHANGE_LOG"

# Only check Python files in src/ app/ tests/ scripts/
case "$FILE" in
    *.py) ;;
    *) exit 0 ;;
esac
case "$FILE" in
    *src/*|*app/*|*tests/*|*scripts/*) ;;
    *) exit 0 ;;
esac

[ -f "$FILE" ] || exit 0

# Lightweight checks. Each issue contributes to the count; >0 sends a reminder.
issues=()

# 1. Hardcoded credentials / secrets
if grep -E -q '(password|passwd|secret|api[_-]?key|token)\s*=\s*["'"'"'][^"'"'"']{4,}' "$FILE" 2>/dev/null \
   && ! grep -q "os.environ\|os.getenv\|load_dotenv\|getpass" "$FILE" 2>/dev/null; then
    issues+=("- [security] 자격증명/토큰이 하드코딩된 것으로 보임 → .env + python-dotenv 사용")
fi

# 2. PHI risk: print of patient identifiers
if grep -E -q 'print\([^)]*(subject_id|hadm_id|stay_id)' "$FILE" 2>/dev/null; then
    issues+=("- [privacy] 환자 식별자 print 감지 → 로그/UI에 노출 금지 (집계만)")
fi

# 3. SQL injection: f-string formatted SQL
if grep -E -q '(execute|read_sql|read_sql_query)\s*\(\s*f["'"'"']' "$FILE" 2>/dev/null; then
    issues+=("- [security] f-string SQL 발견 → parameterized query (%s 또는 SQLAlchemy text+bindparams) 사용")
fi

# 4. bare except
if grep -E -q '^\s*except\s*:' "$FILE" 2>/dev/null; then
    issues+=("- [quality] bare except → 구체 예외 클래스 잡기")
fi

# 5. notebook checkpoints / large data accidentally added
if [[ "$FILE" == *".ipynb_checkpoints"* ]] || [[ "$FILE" == *"/data/"* ]]; then
    issues+=("- [hygiene] data/ 또는 ipynb_checkpoints 경로 → .gitignore 확인")
fi

if [ ${#issues[@]} -eq 0 ]; then
    exit 0
fi

# Emit JSON to surface a system reminder
count=${#issues[@]}
if [ "$count" -le 2 ]; then
    severity="경미 (즉시 수정 권장)"
    advice="지금 바로 위 이슈들을 수정해."
else
    severity="다수 (정밀 검토 필요)"
    advice="code-auditor 에이전트를 호출해서 정밀 검토를 받아."
fi

body=$(printf '[post-tool-use 검사] %s 에서 %d개 이슈 — %s\n%s\n%s\n' \
    "$FILE" "$count" "$severity" "$(printf '%s\n' "${issues[@]}")" "$advice")

# Hook protocol: print JSON with additionalContext to stdout
python - <<PYEOF
import json
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": $(python -c "import json,sys; print(json.dumps('''$body'''))")}}))
PYEOF
