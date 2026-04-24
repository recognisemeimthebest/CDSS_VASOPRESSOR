"""Build mimiciv_derived schema by running mit-lcp/mimic-code concepts_postgres scripts.

Long-running (~30-60 min on first build).
Run via: scripts\\run.bat scripts\\build_derived.py
Logs to: scripts\\build_derived.log
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

PSQL = r"C:\Program Files\PostgreSQL\16\bin\psql.exe"
CONCEPTS_DIR = Path(r"G:\miniproject2\mimic-code\mimic-iv\concepts_postgres")
LOG_PATH = PROJECT_ROOT / "scripts" / "build_derived.log"

DB = dict(
    host=os.environ["MIMIC_DB_HOST"],
    port=os.environ["MIMIC_DB_PORT"],
    user=os.environ["MIMIC_DB_USER"],
    dbname=os.environ["MIMIC_DB_NAME"],
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)


def psql(args: list[str], cwd: Path | None = None, log_handle=None) -> int:
    env = os.environ.copy()
    env["PGPASSWORD"] = DB["password"]
    cmd = [PSQL, "-h", DB["host"], "-p", str(DB["port"]),
           "-U", DB["user"], "-d", DB["dbname"], "-v", "ON_ERROR_STOP=1", *args]
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log_handle,
                          stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> int:
    if not Path(PSQL).exists():
        print(f"psql not found at {PSQL}", file=sys.stderr)
        return 2
    if not CONCEPTS_DIR.exists():
        print(f"mimic-code not cloned at {CONCEPTS_DIR}", file=sys.stderr)
        return 2

    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log:
        log.write(f"Building mimiciv_derived against {DB['dbname']}@{DB['host']}:{DB['port']}\n")
        log.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log.flush()

        # 1. Create schemas (mimiciv_ed as empty placeholder; some scripts reference it in search_path)
        rc = psql(["-c",
                   "CREATE SCHEMA IF NOT EXISTS mimiciv_derived; "
                   "CREATE SCHEMA IF NOT EXISTS mimiciv_ed;"], log_handle=log)
        if rc != 0:
            log.write(f"\nSchema create failed (rc={rc})\n")
            return rc
        log.write("Schemas ready.\n"); log.flush()

        # 2. Helper functions (BigQuery -> PG shims)
        rc = psql(["-f", str(CONCEPTS_DIR / "postgres-functions.sql")], log_handle=log)
        log.write(f"\npostgres-functions.sql rc={rc} at {time.strftime('%H:%M:%S')}\n"); log.flush()

        # 3. Master concepts build (long)
        log.write(f"\nStarting postgres-make-concepts.sql at {time.strftime('%H:%M:%S')}\n"); log.flush()
        rc = psql(["-f", "postgres-make-concepts.sql"], cwd=CONCEPTS_DIR, log_handle=log)

        log.write(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')} (rc={rc})\n")
        return rc


if __name__ == "__main__":
    sys.exit(main())
