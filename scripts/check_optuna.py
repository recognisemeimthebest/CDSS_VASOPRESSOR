"""Quick: count completed Optuna trials per study DB."""
import sqlite3
from pathlib import Path

for db in Path("artifacts/optuna").glob("*.db"):
    try:
        c = sqlite3.connect(str(db))
        total = c.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        done = c.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'").fetchone()[0]
        running = c.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'").fetchone()[0]
        pruned = c.execute("SELECT COUNT(*) FROM trials WHERE state='PRUNED'").fetchone()[0]
        # best value
        best = c.execute("SELECT MAX(value) FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE state='COMPLETE')").fetchone()[0]
        print(f"{db.name}: total={total}  complete={done}  running={running}  pruned={pruned}  best={best}")
    except Exception as e:
        print(f"{db.name}: error {e}")
