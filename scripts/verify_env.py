"""One-shot verification of the conda env: package versions, GPU, DB connection, schemas."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_packages() -> None:
    import torch, streamlit, plotly, psycopg2, pandas, sklearn, sqlalchemy
    print("=== Packages ===")
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"  available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  device    : {props.name}  ({props.total_memory // (1024**3)} GB)")
    print(f"streamlit   : {streamlit.__version__}")
    print(f"plotly      : {plotly.__version__}")
    print(f"psycopg2    : {psycopg2.__version__.split()[0]}")
    print(f"pandas      : {pandas.__version__}")
    print(f"sklearn     : {sklearn.__version__}")
    print(f"sqlalchemy  : {sqlalchemy.__version__}")


def check_db() -> None:
    from src.db import get_conn
    print("\n=== Database ===")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT version();")
        print(f"server      : {cur.fetchone()[0][:80]}")
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast') "
            "ORDER BY schema_name;"
        )
        schemas = [r[0] for r in cur.fetchall()]
        print(f"schemas     : {schemas}")
        for s in schemas:
            cur.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s;",
                (s,),
            )
            print(f"  {s:<25s} {cur.fetchone()[0]} tables")


if __name__ == "__main__":
    check_packages()
    try:
        check_db()
    except Exception as e:
        print(f"\nDB connection failed: {e}")
