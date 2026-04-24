"""MIMIC-IV PostgreSQL connection helpers.

Reads credentials from environment variables (loaded from .env via python-dotenv).
Use get_engine() for SQLAlchemy or get_conn() for raw psycopg2.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_ENV_LOADED = False


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    _ENV_LOADED = True


def _conn_kwargs() -> dict:
    _load_env()
    required = ["MIMIC_DB_HOST", "MIMIC_DB_PORT", "MIMIC_DB_NAME", "MIMIC_DB_USER"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Copy .env.example to .env and fill in.")
    return dict(
        host=os.getenv("MIMIC_DB_HOST"),
        port=int(os.getenv("MIMIC_DB_PORT")),
        dbname=os.getenv("MIMIC_DB_NAME"),
        user=os.getenv("MIMIC_DB_USER"),
        password=os.getenv("MIMIC_DB_PASSWORD") or "",
    )


@contextmanager
def get_conn():
    kwargs = _conn_kwargs()
    conn = psycopg2.connect(**kwargs)
    try:
        yield conn
    finally:
        conn.close()


def get_engine() -> Engine:
    k = _conn_kwargs()
    url = f"postgresql+psycopg2://{k['user']}:{k['password']}@{k['host']}:{k['port']}/{k['dbname']}"
    return create_engine(url)


def ping() -> str:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT version();")
        return cur.fetchone()[0]


if __name__ == "__main__":
    print(ping())
