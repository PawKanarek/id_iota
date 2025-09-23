from __future__ import annotations

import sqlite3
from pathlib import Path

# Connection / minimal helper 
def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """
    Create a SQLite connection with sensible defaults.
    """
    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


# Schema
def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Create tables if not exist. (Matches what the ingester writes + what queries read.)
    NOTE: On-disk incremental ingestion has been removed, so there is no `files` table anymore.
    """
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS miners (
        hotkey TEXT PRIMARY KEY
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS backward_events (
        id INTEGER PRIMARY KEY,
        miner_hotkey TEXT NOT NULL,
        layer INTEGER,
        ts TEXT NOT NULL,                 -- ISO8601 UTC
        since_reset INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS loss_events (
        id INTEGER PRIMARY KEY,
        miner_hotkey TEXT NOT NULL,
        layer INTEGER,
        ts TEXT NOT NULL,                 -- ISO8601 UTC
        loss REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS state_events (
        id INTEGER PRIMARY KEY,
        miner_hotkey TEXT NOT NULL,
        layer INTEGER,
        to_state TEXT NOT NULL,
        ts TEXT NOT NULL                  -- ISO8601 UTC
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS exceptions (
        id INTEGER PRIMARY KEY,
        miner_hotkey TEXT NOT NULL,
        layer INTEGER,
        ts TEXT NOT NULL,                 -- ISO8601 UTC
        ex_type TEXT,
        level TEXT,
        http_endpoint TEXT,
        http_code INTEGER,
        message TEXT
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_bw_miner_ts ON backward_events(miner_hotkey, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_loss_miner_ts ON loss_events(miner_hotkey, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_state_miner_ts ON state_events(miner_hotkey, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_exc_miner_ts ON exceptions(miner_hotkey, ts)")

    conn.commit()
