from __future__ import annotations

import sqlite3
from pathlib import Path

def get_connection(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    tables = [
        "CREATE TABLE IF NOT EXISTS miners (hotkey TEXT PRIMARY KEY)",
        """CREATE TABLE IF NOT EXISTS backward_events (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            ts TEXT NOT NULL, since_reset INTEGER NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS loss_events (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            ts TEXT NOT NULL, loss REAL NOT NULL, activation_id TEXT)""",
        """CREATE TABLE IF NOT EXISTS state_events (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            to_state TEXT NOT NULL, ts TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS exceptions (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            ts TEXT NOT NULL, ex_type TEXT, level TEXT, http_endpoint TEXT,
            http_code INTEGER, message TEXT, message_normalized TEXT,
            line_number INTEGER, source_file TEXT)""",
        """CREATE TABLE IF NOT EXISTS optimization_events (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            ts TEXT NOT NULL, step_number INTEGER, backwards_count INTEGER)""",
        """CREATE TABLE IF NOT EXISTS resource_events (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            ts TEXT NOT NULL, event_type TEXT NOT NULL, value_gb REAL, value_text TEXT)""",
        """CREATE TABLE IF NOT EXISTS registration_events (
            id INTEGER PRIMARY KEY, miner_hotkey TEXT NOT NULL, layer INTEGER,
            ts TEXT NOT NULL, training_epoch INTEGER, status TEXT)""",
    ]

    indices = [
        "CREATE INDEX IF NOT EXISTS idx_bw_miner_ts ON backward_events(miner_hotkey, ts)",
        "CREATE INDEX IF NOT EXISTS idx_loss_miner_ts ON loss_events(miner_hotkey, ts)",
        "CREATE INDEX IF NOT EXISTS idx_state_miner_ts ON state_events(miner_hotkey, ts)",
        "CREATE INDEX IF NOT EXISTS idx_exc_miner_ts ON exceptions(miner_hotkey, ts)",
        "CREATE INDEX IF NOT EXISTS idx_exc_normalized ON exceptions(message_normalized)",
        "CREATE INDEX IF NOT EXISTS idx_opt_miner_ts ON optimization_events(miner_hotkey, ts)",
        "CREATE INDEX IF NOT EXISTS idx_res_miner_ts ON resource_events(miner_hotkey, ts)",
        "CREATE INDEX IF NOT EXISTS idx_reg_miner_ts ON registration_events(miner_hotkey, ts)",
    ]

    for sql in tables + indices:
        cur.execute(sql)

    conn.commit()


def migrate_normalize_messages(conn: sqlite3.Connection) -> None:
    from .ingest import _normalize_exception_message

    cur = conn.cursor()

    cur.execute("PRAGMA table_info(exceptions)")
    columns = [row[1] for row in cur.fetchall()]

    if "message_normalized" not in columns:
        cur.execute("ALTER TABLE exceptions ADD COLUMN message_normalized TEXT")

        cur.execute("SELECT id, message FROM exceptions")
        rows = cur.fetchall()

        for row_id, message in rows:
            if message:
                normalized = _normalize_exception_message(message)
                cur.execute(
                    "UPDATE exceptions SET message_normalized = ? WHERE id = ?",
                    (normalized, row_id)
                )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_exc_normalized ON exceptions(message_normalized)")

        conn.commit()

    if "line_number" not in columns:
        cur.execute("ALTER TABLE exceptions ADD COLUMN line_number INTEGER")
        conn.commit()

    if "source_file" not in columns:
        cur.execute("ALTER TABLE exceptions ADD COLUMN source_file TEXT")
        conn.commit()