from __future__ import annotations
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import sqlite3


def _maybe_filter_miners(df: pd.DataFrame, miners: Optional[List[str]]) -> pd.DataFrame:
    if miners:
        return df[df["miner_hotkey"].isin(miners)].copy()
    return df


def list_miners(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT hotkey FROM miners ORDER BY hotkey")
    return [r[0] for r in cur.fetchall()]


def _to_naive_local(ts_series: pd.Series, tz_name: str) -> pd.Series:
    """
    Parse ISO8601 as UTC -> convert to local tz -> drop tzinfo (naive).
    Robust to malformed values: unparseable entries become NaT.
    """
    local_tz = ZoneInfo(tz_name)
    dt = pd.to_datetime(ts_series, utc=True, errors="coerce", format="ISO8601")
    return dt.dt.tz_convert(local_tz).dt.tz_localize(None)


# Backward events
def query_backward_events(conn: sqlite3.Connection, tz_name: str, miners: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT miner_hotkey, layer, ts AS ts_iso, since_reset
        FROM backward_events
        ORDER BY ts ASC
        """,
        conn,
    )
    if df.empty:
        return df

    df = _maybe_filter_miners(df, miners)
    if df.empty:
        return df

    df["ts_local"] = _to_naive_local(df["ts_iso"], tz_name)
    df = df[df["ts_local"].notna()]
    return df


# Losses
def query_losses(conn: sqlite3.Connection, tz_name: str, miners: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT miner_hotkey, layer, ts AS ts_iso, loss
        FROM loss_events
        ORDER BY ts ASC
        """,
        conn,
    )
    if df.empty:
        return df

    df = _maybe_filter_miners(df, miners)
    if df.empty:
        return df

    df["ts_local"] = _to_naive_local(df["ts_iso"], tz_name)
    df = df[df["ts_local"].notna()]

    return df


# States 
def query_states(conn: sqlite3.Connection, tz_name: str, miners: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT miner_hotkey, layer, to_state, ts AS ts_iso
        FROM state_events
        ORDER BY miner_hotkey ASC, ts ASC
        """,
        conn,
    )
    if df.empty:
        return df

    df = _maybe_filter_miners(df, miners)
    if df.empty:
        return df

    ts_local = _to_naive_local(df["ts_iso"], tz_name)
    df["from_dt_local"] = ts_local
    df = df[df["from_dt_local"].notna()]  # drop rows with bad timestamps

    # 'to' is the next state's 'from' within each miner
    df["to_dt_local"] = df.groupby("miner_hotkey")["from_dt_local"].shift(-1)

    cols = ["miner_hotkey", "layer", "to_state", "from_dt_local", "to_dt_local"]
    return df[cols].copy()


# Exceptions
def query_exceptions(conn: sqlite3.Connection, tz_name: str, miners: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT miner_hotkey, layer, ts AS ts_iso, ex_type, level, http_endpoint, http_code, message
        FROM exceptions
        ORDER BY ts ASC
        """,
        conn,
    )
    if df.empty:
        return df

    df = _maybe_filter_miners(df, miners)
    if df.empty:
        return df

    df["ts_local"] = _to_naive_local(df["ts_iso"], tz_name)
    df = df[df["ts_local"].notna()]

    return df


# Summary:
def build_last_seen_summary(conn: sqlite3.Connection, tz_name: str, miners: Optional[List[str]] = None) -> pd.DataFrame:
    states = query_states(conn, tz_name, miners=miners)
    if states.empty:
        return pd.DataFrame(columns=["miner_hotkey", "last_layer", "last_state", "last_seen"])

    idx = states.groupby("miner_hotkey")["from_dt_local"].idxmax()
    last_rows = states.loc[idx, ["miner_hotkey", "layer", "to_state", "from_dt_local"]].copy()
    last_rows.rename(
        columns={
            "layer": "last_layer",
            "to_state": "last_state",
            "from_dt_local": "last_seen",
        },
        inplace=True,
    )
    return last_rows.sort_values("miner_hotkey").reset_index(drop=True)
