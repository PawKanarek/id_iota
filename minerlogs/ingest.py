from __future__ import annotations

import re
import sqlite3
from typing import Callable, List, Optional, Tuple, Dict
from datetime import timezone
from zoneinfo import ZoneInfo
import io

from .timeutil import parse_log_datetime

# Public API: in-memory uploads ingestion
def ingest_uploaded_files(
    conn: sqlite3.Connection,
    uploaded_files,  # list[streamlit.runtime.uploaded_file_manager.UploadedFile]
    tz_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """
    Parse a list of uploaded files fully in-memory (no disk, no offsets).
    Returns (total_files, total_lines, counters_sum).
    """
    rx = _RegexBundle()
    total_files = 0
    total_lines = 0
    counters_sum: Dict[str, int] = {"backward": 0, "loss": 0, "states": 0, "exceptions": 0}

    for idx, uf in enumerate(uploaded_files, start=1):
        name = getattr(uf, "name", f"uploaded_{idx}")
        try:
            text = uf.getvalue().decode("utf-8", errors="replace")
        except Exception:
            _log(progress_callback, f"[{idx}/{len(uploaded_files)}] Skipping unreadable file: {name}")
            continue

        _log(progress_callback, f"[{idx}/{len(uploaded_files)}] Parsing uploaded file: {name} (size ~{len(text):,} chars)")
        n_lines, _, counters = _ingest_stream(
            conn=conn,
            text=text,
            tz_name=tz_name,
            rx=rx,
            progress_callback=progress_callback
        )
        total_files += 1
        total_lines += n_lines
        for k in counters_sum:
            counters_sum[k] += counters[k]

    _log(progress_callback, f"Done. Parsed {total_files} uploaded file(s), processed ~{total_lines:,} line(s).")
    return total_files, total_lines, counters_sum


# Core ingestion logic against a text buffer
def _ingest_stream(
    conn: sqlite3.Connection,
    text: str,
    tz_name: str,
    rx: "._RegexBundle",
    progress_callback: Optional[Callable[[str], None]],
) -> Tuple[int, int, dict]:
    """
    Same logic as the legacy file parser but reading from an in-memory text buffer.
    Returns (lines_processed, dummy_offset, counters).
    """
    tz_local = ZoneInfo(tz_name)
    miners_cache: set[str] = set()
    last_ctx = {"ts_utc_iso": None, "miner": None, "layer": None}
    counters = {"backward": 0, "loss": 0, "states": 0, "exceptions": 0}

    n_lines = 0
    cur = conn.cursor()
    fh = io.StringIO(text)

    for line in fh:
        n_lines += 1
        m = rx.header.match(line)
        if m:
            dt_str = m.group("dt")
            level = m.group("level")
            msg = m.group("msg")
            ts_local = parse_log_datetime(dt_str)
            if ts_local is None:
                last_ctx = {"ts_utc_iso": None, "miner": None, "layer": None}
                continue
            ts_local = ts_local.replace(tzinfo=tz_local)
            ts_utc_iso = ts_local.astimezone(timezone.utc).isoformat()

            miner = _extract_hotkey(rx, line, msg)
            layer = _extract_layer(rx, line, msg)
            if miner and miner not in miners_cache:
                cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner,))
                miners_cache.add(miner)
            last_ctx = {"ts_utc_iso": ts_utc_iso, "miner": miner, "layer": _int_or_none(layer)}

            # Backward (explicit)
            mb = rx.backward_since_reset.search(msg)
            if mb:
                miner_b = mb.group("miner")
                since_reset = int(mb.group("count"))
                miner_eff = miner or miner_b
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO backward_events (miner_hotkey, layer, ts, since_reset) VALUES (?, ?, ?, ?)",
                    (miner_eff or "", _int_or_none(layer), ts_utc_iso, since_reset),
                )
                counters["backward"] += 1
                continue

            # Loss
            ml = rx.loss.search(msg)
            if ml:
                loss = float(ml.group("loss"))
                if not miner:
                    mminer = rx.msg_miner.search(msg)
                    if mminer:
                        miner = mminer.group("miner")
                        if miner and miner not in miners_cache:
                            cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner,))
                            miners_cache.add(miner)
                if layer is None:
                    mlayer = rx.msg_layer.search(msg)
                    if mlayer:
                        layer = int(mlayer.group("layer"))
                cur.execute(
                    "INSERT INTO loss_events (miner_hotkey, layer, ts, loss) VALUES (?, ?, ?, ?)",
                    (miner or "", _int_or_none(layer), ts_utc_iso, loss),
                )
                counters["loss"] += 1
                continue

            # State line
            ms = rx.state_line.search(msg)
            if ms:
                miner_s = ms.group("miner")
                layer_s = int(ms.group("layer"))
                to_state = ms.group("state")
                miner_eff = miner or miner_s
                layer_eff = _int_or_none(layer if layer is not None else layer_s)
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO state_events (miner_hotkey, layer, to_state, ts) VALUES (?, ?, ?, ?)",
                    (miner_eff or "", layer_eff, to_state, ts_utc_iso),
                )
                counters["states"] += 1
                continue

            # Exceptions
            if level in ("ERROR", "CRITICAL"):
                ex_type, http_endpoint, http_code = _extract_exception_bits(rx, msg)
                cur.execute(
                    "INSERT INTO exceptions (miner_hotkey, layer, ts, ex_type, level, http_endpoint, http_code, message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (miner or "", _int_or_none(layer), ts_utc_iso, ex_type, level, http_endpoint, http_code, msg.strip()),
                )
                counters["exceptions"] += 1
                continue

            continue  # next line

        # continuation line
        mstep = rx.backward_in_step.search(line)
        if mstep:
            since_reset = int(mstep.group("count"))
            miner_tail = rx.tail_hotkey.search(line)
            layer_tail = rx.tail_layer.search(line)
            miner_eff = (miner_tail.group("hotkey") if miner_tail else None) or last_ctx["miner"]
            layer_eff = _int_or_none(int(layer_tail.group("layer")) if layer_tail else last_ctx["layer"])
            ts_eff = last_ctx["ts_utc_iso"]
            if ts_eff and miner_eff:
                if miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO backward_events (miner_hotkey, layer, ts, since_reset) VALUES (?, ?, ?, ?)",
                    (miner_eff, layer_eff, ts_eff, since_reset),
                )
                counters["backward"] += 1
            continue

    conn.commit()
    return n_lines, 0, counters


def _extract_hotkey(rx: "._RegexBundle", line: str, msg: str) -> Optional[str]:
    m = rx.tail_hotkey.search(line)
    if m:
        return m.group("hotkey")
    m = rx.msg_miner.search(msg)
    if m:
        return m.group("miner")
    return None


def _extract_layer(rx: "._RegexBundle", line: str, msg: str) -> Optional[int]:
    m = rx.tail_layer.search(line)
    if m:
        return int(m.group("layer"))
    m = rx.msg_layer.search(msg)
    if m:
        return int(m.group("layer"))
    return None


def _extract_exception_bits(rx: "._RegexBundle", msg: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    ex_type: Optional[str] = None
    endpoint: Optional[str] = None
    code: Optional[int] = None

    if rx.api_exception.search(msg):
        ex_type = "APIException"
    if rx.runtime_error.search(msg):
        ex_type = ex_type or "RuntimeError"

    m = rx.http_endpoint.search(msg)
    if m:
        endpoint = m.group("endpoint")

    m = rx.http_code.search(msg)
    if m:
        try:
            code = int(m.group("code"))
        except ValueError:
            code = None

    if "Error making orchestrator request" in msg:
        ex_type = ex_type or "APIException"

    return ex_type, endpoint, code


# Regex bundle
class _RegexBundle:
    def __init__(self) -> None:
        # Timestamped header: "YYYY-mm-dd HH:MM:SS(.ffffff) | LEVEL | where | message ..."
        self.header = re.compile(
            r"^(?P<dt>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+\|\s+(?P<level>[A-Z]+)\s+\|[^|]*\|\s+(?P<msg>.*)$"
        )

        # Tail dict: ... | {'hotkey': 'XXXXX', 'layer': N}
        self.tail_hotkey = re.compile(r"'hotkey'\s*:\s*'(?P<hotkey>[A-Za-z0-9]+)'")
        self.tail_layer = re.compile(r"'layer'\s*:\s*(?P<layer>\d+)")

        # Message miner/layer (flexible with optional colon)
        self.msg_miner = re.compile(r"\bMiner:?\s*(?P<miner>[A-Za-z0-9]+)\b")
        self.msg_layer = re.compile(r"\bLayer:?\s*(?P<layer>\d+)\b")

        # Backward since reset — two variants
        self.backward_since_reset = re.compile(
            r"Backwards since reset for miner\s+(?P<miner>[A-Za-z0-9]+)\s*:\s*(?P<count>\d+)",
            re.IGNORECASE,
        )
        self.backward_in_step = re.compile(
            r"\bbackwards_since_reset\s*:\s*(?P<count>\d+)",
            re.IGNORECASE,
        )

        # Loss line
        self.loss = re.compile(r"Computed loss\s+(?P<loss>[0-9]*\.?[0-9]+)", re.IGNORECASE)

        # State line
        self.state_line = re.compile(
            r"Miner\s+(?P<miner>[A-Za-z0-9]+)\s+in Layer\s+(?P<layer>\d+)\s+is in state:\s+LayerPhase\.(?P<state>[A-Z_]+)",
            re.IGNORECASE,
        )

        # Exceptions
        self.api_exception = re.compile(r"\bAPI\s*Exception\b|\bAPIException\b", re.IGNORECASE)
        self.runtime_error = re.compile(r"\bRuntimeError\b|\bError during backward step\b", re.IGNORECASE)
        self.http_endpoint = re.compile(r"endpoint\s+(?P<endpoint>/[A-Za-z0-9/_-]+)")
        self.http_code = re.compile(r"(?P<code>\d{3})(?=\s*-\s)")  # first status code before " - "


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        cb(msg)


def _int_or_none(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None
