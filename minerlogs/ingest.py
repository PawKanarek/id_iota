from __future__ import annotations

import re
import sqlite3
from typing import Callable, List, Optional, Tuple, Dict
from datetime import timezone
from zoneinfo import ZoneInfo
import io

from .timeutil import parse_log_datetime

def ingest_uploaded_files(
    conn: sqlite3.Connection,
    uploaded_files,
    tz_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    rx = _RegexBundle()
    total_files = 0
    total_lines = 0
    counters_sum: Dict[str, int] = {"backward": 0, "forward": 0, "loss": 0, "states": 0, "exceptions": 0}

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


def _ingest_stream(
    conn: sqlite3.Connection,
    text: str,
    tz_name: str,
    rx: "._RegexBundle",
    progress_callback: Optional[Callable[[str], None]],
) -> Tuple[int, int, dict]:
    tz_local = ZoneInfo(tz_name)
    miners_cache: set[str] = set()
    last_ctx = {"ts_utc_iso": None, "miner": None, "layer": None}
    counters = {"backward": 0, "forward": 0, "loss": 0, "states": 0, "exceptions": 0}

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

            mf = rx.forward_activation.search(msg)
            if mf:
                activation_id = mf.group(1)
                cur.execute(
                    "INSERT INTO forward_events (miner_hotkey, layer, ts, activation_id) VALUES (?, ?, ?, ?)",
                    (miner or "", _int_or_none(layer), ts_utc_iso, activation_id),
                )
                counters["forward"] += 1
                continue

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

            if level in ("ERROR", "CRITICAL"):
                ex_type, http_endpoint, http_code = _extract_exception_bits(rx, msg)
                cleaned_msg = _clean_message(msg)
                
                if not cleaned_msg or not cleaned_msg.strip() or cleaned_msg.strip() in ('|', '||', '|||'):
                    continue
                
                normalized_msg = _normalize_exception_message(cleaned_msg)
                
                cur.execute(
                    "INSERT INTO exceptions (miner_hotkey, layer, ts, ex_type, level, http_endpoint, http_code, message, message_normalized) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (miner or "", _int_or_none(layer), ts_utc_iso, ex_type, level, http_endpoint, http_code, cleaned_msg, normalized_msg),
                )
                counters["exceptions"] += 1
                continue

            continue

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


def _clean_message(msg: str) -> str:
    msg = msg.strip()
    
    while msg.startswith('|'):
        msg = msg[1:].strip()
    
    if msg.startswith("{") and msg.endswith("}"):
        return ""
    
    last_pipe_brace = msg.rfind(" | {")
    if last_pipe_brace != -1:
        potential_dict = msg[last_pipe_brace + 3:].strip()
        if potential_dict.startswith("{") and potential_dict.endswith("}"):
            return msg[:last_pipe_brace].strip()
    
    return msg


def _normalize_exception_message(msg: str) -> str:
    normalized = msg
    
    prefixes_to_strip = [
        "Error reporting loss: ",
        "Error during backward step: ",
        "Error making orchestrator request: ",
        "Failed to process: ",
        "Exception occurred: ",
        "RuntimeError: ",
    ]
    for prefix in prefixes_to_strip:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    normalized = re.sub(r'\{[^}]*"detail"[^}]*\}', '<JSON>', normalized)
    normalized = re.sub(r'\{[^}]*"error"[^}]*\}', '<JSON>', normalized)
    normalized = re.sub(r'\{[^}]*\}', '<JSON>', normalized)
    
    normalized = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        '<UUID>',
        normalized
    )
    
    normalized = re.sub(r'\b[A-Za-z0-9]{8,}\.\.\.', '<ID>', normalized)
    
    normalized = re.sub(
        r"(hotkey|miner)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9]{8,}['\"]?",
        r"\1: '<HOTKEY>'",
        normalized,
        flags=re.IGNORECASE
    )
    normalized = re.sub(r'\b[A-Za-z0-9]{48}\b', '<HOTKEY>', normalized)
    
    normalized = re.sub(r'after \d+ attempts?', 'after <N> attempts', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'attempt \d+ of \d+', 'attempt <N> of <N>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'retry \d+', 'retry <N>', normalized, flags=re.IGNORECASE)
    
    normalized = re.sub(r'\b(status|code|HTTP)[:\s]*[1-5]\d{2}\b', r'\1: <HTTP_CODE>', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\b[1-5]\d{2}\s*,\s*\{', '<HTTP_CODE>, {', normalized)
    
    normalized = re.sub(r'\b0x[0-9a-fA-F]{8,}\b', '<HEX_ID>', normalized)
    
    normalized = re.sub(
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?',
        '<TIMESTAMP>',
        normalized
    )
    
    normalized = re.sub(r'\b(batch|step|epoch|iteration|layer)\s+\d+\b', r'\1 <N>', normalized, flags=re.IGNORECASE)
    
    normalized = re.sub(r'\b\d+\.\d+\b', '<NUMBER>', normalized)
    
    normalized = re.sub(r'\b\d{7,}\b', '<LARGE_NUMBER>', normalized)
    
    normalized = re.sub(
        r"(activation_id|request_id|task_id|job_id|session_id)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9-]+['\"]?",
        r"\1: '<ID>'",
        normalized,
        flags=re.IGNORECASE
    )
    
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


class _RegexBundle:
    def __init__(self) -> None:
        self.header = re.compile(
            r"^(?P<dt>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+\|\s+(?P<level>[A-Z]+)\s+\|[^|]*\|\s+(?P<msg>.*)$"
        )

        self.tail_hotkey = re.compile(r"'hotkey'\s*:\s*'(?P<hotkey>[A-Za-z0-9]+)'")
        self.tail_layer = re.compile(r"'layer'\s*:\s*(?P<layer>\d+)")

        self.msg_miner = re.compile(r"\bMiner:?\s*(?P<miner>[A-Za-z0-9]+)\b")
        self.msg_layer = re.compile(r"\bLayer:?\s*(?P<layer>\d+)\b")

        self.backward_since_reset = re.compile(
            r"Backwards since reset for miner\s+(?P<miner>[A-Za-z0-9]+)\s*:\s*(?P<count>\d+)",
            re.IGNORECASE,
        )
        self.backward_in_step = re.compile(
            r"\bbackwards_since_reset\s*:\s*(?P<count>\d+)",
            re.IGNORECASE,
        )

        self.forward_activation = re.compile(r"🚀 Starting FORWARD pass.*?activation\s+([a-f0-9\-]+)")

        self.loss = re.compile(r"Computed loss\s+(?P<loss>[0-9]*\.?[0-9]+)", re.IGNORECASE)

        self.state_line = re.compile(
            r"Miner\s+(?P<miner>[A-Za-z0-9]+)\s+in Layer\s+(?P<layer>\d+)\s+is in state:\s+LayerPhase\.(?P<state>[A-Z_]+)",
            re.IGNORECASE,
        )

        self.api_exception = re.compile(r"\bAPI\s*Exception\b|\bAPIException\b", re.IGNORECASE)
        self.runtime_error = re.compile(r"\bRuntimeError\b|\bError during backward step\b", re.IGNORECASE)
        self.http_endpoint = re.compile(r"endpoint\s+(?P<endpoint>/[A-Za-z0-9/_-]+)")
        self.http_code = re.compile(r"(?P<code>\d{3})(?=\s*-\s)")


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        cb(msg)


def _int_or_none(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None