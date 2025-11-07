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
    counters_sum: Dict[str, int] = {
        "backward": 0, "loss": 0, "states": 0, 
        "exceptions": 0, "optimization": 0, "resource": 0, "registration": 0
    }

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
            progress_callback=progress_callback,
            source_file=name
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
    source_file: str = "",
) -> Tuple[int, int, dict]:
    tz_local = ZoneInfo(tz_name)
    miners_cache: set[str] = set()
    last_ctx = {"ts_utc_iso": None, "miner": None, "layer": None}
    counters = {
        "backward": 0, "loss": 0, "states": 0, 
        "exceptions": 0, "optimization": 0, "resource": 0, "registration": 0
    }

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

            ml = rx.loss.search(msg)
            if ml:
                loss = float(ml.group("loss"))
                activation_id = ml.group("activation") if ml.lastindex >= 2 else None
                layer_loss = _int_or_none(ml.group("layer")) if ml.lastindex >= 3 else None
                miner_loss = ml.group("miner") if ml.lastindex >= 4 else None
                
                if not miner_loss:
                    miner_loss = miner
                if not layer_loss:
                    layer_loss = _int_or_none(layer)
                    
                if miner_loss and miner_loss not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_loss,))
                    miners_cache.add(miner_loss)
                    
                cur.execute(
                    "INSERT INTO loss_events (miner_hotkey, layer, ts, loss, activation_id) VALUES (?, ?, ?, ?, ?)",
                    (miner_loss or "", layer_loss, ts_utc_iso, loss, activation_id),
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

            mopt = rx.optimization_step.search(msg)
            if mopt:
                miner_opt = mopt.group("miner")
                backwards = int(mopt.group("backwards"))
                step_num = None
                miner_eff = miner or miner_opt
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO optimization_events (miner_hotkey, layer, ts, step_number, backwards_count) VALUES (?, ?, ?, ?, ?)",
                    (miner_eff or "", _int_or_none(layer), ts_utc_iso, step_num, backwards),
                )
                counters["optimization"] += 1
                continue

            mopt_complete = rx.optimization_complete.search(msg)
            if mopt_complete:
                miner_opt = mopt_complete.group("miner")
                step_num = int(mopt_complete.group("step"))
                miner_eff = miner or miner_opt
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO optimization_events (miner_hotkey, layer, ts, step_number, backwards_count) VALUES (?, ?, ?, ?, ?)",
                    (miner_eff or "", _int_or_none(layer), ts_utc_iso, step_num, None),
                )
                counters["optimization"] += 1
                continue

            mgpu = rx.gpu_memory.search(msg)
            if mgpu:
                memory_gb = float(mgpu.group("memory"))
                event_type = "gpu_memory"
                miner_eff = miner
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO resource_events (miner_hotkey, layer, ts, event_type, value_gb, value_text) VALUES (?, ?, ?, ?, ?, ?)",
                    (miner_eff or "", _int_or_none(layer), ts_utc_iso, event_type, memory_gb, None),
                )
                counters["resource"] += 1
                continue

            mgpu_usage = rx.gpu_memory_usage.search(msg)
            if mgpu_usage:
                used_gb = float(mgpu_usage.group("used"))
                total_gb = float(mgpu_usage.group("total"))
                event_type = "gpu_memory_usage"
                miner_eff = miner
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO resource_events (miner_hotkey, layer, ts, event_type, value_gb, value_text) VALUES (?, ?, ?, ?, ?, ?)",
                    (miner_eff or "", _int_or_none(layer), ts_utc_iso, event_type, used_gb, f"{used_gb}/{total_gb}"),
                )
                counters["resource"] += 1
                continue

            mcache = rx.cache_full.search(msg)
            if mcache:
                miner_cache = mcache.group("miner")
                count = int(mcache.group("count"))
                event_type = "cache_full"
                miner_eff = miner or miner_cache
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO resource_events (miner_hotkey, layer, ts, event_type, value_gb, value_text) VALUES (?, ?, ?, ?, ?, ?)",
                    (miner_eff or "", _int_or_none(layer), ts_utc_iso, event_type, None, str(count)),
                )
                counters["resource"] += 1
                continue

            mreg = rx.registration_success.search(msg)
            if mreg:
                miner_reg = mreg.group("miner")
                layer_reg = int(mreg.group("layer"))
                epoch = int(mreg.group("epoch"))
                miner_eff = miner or miner_reg
                if miner_eff and miner_eff not in miners_cache:
                    cur.execute("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", (miner_eff,))
                    miners_cache.add(miner_eff)
                cur.execute(
                    "INSERT INTO registration_events (miner_hotkey, layer, ts, training_epoch, status) VALUES (?, ?, ?, ?, ?)",
                    (miner_eff or "", layer_reg, ts_utc_iso, epoch, "registered"),
                )
                counters["registration"] += 1
                continue

            if level in ("ERROR", "CRITICAL"):
                ex_type, http_endpoint, http_code = _extract_exception_bits(rx, msg)
                cleaned_msg = _clean_message(msg)
                
                if not cleaned_msg or not cleaned_msg.strip() or cleaned_msg.strip() in ('|', '||', '|||'):
                    continue
                
                normalized_msg = _normalize_exception_message(cleaned_msg)

                cur.execute(
                    "INSERT INTO exceptions (miner_hotkey, layer, ts, ex_type, level, http_endpoint, http_code, message, message_normalized, line_number, source_file) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (miner or "", _int_or_none(layer), ts_utc_iso, ex_type, level, http_endpoint, http_code, cleaned_msg, normalized_msg, n_lines, source_file),
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

        self.loss = re.compile(
            r"(?:📊\s*)?Computed loss\s+(?P<loss>[0-9]*\.?[0-9]+)"
            r"(?:\s+for activation\s+(?P<activation>[a-f0-9\-]+))?"
            r"(?:[^|]*?\bLayer:\s*(?P<layer>\d+))?"
            r"(?:[^|]*?\bMiner:\s*(?P<miner>[A-Za-z0-9]+))?",
            re.IGNORECASE
        )

        self.state_line = re.compile(
            r"Miner\s+(?P<miner>[A-Za-z0-9]+)\s+in Layer\s+(?P<layer>\d+)\s+is in state:\s+(?:LayerPhase\.)?(?P<state>[A-Z_]+)",
            re.IGNORECASE,
        )

        self.optimization_step = re.compile(
            r"Miner\s+(?P<miner>[A-Za-z0-9]+)\s+performing local optimization step after\s+(?P<backwards>\d+)\s+backward",
            re.IGNORECASE,
        )

        self.optimization_complete = re.compile(
            r"(?:✅\s*)?Miner\s+(?P<miner>[A-Za-z0-9]+)\s+completed local optimization step\s+#?(?P<step>\d+)",
            re.IGNORECASE,
        )

        self.gpu_memory = re.compile(
            r"💾\s*GPU memory:\s*(?P<memory>[0-9]*\.?[0-9]+)\s*GB",
            re.IGNORECASE,
        )

        self.gpu_memory_usage = re.compile(
            r"GPU memory usage:\s*(?P<used>[0-9]*\.?[0-9]+)\s*GB\s*/\s*(?P<total>[0-9]*\.?[0-9]+)\s*GB",
            re.IGNORECASE,
        )

        self.cache_full = re.compile(
            r"Miner\s+(?P<miner>[A-Za-z0-9]+)\s+cache full with\s+(?P<count>\d+)\s+activations",
            re.IGNORECASE,
        )

        self.registration_success = re.compile(
            r"✅\s*Miner\s+(?P<miner>[A-Za-z0-9]+)\s+registered successfully in layer\s+(?P<layer>\d+)\s+on training epoch\s+(?P<epoch>\d+)",
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