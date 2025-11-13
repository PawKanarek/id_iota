from __future__ import annotations

import re
import sqlite3
from typing import Callable, List, Optional, Tuple, Dict
from datetime import timezone
from zoneinfo import ZoneInfo
from multiprocessing import Pool, cpu_count

from .timeutil import parse_log_datetime

def _int_or_none(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None

def _process_log_lines(file_path: str, tz_name: str, rx: "_RegexBundle", source_file: str = ""):
    tz_local = ZoneInfo(tz_name)
    miners_cache: set[str] = set()
    last_ctx = {"ts_utc_iso": None, "miner": None, "layer": None}

    counters = {
        "backward": 0, "loss": 0, "states": 0,
        "exceptions": 0, "optimization": 0, "resource": 0, "registration": 0
    }

    n_lines = 0
    min_ts = None
    max_ts = None

    fh = open(file_path, 'r', encoding='utf-8', errors='replace')

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

            if min_ts is None or ts_local < min_ts:
                min_ts = ts_local
            if max_ts is None or ts_local > max_ts:
                max_ts = ts_local

            miner = _extract_hotkey(rx, line, msg)
            layer = _extract_layer(rx, line, msg)
            if miner and miner not in miners_cache:
                yield ("miner", (miner,))
                miners_cache.add(miner)
            last_ctx = {"ts_utc_iso": ts_utc_iso, "miner": miner, "layer": _int_or_none(layer)}

            mb = rx.backward_since_reset.search(msg)
            if mb:
                miner_eff = miner or mb.group("miner")
                if miner_eff and miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("backward", (miner_eff or "", _int_or_none(layer), ts_utc_iso, int(mb.group("count"))))
                counters["backward"] += 1
                continue

            ml = rx.loss.search(msg)
            if ml:
                miner_loss = (ml.group("miner") if ml.lastindex >= 4 else None) or miner
                layer_loss = (_int_or_none(ml.group("layer")) if ml.lastindex >= 3 else None) or _int_or_none(layer)
                activation_id = ml.group("activation") if ml.lastindex >= 2 else None
                if miner_loss and miner_loss not in miners_cache:
                    yield ("miner", (miner_loss,))
                    miners_cache.add(miner_loss)
                yield ("loss", (miner_loss or "", layer_loss, ts_utc_iso, float(ml.group("loss")), activation_id))
                counters["loss"] += 1
                continue

            ms = rx.state_line.search(msg)
            if ms:
                miner_eff = miner or ms.group("miner")
                layer_eff = _int_or_none(layer if layer is not None else int(ms.group("layer")))
                if miner_eff and miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("state", (miner_eff or "", layer_eff, ms.group("state"), ts_utc_iso))
                counters["states"] += 1
                continue

            mopt = rx.optimization_step.search(msg)
            if mopt:
                miner_eff = miner or mopt.group("miner")
                if miner_eff and miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("optimization", (miner_eff or "", _int_or_none(layer), ts_utc_iso, None, int(mopt.group("backwards"))))
                counters["optimization"] += 1
                continue

            mopt_complete = rx.optimization_complete.search(msg)
            if mopt_complete:
                miner_eff = miner or mopt_complete.group("miner")
                if miner_eff and miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("optimization", (miner_eff or "", _int_or_none(layer), ts_utc_iso, int(mopt_complete.group("step")), None))
                counters["optimization"] += 1
                continue

            mgpu = rx.gpu_memory.search(msg)
            if mgpu:
                if miner and miner not in miners_cache:
                    yield ("miner", (miner,))
                    miners_cache.add(miner)
                yield ("resource", (miner or "", _int_or_none(layer), ts_utc_iso, "gpu_memory", float(mgpu.group("memory")), None))
                counters["resource"] += 1
                continue

            mgpu_usage = rx.gpu_memory_usage.search(msg)
            if mgpu_usage:
                used, total = float(mgpu_usage.group("used")), float(mgpu_usage.group("total"))
                if miner and miner not in miners_cache:
                    yield ("miner", (miner,))
                    miners_cache.add(miner)
                yield ("resource", (miner or "", _int_or_none(layer), ts_utc_iso, "gpu_memory_usage", used, f"{used}/{total}"))
                counters["resource"] += 1
                continue

            mcache = rx.cache_full.search(msg)
            if mcache:
                miner_eff = miner or mcache.group("miner")
                if miner_eff and miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("resource", (miner_eff or "", _int_or_none(layer), ts_utc_iso, "cache_full", None, str(int(mcache.group("count")))))
                counters["resource"] += 1
                continue

            mreg = rx.registration_success.search(msg)
            if mreg:
                miner_eff = miner or mreg.group("miner")
                if miner_eff and miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("registration", (miner_eff or "", int(mreg.group("layer")), ts_utc_iso, int(mreg.group("epoch")), "registered"))
                counters["registration"] += 1
                continue

            if level in ("ERROR", "CRITICAL"):
                ex_type, http_endpoint, http_code = _extract_exception_bits(rx, msg)
                cleaned_msg = _clean_message(msg)

                if not cleaned_msg or not cleaned_msg.strip() or cleaned_msg.strip() in ('|', '||', '|||'):
                    continue

                normalized_msg = _normalize_exception_message(cleaned_msg)
                yield ("exception", (miner or "", _int_or_none(layer), ts_utc_iso, ex_type, level, http_endpoint, http_code, cleaned_msg, normalized_msg, n_lines, source_file))
                counters["exceptions"] += 1
                continue

            continue

        mstep = rx.backward_in_step.search(line)
        if mstep:
            miner_tail = rx.tail_hotkey.search(line)
            layer_tail = rx.tail_layer.search(line)
            miner_eff = (miner_tail.group("hotkey") if miner_tail else None) or last_ctx["miner"]
            layer_eff = _int_or_none(int(layer_tail.group("layer")) if layer_tail else last_ctx["layer"])
            if last_ctx["ts_utc_iso"] and miner_eff:
                if miner_eff not in miners_cache:
                    yield ("miner", (miner_eff,))
                    miners_cache.add(miner_eff)
                yield ("backward", (miner_eff, layer_eff, last_ctx["ts_utc_iso"], int(mstep.group("count"))))
                counters["backward"] += 1
            continue

    fh.close()
    yield ("counters", counters)
    yield ("lines", n_lines)
    yield ("min_ts", min_ts)
    yield ("max_ts", max_ts)

def _parse_file_to_batches(args: Tuple[str, str, str]) -> Tuple[str, int, Dict[str, int], Dict[str, List], object, object, int]:
    file_path, name, tz_name = args
    rx = _RegexBundle()

    batches = {
        "miners": [], "backward": [], "loss": [], "state": [],
        "optimization": [], "resource": [], "registration": [], "exception": []
    }

    counters = {}
    n_lines = 0
    min_ts = None
    max_ts = None

    for event_type, event_data in _process_log_lines(file_path, tz_name, rx, name):
        if event_type == "counters":
            counters = event_data
        elif event_type == "lines":
            n_lines = event_data
        elif event_type == "min_ts":
            min_ts = event_data
        elif event_type == "max_ts":
            max_ts = event_data
        elif event_type == "miner":
            batches["miners"].append(event_data)
        else:
            batches[event_type].append(event_data)

    from pathlib import Path
    size_bytes = Path(file_path).stat().st_size if Path(file_path).exists() else 0

    return name, n_lines, counters, batches, min_ts, max_ts, size_bytes

def ingest_log_files(
    conn: sqlite3.Connection,
    file_paths: List,
    tz_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    from pathlib import Path

    rx = _RegexBundle()
    total_files = 0
    total_lines = 0
    counters_sum: Dict[str, int] = {
        "backward": 0, "loss": 0, "states": 0,
        "exceptions": 0, "optimization": 0, "resource": 0, "registration": 0
    }

    file_metadata = {}

    if len(file_paths) <= 4:
        for idx, fpath in enumerate(file_paths, start=1):
            name = Path(fpath).name
            if progress_callback:
                progress_callback(f"[{idx}/{len(file_paths)}] Parsing file: {name}")
            n_lines, _, counters, min_ts, max_ts = _ingest_stream(
                conn=conn,
                file_path=str(fpath),
                tz_name=tz_name,
                rx=rx,
                progress_callback=progress_callback,
                source_file=name
            )
            total_files += 1
            total_lines += n_lines
            for k in counters_sum:
                counters_sum[k] += counters[k]

            if min_ts and max_ts:
                file_metadata[name] = {
                    "path": Path(fpath),
                    "min_timestamp": min_ts,
                    "max_timestamp": max_ts,
                    "size_bytes": Path(fpath).stat().st_size if Path(fpath).exists() else 0,
                }
    else:
        if progress_callback:
            progress_callback(f"Processing {len(file_paths)} files in parallel...")

        path_map = {Path(fpath).name: Path(fpath) for fpath in file_paths}
        args = [(str(fpath), Path(fpath).name, tz_name) for fpath in file_paths]

        with Pool(min(cpu_count(), len(file_paths))) as pool:
            cur = conn.cursor()

            for name, n_lines, counters, batches, min_ts, max_ts, size_bytes in pool.imap_unordered(_parse_file_to_batches, args):
                if batches["miners"]:
                    cur.executemany("INSERT OR IGNORE INTO miners (hotkey) VALUES (?)", batches["miners"])
                if batches["backward"]:
                    cur.executemany("INSERT INTO backward_events (miner_hotkey, layer, ts, since_reset) VALUES (?, ?, ?, ?)", batches["backward"])
                if batches["loss"]:
                    cur.executemany("INSERT INTO loss_events (miner_hotkey, layer, ts, loss, activation_id) VALUES (?, ?, ?, ?, ?)", batches["loss"])
                if batches["state"]:
                    cur.executemany("INSERT INTO state_events (miner_hotkey, layer, to_state, ts) VALUES (?, ?, ?, ?)", batches["state"])
                if batches["optimization"]:
                    cur.executemany("INSERT INTO optimization_events (miner_hotkey, layer, ts, step_number, backwards_count) VALUES (?, ?, ?, ?, ?)", batches["optimization"])
                if batches["resource"]:
                    cur.executemany("INSERT INTO resource_events (miner_hotkey, layer, ts, event_type, value_gb, value_text) VALUES (?, ?, ?, ?, ?, ?)", batches["resource"])
                if batches["registration"]:
                    cur.executemany("INSERT INTO registration_events (miner_hotkey, layer, ts, training_epoch, status) VALUES (?, ?, ?, ?, ?)", batches["registration"])
                if batches["exception"]:
                    cur.executemany("INSERT INTO exceptions (miner_hotkey, layer, ts, ex_type, level, http_endpoint, http_code, message, message_normalized, line_number, source_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batches["exception"])

                conn.commit()

                total_files += 1
                total_lines += n_lines
                for k in counters_sum:
                    counters_sum[k] += counters[k]

                if min_ts and max_ts:
                    file_metadata[name] = {
                        "path": path_map.get(name, Path(name)),
                        "min_timestamp": min_ts,
                        "max_timestamp": max_ts,
                        "size_bytes": size_bytes,
                    }

                if progress_callback:
                    progress_callback(f"[{total_files}/{len(file_paths)}] Completed: {name}")

    if progress_callback:
        progress_callback(f"Done. Parsed {total_files} file(s), processed ~{total_lines:,} line(s).")
    return total_files, total_lines, counters_sum, file_metadata


BATCH_SQL = {
    "miners": "INSERT OR IGNORE INTO miners (hotkey) VALUES (?)",
    "backward": "INSERT INTO backward_events (miner_hotkey, layer, ts, since_reset) VALUES (?, ?, ?, ?)",
    "loss": "INSERT INTO loss_events (miner_hotkey, layer, ts, loss, activation_id) VALUES (?, ?, ?, ?, ?)",
    "state": "INSERT INTO state_events (miner_hotkey, layer, to_state, ts) VALUES (?, ?, ?, ?)",
    "optimization": "INSERT INTO optimization_events (miner_hotkey, layer, ts, step_number, backwards_count) VALUES (?, ?, ?, ?, ?)",
    "resource": "INSERT INTO resource_events (miner_hotkey, layer, ts, event_type, value_gb, value_text) VALUES (?, ?, ?, ?, ?, ?)",
    "registration": "INSERT INTO registration_events (miner_hotkey, layer, ts, training_epoch, status) VALUES (?, ?, ?, ?, ?)",
    "exception": "INSERT INTO exceptions (miner_hotkey, layer, ts, ex_type, level, http_endpoint, http_code, message, message_normalized, line_number, source_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
}

def _flush_single_batch(cur, batch_key: str, batch: list):
    if batch:
        cur.executemany(BATCH_SQL[batch_key], batch)

def _flush_batches(cur, miners_batch, backward_batch, loss_batch, state_batch,
                   optimization_batch, resource_batch, registration_batch, exception_batch):
    _flush_single_batch(cur, "miners", miners_batch)
    _flush_single_batch(cur, "backward", backward_batch)
    _flush_single_batch(cur, "loss", loss_batch)
    _flush_single_batch(cur, "state", state_batch)
    _flush_single_batch(cur, "optimization", optimization_batch)
    _flush_single_batch(cur, "resource", resource_batch)
    _flush_single_batch(cur, "registration", registration_batch)
    _flush_single_batch(cur, "exception", exception_batch)

def _ingest_stream(
    conn: sqlite3.Connection,
    file_path: str,
    tz_name: str,
    rx: "._RegexBundle",
    progress_callback: Optional[Callable[[str], None]],
    source_file: str = "",
) -> Tuple[int, int, dict, object, object]:
    BATCH_SIZE = 1000
    batches = {
        "miners": [], "backward": [], "loss": [], "state": [],
        "optimization": [], "resource": [], "registration": [], "exception": []
    }

    counters = {}
    n_lines = 0
    min_ts = None
    max_ts = None
    cur = conn.cursor()

    for event_type, event_data in _process_log_lines(file_path, tz_name, rx, source_file):
        if event_type == "counters":
            counters = event_data
            continue
        elif event_type == "lines":
            n_lines = event_data
            continue
        elif event_type == "min_ts":
            min_ts = event_data
            continue
        elif event_type == "max_ts":
            max_ts = event_data
            continue

        batch_key = "miners" if event_type == "miner" else event_type
        batches[batch_key].append(event_data)

        if len(batches[batch_key]) >= BATCH_SIZE:
            _flush_single_batch(cur, batch_key, batches[batch_key])
            batches[batch_key].clear()

    _flush_batches(cur, batches["miners"], batches["backward"], batches["loss"], batches["state"],
                   batches["optimization"], batches["resource"], batches["registration"], batches["exception"])
    conn.commit()
    return n_lines, 0, counters, min_ts, max_ts


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


