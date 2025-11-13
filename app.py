#!/usr/bin/env python3
from __future__ import annotations

import re
import sqlite3
import shutil
import tempfile
import atexit
import zipfile
import io
import os
import glob
import base64
import html
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import pandas as pd
import plotly.graph_objects as go
from bisect import bisect_left
from minerlogs.timeutil import parse_log_datetime

APP_TITLE = "ID_IOTA — Interactive Diagnostic for IOTA"
_LOGO = Path(__file__).resolve().parent / "assets" / "logo.png"
TZ_NAME = "UTC"

HEADER_RX = re.compile(
    r"^(?P<dt>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+\|\s+(?P<level>[A-Z]+)\s+\|[^|]*\|\s+(?P<msg>.*)$"
)

def _ts_key(dt) -> tuple[int, int, int, int, int, int, int]:
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)

def _load_file_lines(filepath: Path) -> List[str]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except Exception:
        return []

def _build_single_header_index(fpath: Path) -> Dict[str, Any]:
    lines = _load_file_lines(fpath)
    keys, levels, line_idx = [], [], []
    for i, line in enumerate(lines):
        m = HEADER_RX.match(line)
        if not m:
            continue
        dt = parse_log_datetime(m.group("dt"))
        if dt is None:
            continue
        keys.append(_ts_key(dt))
        levels.append(m.group("level"))
        line_idx.append(i)
    return {
        "name": fpath.name,
        "path": fpath,
        "keys": keys,
        "levels": levels,
        "line_idx": line_idx,
    }

def _build_header_indices(file_paths: List[Path]) -> List[Dict[str, Any]]:
    if len(file_paths) <= 4:
        return [_build_single_header_index(fpath) for fpath in file_paths]

    with Pool(min(cpu_count(), len(file_paths))) as pool:
        idx_list = pool.map(_build_single_header_index, file_paths)
    return idx_list

def _find_exception_line_and_context(
    ts_local: pd.Timestamp,
    level: str,
    message: str,
    file_paths: List[Path],
    pre_lines: int = 100,
    post_lines: int = 25,
) -> Tuple[Optional[str], List[str], List[str], Optional[int]]:
    """Find exception line and return (filename, before_lines, after_lines, line_number)"""
    idx_pack = st.session_state.get("report_index", [])
    key = _ts_key(ts_local.to_pydatetime())

    # Use multiple parts of message for better matching
    msg_parts = []
    if message:
        msg_clean = message.strip()
        # Try to extract distinctive parts
        if len(msg_clean) > 20:
            msg_parts.append(msg_clean[:40])  # First 40 chars
        if len(msg_clean) > 80:
            msg_parts.append(msg_clean[40:80])  # Middle part
        if len(msg_clean) <= 20:
            msg_parts.append(msg_clean)

    for idxf in idx_pack:
        keys = idxf["keys"]
        if not keys:
            continue
        pos = bisect_left(keys, key)
        # Search in a wider window around the timestamp
        start = max(0, pos - 5)
        stop = min(len(keys), pos + 6)

        best_match = None
        best_score = 0

        for j in range(start, stop):
            # Must match timestamp and level
            if keys[j] != key or idxf["levels"][j] != level:
                continue

            lines = _load_file_lines(idxf["path"])
            if not lines:
                continue

            li = idxf["line_idx"][j]
            line = lines[li]

            # Score the match quality
            score = 0
            if msg_parts:
                for part in msg_parts:
                    if part and part in line:
                        score += len(part)

            # If we have any message match, prefer it
            if score > best_score:
                best_score = score
                best_match = li
            elif score == 0 and best_match is None:
                # No message match yet, use first timestamp+level match
                best_match = li

        if best_match is not None:
            s = max(0, best_match - pre_lines)
            e = min(len(lines), best_match + 1 + post_lines)
            return idxf["name"], lines[s:best_match], lines[best_match:e], best_match + 1

    return None, [], [], None


def _build_single_file_metadata(fpath: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    lines = _load_file_lines(fpath)
    timestamps = []
    for line in lines:
        m = HEADER_RX.match(line)
        if not m:
            continue
        dt = parse_log_datetime(m.group("dt"))
        if dt:
            timestamps.append(dt)

    if timestamps:
        return (fpath.name, {
            "path": fpath,
            "min_timestamp": min(timestamps),
            "max_timestamp": max(timestamps),
            "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
        })
    return (fpath.name, None)

def _build_file_metadata(file_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    if len(file_paths) <= 4:
        metadata = {}
        for fpath in file_paths:
            name, meta = _build_single_file_metadata(fpath)
            if meta:
                metadata[name] = meta
        return metadata

    with Pool(min(cpu_count(), len(file_paths))) as pool:
        results = pool.map(_build_single_file_metadata, file_paths)

    metadata = {}
    for name, meta in results:
        if meta:
            metadata[name] = meta
    return metadata


def _get_files_in_range(
    file_metadata: Dict[str, Dict[str, Any]],
    start_dt: datetime,
    end_dt: datetime,
) -> List[Path]:
    matching_files = []
    for name, meta in file_metadata.items():
        file_min = meta["min_timestamp"]
        file_max = meta["max_timestamp"]
        if file_min <= end_dt and file_max >= start_dt:
            matching_files.append(meta["path"])
    return matching_files


def _create_logs_zip(file_paths: List[Path], time_range: Optional[Tuple[datetime, datetime]] = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in file_paths:
            if fpath.exists():
                zf.write(fpath, fpath.name)
    buffer.seek(0)
    return buffer.read()


def _get_zip_filename(time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
    if time_range:
        start_str = time_range[0].strftime("%Y-%m-%d_%H-%M")
        end_str = time_range[1].strftime("%Y-%m-%d_%H-%M")
        return f"logs_selected_{start_str}_to_{end_str}.zip"
    else:
        return f"logs_all_{datetime.now().strftime('%Y-%m-%d')}.zip"


def _apply_exception_filters(df: pd.DataFrame, filters: List[str]) -> pd.DataFrame:
    if df.empty or not filters:
        return df
    
    mask = pd.Series([True] * len(df), index=df.index)
    for filter_pattern in filters:
        if filter_pattern.strip():
            pattern_lower = filter_pattern.lower()
            mask &= ~df["message_normalized"].astype(str).str.lower().str.contains(pattern_lower, regex=False, na=False)
    
    return df[mask].copy()

def _count_exceptions_per_filter(df: pd.DataFrame, filters: List[str]) -> Dict[str, int]:
    if df.empty or not filters:
        return {}
    
    counts = {}
    for filter_pattern in filters:
        if filter_pattern.strip():
            pattern_lower = filter_pattern.lower()
            match_mask = df["message_normalized"].astype(str).str.lower().str.contains(pattern_lower, regex=False, na=False)
            counts[filter_pattern] = match_mask.sum()
    
    return counts


def _find_file_containing_exception(
    ts_local: pd.Timestamp,
    miner: str,
    message: str,
    file_paths: List[Path]
) -> Optional[Path]:
    """Find which file contains the exception by timestamp (DEPRECATED - use _match_source_file_to_path instead)"""
    if "file_metadata" not in st.session_state:
        return file_paths[0] if file_paths else None

    file_metadata = st.session_state["file_metadata"]
    target_dt = ts_local.to_pydatetime()

    for fname, meta in file_metadata.items():
        if meta["min_timestamp"] <= target_dt <= meta["max_timestamp"]:
            return meta["path"]

    return file_paths[0] if file_paths else None


def _match_source_file_to_path(
    source_file: str,
    file_paths: List[Path]
) -> Optional[Path]:
    """Match a source file name from the database to a full Path object.

    Args:
        source_file: The filename stored in the database (e.g., "miner.log")
        file_paths: List of available file paths from session state

    Returns:
        The matching Path object, or None if not found
    """
    if not source_file or not file_paths:
        return None

    # Direct name match (most common case)
    for path in file_paths:
        if path.name == source_file:
            return path

    # Fallback: check if source_file is a substring of any filename
    for path in file_paths:
        if source_file in str(path):
            return path

    return None


def _generate_log_html(
    file_path: Path,
    target_timestamp: pd.Timestamp,
    miner_hotkey: str,
    anti_doxx: bool,
    mask_map: Dict[str, str],
    target_line_number: Optional[int] = None
) -> str:
    """Generate HTML file with highlighted exception"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return "<html><body><h1>Error loading file</h1></body></html>"

    # Find target line - use provided line number or search
    target_line_idx = None
    if target_line_number is not None:
        # Line numbers are 1-indexed, convert to 0-indexed
        target_line_idx = target_line_number - 1
        # Validate line number is within bounds
        if target_line_idx < 0 or target_line_idx >= len(lines):
            target_line_idx = None

    # Only use timestamp fallback if no line number was provided at all
    if target_line_number is None:
        # Fallback: search by timestamp and miner (for backward compatibility)
        target_ts_str = target_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        for idx, line in enumerate(lines):
            m = HEADER_RX.match(line)
            if m:
                line_dt = m.group("dt")
                if target_ts_str in line_dt and miner_hotkey in line:
                    target_line_idx = idx
                    break
    
    # Build HTML
    html_lines = ['''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>''' + file_path.name + '''</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        .header {
            position: sticky;
            top: 0;
            background-color: #2d2d2d;
            padding: 10px;
            border-bottom: 2px solid #444;
            margin-bottom: 20px;
        }
        .log-line {
            white-space: pre;
            line-height: 1.4;
            padding: 2px 0;
        }
        .log-line-number {
            display: inline-block;
            width: 60px;
            color: #858585;
            text-align: right;
            margin-right: 15px;
            user-select: none;
        }
        .exception-line {
            background-color: #5a1e1e;
            border-left: 4px solid #f48771;
            padding: 2px;
            margin-left: -4px;
        }
        .exception-line .log-content {
            background-color: #f48771;
            color: #1e1e1e;
            padding: 2px 4px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>''' + file_path.name + '''</h2>
        <p>File size: ''' + f"{file_path.stat().st_size / (1024*1024):.2f}" + ''' MB | Lines: ''' + str(len(lines)) + '''</p>''']
    
    if target_line_idx is not None:
        html_lines.append(f'        <p>🎯 Exception highlighted at line {target_line_idx + 1}</p>')
    
    html_lines.append('    </div>')
    html_lines.append('    <div class="log-container">')
    
    # Add all lines
    for idx, line in enumerate(lines):
        line_content = line.rstrip('\n')
        # Escape HTML
        line_content = line_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Apply anti-doxx masking if needed
        if anti_doxx:
            for real, masked in mask_map.items():
                line_content = line_content.replace(real, masked)
        
        line_number = idx + 1
        is_exception_line = (idx == target_line_idx)
        
        if is_exception_line:
            html_lines.append(
                f'<div class="log-line exception-line" id="target-line">'
                f'<span class="log-line-number">{line_number}</span>'
                f'<span class="log-content">{line_content}</span>'
                f'</div>'
            )
        else:
            html_lines.append(
                f'<div class="log-line">'
                f'<span class="log-line-number">{line_number}</span>'
                f'{line_content}'
                f'</div>'
            )
    
    html_lines.append('    </div>')
    
    # Add JavaScript to scroll to target line
    if target_line_idx is not None:
        html_lines.append('''
    <script>
    window.addEventListener('load', function() {
        var target = document.getElementById('target-line');
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
    </script>''')
    
    html_lines.append('</body>')
    html_lines.append('</html>')
    
    return '\n'.join(html_lines)


def _clear_all_tables(conn):
    """Clear all data from database tables"""
    tables = ["miners", "backward_events", "loss_events", "state_events",
              "exceptions", "optimization_events", "resource_events", "registration_events"]
    for table in tables:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()

def _load_and_process_files(conn, uploaded_files, file_paths, progress_callback=None):
    """Load files into DB and build indices"""
    import streamlit as st
    from minerlogs.ingest import ingest_uploaded_files

    st.session_state["uploaded_file_paths"] = file_paths
    if "report_index" in st.session_state:
        del st.session_state["report_index"]

    with st.spinner("Building file metadata..."):
        st.session_state["file_metadata"] = _build_file_metadata(file_paths)

    with st.spinner("Indexing log files for fast context extraction..."):
        st.session_state["report_index"] = _build_header_indices(file_paths)

    with st.spinner("Parsing logs..."):
        total_files, total_lines, counters = ingest_uploaded_files(
            conn=conn,
            uploaded_files=uploaded_files,
            tz_name=TZ_NAME,
            progress_callback=progress_callback,
        )

    st.success(f"Parsed {total_files} file(s), ~{total_lines:,} line(s).")
    st.write("**Events captured:**")
    for event_type, count in counters.items():
        st.write(f"- {event_type}: {count:,}")

def init_session():
    import streamlit as st
    from minerlogs.db import ensure_schema, get_connection, migrate_normalize_messages

    def cleanup_session():
        temp_dir = st.session_state.get("temp_dir")
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        db_path = st.session_state.get("db_path")
        if db_path and Path(db_path).exists():
            try:
                conn = st.session_state.get("conn")
                if conn:
                    conn.close()
                Path(db_path).unlink()
            except Exception:
                pass

    if "session_initialized" not in st.session_state:
        cleanup_session()

        temp_dir = tempfile.mkdtemp(prefix="id_iota_")
        st.session_state["temp_dir"] = temp_dir

        db_path = Path(temp_dir) / "logs.db"
        st.session_state["db_path"] = str(db_path)
        conn = get_connection(db_path)
        ensure_schema(conn)
        migrate_normalize_messages(conn)
        st.session_state["conn"] = conn

        st.session_state["session_initialized"] = True
        st.session_state["uploaded_file_paths"] = []
        st.session_state["file_metadata"] = {}
        st.session_state["time_selection"] = {"start": None, "end": None, "active": False}
        st.session_state["filter_expander_open"] = False
        st.session_state["exception_filters"] = ["Miner is moving state from"]

        atexit.register(cleanup_session)


# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_sidebar(conn):
    import streamlit as st

    logs_dir = os.getenv("LOGS_DIR", "")

    if logs_dir and os.path.isdir(logs_dir):
        st.info(f"📁 Auto-loading from: `{logs_dir}`")
        if st.button("Reload logs from folder", type="primary", width='stretch'):
            log_files = sorted(glob.glob(os.path.join(logs_dir, "**/*.log"), recursive=True))

            if not log_files:
                st.warning(f"No .log files found in {logs_dir}")
            else:
                _clear_all_tables(conn)

                temp_dir = Path(st.session_state["temp_dir"])
                for old_file in st.session_state.get("uploaded_file_paths", []):
                    if old_file.exists():
                        old_file.unlink()

                class FileWrapper:
                    def __init__(self, name, content):
                        self.name = name
                        self._content = content
                    def getvalue(self):
                        return self._content

                file_paths, uploaded_files = [], []
                for log_path in log_files:
                    with open(log_path, "rb") as f:
                        content = f.read()
                    uploaded_files.append(FileWrapper(os.path.basename(log_path), content))
                    dest_path = temp_dir / os.path.basename(log_path)
                    dest_path.write_bytes(content)
                    file_paths.append(dest_path)

                _load_and_process_files(conn, uploaded_files, file_paths, lambda msg: st.sidebar.write(msg))

        st.divider()
        st.caption("Or upload files manually:")

    uploaded = st.file_uploader("Drop .log files here", type=["log"], accept_multiple_files=True)
    if st.button("Load uploaded logs", type="primary" if not logs_dir else "secondary", width='stretch'):
        if not uploaded:
            st.warning("Please upload at least one log file.")
        else:
            _clear_all_tables(conn)

            temp_dir = Path(st.session_state["temp_dir"])
            for old_file in st.session_state.get("uploaded_file_paths", []):
                if old_file.exists():
                    old_file.unlink()

            file_paths = []
            for idx, uf in enumerate(uploaded, start=1):
                name = getattr(uf, "name", f"uploaded_{idx}.log")
                file_path = temp_dir / name
                file_path.write_bytes(uf.getvalue())
                file_paths.append(file_path)

            _load_and_process_files(conn, uploaded, file_paths, lambda msg: st.sidebar.write(msg))


def render_miner_selector(conn, anti_doxx, mask_map_full):
    import streamlit as st
    from minerlogs.queries import list_miners

    miners_available = list_miners(conn)
    miners_filtered = [m for m in miners_available if isinstance(m, str) and len(m) == 8 and m not in ["shutdown"]]

    if not miners_filtered:
        st.info("No logs with valid 8-char hotkeys yet. Upload log files in the sidebar and click **Load uploaded logs**.")
        st.stop()

    def mask_hotkey(h: str) -> str:
        if not anti_doxx:
            return h
        return mask_map_full.get(h, "hidden")

    selected_miners: List[str] = st.multiselect(
        "Select miner(s) (hotkeys)",
        options=sorted(miners_filtered),
        default=sorted(miners_filtered),
        format_func=mask_hotkey,
    )

    if not selected_miners:
        st.warning("Select at least one miner.")
        st.stop()

    return selected_miners


def render_emissions_section():
    import streamlit as st
    import bittensor as bt

    @st.cache_data(ttl=300)
    def fetch_emission_data(network: str = "finney", netuid: int = 9) -> Dict[str, Dict[str, float]]:
        try:
            subtensor = bt.subtensor(network)
            metagraph = subtensor.metagraph(netuid, lite=True)

            emission_map = {}
            for uid in metagraph.uids.tolist():
                hotkey = metagraph.hotkeys[uid]
                emission = metagraph.emission[uid].item() * 20
                emission_map[hotkey] = {
                    "uid": int(uid),
                    "emission": emission
                }

            return emission_map
        except Exception as e:
            st.warning(f"Failed to fetch emission data: {e}")
            return {}

    emission_map = {}
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        fetch_emissions = st.checkbox("Fetch emissions", value=False, help="Fetch emission data from blockchain (cached for 5 min)")
    with col2:
        if fetch_emissions:
            netuid = st.number_input("NetUID", value=9, min_value=1, max_value=100, step=1)
    with col3:
        if fetch_emissions:
            network = st.selectbox("Network", ["finney", "test"], index=0)

    if fetch_emissions:
        with st.spinner("Fetching emission data from blockchain..."):
            emission_map = fetch_emission_data(network, netuid)
        if emission_map:
            st.success(f"✅ Loaded emissions for {len(emission_map)} hotkeys")

    return emission_map


def render_summary_table(conn, selected_miners, emission_map, anti_doxx, mask_hotkey):
    import streamlit as st
    from minerlogs.queries import build_last_seen_summary

    def match_hotkey_emission(short_hotkey: str, emission_map: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
        for full_hotkey, data in emission_map.items():
            if full_hotkey.startswith(short_hotkey):
                return data
        return None

    summary_df = build_last_seen_summary(conn, TZ_NAME, miners=selected_miners)
    if not summary_df.empty:
        if emission_map:
            emission_data = summary_df["miner_hotkey"].apply(
                lambda hk: match_hotkey_emission(hk, emission_map)
            )

            summary_df["uid"] = emission_data.apply(
                lambda x: int(x["uid"]) if x is not None else None
            )

            summary_df["Alpha per 1D"] = emission_data.apply(
                lambda x: x["emission"] if x is not None else 0.0
            )
            summary_df["Alpha per 1D"] = summary_df["Alpha per 1D"].fillna(0.0)
            summary_df["Alpha per 1D"] = summary_df["Alpha per 1D"].round(2)

        if anti_doxx:
            summary_df = summary_df.copy()
            summary_df["miner_hotkey"] = summary_df["miner_hotkey"].map(mask_hotkey)

        if emission_map:
            column_order = ["miner_hotkey", "uid", "last_layer", "last_state", "last_seen", "Alpha per 1D"]
            summary_df = summary_df[column_order]

        st.dataframe(
            summary_df.reset_index(drop=True),
            hide_index=True,
            width='stretch',
        )
    else:
        st.info("No state info yet for selected miners.")


def render_backward_chart(bw_df, loss_df, exc_df, time_selection, anti_doxx, mask_map_full):
    import streamlit as st

    st.subheader("Backward passes over time")

    def _plot_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        msk = df["miner_hotkey"].fillna("").astype(str).str.len() == 8
        return df[msk].copy()

    bw_plot = _plot_filter(bw_df)
    loss_plot = _plot_filter(loss_df)

    all_layers_available = set()
    if not bw_plot.empty:
        all_layers_available.update(bw_plot["layer"].dropna().unique())
    if not loss_plot.empty:
        all_layers_available.update(loss_plot["layer"].dropna().unique())
    all_layers_available = sorted([int(l) for l in all_layers_available if pd.notna(l)])

    if "selected_layers" not in st.session_state:
        st.session_state["selected_layers"] = set(all_layers_available)

    if all_layers_available:
        st.markdown("**Filter by layer:**")
        cols = st.columns(len(all_layers_available))

        for idx, layer in enumerate(all_layers_available):
            with cols[idx]:
                checked = layer in st.session_state["selected_layers"]
                new_checked = st.checkbox(f"Layer{layer}", value=checked, key=f"layer_cb_{layer}")
                if new_checked and layer not in st.session_state["selected_layers"]:
                    st.session_state["selected_layers"].add(layer)
                    st.rerun()
                elif not new_checked and layer in st.session_state["selected_layers"]:
                    st.session_state["selected_layers"].discard(layer)
                    st.rerun()

    if not bw_plot.empty:
        bw_plot = bw_plot[bw_plot["layer"].isin(st.session_state["selected_layers"])]
    if not loss_plot.empty:
        loss_plot = loss_plot[loss_plot["layer"].isin(st.session_state["selected_layers"])]

    if bw_plot.empty and loss_plot.empty:
        st.info("No backward or loss events found for selected miners and layers.")
        return

    fig = go.Figure()

    color_map = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    miner_idx = 0

    for miner in sorted(bw_plot["miner_hotkey"].unique()):
        if miner not in color_map:
            color_map[miner] = color_palette[miner_idx % len(color_palette)]
            miner_idx += 1

    for miner, mdf in bw_plot.groupby("miner_hotkey"):
        mdf = mdf.sort_values("ts_local")
        mdf = mdf.loc[mdf["since_reset"].ne(mdf["since_reset"].shift())]

        raw_label = mask_map_full.get(miner, miner) if anti_doxx else miner
        label = raw_label if (isinstance(raw_label, str) and raw_label.strip()) else "unknown"

        for layer in mdf["layer"].dropna().unique():
            layer_data = mdf[mdf["layer"] == layer]
            fig.add_trace(
                go.Scatter(
                    x=layer_data["ts_local"],
                    y=layer_data["since_reset"],
                    mode="lines+markers",
                    name=f"{label} (L{int(layer)})",
                    legendgroup=f"layer_{int(layer)}",
                    line=dict(color=color_map[miner]),
                    marker=dict(color=color_map[miner]),
                    hovertemplate="%{x}<br>backwards_since_reset=%{y}<extra></extra>",
                )
            )

    if not loss_plot.empty:
        for miner, ldf in loss_plot.groupby("miner_hotkey"):
            ldf = ldf.sort_values("ts_local")
            raw_label = mask_map_full.get(miner, miner) if anti_doxx else miner
            label = raw_label if (isinstance(raw_label, str) and raw_label.strip()) else "unknown"

            for layer in ldf["layer"].dropna().unique():
                layer_data = ldf[ldf["layer"] == layer]
                fig.add_trace(
                    go.Scatter(
                        x=layer_data["ts_local"],
                        y=layer_data["loss"],
                        mode="markers",
                        name=f"{label} (L{int(layer)} loss)",
                        legendgroup=f"layer_{int(layer)}",
                        yaxis="y2",
                        marker=dict(symbol="circle-open", size=8, line=dict(width=2, color=color_map.get(miner, color_palette[0]))),
                        hovertemplate="%{x}<br>loss=%{y:.4f}<extra></extra>",
                    )
                )

    if not exc_df.empty:
        fig.update_layout(
            yaxis3=dict(
                title="",
                overlaying="y",
                side="right",
                range=[0, 1],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
        )
        miner_masked = exc_df["miner_hotkey"].astype(str).map(lambda h: mask_map_full.get(h, h) if anti_doxx else h)

        def mask_text_hotkeys_local(text: str) -> str:
            if not anti_doxx or not isinstance(text, str):
                return text
            out = text
            for real, masked in mask_map_full.items():
                out = out.replace(real, masked)
            return out

        message_masked = exc_df["message"].astype(str).map(mask_text_hotkeys_local)
        custom = pd.concat([miner_masked, message_masked], axis=1).values

        fig.add_trace(
            go.Scatter(
                x=exc_df["ts_local"],
                y=[0.05] * len(exc_df),
                mode="markers",
                name="exceptions",
                marker=dict(symbol="x", size=10, color="red"),
                hovertemplate=("time=%{x}<br>miner=%{customdata[0]}<br>message=%{customdata[1]}<extra></extra>"),
                customdata=custom,
                yaxis="y3",
            )
        )

    if time_selection["active"]:
        fig.add_vrect(
            x0=time_selection["start"],
            x1=time_selection["end"],
            fillcolor="lightblue",
            opacity=0.2,
            layer="below",
            line_width=2,
            line_color="blue",
        )

    layout_kwargs = dict(
        title=None,
        xaxis_title="time",
        yaxis_title="backwards_since_reset",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=40, b=40),
        dragmode="select",
    )

    if not loss_plot.empty:
        fig.update_layout(
            **layout_kwargs,
            yaxis2=dict(title="loss", overlaying="y", side="right", showgrid=False),
        )
    else:
        fig.update_layout(**layout_kwargs)

    selected = st.plotly_chart(fig, width='stretch', on_select="rerun", selection_mode="box")

    if selected and selected.selection and "box" in selected.selection:
        box_selection = selected.selection["box"]
        if box_selection and len(box_selection) > 0:
            x_range = box_selection[0].get("x", [])
            if len(x_range) == 2:
                start_dt = pd.to_datetime(x_range[0])
                end_dt = pd.to_datetime(x_range[1])
                if start_dt > end_dt:
                    start_dt, end_dt = end_dt, start_dt
                st.session_state["time_selection"] = {
                    "start": start_dt.to_pydatetime(),
                    "end": end_dt.to_pydatetime(),
                    "active": True,
                }
                st.rerun()


def render_time_range_widget(time_selection, file_metadata):
    import streamlit as st

    if not time_selection["active"]:
        return

    with st.container(border=True):
        st.markdown("### 📅 Active Time Range Filter")
        col1, col2 = st.columns([3, 1])
        with col1:
            start_str = time_selection["start"].strftime("%Y-%m-%d %H:%M:%S")
            end_str = time_selection["end"].strftime("%Y-%m-%d %H:%M:%S")
            duration = time_selection["end"] - time_selection["start"]
            hours = duration.total_seconds() / 3600
            st.write(f"**From:** {start_str}")
            st.write(f"**To:** {end_str}")
            st.write(f"**Duration:** {hours:.1f}h")

            if file_metadata:
                filtered_files = _get_files_in_range(
                    file_metadata,
                    time_selection["start"],
                    time_selection["end"],
                )
                total_files = len(file_metadata)
                st.write(f"**Files in range:** {len(filtered_files)} of {total_files}")

        with col2:
            if st.button("Clear Selection", width='stretch'):
                st.session_state["time_selection"] = {"start": None, "end": None, "active": False}
                st.rerun()

    if file_metadata:
        filtered_files = _get_files_in_range(
            file_metadata,
            time_selection["start"],
            time_selection["end"],
        )
        if filtered_files:
            st.markdown("### 📦 Download Filtered Data")
            col1, col2 = st.columns(2)
            time_range = (time_selection["start"], time_selection["end"])
            with col1:
                zip_data = _create_logs_zip(filtered_files, time_range)
                zip_name = _get_zip_filename(time_range)
                st.download_button(
                    label=f"⬇ Download Selected Logs ({len(filtered_files)} files)",
                    data=zip_data,
                    file_name=zip_name,
                    mime="application/zip",
                    width='stretch',
                    type="primary",
                )
            with col2:
                exc_df = st.session_state.get("current_exc_df", pd.DataFrame())
                if not exc_df.empty:
                    st.info(f"📄 Report will include {len(exc_df)} exceptions")
                else:
                    st.warning("No exceptions in selected time range")
        else:
            st.warning("No log files found in selected time range")


def render_gpu_section(res_df, anti_doxx, mask_hotkey):
    import streamlit as st

    st.divider()
    st.subheader("GPU Memory Usage")

    if not res_df.empty:
        gpu_df = res_df[res_df["event_type"].isin(["gpu_memory", "gpu_memory_usage"])]

        if not gpu_df.empty:
            peak_memory = gpu_df.groupby("miner_hotkey")["value_gb"].max().sort_values(ascending=False)

            if not peak_memory.empty:
                st.markdown("**Peak GPU Memory Usage:**")
                for miner, peak_gb in peak_memory.items():
                    display_miner = mask_hotkey(miner) if anti_doxx else miner
                    st.write(f"• **{display_miner}**: {peak_gb:.2f} GB")
            else:
                st.info("No GPU memory data available")
        else:
            st.info("No GPU memory events captured")
    else:
        st.info("No resource events captured")


def render_exception_filters(exc_df_raw):
    import streamlit as st

    filter_counts = _count_exceptions_per_filter(exc_df_raw, st.session_state["exception_filters"])
    expander_open = st.session_state.get("filter_expander_open", False)

    with st.expander(f"🔍 Exception Filters ({len(st.session_state['exception_filters'])} active)", expanded=expander_open):
        st.caption("Exceptions containing these patterns will be hidden from the display and report.")

        filters = st.session_state["exception_filters"]

        if filters:
            st.markdown("**Active filters:**")
            for idx, filter_text in enumerate(filters):
                col1, col2, col3 = st.columns([4, 1, 0.6])
                with col1:
                    display_text = filter_text if len(filter_text) <= 80 else filter_text[:77] + "..."
                    st.text(display_text)
                with col2:
                    count = filter_counts.get(filter_text, 0)
                    st.caption(f"🚫 {count}")
                with col3:
                    if st.button("✕", key=f"remove_filter_{idx}", help="Remove this filter"):
                        st.session_state["exception_filters"].pop(idx)
                        st.session_state["filter_expander_open"] = True
                        st.rerun()
        else:
            st.info("No active filters. All exceptions will be shown.")

        st.markdown("**Add new filter:**")
        new_filter = st.text_input(
            "Pattern to filter",
            placeholder="Enter text pattern to filter out...",
            label_visibility="collapsed",
            key="new_filter_input"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add Filter", type="primary"):
                if new_filter and new_filter.strip():
                    if new_filter in st.session_state["exception_filters"]:
                        st.warning("This filter already exists.")
                    else:
                        st.session_state["exception_filters"].append(new_filter)
                        st.session_state["filter_expander_open"] = True
                        st.success("Filter added!")
                        st.rerun()
                else:
                    st.warning("Please enter a filter pattern.")

    if not expander_open:
        st.session_state["filter_expander_open"] = False


def render_exceptions_section(exc_df, exc_df_raw, grouped, anti_doxx, file_paths, mask_hotkey, mask_text_hotkeys, mask_map_full):
    import streamlit as st

    st.subheader("Exceptions (grouped — most frequent first)")

    total_exc_raw = len(exc_df_raw)
    total_exc_filtered = len(exc_df)
    filtered_count = total_exc_raw - total_exc_filtered

    if filtered_count > 0:
        st.caption(f"Showing {total_exc_filtered} of {total_exc_raw} exceptions ({filtered_count} hidden by filters)")
    elif total_exc_filtered > 0:
        st.caption(f"Showing all {total_exc_filtered} exceptions (no filters applied)")

    time_selection = st.session_state.get("time_selection", {"start": None, "end": None, "active": False})

    if exc_df.empty:
        if time_selection["active"]:
            st.info("No exceptions captured for selected miners in the selected time range (after filtering).")
        else:
            st.info("No exceptions captured for selected miners (after filtering).")
        return

    show_normalized = st.checkbox(
        "Show normalized messages (groups similar exceptions together)",
        value=True,
        help="Normalized messages replace variable parts (UUIDs, IDs, timestamps) with placeholders for better grouping"
    )

    st.markdown("**Click 🚫 to filter, ▼ to expand and view individual exceptions:**")

    col_btn, col_expand, col_miners, col_count, col_msg = st.columns([0.4, 0.4, 1.5, 0.6, 5.1])
    with col_btn:
        st.markdown("**🚫**")
    with col_expand:
        st.markdown("**▼**")
    with col_miners:
        st.markdown("**Miners**")
    with col_count:
        st.markdown("**Count**")
    with col_msg:
        st.markdown("**Message**")

    for idx, row in grouped.iterrows():
        matching_exceptions = exc_df[exc_df["message_normalized"] == row["message_normalized"]].copy()
        matching_exceptions = matching_exceptions.sort_values("ts_local", ascending=False)

        col_btn, col_expand, col_miners, col_count, col_msg = st.columns([0.4, 0.4, 1.5, 0.6, 5.1])

        with col_btn:
            filter_key = f"filter_{idx}"
            if st.button("🚫", key=filter_key, help="Add to filters", use_container_width=False):
                filter_text = str(row["message_normalized"])
                if filter_text not in st.session_state["exception_filters"]:
                    st.session_state["exception_filters"].append(filter_text)
                    st.rerun()

        with col_expand:
            expand_key = f"expand_{idx}"
            expand_state_key = f"expand_state_{idx}"
            is_expanded = st.session_state.get(expand_state_key, False)
            if st.button("▼" if not is_expanded else "▲", key=expand_key, help="Expand to see individual exceptions", use_container_width=False):
                st.session_state[expand_state_key] = not is_expanded
                st.rerun()

        with col_miners:
            miners_list = row["miner_hotkey"]
            if len(miners_list) <= 3:
                miner_names = [mask_hotkey(m) if anti_doxx else m for m in miners_list]
                miner_display = ", ".join(miner_names)
            else:
                miner_names = [mask_hotkey(m) if anti_doxx else m for m in miners_list[:2]]
                miner_display = f"{', '.join(miner_names)}... +{len(miners_list)-2}"
            st.text(miner_display)

        with col_count:
            st.text(str(int(row["count"])))

        with col_msg:
            if show_normalized:
                msg_display = mask_text_hotkeys(str(row["message_normalized"])) if anti_doxx else str(row["message_normalized"])
            else:
                msg_display = mask_text_hotkeys(str(row["message"])) if anti_doxx else str(row["message"])
            msg_short = msg_display if len(msg_display) <= 120 else msg_display[:117] + "..."
            st.text(msg_short)

        if st.session_state.get(expand_state_key, False):
            with st.container(border=True):
                st.caption(f"Individual occurrences ({len(matching_exceptions)} total):")

                display_count = min(50, len(matching_exceptions))

                col1_h, col2_h, col3_h, col4_h, col5_h = st.columns([0.5, 1.8, 1.5, 0.8, 3.4])
                with col1_h:
                    st.caption("**View**")
                with col2_h:
                    st.caption("**Timestamp**")
                with col3_h:
                    st.caption("**Miner**")
                with col4_h:
                    st.caption("**Line**")
                with col5_h:
                    st.caption("**Message**")

                for exc_idx, exc_row in matching_exceptions.head(display_count).iterrows():
                    col1, col2, col3, col4, col5 = st.columns([0.5, 1.8, 1.5, 0.8, 3.4])

                    line_num = exc_row.get("line_number")
                    source_file = exc_row.get("source_file", "")

                    with col1:
                        file_found = _match_source_file_to_path(source_file, file_paths)

                        if not file_found:
                            file_found = _find_file_containing_exception(
                                exc_row["ts_local"],
                                exc_row["miner_hotkey"],
                                str(exc_row.get("message", "")),
                                file_paths
                            )

                        if file_found and file_found.exists():
                            log_filename = file_found.name

                            modal_key = f"show_log_{idx}_{exc_idx}"
                            button_key = f"view_exp_{idx}_{exc_idx}"

                            tooltip = f"View log: {source_file} (line {line_num})" if source_file else f"View log (line {line_num})"
                            if st.button("📋", key=button_key, help=tooltip):
                                st.session_state[modal_key] = True
                                st.session_state[f"line_num_{modal_key}"] = line_num
                                st.session_state[f"source_file_{modal_key}"] = source_file
                                st.session_state[f"resolved_path_{modal_key}"] = file_found
                        elif source_file:
                            st.caption(f"⚠️ {source_file[:10]}...")

                    with col2:
                        st.text(exc_row["ts_local"].strftime("%Y-%m-%d %H:%M:%S"))

                    with col3:
                        miner_display = mask_hotkey(exc_row["miner_hotkey"]) if anti_doxx else exc_row["miner_hotkey"]
                        st.text(miner_display)

                    with col4:
                        st.text(f"L{line_num}" if line_num else "-")

                    with col5:
                        msg_display = mask_text_hotkeys(str(exc_row["message"])) if anti_doxx else str(exc_row["message"])
                        msg_short = msg_display if len(msg_display) <= 80 else msg_display[:77] + "..."
                        st.text(msg_short)

                if len(matching_exceptions) > display_count:
                    st.caption(f"Showing first {display_count} of {len(matching_exceptions)} occurrences")

    # Display log viewer modals
    for idx, row in grouped.iterrows():
        matching_exceptions = exc_df[exc_df["message_normalized"] == row["message_normalized"]].copy()
        matching_exceptions = matching_exceptions.sort_values("ts_local", ascending=False)
        display_count = min(50, len(matching_exceptions))

        for exc_idx, exc_row in matching_exceptions.head(display_count).iterrows():
            modal_key = f"show_log_{idx}_{exc_idx}"

            if st.session_state.get(modal_key, False):
                line_num = st.session_state.get(f"line_num_{modal_key}")
                source_file = st.session_state.get(f"source_file_{modal_key}", "")
                file_found = st.session_state.get(f"resolved_path_{modal_key}")

                if not file_found:
                    file_found = _match_source_file_to_path(source_file, file_paths)

                    if not file_found:
                        file_found = _find_file_containing_exception(
                            exc_row["ts_local"],
                            exc_row["miner_hotkey"],
                            str(exc_row.get("message", "")),
                            file_paths
                        )

                if file_found and file_found.exists():
                    st.markdown("---")
                    st.markdown(f"### 📋 Viewing: {file_found.name}" + (f" (line {line_num})" if line_num else ""))

                    col1, col2 = st.columns([6, 1])
                    with col2:
                        if st.button("Close", key=f"close_{modal_key}"):
                            del st.session_state[modal_key]
                            st.rerun()

                    html_content = _generate_log_html(
                        file_found,
                        exc_row["ts_local"],
                        exc_row["miner_hotkey"],
                        anti_doxx,
                        mask_map_full,
                        target_line_number=line_num
                    )

                    iframe_html = f'<div id="log_viewer_{idx}_{exc_idx}">{html_content}</div>'
                    st.components.v1.html(iframe_html, height=800, scrolling=True)

                    st.markdown(
                        f"""
                        <script>
                            setTimeout(function() {{
                                var viewer = document.getElementById('log_viewer_{idx}_{exc_idx}');
                                if (viewer) {{
                                    viewer.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                                }}
                            }}, 100);
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
                else:
                    st.markdown("---")
                    st.error(f"❌ Cannot display log viewer: File '{source_file}' not found or no longer exists.")
                    st.caption(f"Expected file: {source_file}")
                    st.caption(f"Available files: {', '.join(p.name for p in file_paths)}")
                    if st.button("Close", key=f"close_error_{modal_key}"):
                        del st.session_state[modal_key]
                        st.rerun()
                    st.markdown("---")


def render_report_section(grouped, exc_df, exc_df_raw, selected_miners, anti_doxx, file_paths, mask_hotkey, mask_text_hotkeys):
    import streamlit as st

    st.markdown("### Text report")
    gen_report = st.button("Generate exceptions report (.txt)", type="primary", width='stretch')

    if not gen_report:
        st.caption("Click the button to build the text report.")
        return

    if not file_paths:
        st.warning("No log files available. Please upload logs first.")
        st.stop()

    time_selection = st.session_state.get("time_selection", {"start": None, "end": None, "active": False})

    def _sig_tuple(row: pd.Series) -> Tuple:
        return (row["message_normalized"],)

    sig_to_exid: Dict[Tuple, int] = {}
    for i, row in grouped.iterrows():
        sig_to_exid[_sig_tuple(row)] = i + 1

    lines: List[str] = []
    lines.append("=== EXCEPTIONS REPORT ===")
    if time_selection["active"]:
        start_str = time_selection["start"].strftime("%Y-%m-%d %H:%M:%S")
        end_str = time_selection["end"].strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Time Range: {start_str} to {end_str}")
    lines.append(f"Selected miners: {', '.join(mask_hotkey(m) if anti_doxx else m for m in selected_miners)}")
    if st.session_state["exception_filters"]:
        lines.append(f"Active exception filters: {len(st.session_state['exception_filters'])}")
        for filt in st.session_state["exception_filters"]:
            lines.append(f"  - {filt}")
    total_exc_raw = len(exc_df_raw)
    if time_selection["active"]:
        lines.append(f"Total exceptions: {len(exc_df)} (filtered from {total_exc_raw})")
    else:
        lines.append(f"Total exceptions: {len(exc_df)} (filtered from {total_exc_raw})")
    lines.append("")

    for i, row in grouped.iterrows():
        ex_id = i + 1

        miners_list = row["miner_hotkey"]
        if anti_doxx:
            miners_display = f"{len(miners_list)} miner(s)"
        else:
            if len(miners_list) <= 5:
                miners_display = ", ".join(miners_list)
            else:
                miners_display = ", ".join(miners_list[:5]) + f" (+{len(miners_list)-5} more)"

        msg_norm = mask_text_hotkeys(str(row["message_normalized"])) if anti_doxx else str(row["message_normalized"])
        msg_example = mask_text_hotkeys(str(row["message"])) if anti_doxx else str(row["message"])

        lines.append(f"EX_ID {ex_id}: count={int(row['count'])} | miners={miners_display}")
        lines.append(f"Normalized: {msg_norm}")
        lines.append(f"Example: {msg_example}")
        lines.append("")

    lines.append("== Full exceptions ==")
    exc_sorted = exc_df.sort_values("ts_local")
    keep_chunks = []
    for _, g in exc_sorted.groupby("message_normalized", dropna=False):
        keep_chunks.append(g.tail(10) if len(g) > 10 else g)
    exc_filtered = pd.concat(keep_chunks).sort_values("ts_local")

    if len(exc_filtered) < len(exc_sorted):
        lines.append("NOTE: For exception types with > 10 occurrences, only the last 10 full exceptions are included below.")
        lines.append("")

    total_exc = len(exc_filtered)
    prog = st.progress(0, text="Gathering context from log files…")

    for idx, r in exc_filtered.reset_index(drop=True).iterrows():
        key = (r["message_normalized"] if pd.notna(r["message_normalized"]) else "",)
        ex_id = sig_to_exid.get(key, -1)
        level_val = str(r.get("level", ""))

        src_name, before, after, line_num = _find_exception_line_and_context(
            ts_local=r["ts_local"],
            level=level_val,
            message=str(r.get("message", "")),
            file_paths=file_paths,
            pre_lines=100,
            post_lines=25,
        )

        lines.append(f"EX_ID {ex_id}:")
        if src_name is None:
            lines.append("(context unavailable — header not found in logs)")
            flat_msg = mask_text_hotkeys(str(r["message"])) if anti_doxx else str(r["message"])
            miner_line = mask_hotkey(str(r["miner_hotkey"])) if anti_doxx else str(r["miner_hotkey"])
            lines.append(f"{r['ts_local']} | miner={miner_line} | message={flat_msg}")
            lines.append("")
        else:
            lines.append(f"(source: {src_name})")
            lines.append("(...) 100 lines before exception ")
            for ln in before:
                lines.append(mask_text_hotkeys(ln) if anti_doxx else ln)
            lines.append("(...) 25 lines after exception")
            for ln in after:
                lines.append(mask_text_hotkeys(ln) if anti_doxx else ln)
            lines.append("=" * 8)
            lines.append("")

        if total_exc:
            prog.progress((idx + 1) / total_exc, text=f"Gathering context… {idx + 1}/{total_exc}")

    prog.empty()

    if "report_index" in st.session_state:
        del st.session_state["report_index"]

    report_text = "\n".join(lines)

    st.download_button(
        label="Download exceptions report (.txt)",
        data=report_text,
        file_name="exceptions_report.txt",
        mime="text/plain",
        width='stretch',
    )


if __name__ == "__main__":
    import streamlit as st
    from minerlogs.queries import (
        list_miners,
        query_backward_events,
        query_losses,
        query_exceptions,
        query_resource_events,
    )

    # Setup
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon=_LOGO)
    init_session()
    conn = st.session_state["conn"]

    # Sidebar
    with st.sidebar:
        render_sidebar(conn)

    # Header
    with st.container(horizontal=True):
        st.image(_LOGO, width=177)
        with st.container():
            st.title(APP_TITLE)
            st.caption("Logs so clear, even an idiota gets it.")

    anti_doxx = st.checkbox("anti-doxx", value=False)

    # Miner selection
    miners_available = list_miners(conn)
    miners_filtered = [m for m in miners_available if isinstance(m, str) and len(m) == 8 and m not in ["shutdown"]]

    if not miners_filtered:
        st.info("No logs with valid 8-char hotkeys yet. Upload log files in the sidebar and click **Load uploaded logs**.")
        st.stop()

    def make_hotkey_mask(values: List[str]) -> Dict[str, str]:
        return {v: f"hotkey_{i + 1}" for i, v in enumerate(dict.fromkeys(values))}

    mask_map_full = make_hotkey_mask(sorted(miners_filtered))

    def mask_hotkey(h: str) -> str:
        if not anti_doxx:
            return h
        return mask_map_full.get(h, "hidden")

    selected_miners = render_miner_selector(conn, anti_doxx, mask_map_full)

    # Emissions
    emission_map = render_emissions_section()

    st.divider()

    # Summary table
    render_summary_table(conn, selected_miners, emission_map, anti_doxx, mask_hotkey)

    st.divider()

    # Query data
    bw_df = query_backward_events(conn, TZ_NAME, miners=selected_miners)
    loss_df = query_losses(conn, TZ_NAME, miners=selected_miners)
    res_df = query_resource_events(conn, TZ_NAME, miners=selected_miners)

    time_selection = st.session_state.get("time_selection", {"start": None, "end": None, "active": False})
    time_range = None
    if time_selection["active"]:
        time_range = (time_selection["start"], time_selection["end"])

    exc_df_raw = query_exceptions(conn, TZ_NAME, miners=selected_miners, time_range=time_range)
    exc_df = _apply_exception_filters(exc_df_raw, st.session_state["exception_filters"])

    # Store exc_df for time range widget
    st.session_state["current_exc_df"] = exc_df

    # Create mask_text_hotkeys function
    _hotkey_kv_rx = re.compile(r"('hotkey'\s*:\s*')([A-Za-z0-9]{8})(')")

    def mask_text_hotkeys(text: str) -> str:
        if not anti_doxx or not isinstance(text, str):
            return text

        def repl(m: re.Match) -> str:
            hv = m.group(2)
            masked = mask_hotkey(hv)
            return f"{m.group(1)}{masked}{m.group(3)}"

        out = _hotkey_kv_rx.sub(repl, text)

        for real, masked in mask_map_full.items():
            if real in selected_miners:
                out = out.replace(real, masked)
        return out

    # Backward passes chart
    render_backward_chart(bw_df, loss_df, exc_df, time_selection, anti_doxx, mask_map_full)

    # Time range widget
    render_time_range_widget(time_selection, st.session_state.get("file_metadata", {}))

    # GPU section
    render_gpu_section(res_df, anti_doxx, mask_hotkey)

    # Exception filters
    render_exception_filters(exc_df_raw)

    # Exceptions section
    grouped = (
        exc_df.groupby("message_normalized", dropna=False)
        .agg({
            "message": "first",
            "miner_hotkey": lambda x: list(x.unique()),
            "ts_local": "count",
        })
        .rename(columns={"ts_local": "count"})
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    file_paths = st.session_state.get("uploaded_file_paths", [])
    render_exceptions_section(exc_df, exc_df_raw, grouped, anti_doxx, file_paths, mask_hotkey, mask_text_hotkeys, mask_map_full)

    # Report section
    render_report_section(grouped, exc_df, exc_df_raw, selected_miners, anti_doxx, file_paths, mask_hotkey, mask_text_hotkeys)
