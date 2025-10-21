#!/usr/bin/env python3
from __future__ import annotations

import re
import sqlite3
import shutil
import tempfile
import atexit
import zipfile
import io
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bisect import bisect_left
from minerlogs.timeutil import parse_log_datetime

from minerlogs.db import ensure_schema, get_connection
from minerlogs.ingest import ingest_uploaded_files
from minerlogs.queries import (
    list_miners,
    query_backward_events,
    query_losses,
    query_states,
    query_exceptions,
    build_last_seen_summary,
)

APP_TITLE = "ID_IOTA — Interactive Diagnostic for IOTA"
_LOGO = Path(__file__).resolve().parent / "assets" / "logo.png"
TZ_NAME = "UTC" 

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon=_LOGO)

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

def _build_header_indices(file_paths: List[Path]) -> List[Dict[str, Any]]:
    idx_list = []
    for fpath in file_paths:
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
        idx_list.append({
            "name": fpath.name,
            "path": fpath,
            "keys": keys,
            "levels": levels,
            "line_idx": line_idx,
        })
    return idx_list

def _find_exception_line_and_context(
    ts_local: pd.Timestamp,
    level: str,
    message: str,
    file_paths: List[Path],
    pre_lines: int = 100,
    post_lines: int = 25,
) -> Tuple[Optional[str], List[str], List[str]]:
    if "report_index" not in st.session_state:
        with st.spinner("Indexing log files for context extraction..."):
            st.session_state["report_index"] = _build_header_indices(file_paths)
    
    idx_pack = st.session_state.get("report_index", [])
    key = _ts_key(ts_local.to_pydatetime())
    msg_prefix = (message or "").strip()[:80]
    
    for idxf in idx_pack:
        keys = idxf["keys"]
        if not keys:
            continue
        pos = bisect_left(keys, key)
        start = max(0, pos - 3)
        stop = min(len(keys), pos + 4)
        for j in range(start, stop):
            if keys[j] != key or idxf["levels"][j] != level:
                continue
            lines = _load_file_lines(idxf["path"])
            if not lines:
                continue
            li = idxf["line_idx"][j]
            line = lines[li]
            if msg_prefix and msg_prefix not in line:
                continue
            s = max(0, li - pre_lines)
            e = min(len(lines), li + 1 + post_lines)
            return idxf["name"], lines[s:li], lines[li:e]
    
    return None, [], []


def _build_file_metadata(file_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    metadata = {}
    for fpath in file_paths:
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
            metadata[fpath.name] = {
                "path": fpath,
                "min_timestamp": min(timestamps),
                "max_timestamp": max(timestamps),
                "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
            }
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
        if file_min >= start_dt and file_max <= end_dt:
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
            mask &= ~df["message"].astype(str).str.lower().str.contains(pattern_lower, regex=False, na=False)
    
    return df[mask].copy()


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

def init_session():
    if "session_initialized" not in st.session_state:
        cleanup_session()
        
        temp_dir = tempfile.mkdtemp(prefix="id_iota_")
        st.session_state["temp_dir"] = temp_dir
        
        db_path = Path(temp_dir) / "logs.db"
        st.session_state["db_path"] = str(db_path)
        conn = get_connection(db_path)
        ensure_schema(conn)
        st.session_state["conn"] = conn
        
        st.session_state["session_initialized"] = True
        st.session_state["uploaded_file_paths"] = []
        st.session_state["file_metadata"] = {}
        st.session_state["time_selection"] = {"start": None, "end": None, "active": False}
        st.session_state["exception_filters"] = ["Miner is moving state from LayerPhase.TRAINING to LayerPhase.WEIGHTS_UPLOADING"]
        
        atexit.register(cleanup_session)

init_session()
conn = st.session_state["conn"]

with st.sidebar:
    uploaded = st.file_uploader("Drop .log files here", type=["log"], accept_multiple_files=True)
    if st.button("Load uploaded logs", type="primary", width='stretch'):
        if not uploaded:
            st.warning("Please upload at least one log file.")
        else:
            conn.execute("DELETE FROM miners")
            conn.execute("DELETE FROM backward_events")
            conn.execute("DELETE FROM loss_events")
            conn.execute("DELETE FROM state_events")
            conn.execute("DELETE FROM exceptions")
            conn.commit()
            
            temp_dir = Path(st.session_state["temp_dir"])
            for old_file in st.session_state.get("uploaded_file_paths", []):
                if old_file.exists():
                    old_file.unlink()
            
            file_paths = []
            for idx, uf in enumerate(uploaded, start=1):
                name = getattr(uf, "name", f"uploaded_{idx}.log")
                file_path = temp_dir / name
                with open(file_path, "wb") as f:
                    f.write(uf.getvalue())
                file_paths.append(file_path)
            
            st.session_state["uploaded_file_paths"] = file_paths
            
            if "report_index" in st.session_state:
                del st.session_state["report_index"]
            
            with st.spinner("Building file metadata..."):
                st.session_state["file_metadata"] = _build_file_metadata(file_paths)
            
            with st.spinner("Parsing uploaded logs..."):
                total_files, total_lines, counters = ingest_uploaded_files(
                    conn=conn,
                    uploaded_files=uploaded,
                    tz_name=TZ_NAME,
                    progress_callback=lambda msg: st.sidebar.write(msg),
                )
            st.success(f"Parsed {total_files} file(s), ~{total_lines:,} line(s).")

with st.container(horizontal=True):
    st.image(_LOGO, width=177)
    with st.container():
        st.title(APP_TITLE)
        st.caption("Logs so clear, even an idiota gets it.")

anti_doxx = st.checkbox("anti-doxx", value=False)

miners_available = list_miners(conn)
miners_filtered = [m for m in miners_available if isinstance(m, str) and len(m) == 8 and m not in ["shutdown"]]

if not miners_filtered:
    st.info("No logs with valid 8-char hotkeys yet. Upload log files in the sidebar and click **Load uploaded logs**.")
    st.stop()

def make_hotkey_mask(values: List[str]) -> Dict[str, str]:
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    return {v: f"hotkey_{i + 1}" for i, v in enumerate(uniq)}

mask_map_full = make_hotkey_mask(sorted(miners_filtered))

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

st.divider()

summary_df = build_last_seen_summary(conn, TZ_NAME, miners=selected_miners)
if not summary_df.empty:
    if anti_doxx:
        summary_df = summary_df.copy()
        summary_df["miner_hotkey"] = summary_df["miner_hotkey"].map(mask_hotkey)
    st.dataframe(
        summary_df.reset_index(drop=True),
        hide_index=True,
        width='stretch',
    )
else:
    st.info("No state info yet for selected miners.")

st.divider()

bw_df = query_backward_events(conn, TZ_NAME, miners=selected_miners)
loss_df = query_losses(conn, TZ_NAME, miners=selected_miners)
state_df = query_states(conn, TZ_NAME, miners=selected_miners)

time_selection = st.session_state.get("time_selection", {"start": None, "end": None, "active": False})
time_range = None
if time_selection["active"]:
    time_range = (time_selection["start"], time_selection["end"])

exc_df_raw = query_exceptions(conn, TZ_NAME, miners=selected_miners, time_range=time_range)
exc_df = _apply_exception_filters(exc_df_raw, st.session_state["exception_filters"])

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

st.subheader("Backward passes over time")

def _plot_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    msk = df["miner_hotkey"].fillna("").astype(str).str.len() == 8
    return df[msk].copy()

bw_plot = _plot_filter(bw_df)
loss_plot = _plot_filter(loss_df)

if bw_plot.empty:
    st.info("No backward events found for selected miners.")
else:
    fig = go.Figure()

    for miner, mdf in bw_plot.groupby("miner_hotkey"):
        mdf = mdf.sort_values("ts_local")
        mdf = mdf.loc[mdf["since_reset"].ne(mdf["since_reset"].shift())]

        raw_label = mask_map_full.get(miner, miner) if anti_doxx else miner
        label = raw_label if (isinstance(raw_label, str) and raw_label.strip()) else "unknown"

        fig.add_trace(
            go.Scatter(
                x=mdf["ts_local"],
                y=mdf["since_reset"],
                mode="lines+markers",
                name=label,
                hovertemplate="%{x}<br>backwards_since_reset=%{y}<extra></extra>",
            )
        )

    if not loss_plot.empty:
        for miner, ldf in loss_plot.groupby("miner_hotkey"):
            ldf = ldf.sort_values("ts_local")
            raw_label = mask_map_full.get(miner, miner) if anti_doxx else miner
            label = raw_label if (isinstance(raw_label, str) and raw_label.strip()) else "unknown"

            fig.add_trace(
                go.Scatter(
                    x=ldf["ts_local"],
                    y=ldf["loss"],
                    mode="markers",
                    name=f"loss {label}",
                    yaxis="y2",
                    marker=dict(symbol="circle-open"),
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
        miner_masked = exc_df["miner_hotkey"].astype(str).map(mask_hotkey if anti_doxx else (lambda x: x))
        message_masked = exc_df["message"].astype(str).map(mask_text_hotkeys if anti_doxx else (lambda x: x))
        custom = pd.concat([miner_masked, message_masked], axis=1).values

        fig.add_trace(
            go.Scatter(
                x=exc_df["ts_local"],
                y=[0.05] * len(exc_df),
                mode="markers",
                name="exceptions",
                marker=dict(symbol="x", size=10),
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
        margin=dict(l=40, r=40, t=20, b=40),
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
    
    if time_selection["active"]:
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
                
                file_metadata = st.session_state.get("file_metadata", {})
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
        
        file_metadata = st.session_state.get("file_metadata", {})
        if file_metadata:
            filtered_files = _get_files_in_range(
                file_metadata,
                time_selection["start"],
                time_selection["end"],
            )
            if filtered_files:
                st.markdown("### 📦 Download Filtered Data")
                col1, col2 = st.columns(2)
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
                    if not exc_df.empty:
                        st.info(f"📄 Report will include {len(exc_df)} exceptions")
                    else:
                        st.warning("No exceptions in selected time range")
            else:
                st.warning("No log files found in selected time range")

st.divider()

with st.expander(f"🔍 Exception Filters ({len(st.session_state['exception_filters'])} active)", expanded=False):
    st.caption("Exceptions containing these patterns will be hidden from the display and report.")
    
    filters = st.session_state["exception_filters"]
    
    if filters:
        st.markdown("**Active filters:**")
        for idx, filter_text in enumerate(filters):
            col1, col2 = st.columns([5, 1])
            with col1:
                display_text = filter_text if len(filter_text) <= 100 else filter_text[:97] + "..."
                st.text(display_text)
            with col2:
                if st.button("✕", key=f"remove_filter_{idx}", help="Remove this filter"):
                    st.session_state["exception_filters"].pop(idx)
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
                    st.success("Filter added!")
                    st.rerun()
            else:
                st.warning("Please enter a filter pattern.")

st.subheader("Exceptions (grouped — most frequent first)")

total_exc_raw = len(exc_df_raw)
total_exc_filtered = len(exc_df)
filtered_count = total_exc_raw - total_exc_filtered

if filtered_count > 0:
    st.caption(f"Showing {total_exc_filtered} of {total_exc_raw} exceptions ({filtered_count} hidden by filters)")
elif total_exc_filtered > 0:
    st.caption(f"Showing all {total_exc_filtered} exceptions (no filters applied)")

if exc_df.empty:
    if time_selection["active"]:
        st.info("No exceptions captured for selected miners in the selected time range (after filtering).")
    else:
        st.info("No exceptions captured for selected miners (after filtering).")
else:
    group_cols = ["message", "miner_hotkey", "layer"]

    grouped = (
        exc_df[group_cols]
        .fillna("")
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    display_cols = ["miner_hotkey", "count", "message"]
    show_grouped = grouped[display_cols].copy()

    if anti_doxx:
        show_grouped["miner_hotkey"] = show_grouped["miner_hotkey"].astype(str).map(mask_hotkey)
        show_grouped["message"] = show_grouped["message"].astype(str).map(mask_text_hotkeys)

    st.dataframe(
        show_grouped,
        width='stretch',
        hide_index=True,
        column_config={
            "miner_hotkey": st.column_config.TextColumn("miner_hotkey"),
            "count": st.column_config.NumberColumn("count"),
            "message": st.column_config.TextColumn("message"),
        },
    )

    st.markdown("### Text report")
    gen_report = st.button("Generate exceptions report (.txt)", type="primary", width='stretch')

    if gen_report:
        file_paths: List[Path] = st.session_state.get("uploaded_file_paths", [])
        
        if not file_paths:
            st.warning("No log files available. Please upload logs first.")
            st.stop()

        time_selection = st.session_state.get("time_selection", {"start": None, "end": None, "active": False})

        def _sig_tuple(row: pd.Series) -> Tuple:
            return tuple(row[c] for c in ["message", "miner_hotkey", "layer"])
        
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
        if time_selection["active"]:
            lines.append(f"Total exceptions: {len(exc_df)} (filtered from {total_exc_raw})")
        else:
            lines.append(f"Total exceptions: {len(exc_df)} (filtered from {total_exc_raw})")
        lines.append("")

        for i, row in grouped.iterrows():
            ex_id = i + 1
            miner_line = mask_hotkey(str(row["miner_hotkey"])) if anti_doxx else str(row["miner_hotkey"])
            layer = str(row["layer"])
            msg = mask_text_hotkeys(str(row["message"])) if anti_doxx else str(row["message"])
            lines.append(f"EX_ID {ex_id}: count={int(row['count'])} | miner={miner_line} | layer={layer}")
            lines.append(f"message: {msg}")
            lines.append("")

        lines.append("== Full exceptions ==")
        exc_sorted = exc_df.sort_values("ts_local")
        keep_chunks = []
        for _, g in exc_sorted.groupby(group_cols, dropna=False):
            keep_chunks.append(g.tail(10) if len(g) > 10 else g)
        exc_filtered = pd.concat(keep_chunks).sort_values("ts_local")
        
        if len(exc_filtered) < len(exc_sorted):
            lines.append("NOTE: For exception types with > 10 occurrences, only the last 10 full exceptions are included below.")
            lines.append("")

        total_exc = len(exc_filtered)
        prog = st.progress(0, text="Gathering context from log files…")

        for idx, r in exc_filtered.reset_index(drop=True).iterrows():
            key = tuple((r[c] if pd.notna(r[c]) else "") for c in group_cols)
            ex_id = sig_to_exid.get(key, -1)
            level_val = str(r.get("level", ""))

            src_name, before, after = _find_exception_line_and_context(
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
                lines.append(f"{r['ts_local']} | miner={miner_line} | layer={r['layer']} | message={flat_msg}")
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
    else:
        st.caption("Click the button to build the text report.")