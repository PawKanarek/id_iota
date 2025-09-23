#!/usr/bin/env python3
from __future__ import annotations

import re
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from bisect import bisect_left
from minerlogs.timeutil import parse_log_datetime  # reuse your parser

# Local modules
from minerlogs.db import ensure_schema
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

def _build_header_indices(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx_list = []
    for f in files:
        lines = f.get("lines", [])
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
            "name": f.get("name", "unknown"),
            "lines": lines,
            "keys": keys,
            "levels": levels,
            "line_idx": line_idx,
        })
    return idx_list

def _ensure_index() -> None:
    """Build the per-file timestamp index only when needed (or when files change)."""
    files = st.session_state.get("uploaded_texts", [])
    fp = tuple((f.get("name"), len(f.get("lines", []))) for f in files)
    if st.session_state.get("index_fingerprint") == fp and st.session_state.get("uploaded_index"):
        return
    with st.spinner("Indexing log headers for fast reporting…"):
        st.session_state["uploaded_index"] = _build_header_indices(files)
        st.session_state["index_fingerprint"] = fp
    
    
def _find_exception_line_and_context(
    ts_local: pd.Timestamp,
    level: str,
    message: str,
    files: List[Dict[str, Any]],
    pre_lines: int = 100,
    post_lines: int = 25,
) -> Tuple[Optional[str], List[str], List[str]]:
    # Fast path with index
    idx_pack = st.session_state.get("uploaded_index", [])
    if idx_pack:
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
                li = idxf["line_idx"][j]
                line = idxf["lines"][li]
                if msg_prefix and msg_prefix not in line:
                    continue
                s = max(0, li - pre_lines)
                e = min(len(idxf["lines"]), li + 1 + post_lines)
                return idxf["name"], idxf["lines"][s:li], idxf["lines"][li:e]

    # Slow fallback (rare)
    rx = _ts_regex_for_line(ts_local, level)
    msg_prefix = (message or "").strip()[:80]
    msg_prefix_esc = re.escape(msg_prefix) if msg_prefix else None
    for f in files:
        lines = f.get("lines", [])
        if not lines:
            continue
        for i, line in enumerate(lines):
            if rx.match(line):
                if msg_prefix_esc and msg_prefix and re.search(msg_prefix_esc, line) is None:
                    continue
                s = max(0, i - pre_lines)
                return f.get("name", "unknown"), lines[s:i], lines[i : min(len(lines), i + 1 + post_lines)]
    return None, [], []


# DB connection
if "conn" not in st.session_state:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    ensure_schema(conn)
    st.session_state["conn"] = conn

conn = st.session_state["conn"]

# Sidebar: uploader
with st.sidebar:
    uploaded = st.file_uploader("Drop .log files here", type=["log"], accept_multiple_files=True)
    if st.button("Load uploaded logs", type="primary", width='stretch'):
        if not uploaded:
            st.warning("Please upload at least one log file.")
        else:
            # Keep raw texts for context-rich TXT reports (±N lines extraction)
            raw_texts = []
            for idx, uf in enumerate(uploaded, start=1):
                name = getattr(uf, "name", f"uploaded_{idx}")
                try:
                    text = uf.getvalue().decode("utf-8", errors="replace")
                except Exception:
                    text = ""
                raw_texts.append({"name": name, "text": text, "lines": text.splitlines()})
            st.session_state["uploaded_texts"] = raw_texts

            # Ingest uploads
            with st.spinner("Parsing uploaded logs..."):
                total_files, total_lines, counters = ingest_uploaded_files(
                    conn=conn,
                    uploaded_files=uploaded,
                    tz_name=TZ_NAME,
                    progress_callback=lambda msg: st.sidebar.write(msg),
                )
            st.success(f"Parsed {total_files} file(s), ~{total_lines:,} line(s). ")

# --------------------------------------------------------------------
# Main page
# ---------------- ---------------------------------------------------- 
with st.container(horizontal=True, horizontal_alignment="left"):
    st.image(_LOGO, width=177)
    with st.container():
        st.title(APP_TITLE)
        st.caption("Logs so clear, even an idiota gets it.")

anti_doxx = st.checkbox("anti-doxx", value=False)

# Miner list
miners_available = list_miners(conn)
miners_filtered = [m for m in miners_available if isinstance(m, str) and len(m) == 8 and m not in ["shutdown"]]

if not miners_filtered:
    st.info("No logs with valid 8-char hotkeys yet. Upload log files in the sidebar and click **Load uploaded logs**.")
    st.stop()

# Controls row: just the multiselect; 
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

# --------------------------------------------------------------------
# Summary table. Mask hotkeys if anti-doxx.
# --------------------------------------------------------------------
summary_df = build_last_seen_summary(conn, TZ_NAME, miners=selected_miners)
if not summary_df.empty:
    if anti_doxx:
        summary_df = summary_df.copy()
        summary_df["miner_hotkey"] = summary_df["miner_hotkey"].map(mask_hotkey)
    st.dataframe(
        summary_df.reset_index(drop=True),
        hide_index=True,
        width="stretch",
    )
else:
    st.info("No state info yet for selected miners.")

st.divider()

# --------------------------------------------------------------------
# Query full timeline
# --------------------------------------------------------------------
bw_df = query_backward_events(conn, TZ_NAME, miners=selected_miners)
loss_df = query_losses(conn, TZ_NAME, miners=selected_miners)
state_df = query_states(conn, TZ_NAME, miners=selected_miners)
exc_df = query_exceptions(conn, TZ_NAME, miners=selected_miners)  # use directly

# Helper: mask hotkeys inside arbitrary text
_hotkey_kv_rx = re.compile(r"('hotkey'\s*:\s*')([A-Za-z0-9]{8})(')")

def mask_text_hotkeys(text: str) -> str:
    if not anti_doxx or not isinstance(text, str):
        return text

    def repl(m: re.Match) -> str:
        hv = m.group(2)
        masked = mask_hotkey(hv)
        return f"{m.group(1)}{masked}{m.group(3)}"

    out = _hotkey_kv_rx.sub(repl, text)

    # Also mask any selected miner occurrences verbatim
    for real, masked in mask_map_full.items():
        if real in selected_miners:
            out = out.replace(real, masked)
    return out

# --------------------------------------------------------------------
# Chart: Backwards since reset over time
# --------------------------------------------------------------------
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

    # Backward traces
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

    # Loss points
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

    # Exception markers
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
        custom = pd.concat(
            [
                miner_masked,
                message_masked,
            ],
            axis=1,
        ).values

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

    # Layout
    layout_kwargs = dict(
        title=None,
        xaxis_title="time",
        yaxis_title="backwards_since_reset",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    if not loss_plot.empty:
        fig.update_layout(
            **layout_kwargs,
            yaxis2=dict(title="loss", overlaying="y", side="right", showgrid=False),
        )
    else:
        fig.update_layout(**layout_kwargs)

    st.plotly_chart(fig, width='stretch')

# --------------------------------------------------------------------
# Exceptions table
# --------------------------------------------------------------------
st.subheader("Exceptions (grouped — most frequent first)")
if exc_df.empty:
    st.info("No exceptions captured for selected miners.")
else:
    # Simplified grouping signature
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

    # Display: miner_hotkey, count, message
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
    # --- Keep your grouped table code above ---

    st.markdown("### Text report")
    gen_report = st.button("Generate exceptions report (.txt)", type="primary", use_container_width=True)

    if gen_report:
        _ensure_index()  # build the index only now

        uploaded_texts: List[Dict[str, Any]] = st.session_state.get("uploaded_texts", [])

        # Map groups to EX_IDs
        def _sig_tuple(row: pd.Series) -> Tuple:
            return tuple(row[c] for c in ["message", "miner_hotkey", "layer"])
        sig_to_exid: Dict[Tuple, int] = {}
        for i, row in grouped.iterrows():
            sig_to_exid[_sig_tuple(row)] = i + 1

        lines: List[str] = []
        lines.append("=== EXCEPTIONS REPORT ===")
        lines.append(f"Selected miners: {', '.join(mask_hotkey(m) if anti_doxx else m for m in selected_miners)}")
        lines.append(f"Total exceptions: {len(exc_df)}")
        lines.append("")

        # Groups with counts (unchanged)
        for i, row in grouped.iterrows():
            ex_id = i + 1
            miner_line = mask_hotkey(str(row["miner_hotkey"])) if anti_doxx else str(row["miner_hotkey"])
            layer = str(row["layer"])
            msg = mask_text_hotkeys(str(row["message"])) if anti_doxx else str(row["message"])
            lines.append(f"EX_ID {ex_id}: count={int(row['count'])} | miner={miner_line} | layer={layer}")
            lines.append(f"message: {msg}")
            lines.append("")

        # --- Only last 10 full contexts per type ---
        lines.append("== Full exceptions ==")
        group_cols = ["message", "miner_hotkey", "layer"]
        exc_sorted = exc_df.sort_values("ts_local")
        keep_chunks = []
        for _, g in exc_sorted.groupby(group_cols, dropna=False):
            keep_chunks.append(g.tail(10) if len(g) > 10 else g)
        exc_filtered = pd.concat(keep_chunks).sort_values("ts_local")
        if len(exc_filtered) < len(exc_sorted):
            lines.append("NOTE: For exception types with > 10 occurrences, only the last 10 full exceptions are included below.")
            lines.append("")

        total_exc = len(exc_filtered)
        prog = st.progress(0, text="Gathering context from raw logs…")

        for idx, r in exc_filtered.reset_index(drop=True).iterrows():
            key = tuple((r[c] if pd.notna(r[c]) else "") for c in group_cols)
            ex_id = sig_to_exid.get(key, -1)
            level_val = str(r.get("level", ""))

            src_name, before, after = _find_exception_line_and_context(
                ts_local=r["ts_local"],
                level=level_val,
                message=str(r.get("message", "")),
                files=uploaded_texts,
                pre_lines=100,
                post_lines=25,
            )

            lines.append(f"EX_ID {ex_id}:")
            if src_name is None:
                lines.append("(context unavailable — raw logs not loaded in this session or header not found)")
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
        report_text = "\n".join(lines)

        st.download_button(
            label="Download exceptions report (.txt)",
            data=report_text,
            file_name="exceptions_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.caption("Click the button to build the text report.")

