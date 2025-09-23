from __future__ import annotations

from datetime import datetime

# Centralized log timestamp parsing (local, naive). We accept 0–6 microseconds.
# Examples:
# - "2025-09-20 16:05:01.597803"
# - "2025-09-20 16:05:01"
def parse_log_datetime(ts: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None
