# -*- coding: utf-8 -*-
"""
Time parsing and window utilities for timeline processing.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Any


def parse_time_any(x: Any) -> Optional[datetime]:
    """
    Parse time from various formats:
      - ISO string: '2023-05-01T12:00:00+00:00' or '2023-05-01'
      - List: [YYYY, M, D, ...] (only first 3 elements used)
      - Invalid inputs -> None
    """
    if x is None:
        return None
    
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        try:
            y, m, d = int(x[0]), int(x[1]), int(x[2])
            return datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            return None
    
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            if "T" in s:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.strptime(s, "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    
    return None


def midpoint(a: datetime, b: datetime) -> datetime:
    """Calculate the midpoint between two datetimes."""
    if a > b:
        a, b = b, a
    delta = b - a
    return a + delta / 2


def build_windows(sorted_times: List[datetime]) -> List[Tuple[datetime, datetime]]:
    """
    Build K half-open intervals [L_k, R_k) from sorted gold node times.
    
    Boundaries:
      - L_0 = -infinity (1900-01-01)
      - R_{K-1} = +infinity (2999-01-01)
      - R_k = midpoint(t_k, t_{k+1}); L_k = R_{k-1}
    
    Args:
        sorted_times: List of datetime objects in ascending order
        
    Returns:
        List of (L, R) tuples representing time windows
    """
    if not sorted_times:
        return []
    
    INF_PAST = datetime(1900, 1, 1, tzinfo=timezone.utc)
    INF_FUTR = datetime(2999, 1, 1, tzinfo=timezone.utc)
    
    K = len(sorted_times)
    mids = [midpoint(sorted_times[i], sorted_times[i+1]) for i in range(K-1)]
    
    Ls = [INF_PAST] + mids
    Rs = mids + [INF_FUTR]
    
    return list(zip(Ls, Rs))


def in_window(dt: datetime, win: Tuple[datetime, datetime]) -> bool:
    """Check if a datetime falls within a time window [L, R)."""
    L, R = win
    return (dt >= L) and (dt < R)


def daydiff(a: datetime, b: datetime) -> int:
    """Calculate absolute day difference between two datetimes."""
    return abs(int((a - b).total_seconds() // 86400))


def same_day(a: datetime, b: datetime) -> bool:
    """Check if two datetimes are on the same day (UTC)."""
    a = a.astimezone(timezone.utc)
    b = b.astimezone(timezone.utc)
    return (a.year, a.month, a.day) == (b.year, b.month, b.day)
