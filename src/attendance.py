"""In-memory attendance log: records who was recognized and when, with a
cooldown so the same person isn't re-logged on every frame they appear in.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

DEFAULT_COOLDOWN_SECONDS = 60.0


@dataclass
class AttendanceEntry:
    name: str
    timestamp: float


class AttendanceLog:
    def __init__(self, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._entries: list[AttendanceEntry] = []
        self._last_seen: dict[str, float] = {}

    def mark_present(self, name: str, now: float | None = None) -> bool:
        """Record attendance for `name`. Returns True if a new entry was
        logged, False if suppressed by the cooldown window.
        """
        now = time.time() if now is None else now
        last = self._last_seen.get(name)
        if last is not None and (now - last) < self.cooldown_seconds:
            return False
        self._last_seen[name] = now
        self._entries.append(AttendanceEntry(name=name, timestamp=now))
        return True

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(columns=["name", "timestamp"])
        return pd.DataFrame(
            [
                {"name": e.name, "timestamp": pd.Timestamp(e.timestamp, unit="s")}
                for e in self._entries
            ]
        )

    def to_csv(self) -> str:
        return self.to_dataframe().to_csv(index=False)

    def __len__(self) -> int:
        return len(self._entries)
