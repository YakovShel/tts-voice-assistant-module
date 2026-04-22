from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from app.config import HISTORY_FILE

_HISTORY_LOCK = threading.Lock()


def log_history(record: dict) -> None:
    record = dict(record)
    if 'timestamp' not in record:
        record['timestamp'] = datetime.now(timezone.utc).isoformat()

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False) + '\n'
    with _HISTORY_LOCK:
        with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(payload)
            f.flush()


def read_history(limit: int = 20) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []

    records: list[dict] = []
    with _HISTORY_LOCK:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-limit:][::-1]
