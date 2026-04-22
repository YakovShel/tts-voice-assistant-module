import json

from app.config import HISTORY_FILE


def compute_stats() -> dict:
    if not HISTORY_FILE.exists():
        return {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_hit_rate': 0,
            'avg_generation_time': 0,
            'file_requests': 0,
            'stream_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
        }

    total = 0
    cache_hits = 0
    durations = []
    file_requests = 0
    stream_requests = 0
    completed_requests = 0
    failed_requests = 0

    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if record.get('cache_hit'):
                cache_hits += 1
            duration = record.get('duration_sec')
            if isinstance(duration, (int, float)):
                durations.append(duration)
            mode = record.get('mode')
            if mode == 'file':
                file_requests += 1
            elif mode == 'stream':
                stream_requests += 1
            if record.get('completed'):
                completed_requests += 1
            else:
                failed_requests += 1

    avg_time = sum(durations) / len(durations) if durations else 0
    return {
        'total_requests': total,
        'cache_hits': cache_hits,
        'cache_hit_rate': round(cache_hits / total, 3) if total else 0,
        'avg_generation_time': round(avg_time, 3),
        'file_requests': file_requests,
        'stream_requests': stream_requests,
        'completed_requests': completed_requests,
        'failed_requests': failed_requests,
    }
