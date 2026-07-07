from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

_REQUEST_PATH = Path("./data/amnesia_request.json")


def request_amnesia() -> str:
    """Called by the dashboard. Records a new wipe request and returns its id."""
    _REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex[:8]
    tmp = _REQUEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"id": request_id, "ts": time.time()}), encoding="utf-8")
    tmp.replace(_REQUEST_PATH)
    return request_id


def pop_amnesia_request(last_seen_id: str | None) -> str | None:
    """Called by the workload. Returns a new request id once, else None."""
    if not _REQUEST_PATH.exists():
        return None
    try:
        data = json.loads(_REQUEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    request_id = data.get("id")
    if request_id and request_id != last_seen_id:
        return request_id
    return None


def clear_amnesia_request() -> None:
    if _REQUEST_PATH.exists():
        try:
            _REQUEST_PATH.unlink()
        except OSError:
            pass
