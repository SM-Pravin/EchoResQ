# modules/logger.py
import json
import os
from datetime import datetime, timezone
import traceback as _traceback

LOG_DIR = "logs"
JSON_LOG_FILE = os.path.join(LOG_DIR, "system.log")
TXT_LOG_FILE = os.path.join(LOG_DIR, "system.txt")
os.makedirs(LOG_DIR, exist_ok=True)

def log_call(caller_id, transcript, final_emotion, distress_token, scores, reason=None):
    # Use timezone-aware UTC timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "timestamp": timestamp,
        "caller_id": caller_id,
        "transcript": transcript,
        "final_emotion": final_emotion,
        "distress_token": distress_token,
        "scores": scores,
        "reason": reason
    }
    # Append structured JSON log
    with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Human-readable log
    with open(TXT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f" Emergency Call Log - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Caller ID     : {caller_id}\n")
        f.write(f"Transcript    : {transcript}\n")
        f.write(f"Final Emotion : {str(final_emotion).upper()}\n")
        f.write(f"Distress      : {str(distress_token).upper()}\n")
        if reason:
            f.write(f"Reason        : {reason}\n")
        f.write("="*50 + "\n")


def _write_json(entry: dict):
    with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_error(context: str, exc: Exception | None = None, extra: dict | None = None):
    """Log an error with optional exception and extra context."""
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "level": "ERROR",
        "context": context,
    }
    if exc is not None:
        payload["error"] = str(exc)
        payload["traceback"] = _traceback.format_exc()
    if extra:
        payload["extra"] = extra
    _write_json(payload)

    with open(TXT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[ERROR] {timestamp} | {context}\n")
        if exc is not None:
            f.write(f"  Exception: {exc}\n")
            f.write(f"  Traceback: {_traceback.format_exc()}\n")


def log_warning(context: str, extra: dict | None = None):
    """Log a warning with optional extra context."""
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "level": "WARNING",
        "context": context,
    }
    if extra:
        payload["extra"] = extra
    _write_json(payload)

    with open(TXT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[WARN ] {timestamp} | {context}\n")
