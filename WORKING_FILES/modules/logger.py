# modules/logger.py
import json
import os
from datetime import datetime, timezone

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
