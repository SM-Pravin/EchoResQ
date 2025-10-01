# modules/logger.py
import json
import os
from datetime import datetime

LOG_DIR = "logs"
JSON_LOG_FILE = os.path.join(LOG_DIR, "system.log")
TXT_LOG_FILE = os.path.join(LOG_DIR, "system.txt")
os.makedirs(LOG_DIR, exist_ok=True)

def log_call(caller_id, transcript, final_emotion, distress_token, scores, reason=None):
    timestamp = datetime.utcnow().isoformat() + "Z"
    entry = {
        "timestamp": timestamp,
        "caller_id": caller_id,
        "transcript": transcript,
        "final_emotion": final_emotion,
        "distress_token": distress_token,
        "scores": scores,
        "reason": reason
    }
    with open(JSON_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Human readable
    with open(TXT_LOG_FILE, "a") as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f" Emergency Call Log - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Caller ID     : {caller_id}\n")
        f.write(f"Transcript    : {transcript}\n")
        f.write(f"Final Emotion : {final_emotion.upper()}\n")
        f.write(f"Distress      : {distress_token.upper()}\n")
        if reason:
            f.write(f"Reason        : {reason}\n")
        f.write("="*50 + "\n")

