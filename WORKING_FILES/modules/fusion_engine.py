# modules/fusion_engine.py
"""
Fuse audio & text emotion dicts into canonical labels.
Returns: (confidence: float, final_emotion: str, fused_scores: dict)
canonical labels: 'angry', 'happy', 'neutral', 'sad'
"""
import numpy as np

CANONICAL = ["angry", "happy", "neutral", "sad"]

def _normalize_to_canonical(scores: dict):
    """Map arbitrary model labels to canonical labels (take max match)."""
    out = {k: 0.0 for k in CANONICAL}
    if not scores:
        return out

    for label, score in scores.items():
        l = label.lower()
        if "ang" in l or "anger" in l:
            out["angry"] = max(out["angry"], score)
        elif "joy" in l or "happy" in l:
            out["happy"] = max(out["happy"], score)
        elif "neu" in l or "neutral" in l:
            out["neutral"] = max(out["neutral"], score)
        elif "sad" in l or "sadness" in l:
            out["sad"] = max(out["sad"], score)
        elif "fear" in l:
            # map fear -> sad (conservative)
            out["sad"] = max(out["sad"], score * 0.9)
        else:
            # if label unknown, try to push into neutral conservatively
            out["neutral"] = max(out["neutral"], score * 0.5)
    return out

def fuse_emotions(audio_scores: dict, text_scores: dict, w_audio=0.5, w_text=0.5):
    a = _normalize_to_canonical(audio_scores or {})
    t = _normalize_to_canonical(text_scores or {})

    fused = {k: w_audio * a.get(k, 0.0) + w_text * t.get(k, 0.0) for k in CANONICAL}
    total = sum(fused.values())
    if total > 0:
        for k in fused:
            fused[k] = fused[k] / total
        final = max(fused, key=fused.get)
        confidence = float(fused[final])
        return confidence, final, fused
    else:
        # nothing available
        return 0.0, "unknown", {k: 0.0 for k in CANONICAL}

