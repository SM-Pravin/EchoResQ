# modules/emotion_text.py
import numpy as np
from modules.model_loader import text_classifier

def analyze_text_emotion(text: str):
    """
    Returns: dict[label -> score] from text classifier pipeline.
    If model unavailable or empty text -> returns empty dict.
    """
    if not text or text.strip() == "":
        return {}

    if text_classifier is None:
        return {}

    try:
        # pipeline returns list of {label, score}
        results = text_classifier(text, top_k=None)
        return {r["label"].lower(): float(r["score"]) for r in results}
    except Exception:
        return {}

