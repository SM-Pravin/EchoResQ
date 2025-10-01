# modules/distress_mapper.py
def get_distress_token(emotion: str, confidence: float):
    """
    Map (emotion, confidence) -> one of:
      'peak emergency distress', 'high distress', 'medium distress', 'low distress'
    Always returns a valid token (no 'undefined').
    """
    if emotion is None:
        return "low distress"

    e = str(emotion).lower()
    c = float(confidence or 0.0)

    # Peak emergency if strongly angry/fearful with very high confidence
    if (e in ["angry", "anger"] and c >= 0.90):
        return "peak emergency distress"

    # High distress for angry/sad/fear with good confidence
    if e in ["angry", "sad", "fear", "fearful"] and c >= 0.60:
        return "high distress"

    # Medium distress if moderate confidence
    if c >= 0.40:
        return "medium distress"

    # Default
    return "low distress"

