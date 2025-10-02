# modules/keyword_detector.py
# simple keyword -> distress mapping; no prints inside
KEYWORD_MAP = {
    "heart attack": "peak emergency distress",
    "not breathing": "peak emergency distress",
    "can't breathe": "peak emergency distress",
    "unconscious": "high distress",
    "bleeding": "high distress",
    "fire": "peak emergency distress",
    "explosion": "peak emergency distress",
    "gunshot": "peak emergency distress",
    "trapped": "high distress",
    "collapsed": "high distress",
    "accident": "medium distress",
    "help me": "high distress",
}

# Emergency keywords list for streaming detection
EMERGENCY_KEYWORDS = list(KEYWORD_MAP.keys()) + [
    "emergency", "help", "911", "urgent", "crisis", "dying", "suicide", 
    "kill", "murder", "rape", "attack", "bomb", "knife", "police", 
    "ambulance", "medical", "hurt", "pain", "sick", "overdose"
]

SEVERITY_ORDER = {
    "peak emergency distress": 1,
    "high distress": 2,
    "medium distress": 3,
    "low distress": 4
}

def severity_level(distress_token: str) -> int:
    return SEVERITY_ORDER.get(distress_token, 99)

def check_keywords(transcript: str, current_distress: str):
    """
    Return a possibly-upgraded distress token (no prints).
    """
    if not transcript:
        return current_distress
    t = transcript.lower()
    for phrase, mapped in KEYWORD_MAP.items():
        if phrase in t:
            if severity_level(mapped) < severity_level(current_distress):
                return mapped
    return current_distress

