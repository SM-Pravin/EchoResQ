# modules/keyword_detector.py
"""
Enhanced keyword detection with configurable keywords and scoring.
"""
import re
from typing import List, Dict, Any
from modules.config_manager import get_config_manager

# Default keyword mappings
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


def extract_keywords(transcript: str, config=None) -> List[str]:
    """
    Extract emergency and medical keywords from transcript text.
    
    Args:
        transcript: The text to analyze
        config: Configuration object (optional)
    
    Returns:
        List of detected keywords
    """
    if not transcript:
        return []
    
    if config is None:
        config = get_config_manager().config
    
    # Get keywords from config
    emergency_keywords = config.keywords.emergency_keywords
    medical_keywords = config.keywords.medical_keywords
    all_keywords = emergency_keywords + medical_keywords
    
    # Normalize transcript
    text_lower = transcript.lower()
    
    # Find keywords
    detected = []
    for keyword in all_keywords:
        if keyword.lower() in text_lower:
            detected.append(keyword)
    
    # Also check for multi-word phrases from KEYWORD_MAP
    for phrase in KEYWORD_MAP.keys():
        if phrase.lower() in text_lower and phrase not in detected:
            detected.append(phrase)
    
    return detected


def calculate_keyword_score(keywords: List[str], config=None) -> float:
    """
    Calculate keyword-based distress score.
    
    Args:
        keywords: List of detected keywords
        config: Configuration object (optional)
    
    Returns:
        Keyword distress score (0-1)
    """
    if not keywords:
        return 0.0
    
    if config is None:
        config = get_config_manager().config
    
    # Base score from number of keywords
    base_score = len(keywords) * 0.1
    
    # Add severity weights
    severity_boost = 0.0
    for keyword in keywords:
        if keyword in KEYWORD_MAP:
            severity = KEYWORD_MAP[keyword]
            if severity == "peak emergency distress":
                severity_boost += 0.3
            elif severity == "high distress":
                severity_boost += 0.2
            elif severity == "medium distress":
                severity_boost += 0.1
    
    # Apply boost factor from config
    boost_factor = config.keywords.keyword_boost_factor
    total_score = (base_score + severity_boost) * boost_factor
    
    return min(1.0, total_score)

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

