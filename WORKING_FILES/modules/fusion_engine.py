# modules/fusion_engine.py
"""
Enhanced fusion engine for Emergency AI with configurable distress scoring.
Combines keyword detection, emotion analysis, and sound events for comprehensive distress assessment.
Returns: (confidence: float, final_emotion: str, fused_scores: dict)
canonical labels: 'angry', 'happy', 'neutral', 'sad', 'fear', 'disgust', 'surprise'
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from modules.config_manager import get_config_manager

# Extended canonical emotions for better distress detection
CANONICAL = ["angry", "happy", "neutral", "sad", "fear", "disgust", "surprise"]

# Distress weights for different emotions (higher = more distressing)
EMOTION_DISTRESS_WEIGHTS = {
    "fear": 0.9,
    "angry": 0.8,
    "sad": 0.7,
    "disgust": 0.6,
    "surprise": 0.4,
    "neutral": 0.2,
    "happy": 0.1
}

def _normalize_to_canonical(scores: dict):
    """Map arbitrary model labels to canonical labels with enhanced mapping."""
    out = {k: 0.0 for k in CANONICAL}
    if not scores:
        return out

    for label, score in scores.items():
        l = label.lower()
        if "ang" in l or "anger" in l:
            out["angry"] = max(out["angry"], score)
        elif "joy" in l or "happy" in l or "happiness" in l:
            out["happy"] = max(out["happy"], score)
        elif "neu" in l or "neutral" in l:
            out["neutral"] = max(out["neutral"], score)
        elif "sad" in l or "sadness" in l:
            out["sad"] = max(out["sad"], score)
        elif "fear" in l or "scared" in l or "afraid" in l:
            out["fear"] = max(out["fear"], score)
        elif "disgust" in l or "revulsion" in l:
            out["disgust"] = max(out["disgust"], score)
        elif "surprise" in l or "surprised" in l or "shock" in l:
            out["surprise"] = max(out["surprise"], score)
        else:
            # If label unknown, distribute to neutral with reduced weight
            out["neutral"] = max(out["neutral"], score * 0.3)
    return out


def calculate_distress_score(fused_emotions: Dict[str, float], 
                           keywords_detected: List[str] = None,
                           sound_events: List[Dict[str, Any]] = None,
                           confidence: float = 0.0,
                           config=None) -> Tuple[float, str]:
    """
    Calculate comprehensive distress score (0-1) from multiple inputs.
    
    Args:
        fused_emotions: Dictionary of emotion scores
        keywords_detected: List of emergency keywords found
        sound_events: List of detected sound events
        confidence: Overall confidence in the analysis
        config: Configuration object
    
    Returns:
        Tuple of (distress_score, distress_level)
    """
    if config is None:
        config = get_config_manager().config
    
    # Base emotion distress score
    emotion_distress = 0.0
    for emotion, score in fused_emotions.items():
        weight = EMOTION_DISTRESS_WEIGHTS.get(emotion, 0.2)
        emotion_distress += score * weight
    
    # Apply emotion weight from config
    emotion_contribution = emotion_distress * config.fusion.emotion_weight
    
    # Keyword contribution
    keyword_contribution = 0.0
    if keywords_detected:
        keyword_score = len(keywords_detected) * 0.15  # Each keyword adds distress
        keyword_score = min(keyword_score, 0.5)  # Cap at 0.5
        
        # Boost for high-priority keywords
        emergency_keywords = config.keywords.emergency_keywords + config.keywords.medical_keywords
        high_priority_count = sum(1 for kw in keywords_detected if kw.lower() in [k.lower() for k in emergency_keywords])
        keyword_score += high_priority_count * 0.1
        
        keyword_contribution = keyword_score * config.fusion.keyword_weight
    
    # Sound event contribution
    sound_contribution = 0.0
    if sound_events:
        # Analyze sound events for distress indicators
        distressing_sounds = ['scream', 'crying', 'breaking', 'siren', 'alarm', 'crash', 'explosion']
        for event in sound_events:
            if isinstance(event, dict):
                sound_label = event.get('label', '').lower()
                sound_confidence = event.get('confidence', 0.0)
                
                if any(ds in sound_label for ds in distressing_sounds):
                    sound_contribution += sound_confidence * 0.2
        
        sound_contribution = min(sound_contribution, 0.4) * config.fusion.sound_event_weight
    
    # Confidence boost (higher confidence in distressing content increases score)
    confidence_boost = 0.0
    if confidence > config.fusion.confidence_threshold:
        confidence_boost = (confidence - config.fusion.confidence_threshold) * 0.1
    
    # Combine all contributions
    base_score = emotion_contribution + keyword_contribution + sound_contribution + confidence_boost
    
    # Apply sensitivity adjustments
    sensitivity = config.fusion.sensitivity
    if sensitivity in config.fusion.sensitivity_adjustments:
        adjustments = config.fusion.sensitivity_adjustments[sensitivity]
        # Adjust the final score based on sensitivity
        if sensitivity == "high":
            base_score *= 1.2  # Increase sensitivity
        elif sensitivity == "low":
            base_score *= 0.8  # Decrease sensitivity
    
    # Normalize to 0-1 range
    distress_score = min(1.0, max(0.0, base_score))
    
    # Determine distress level
    if distress_score >= config.fusion.distress_threshold:
        if distress_score >= 0.8:
            distress_level = "critical"
        elif distress_score >= 0.6:
            distress_level = "high distress"
        else:
            distress_level = "medium distress"
    else:
        distress_level = "low distress"
    
    return distress_score, distress_level


def enhanced_fuse_emotions(audio_scores: dict, 
                         text_scores: dict, 
                         keywords_detected: List[str] = None,
                         sound_events: List[Dict[str, Any]] = None,
                         config=None) -> Tuple[float, str, dict, float, str]:
    """
    Enhanced emotion fusion with distress scoring.
    
    Returns:
        Tuple of (confidence, final_emotion, fused_scores, distress_score, distress_level)
    """
    if config is None:
        config = get_config_manager().config
    
    # Normalize emotions to canonical labels
    a = _normalize_to_canonical(audio_scores or {})
    t = _normalize_to_canonical(text_scores or {})
    
    # Get weights from config or use defaults
    w_audio = getattr(config.fusion, 'audio_weight', 0.5)
    w_text = getattr(config.fusion, 'text_weight', 0.5)
    
    # Fuse emotions
    fused = {k: w_audio * a.get(k, 0.0) + w_text * t.get(k, 0.0) for k in CANONICAL}
    
    # Normalize fused scores
    total = sum(fused.values())
    if total > 0:
        for k in fused:
            fused[k] = fused[k] / total
        final = max(fused, key=fused.get)
        confidence = float(fused[final])
    else:
        # No emotion data available
        fused = {k: 1.0/len(CANONICAL) for k in CANONICAL}  # Uniform distribution
        final = "neutral"
        confidence = 0.0
    
    # Calculate distress score
    distress_score, distress_level = calculate_distress_score(
        fused, keywords_detected, sound_events, confidence, config
    )
    
    return confidence, final, fused, distress_score, distress_level

# Legacy function for backward compatibility
def fuse_emotions(audio_scores: dict, text_scores: dict, w_audio=0.5, w_text=0.5):
    """Legacy emotion fusion function for backward compatibility."""
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


def get_distress_analysis(audio_scores: dict = None,
                         text_scores: dict = None,
                         transcript: str = "",
                         sound_events: List[Dict[str, Any]] = None,
                         config=None) -> Dict[str, Any]:
    """
    Comprehensive distress analysis combining all available inputs.
    
    Returns:
        Dictionary containing all analysis results including distress score
    """
    if config is None:
        config = get_config_manager().config
    
    # Detect keywords from transcript
    keywords_detected = []
    if transcript and config.keywords.enable_keyword_detection:
        from modules.keyword_detector import extract_keywords
        try:
            if hasattr(extract_keywords, '__call__'):
                keywords_detected = extract_keywords(transcript, config)
            else:
                # Simple keyword detection fallback
                all_keywords = config.keywords.emergency_keywords + config.keywords.medical_keywords
                keywords_detected = [kw for kw in all_keywords if kw.lower() in transcript.lower()]
        except Exception as e:
            print(f"[WARNING] Keyword detection error: {e}")
            keywords_detected = []
    
    # Enhanced emotion fusion with distress scoring
    confidence, final_emotion, fused_scores, distress_score, distress_level = enhanced_fuse_emotions(
        audio_scores, text_scores, keywords_detected, sound_events, config
    )
    
    return {
        'confidence': confidence,
        'final_emotion': final_emotion,
        'fused_scores': fused_scores,
        'distress_score': distress_score,
        'distress_level': distress_level,
        'keywords_detected': keywords_detected,
        'sound_events': sound_events or [],
        'audio_scores': audio_scores or {},
        'text_scores': text_scores or {},
        'config_sensitivity': config.fusion.sensitivity,
        'thresholds': {
            'distress_threshold': config.fusion.distress_threshold,
            'confidence_threshold': config.fusion.confidence_threshold
        }
    }

