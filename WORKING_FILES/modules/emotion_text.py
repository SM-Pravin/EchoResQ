# modules/emotion_text.py
import numpy as np
import torch
from modules.model_loader import text_classifier, get_model
from modules import env_config as cfg

def analyze_text_emotion(text: str):
    """
    Returns: dict[label -> score] from text classifier pipeline.
    If model unavailable or empty text -> returns empty dict.
    """
    if not text or text.strip() == "":
        return {}

    # Prefer ONNX if enabled and available
    use_onnx = cfg.get_use_onnx_text(cfg.get_use_onnx(False))
    if use_onnx:
        try:
            session = get_model('text_onnx')
            if session is not None:
                # Need tokenizer to prepare inputs in same way; reuse HF tokenizer from PyTorch pipeline if available
                if text_classifier is not None and hasattr(text_classifier, 'tokenizer'):
                    tok = text_classifier.tokenizer
                else:
                    # Fallback: try to load tokenizer from local model dir
                    from transformers import AutoTokenizer
                    from modules.model_loader import DISTILROBERTA_PATH
                    tok = AutoTokenizer.from_pretrained(DISTILROBERTA_PATH, local_files_only=True)
                enc = tok(
                    text,
                    return_tensors="np",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                )
                inputs = {k: v.astype(np.int64) for k, v in enc.items()}
                outputs = session.run(None, inputs)
                logits = outputs[0]
                probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
                # Build labels mapping; prefer pipeline model config if present
                if text_classifier is not None and hasattr(text_classifier.model, 'config') and hasattr(text_classifier.model.config, 'id2label'):
                    id2label = text_classifier.model.config.id2label
                else:
                    # Reasonable defaults
                    id2label = {i: f"label_{i}" for i in range(probs.shape[-1])}
                return {str(id2label[int(i)]).lower(): float(probs[int(i)]) for i in range(len(probs))}
        except Exception:
            # Fall through to PyTorch pipeline
            pass

    # PyTorch pipeline fallback
    if text_classifier is None:
        return {}
    try:
        results = text_classifier(text, top_k=None)
        return {r["label"].lower(): float(r["score"]) for r in results}
    except Exception:
        return {}

