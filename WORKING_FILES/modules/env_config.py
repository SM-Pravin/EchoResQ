import os
from typing import Dict, Any

# Snapshot constants (initialized at import time) for backward compatibility
PARALLEL_MAX_WORKERS = int(os.getenv("PARALLEL_MAX_WORKERS", "4"))
ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "false").lower() == "true"
AUDIO_BATCH_SIZE = int(os.getenv("AUDIO_BATCH_SIZE", "8"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/default_model.pt")
USE_ONNX = os.getenv("USE_ONNX", "false").lower() == "true"
USE_ONNX_AUDIO = os.getenv("USE_ONNX_AUDIO", "false").lower() == "true"
USE_ONNX_TEXT = os.getenv("USE_ONNX_TEXT", "false").lower() == "true"
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", "models/optimized")
USE_ONNX_CUDA = os.getenv("USE_ONNX_CUDA", "auto")  # "true"|"false"|"auto"

# New: dynamic getters to read latest env values at call time
def get_parallel_max_workers(default: int = None) -> int:
    try:
        v = os.getenv("PARALLEL_MAX_WORKERS")
        if v is not None:
            return max(1, int(v))
    except Exception:
        pass
    return max(1, int(default if default is not None else PARALLEL_MAX_WORKERS))


def get_enable_batch_processing(default: bool = None) -> bool:
    v = os.getenv("ENABLE_BATCH_PROCESSING")
    if v is not None:
        return str(v).lower() == "true"
    return ENABLE_BATCH_PROCESSING if default is None else bool(default)


def get_audio_batch_size(default: int = None) -> int:
    try:
        v = os.getenv("AUDIO_BATCH_SIZE")
        if v is not None:
            return max(1, int(v))
    except Exception:
        pass
    return max(1, int(default if default is not None else AUDIO_BATCH_SIZE))


def get_model_path(default: str | None = None) -> str:
    v = os.getenv("MODEL_PATH")
    if v is not None:
        return v
    return default if default is not None else MODEL_PATH


def get_use_onnx(default: bool | None = None) -> bool:
    v = os.getenv("USE_ONNX")
    if v is not None:
        return str(v).lower() == "true"
    return USE_ONNX if default is None else bool(default)


def get_use_onnx_audio(default: bool | None = None) -> bool:
    v = os.getenv("USE_ONNX_AUDIO")
    if v is not None:
        return str(v).lower() == "true"
    # inherit from USE_ONNX if set and specific flag missing
    return (USE_ONNX or USE_ONNX_AUDIO) if default is None else bool(default)


def get_use_onnx_text(default: bool | None = None) -> bool:
    v = os.getenv("USE_ONNX_TEXT")
    if v is not None:
        return str(v).lower() == "true"
    return (USE_ONNX or USE_ONNX_TEXT) if default is None else bool(default)


def get_onnx_model_dir(default: str | None = None) -> str:
    v = os.getenv("ONNX_MODEL_DIR")
    if v is not None:
        return v
    return default if default is not None else ONNX_MODEL_DIR


def get_use_onnx_cuda(default: str | None = None) -> str:
    v = os.getenv("USE_ONNX_CUDA")
    if v is not None:
        return v
    return default if default is not None else USE_ONNX_CUDA


def get_config_snapshot() -> Dict[str, Any]:
    """Return a dict snapshot of the current effective configuration."""
    return {
        "PARALLEL_MAX_WORKERS": get_parallel_max_workers(PARALLEL_MAX_WORKERS),
        "ENABLE_BATCH_PROCESSING": get_enable_batch_processing(ENABLE_BATCH_PROCESSING),
        "AUDIO_BATCH_SIZE": get_audio_batch_size(AUDIO_BATCH_SIZE),
        "MODEL_PATH": get_model_path(MODEL_PATH),
        "USE_ONNX": get_use_onnx(USE_ONNX),
        "USE_ONNX_AUDIO": get_use_onnx_audio(USE_ONNX_AUDIO),
        "USE_ONNX_TEXT": get_use_onnx_text(USE_ONNX_TEXT),
        "ONNX_MODEL_DIR": get_onnx_model_dir(ONNX_MODEL_DIR),
        "USE_ONNX_CUDA": get_use_onnx_cuda(USE_ONNX_CUDA),
    }


def print_config():
    cfg = get_config_snapshot()
    for k, v in cfg.items():
        print(f"{k}: {v}")

# Environment Configuration Module
