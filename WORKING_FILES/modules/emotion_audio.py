# modules/emotion_audio.py
import torch
import numpy as np
import librosa
from modules.model_loader import audio_feature_extractor, wav2vec_model, TORCH_DEVICE, get_model
from modules import env_config as cfg

def analyze_audio_emotion(audio_input, sr=16000):
    """
    Returns dict {label: score}.
    Accepts:
      - a str path to a file
      - a numpy ndarray (mono, float32, range -1..1)
      - a dict with keys { 'data': ndarray, 'sr': int }
    If model missing -> {}.
    """
    afe = audio_feature_extractor or get_model('audio_feature_extractor')
    w2v = wav2vec_model or get_model('wav2vec_model')
    if afe is None and not cfg.get_use_onnx_audio(cfg.get_use_onnx(False)):
        return {}

    try:
        # get waveform and sampling rate
        if isinstance(audio_input, str):
            speech, s = librosa.load(audio_input, sr=sr, mono=True)
        elif isinstance(audio_input, dict) and 'data' in audio_input:
            speech = np.asarray(audio_input['data'])
            s = int(audio_input.get('sr', sr))
        elif isinstance(audio_input, np.ndarray):
            speech = audio_input
            s = sr
        else:
            return {}

        # ensure float32 and 1D
        speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        # Prefer ONNX if enabled and available
        if cfg.get_use_onnx_audio(cfg.get_use_onnx(False)):
            try:
                session = get_model('wav2vec_onnx')
                if session is not None and afe is not None:
                    enc = afe(speech, sampling_rate=s, return_tensors="np", padding=True)
                    # ONNX expects int64 for attention mask and float32 for input values
                    onnx_inputs = {
                        'input_values': enc['input_values'].astype(np.float32),
                        'attention_mask': enc.get('attention_mask', np.ones_like(enc['input_values']).astype(np.int64)).astype(np.int64)
                    }
                    outputs = session.run(None, onnx_inputs)
                    logits = outputs[0]
                    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
                    # Map labels
                    if w2v is not None and hasattr(w2v, 'config') and hasattr(w2v.config, 'id2label'):
                        id2label = w2v.config.id2label
                    else:
                        id2label = {i: f'label_{i}' for i in range(probs.shape[-1])}
                    return {str(id2label[int(i)]).lower(): float(probs[int(i)]) for i in range(len(probs))}
            except Exception:
                # Fall back to PyTorch path
                pass

        # PyTorch fallback path
        if afe is None or w2v is None:
            return {}
        inputs = afe(speech, sampling_rate=s, return_tensors="pt", padding=True)
        # Move tensors to model device if possible
        device = TORCH_DEVICE
        try:
            device = next(w2v.parameters()).device
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)
        except Exception:
            pass

        with torch.inference_mode():
            try:
                from torch.cuda.amp import autocast
                use_amp = device.type == 'cuda'
            except Exception:
                use_amp = False
            if use_amp:
                with autocast():
                    outputs = w2v(**inputs)
            else:
                outputs = w2v(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        labels = w2v.config.id2label
        return {labels[int(i)].lower(): float(probs[int(i)]) for i in range(len(probs))}
    except Exception as e:
        print(f" ⚠️ Failed audio emotion analysis: {e}")
        return {}


def analyze_audio_emotion_batch(audio_inputs, sr=16000, max_batch_size=8):
    """
    Batched version of audio emotion analysis for efficiency.
    
    Args:
        audio_inputs: list of audio inputs (same format as analyze_audio_emotion)
        sr: target sampling rate
        max_batch_size: maximum batch size for inference
    
    Returns:
        list of dicts {label: score} corresponding to each input
    """
    afe = audio_feature_extractor or get_model('audio_feature_extractor')
    w2v = wav2vec_model or get_model('wav2vec_model')
    use_onnx = cfg.get_use_onnx_audio(cfg.get_use_onnx(False))
    if afe is None and not use_onnx:
        return [{}] * len(audio_inputs)
    
    if not audio_inputs:
        return []
    
    # Process in batches
    results = []
    try:
        device = TORCH_DEVICE
        if w2v is not None:
            try:
                device = next(w2v.parameters()).device
            except Exception:
                pass
        labels = (w2v.config.id2label if (w2v is not None and hasattr(w2v, 'config')) else None)
        
        for i in range(0, len(audio_inputs), max_batch_size):
            batch = audio_inputs[i:i + max_batch_size]
            batch_results = []
            
            try:
                # Prepare batch
                speeches = []
                for audio_input in batch:
                    if isinstance(audio_input, str):
                        speech, s = librosa.load(audio_input, sr=sr, mono=True)
                    elif isinstance(audio_input, dict) and 'data' in audio_input:
                        speech = np.asarray(audio_input['data'], dtype=np.float32)
                        s = int(audio_input.get('sr', sr))
                    elif isinstance(audio_input, np.ndarray):
                        speech = audio_input.astype(np.float32)
                        s = sr
                    else:
                        speech = np.array([], dtype=np.float32)
                        s = sr
                    
                    # ensure float32 and 1D
                    if speech.ndim > 1:
                        speech = speech.mean(axis=1)
                    
                    speeches.append(speech)
                
                if use_onnx:
                    try:
                        session = get_model('wav2vec_onnx')
                        if session is not None and afe is not None:
                            batch_inputs_np = afe(
                                speeches,
                                sampling_rate=sr,
                                return_tensors="np",
                                padding=True
                            )
                            onnx_inputs = {
                                'input_values': batch_inputs_np['input_values'].astype(np.float32),
                                'attention_mask': batch_inputs_np.get('attention_mask', np.ones_like(batch_inputs_np['input_values']).astype(np.int64)).astype(np.int64)
                            }
                            outputs = session.run(None, onnx_inputs)
                            logits = outputs[0]
                            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                        else:
                            raise RuntimeError("ONNX session or feature extractor missing")
                    except Exception:
                        # Fall back to PyTorch below
                        use_onnx = False
                if not use_onnx:
                    # Batch feature extraction
                    batch_inputs = afe(
                        speeches, 
                        sampling_rate=sr, 
                        return_tensors="pt", 
                        padding=True
                    )
                    # Move to device
                    for k, v in batch_inputs.items():
                        if hasattr(v, "to"):
                            batch_inputs[k] = v.to(device)
                    # Batch inference
                    with torch.inference_mode():
                        try:
                            from torch.cuda.amp import autocast
                            use_amp = device.type == 'cuda'
                        except Exception:
                            use_amp = False
                        if use_amp:
                            with autocast():
                                outputs = w2v(**batch_inputs)
                        else:
                            outputs = w2v(**batch_inputs)
                        logits = outputs.logits
                        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                
                # Convert to results
                for j in range(len(batch)):
                    if j < probs.shape[0]:
                        if labels is None:
                            labels = {k: f'label_{k}' for k in range(probs.shape[1])}
                        result = {str(labels[int(k)]).lower(): float(probs[j, k]) for k in range(probs.shape[1])}
                    else:
                        result = {}
                    batch_results.append(result)
                
            except Exception as e:
                print(f" ⚠️ Failed batch audio emotion analysis: {e}")
                # Fallback to individual processing for this batch
                for audio_input in batch:
                    batch_results.append(analyze_audio_emotion(audio_input, sr=sr))
            
            results.extend(batch_results)
            
    except Exception as e:
        print(f" ⚠️ Critical error in batch processing: {e}")
        # Complete fallback
        results = [analyze_audio_emotion(audio_input, sr=sr) for audio_input in audio_inputs]
    
    return results
