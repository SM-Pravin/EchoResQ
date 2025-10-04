# modules/emotion_audio.py
import torch
import numpy as np
import librosa
from modules.model_loader import audio_feature_extractor, wav2vec_model, TORCH_DEVICE, get_model
from modules import env_config as cfg


def analyze_audio_emotion_buffer(audio_buffer, sr=16000):
    """
    Analyze emotion from AudioBuffer object directly.
    Returns dict {label: score}.
    """
    from modules.in_memory_audio import AudioBuffer
    
    afe = audio_feature_extractor or get_model('audio_feature_extractor')
    w2v = wav2vec_model or get_model('wav2vec_model')
    if afe is None and not cfg.get_use_onnx_audio(cfg.get_use_onnx(False)):
        return {}
    
    try:
        # Get audio data from buffer
        speech = audio_buffer.data
        s = audio_buffer.sample_rate
        
        # Resample if needed
        if s != sr:
            import librosa
            speech = librosa.resample(speech, orig_sr=s, target_sr=sr)
        
        # Ensure mono
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        
        # Process with existing logic
        speech = np.asarray(speech, dtype=np.float32)
        
        if len(speech) == 0:
            return {}
        
        # Normalize
        speech = speech / (np.max(np.abs(speech)) + 1e-9)
        
        # Use existing model processing
        if cfg.get_use_onnx_audio(cfg.get_use_onnx(False)):
            return _analyze_with_onnx_audio(speech, sr)
        else:
            return _analyze_with_torch_audio(speech, sr, afe, w2v)
            
    except Exception as e:
        print(f"❌ Buffer audio emotion error: {e}")
        return {}


def _analyze_with_torch_audio(speech, sr, afe, w2v):
    """Helper function for torch-based audio emotion analysis."""
    try:
        # Use wav2vec2 if available
        if w2v is not None:
            inputs = afe(speech, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = w2v(**inputs)
                hidden_states = outputs.last_hidden_state
                # Simple emotion classification (this is a placeholder)
                # In a real implementation, you'd use a trained emotion classifier
                pooled = torch.mean(hidden_states, dim=1)
                # Mock emotion scores for demonstration
                emotions = {
                    'happy': float(torch.sigmoid(pooled[0, 0]).cpu()),
                    'sad': float(torch.sigmoid(pooled[0, 1] if pooled.shape[1] > 1 else pooled[0, 0]).cpu()),
                    'angry': float(torch.sigmoid(pooled[0, 2] if pooled.shape[1] > 2 else pooled[0, 0]).cpu()),
                    'fear': float(torch.sigmoid(pooled[0, 3] if pooled.shape[1] > 3 else pooled[0, 0]).cpu()),
                    'neutral': float(torch.sigmoid(pooled[0, 4] if pooled.shape[1] > 4 else pooled[0, 0]).cpu())
                }
                
                # Normalize to sum to 1
                total = sum(emotions.values())
                if total > 0:
                    emotions = {k: v/total for k, v in emotions.items()}
                
                return emotions
        
        return {}
        
    except Exception as e:
        print(f"❌ Torch audio analysis error: {e}")
        return {}


def _analyze_with_onnx_audio(speech, sr):
    """Helper function for ONNX-based audio emotion analysis."""
    try:
        # Placeholder for ONNX implementation
        # This would use the ONNX runtime for inference
        return {
            'neutral': 0.4,
            'happy': 0.3,
            'sad': 0.1,
            'angry': 0.1,
            'fear': 0.1
        }
    except Exception as e:
        print(f"❌ ONNX audio analysis error: {e}")
        return {}

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
