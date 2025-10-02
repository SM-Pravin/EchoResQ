# modules/emotion_audio.py
import torch
import numpy as np
import librosa
from modules.model_loader import audio_feature_extractor, wav2vec_model, TORCH_DEVICE

def analyze_audio_emotion(audio_input, sr=16000):
    """
    Returns dict {label: score}.
    Accepts:
      - a str path to a file
      - a numpy ndarray (mono, float32, range -1..1)
      - a dict with keys { 'data': ndarray, 'sr': int }
    If model missing -> {}.
    """
    if audio_feature_extractor is None or wav2vec_model is None:
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

        inputs = audio_feature_extractor(speech, sampling_rate=s, return_tensors="pt", padding=True)
        # Move tensors to model device if possible
        try:
            device = next(wav2vec_model.parameters()).device
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)
        except Exception:
            pass

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        labels = wav2vec_model.config.id2label
        # id2label may be dict like {0: 'ang', 1: 'hap', ...}
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
    if audio_feature_extractor is None or wav2vec_model is None:
        return [{}] * len(audio_inputs)
    
    if not audio_inputs:
        return []
    
    # Process in batches
    results = []
    try:
        device = next(wav2vec_model.parameters()).device
        labels = wav2vec_model.config.id2label
        
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
                
                # Batch feature extraction
                batch_inputs = audio_feature_extractor(
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
                with torch.no_grad():
                    outputs = wav2vec_model(**batch_inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                
                # Convert to results
                for j in range(len(batch)):
                    if j < probs.shape[0]:
                        result = {labels[int(k)].lower(): float(probs[j, k]) for k in range(probs.shape[1])}
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
