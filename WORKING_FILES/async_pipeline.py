"""
Async-enabled analysis pipeline for Emergency AI.
Provides true concurrent processing of STT, emotion detection, and sound analysis
with better resource management and lower latency.
"""

import asyncio
import os
import tempfile
import warnings
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Union
import threading
from contextlib import asynccontextmanager

# Environment tweaks should be set early
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore")
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import numpy as np
import soundfile as sf
import librosa
from modules.audio_preprocessing import preprocess_audio, split_audio_chunks
from modules.speech_to_text import transcribe_audio
from modules.emotion_audio import analyze_audio_emotion, analyze_audio_emotion_batch
from modules.emotion_text import analyze_text_emotion
from modules.fusion_engine import fuse_emotions
from modules.distress_mapper import get_distress_token
from modules.keyword_detector import check_keywords, severity_level
from modules.sound_event_detector import analyze_sound_events
from modules.logger import log_call, log_error
from modules.model_loader import get_model
from modules.env_config import (
    get_enable_batch_processing,
    get_audio_batch_size,
    get_parallel_max_workers,
)

# Parameters
CHUNK_SIZE_SECONDS = 30
HOP_SIZE_SECONDS = 15
SILENCE_RMS_THRESHOLD_RATIO = 0.06
TEMP_DIR = "tmp_chunks"

# Threading context for model loading in workers
_thread_local = threading.local()


class AsyncAudioProcessor:
    """Async audio processing pipeline with concurrent inference."""
    
    def __init__(self, max_concurrent_tasks: int = None):
        self.max_concurrent_tasks = max_concurrent_tasks or get_parallel_max_workers()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self.process_pool = None  # Lazy initialization
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up thread pools and resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
    
    def _get_process_pool(self):
        """Lazy initialization of process pool."""
        if self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(
                max_workers=min(self.max_concurrent_tasks, 4),  # Limit process workers
                mp_context=None  # Use default context
            )
        return self.process_pool
    
    async def run_in_thread(self, func, *args, **kwargs):
        """Run a function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    async def run_in_process(self, func, *args, **kwargs):
        """Run a function in process pool."""
        loop = asyncio.get_event_loop()
        process_pool = self._get_process_pool()
        return await loop.run_in_executor(process_pool, func, *args, **kwargs)
    
    async def process_audio_concurrent(
        self, 
        audio_file: str, 
        fast_mode: bool = False,
        return_chunks_details: bool = False,
        chunk_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process audio file with concurrent STT, emotion, and sound analysis.
        
        Args:
            audio_file: Path to audio file
            fast_mode: Skip detailed chunk analysis
            return_chunks_details: Include per-chunk details in results
            chunk_callback: Optional callback for chunk results
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "caller_id": "UNKNOWN", "transcript": "", "emotion": "UNKNOWN",
            "confidence": 0.0, "distress": "low distress", "sounds": [],
            "reason": "", "error": None, "fused_scores": {}, "chunks": []
        }
        
        fixed_file = None
        try:
            # Preprocess audio
            temp_dir = tempfile.gettempdir()
            base_name = os.path.basename(audio_file)
            fixed_file = os.path.join(temp_dir, f"async_fixed_{base_name}")
            
            await self.run_in_thread(preprocess_audio, audio_file, fixed_file)
            
            caller_id = os.path.basename(audio_file).split(".")[0].upper()
            results["caller_id"] = caller_id
            
            if fast_mode:
                return await self._process_fast_mode(fixed_file, results)
            
            # Concurrent processing of different analysis types
            tasks = []
            
            # Task 1: Full-file transcription
            transcript_task = asyncio.create_task(
                self.run_in_thread(transcribe_audio, fixed_file),
                name="transcription"
            )
            tasks.append(transcript_task)
            
            # Task 2: Prepare chunks for analysis
            chunks_task = asyncio.create_task(
                self.run_in_thread(
                    split_audio_chunks, 
                    fixed_file, 
                    max_chunk=CHUNK_SIZE_SECONDS,
                    overlap=HOP_SIZE_SECONDS,
                    in_memory=True
                ),
                name="chunking"
            )
            tasks.append(chunks_task)
            
            # Wait for initial tasks
            transcript, chunks = await asyncio.gather(*tasks)
            results["transcript"] = transcript or ""
            
            # Task 3: Text emotion analysis (depends on transcript)
            text_emotion_task = asyncio.create_task(
                self.run_in_thread(analyze_text_emotion, transcript or ""),
                name="text_emotion"
            )
            
            # Task 4: Concurrent chunk analysis
            if chunks:
                chunk_analysis_task = asyncio.create_task(
                    self._analyze_chunks_concurrent(chunks, chunk_callback),
                    name="chunk_analysis"
                )
                
                # Wait for remaining analysis
                text_scores, chunk_results = await asyncio.gather(
                    text_emotion_task,
                    chunk_analysis_task
                )
            else:
                text_scores = await text_emotion_task
                chunk_results = {
                    "aggregated_fused": {"angry": 0.0, "happy": 0.0, "neutral": 1.0, "sad": 0.0},
                    "final_distress": "low distress",
                    "final_emotion": "neutral",
                    "sound_events": [],
                    "chunks": [],
                    "reason": "no chunks to analyze"
                }
            
            # Combine results
            audio_scores = chunk_results["aggregated_fused"]
            confidence, emotion, fused_scores = fuse_emotions(audio_scores, text_scores)
            distress = get_distress_token(emotion, confidence)
            distress = check_keywords(transcript or "", distress)
            
            # Use chunk analysis results if more severe
            if severity_level(chunk_results["final_distress"]) < severity_level(distress):
                distress = chunk_results["final_distress"]
                emotion = chunk_results["final_emotion"]
            
            results.update({
                "emotion": emotion,
                "confidence": confidence,
                "distress": distress,
                "sounds": chunk_results["sound_events"],
                "reason": chunk_results["reason"],
                "fused_scores": fused_scores
            })
            
            if return_chunks_details:
                results["chunks"] = chunk_results["chunks"]
            
            # Log the call
            await self.run_in_thread(
                log_call, caller_id, transcript or "", emotion, distress,
                {"fused": fused_scores, "sounds": chunk_results["sound_events"]},
                chunk_results["reason"]
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Async processing error: {e}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            results["error"] = error_msg
            return results
            
        finally:
            # Cleanup
            if fixed_file and os.path.exists(fixed_file):
                try:
                    os.remove(fixed_file)
                except Exception:
                    pass
    
    async def _process_fast_mode(self, fixed_file: str, results: Dict) -> Dict:
        """Process audio in fast mode with minimal analysis."""
        try:
            # Concurrent fast analysis
            tasks = [
                asyncio.create_task(
                    self.run_in_thread(transcribe_audio, fixed_file),
                    name="fast_transcription"
                ),
                asyncio.create_task(
                    self.run_in_thread(analyze_audio_emotion, fixed_file),
                    name="fast_audio_emotion"
                )
            ]
            
            transcript, audio_scores = await asyncio.gather(*tasks)
            
            # Text emotion analysis
            text_scores = await self.run_in_thread(analyze_text_emotion, transcript or "")
            
            confidence, emotion, fused_scores = fuse_emotions(audio_scores, text_scores)
            distress = get_distress_token(emotion, confidence)
            distress = check_keywords(transcript or "", distress)
            
            results.update({
                "transcript": transcript or "",
                "emotion": emotion,
                "confidence": confidence,
                "distress": distress,
                "reason": "fast mode: concurrent processing",
                "fused_scores": fused_scores
            })
            
            return results
            
        except Exception as e:
            results["error"] = f"Fast mode error: {e}"
            return results
    
    async def _analyze_chunks_concurrent(
        self, 
        chunks: List[Dict], 
        chunk_callback: Optional[Callable] = None
    ) -> Dict:
        """Analyze chunks with concurrent processing."""
        if not chunks:
            return {
                "aggregated_fused": {"angry": 0.0, "happy": 0.0, "neutral": 1.0, "sad": 0.0},
                "final_distress": "low distress",
                "final_emotion": "neutral",
                "sound_events": [],
                "chunks": [],
                "reason": "no chunks provided"
            }
        
        canonical_labels = ["angry", "happy", "neutral", "sad"]
        aggregated_fused = {k: 0.0 for k in canonical_labels}
        total_weight = 0.0
        sound_events_all = []
        final_distress = "low distress"
        final_emotion = "neutral"
        reason = None
        chunk_results = []
        
        # Calculate RMS values for weighting
        max_rms = 0.0
        chunk_rms_map = {}
        for idx, chunk in enumerate(chunks):
            y = chunk.get("data", np.array([], dtype=np.float32))
            rms = float(np.sqrt(np.mean(y**2))) if y.size > 0 else 0.0
            chunk_rms_map[idx] = rms
            if rms > max_rms:
                max_rms = rms
        
        silence_thresh = max_rms * SILENCE_RMS_THRESHOLD_RATIO if max_rms > 0 else 0.0
        
        # Process chunks concurrently
        async def process_chunk(idx: int, chunk: Dict) -> Dict:
            """Process a single chunk."""
            async with self.semaphore:  # Limit concurrent chunk processing
                try:
                    y = chunk.get("data", np.array([], dtype=np.float32))
                    sr = int(chunk.get("sr", 16000) or 16000)
                    rms = chunk_rms_map.get(idx, 0.0)
                    
                    start_s = chunk.get("start_s", 0.0)
                    end_s = chunk.get("end_s", 0.0)
                    
                    # Skip silent chunks
                    if rms <= silence_thresh:
                        return {
                            "index": idx, "start_s": start_s, "end_s": end_s,
                            "rms": rms, "win_emotion": None, "win_conf": 0.0,
                            "win_fused": {k: 0.0 for k in canonical_labels},
                            "win_distress": "low distress", "sounds": [],
                            "silent": True
                        }
                    
                    # Concurrent emotion and sound analysis
                    emotion_task = asyncio.create_task(
                        self.run_in_thread(analyze_audio_emotion, {"data": y, "sr": sr}),
                        name=f"emotion_chunk_{idx}"
                    )
                    sound_task = asyncio.create_task(
                        self.run_in_thread(analyze_sound_events, {"data": y, "sr": sr}),
                        name=f"sound_chunk_{idx}"
                    )
                    
                    audio_scores, win_sounds = await asyncio.gather(emotion_task, sound_task)
                    
                    win_conf, win_emotion, win_fused = fuse_emotions(audio_scores, {})
                    win_distress = get_distress_token(win_emotion, win_conf)
                    
                    return {
                        "index": idx, "start_s": start_s, "end_s": end_s,
                        "rms": rms, "win_emotion": win_emotion, "win_conf": win_conf,
                        "win_fused": win_fused, "win_distress": win_distress,
                        "sounds": win_sounds or [], "silent": False
                    }
                    
                except Exception as e:
                    log_error(f"chunk_processing_{idx}", e)
                    return {
                        "index": idx, "start_s": chunk.get("start_s", 0.0),
                        "end_s": chunk.get("end_s", 0.0), "rms": 0.0,
                        "win_emotion": None, "win_conf": 0.0,
                        "win_fused": {k: 0.0 for k in canonical_labels},
                        "win_distress": "low distress", "sounds": [],
                        "error": str(e)
                    }
        
        # Process all chunks concurrently
        chunk_tasks = [process_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
        processed_chunks = await asyncio.gather(*chunk_tasks)
        
        # Aggregate results
        for chunk_result in processed_chunks:
            if chunk_callback:
                try:
                    if asyncio.iscoroutinefunction(chunk_callback):
                        await chunk_callback(chunk_result)
                    else:
                        await self.run_in_thread(chunk_callback, chunk_result)
                except Exception:
                    pass
            
            chunk_results.append(chunk_result)
            
            if chunk_result.get("silent"):
                continue
                
            rms = chunk_result["rms"]
            win_fused = chunk_result["win_fused"]
            win_distress = chunk_result["win_distress"]
            win_emotion = chunk_result["win_emotion"]
            win_sounds = chunk_result["sounds"]
            
            # Update aggregated scores
            weight = rms if max_rms > 0 else 1.0
            for k in canonical_labels:
                aggregated_fused[k] += win_fused.get(k, 0.0) * weight
            total_weight += weight
            
            # Track sound events
            if win_sounds:
                sound_events_all.extend(win_sounds)
            
            # Update final distress/emotion
            if severity_level(win_distress) < severity_level(final_distress):
                final_distress = win_distress
                final_emotion = win_emotion
                reason = f"emotion escalation (chunk {chunk_result['index']}): {win_emotion} ({chunk_result['win_conf']:.2f})"
        
        # Normalize aggregated scores
        if total_weight > 0:
            for k in canonical_labels:
                aggregated_fused[k] = aggregated_fused[k] / total_weight
        
        return {
            "aggregated_fused": aggregated_fused,
            "final_distress": final_distress,
            "final_emotion": final_emotion,
            "sound_events": sound_events_all,
            "chunks": chunk_results,
            "reason": reason or "concurrent chunk analysis completed"
        }


# Global async processor instance
_async_processor = None


async def get_async_processor() -> AsyncAudioProcessor:
    """Get or create global async processor."""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncAudioProcessor()
    return _async_processor


async def process_audio_file_async(
    audio_file: str,
    fast_mode: bool = False,
    return_chunks_details: bool = False,
    chunk_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Async version of process_audio_file with concurrent processing.
    
    Args:
        audio_file: Path to audio file
        fast_mode: Skip detailed chunk analysis
        return_chunks_details: Include per-chunk details
        chunk_callback: Optional callback for chunk results
        
    Returns:
        Analysis results dictionary
    """
    processor = await get_async_processor()
    return await processor.process_audio_concurrent(
        audio_file, fast_mode, return_chunks_details, chunk_callback
    )


def process_audio_file_sync_wrapper(
    audio_file: str,
    fast_mode: bool = False,
    return_chunks_details: bool = False,
    chunk_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for async processing.
    Creates new event loop if none exists.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use a different approach
            # This is common in Jupyter/Streamlit environments
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    process_audio_file_async(audio_file, fast_mode, return_chunks_details, chunk_callback)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                process_audio_file_async(audio_file, fast_mode, return_chunks_details, chunk_callback)
            )
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(
            process_audio_file_async(audio_file, fast_mode, return_chunks_details, chunk_callback)
        )


# Cleanup function for graceful shutdown
async def cleanup_async_processor():
    """Clean up global async processor."""
    global _async_processor
    if _async_processor:
        await _async_processor.cleanup()
        _async_processor = None


def cleanup_async_processor_sync():
    """Synchronous cleanup wrapper."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(cleanup_async_processor())
    except RuntimeError:
        asyncio.run(cleanup_async_processor())