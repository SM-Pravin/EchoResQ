"""
Real-time audio streaming and processing module for Emergency AI.
Supports live microphone input with partial transcription and emotion analysis.
"""

import os
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import streamlit as st

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[WARNING] PyAudio not available. Install with: pip install pyaudio")

try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("[WARNING] streamlit-webrtc not available. Install with: pip install streamlit-webrtc")

from modules.config_manager import get_config_manager
from modules.in_memory_audio import AudioBuffer, get_audio_processor
from modules.speech_to_text import transcribe_audio_buffer
from modules.emotion_audio import analyze_audio_emotion_buffer
from modules.emotion_text import analyze_text_emotion
from modules.fusion_engine import fuse_emotions
from modules.keyword_detector import check_keywords
from modules.distress_mapper import get_distress_token


@dataclass
class StreamingResult:
    """Result from real-time processing."""
    timestamp: float
    transcript: str
    partial_transcript: str
    audio_emotion: Dict[str, float]
    text_emotion: Dict[str, float]
    fused_emotion: Dict[str, float]
    distress_score: float
    distress_token: str
    confidence: float
    keywords_detected: List[str]
    is_final: bool = False


class AudioStreamProcessor:
    """Processes audio stream chunks in real-time."""
    
    def __init__(self, config=None):
        self.config = config or get_config_manager().config
        self.audio_processor = get_audio_processor()
        
        # Streaming parameters
        self.sample_rate = self.config.audio.sample_rate
        self.chunk_duration = self.config.streaming.min_chunk_duration
        self.buffer_duration = self.config.streaming.buffer_size_ms / 1000.0
        
        # Processing state
        self.audio_buffer = []
        self.accumulated_transcript = ""
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks
        self.on_result_callback: Optional[Callable[[StreamingResult], None]] = None
        self.on_error_callback: Optional[Callable[[Exception], None]] = None
    
    def start_processing(self):
        """Start the background processing thread."""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the background processing thread."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def add_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = None):
        """Add audio chunk to processing queue."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Create AudioBuffer from chunk
        audio_buffer = AudioBuffer(
            data=audio_data,
            sample_rate=sample_rate,
            metadata={'chunk_type': 'stream', 'timestamp': time.time()}
        )
        
        if not self.processing_queue.full():
            self.processing_queue.put(audio_buffer)
    
    def get_latest_result(self) -> Optional[StreamingResult]:
        """Get the latest processing result."""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop running in background thread."""
        while self.is_processing:
            try:
                # Get audio chunk with timeout
                try:
                    audio_buffer = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the chunk
                result = self._process_audio_chunk(audio_buffer)
                
                if result:
                    # Put result in queue
                    self.results_queue.put(result)
                    
                    # Call callback if provided
                    if self.on_result_callback:
                        try:
                            self.on_result_callback(result)
                        except Exception as e:
                            if self.on_error_callback:
                                self.on_error_callback(e)
                
                self.processing_queue.task_done()
                
            except Exception as e:
                if self.on_error_callback:
                    self.on_error_callback(e)
                time.sleep(0.1)  # Prevent tight error loop
    
    def _process_audio_chunk(self, audio_buffer: AudioBuffer) -> Optional[StreamingResult]:
        """Process a single audio chunk."""
        try:
            timestamp = time.time()
            
            # Transcribe audio
            transcript = ""
            try:
                if hasattr(transcribe_audio_buffer, '__call__'):
                    transcript = transcribe_audio_buffer(audio_buffer)
                else:
                    # Fallback to file-based transcription
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
                        self.audio_processor.save_buffer_to_file(audio_buffer, tmp.name)
                        from modules.speech_to_text import transcribe_audio
                        transcript = transcribe_audio(tmp.name)
            except Exception as e:
                print(f"[WARNING] Transcription error: {e}")
                transcript = ""
            
            # Skip if no meaningful transcript
            if not transcript or len(transcript.strip()) < 3:
                return None
            
            # Update accumulated transcript
            self.accumulated_transcript = (self.accumulated_transcript + " " + transcript).strip()
            
            # Audio emotion analysis
            audio_emotion = {}
            try:
                if hasattr(analyze_audio_emotion_buffer, '__call__'):
                    audio_emotion = analyze_audio_emotion_buffer(audio_buffer)
                else:
                    # Fallback to file-based analysis
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
                        self.audio_processor.save_buffer_to_file(audio_buffer, tmp.name)
                        from modules.emotion_audio import analyze_audio_emotion
                        audio_emotion = analyze_audio_emotion(tmp.name)
            except Exception as e:
                print(f"[WARNING] Audio emotion error: {e}")
                audio_emotion = {}
            
            # Text emotion analysis
            text_emotion = {}
            try:
                text_emotion = analyze_text_emotion(transcript)
            except Exception as e:
                print(f"[WARNING] Text emotion error: {e}")
                text_emotion = {}
            
            # Fusion
            confidence, dominant_emotion, fused_emotion = fuse_emotions(audio_emotion, text_emotion)
            
            # Keyword detection
            keywords_detected = []
            distress_token_from_keywords = ""
            try:
                keywords_detected = []  # Placeholder - implement keyword extraction
                distress_token_from_keywords = check_keywords(transcript, "low distress")
            except Exception as e:
                print(f"[WARNING] Keyword detection error: {e}")
            
            # Distress scoring
            try:
                distress_token = get_distress_token(dominant_emotion, confidence)
                if distress_token_from_keywords and distress_token_from_keywords != "low distress":
                    distress_token = distress_token_from_keywords
            except Exception as e:
                print(f"[WARNING] Distress mapping error: {e}")
                distress_token = "unknown"
            
            # Calculate distress score (0-1)
            distress_score = self._calculate_distress_score(
                fused_emotion, confidence, keywords_detected, distress_token
            )
            
            return StreamingResult(
                timestamp=timestamp,
                transcript=transcript,
                partial_transcript=self.accumulated_transcript,
                audio_emotion=audio_emotion,
                text_emotion=text_emotion,
                fused_emotion=fused_emotion,
                distress_score=distress_score,
                distress_token=distress_token,
                confidence=confidence,
                keywords_detected=keywords_detected,
                is_final=False
            )
            
        except Exception as e:
            if self.on_error_callback:
                self.on_error_callback(e)
            return None
    
    def _calculate_distress_score(self, fused_emotion: Dict[str, float], 
                                confidence: float, keywords: List[str], 
                                distress_token: str) -> float:
        """Calculate distress score from 0-1."""
        try:
            base_score = 0.0
            
            # Emotion contribution
            if fused_emotion:
                # Higher scores for negative emotions
                negative_emotions = ['angry', 'fear', 'sad', 'disgust']
                for emotion, score in fused_emotion.items():
                    if emotion.lower() in negative_emotions:
                        base_score += score * 0.3
            
            # Confidence contribution
            base_score += confidence * 0.2
            
            # Keyword contribution
            base_score += len(keywords) * 0.1
            
            # Distress token contribution
            if distress_token == "high distress":
                base_score += 0.4
            elif distress_token == "medium distress":
                base_score += 0.2
            
            # Apply sensitivity adjustments
            sensitivity = self.config.fusion.sensitivity
            if sensitivity == "high":
                base_score *= 1.2
            elif sensitivity == "low":
                base_score *= 0.8
            
            return min(1.0, max(0.0, base_score))
            
        except Exception:
            return 0.0


class WebRTCAudioProcessor(AudioProcessorBase):
    """WebRTC audio processor for streamlit-webrtc."""
    
    def __init__(self, stream_processor: AudioStreamProcessor):
        self.stream_processor = stream_processor
        self.sample_rate = 16000
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frame."""
        try:
            # Convert frame to numpy array
            audio_data = frame.to_ndarray().flatten().astype(np.float32)
            
            # Normalize if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Add to processor
            self.stream_processor.add_audio_chunk(audio_data, frame.sample_rate)
            
        except Exception as e:
            print(f"[WARNING] WebRTC processing error: {e}")
        
        return frame


def create_streamlit_audio_interface(config=None) -> Optional[AudioStreamProcessor]:
    """Create Streamlit interface for real-time audio processing."""
    if not WEBRTC_AVAILABLE:
        st.error("[WARNING] Real-time audio requires streamlit-webrtc. Install with: pip install streamlit-webrtc")
        return None
    
    config = config or get_config_manager().config
    
    # Create processor
    stream_processor = AudioStreamProcessor(config)
    
    # WebRTC settings
    client_settings = ClientSettings(
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": False,
            "audio": {
                "sampleRate": config.audio.sample_rate,
                "channelCount": 1,
                "autoGainControl": True,
                "echoCancellation": True,
                "noiseSuppression": True
            }
        }
    )
    
    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="emergency-audio-stream",
        mode=st.cache_data,
        audio_processor_factory=lambda: WebRTCAudioProcessor(stream_processor),
        client_settings=client_settings,
        async_processing=True
    )
    
    # Handle streaming state
    if webrtc_ctx.state.playing:
        stream_processor.start_processing()
        return stream_processor
    else:
        stream_processor.stop_processing()
        return None


def create_mock_audio_interface(config=None) -> AudioStreamProcessor:
    """Create mock audio interface for testing without microphone."""
    config = config or get_config_manager().config
    
    stream_processor = AudioStreamProcessor(config)
    
    st.info("🎤 Mock Audio Mode - Use the buttons below to simulate audio input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😰 Simulate Distress Call"):
            # Generate mock distressed audio
            mock_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1  # 3 seconds
            stream_processor.add_audio_chunk(mock_audio)
            
            # Mock result
            mock_result = StreamingResult(
                timestamp=time.time(),
                transcript="Help me please, there's been an accident!",
                partial_transcript="Help me please, there's been an accident!",
                audio_emotion={"fear": 0.8, "sad": 0.6, "angry": 0.3},
                text_emotion={"fear": 0.9, "sad": 0.4},
                fused_emotion={"fear": 0.85, "sad": 0.5, "angry": 0.3},
                distress_score=0.85,
                distress_token="high distress",
                confidence=0.87,
                keywords_detected=["help", "accident"],
                is_final=True
            )
            stream_processor.results_queue.put(mock_result)
    
    with col2:
        if st.button("😊 Simulate Normal Call"):
            # Generate mock normal audio
            mock_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.05  # 2 seconds
            stream_processor.add_audio_chunk(mock_audio)
            
            # Mock result
            mock_result = StreamingResult(
                timestamp=time.time(),
                transcript="Hi, I just wanted to check if everything is okay.",
                partial_transcript="Hi, I just wanted to check if everything is okay.",
                audio_emotion={"happy": 0.6, "neutral": 0.8},
                text_emotion={"neutral": 0.9, "happy": 0.4},
                fused_emotion={"neutral": 0.85, "happy": 0.5},
                distress_score=0.15,
                distress_token="low distress",
                confidence=0.75,
                keywords_detected=[],
                is_final=True
            )
            stream_processor.results_queue.put(mock_result)
    
    with col3:
        if st.button("[EMERGENCY] Simulate Emergency"):
            # Generate mock emergency audio
            mock_audio = np.random.randn(16000 * 4).astype(np.float32) * 0.2  # 4 seconds
            stream_processor.add_audio_chunk(mock_audio)
            
            # Mock result
            mock_result = StreamingResult(
                timestamp=time.time(),
                transcript="Emergency! There's a fire in the building! Send help immediately!",
                partial_transcript="Emergency! There's a fire in the building! Send help immediately!",
                audio_emotion={"fear": 0.95, "angry": 0.7, "sad": 0.4},
                text_emotion={"fear": 0.98, "angry": 0.6},
                fused_emotion={"fear": 0.96, "angry": 0.65, "sad": 0.4},
                distress_score=0.96,
                distress_token="high distress",
                confidence=0.94,
                keywords_detected=["emergency", "fire", "help"],
                is_final=True
            )
            stream_processor.results_queue.put(mock_result)
    
    return stream_processor


def display_streaming_results(stream_processor: AudioStreamProcessor, 
                            placeholder_container):
    """Display real-time streaming results in Streamlit."""
    config = get_config_manager().config
    
    # Get latest result
    latest_result = stream_processor.get_latest_result()
    
    if latest_result:
        with placeholder_container.container():
            # Timestamp and status
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.metric("⏰ Timestamp", 
                         time.strftime("%H:%M:%S", time.localtime(latest_result.timestamp)))
            
            with col2:
                # Distress level with color coding
                distress_color = "🔴" if latest_result.distress_score > 0.7 else \
                               "🟡" if latest_result.distress_score > 0.4 else "🟢"
                st.metric(f"{distress_color} Distress Score", 
                         f"{latest_result.distress_score:.2f}", 
                         latest_result.distress_token)
            
            with col3:
                st.metric("[TARGET] Confidence", f"{latest_result.confidence:.2f}")
            
            # Transcript
            st.subheader("📝 Live Transcript")
            transcript_container = st.container()
            with transcript_container:
                # Show partial transcript with highlighting
                if latest_result.partial_transcript:
                    # Highlight recent additions
                    recent_text = latest_result.transcript
                    if recent_text in latest_result.partial_transcript:
                        highlighted = latest_result.partial_transcript.replace(
                            recent_text, f"**{recent_text}**"
                        )
                        st.markdown(highlighted)
                    else:
                        st.write(latest_result.partial_transcript)
                else:
                    st.write(latest_result.transcript)
            
            # Emotions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎵 Audio Emotions")
                if latest_result.audio_emotion:
                    for emotion, score in latest_result.audio_emotion.items():
                        st.progress(score, text=f"{emotion.title()}: {score:.2f}")
                else:
                    st.write("No audio emotion data")
            
            with col2:
                st.subheader("📄 Text Emotions")
                if latest_result.text_emotion:
                    for emotion, score in latest_result.text_emotion.items():
                        st.progress(score, text=f"{emotion.title()}: {score:.2f}")
                else:
                    st.write("No text emotion data")
            
            # Fused emotions
            if latest_result.fused_emotion:
                st.subheader("🔗 Fused Emotions")
                emotion_cols = st.columns(len(latest_result.fused_emotion))
                for i, (emotion, score) in enumerate(latest_result.fused_emotion.items()):
                    with emotion_cols[i]:
                        st.metric(emotion.title(), f"{score:.2f}")
            
            # Keywords
            if latest_result.keywords_detected:
                st.subheader("[SEARCH] Keywords Detected")
                keyword_tags = " ".join([f"`{kw}`" for kw in latest_result.keywords_detected])
                st.markdown(keyword_tags)
            
            # Separator
            st.divider()
        
        return True
    
    return False