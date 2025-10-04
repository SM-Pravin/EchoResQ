"""
Advanced visualization module for Emergency AI developer insights.
Provides waveform visualization, performance metrics, and confidence analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import librosa.display
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime, timedelta
import tempfile

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.enhanced_logger import get_logger, PerformanceTracker
from modules.config_manager import get_config_manager
from analysis_pipeline import process_audio_file


class WaveformVisualizer:
    """Advanced waveform visualization with event overlays."""
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config_manager().config
    
    def create_waveform_plot(self, audio_file: str, analysis_results: Dict[str, Any] = None) -> go.Figure:
        """Create interactive waveform plot with event overlays."""
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            duration = len(audio) / sr
            time_axis = np.linspace(0, duration, len(audio))
        except Exception as e:
            st.error(f"Failed to load audio file: {e}")
            return go.Figure()
        
        # Create subplot with secondary y-axis for confidence
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Waveform with Events', 'Spectral Features', 'Confidence & Emotions'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. Main waveform
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=audio,
                mode='lines',
                name='Audio Waveform',
                line=dict(color='steelblue', width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 2. Spectral features (RMS energy)
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        fig.add_trace(
            go.Scatter(
                x=rms_times,
                y=rms,
                mode='lines',
                name='RMS Energy',
                line=dict(color='orange', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Add spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_times = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
        
        fig.add_trace(
            go.Scatter(
                x=spectral_times,
                y=spectral_centroids / 1000,  # Convert to kHz for readability
                mode='lines',
                name='Spectral Centroid (kHz)',
                line=dict(color='green', width=1.5),
                yaxis='y3'
            ),
            row=2, col=1
        )
        
        # 3. Add analysis results overlays if available
        if analysis_results:
            self._add_analysis_overlays(fig, analysis_results, duration)
        
        # Update layout
        fig.update_layout(
            title=f"Audio Analysis: {os.path.basename(audio_file)}",
            height=800,
            showlegend=True,
            hovermode='x unified',
            font=dict(size=12)
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="RMS Energy", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=3, col=1)
        
        return fig
    
    def _add_analysis_overlays(self, fig: go.Figure, results: Dict[str, Any], duration: float):
        """Add analysis results as overlays on the waveform."""
        
        # Voice Activity Detection regions
        if 'voice_segments' in results:
            for segment in results['voice_segments']:
                fig.add_vrect(
                    x0=segment.get('start', 0),
                    x1=segment.get('end', duration),
                    fillcolor="rgba(0, 255, 0, 0.2)",
                    layer="below",
                    line_width=0,
                    annotation_text="Voice Activity",
                    row=1, col=1
                )
        
        # Distress indicators
        if 'distress_score' in results and results['distress_score'] > 0.7:
            # Highlight high distress periods
            fig.add_hline(
                y=results['distress_score'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Distress: {results['distress_score']:.2f}",
                row=3, col=1
            )
        
        # Confidence timeline
        if 'confidence_timeline' in results:
            confidence_data = results['confidence_timeline']
            times = [point['time'] for point in confidence_data]
            confidences = [point['confidence'] for point in confidence_data]
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=confidences,
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='purple', width=2),
                    marker=dict(size=4)
                ),
                row=3, col=1
            )
        
        # Emotion indicators
        if 'emotions' in results:
            emotions = results['emotions']
            if isinstance(emotions, dict):
                # Create emotion intensity timeline
                emotion_colors = {
                    'anger': 'red',
                    'fear': 'orange', 
                    'sadness': 'blue',
                    'joy': 'green',
                    'surprise': 'yellow',
                    'disgust': 'brown',
                    'neutral': 'gray'
                }
                
                for emotion, intensity in emotions.items():
                    if intensity > 0.3:  # Only show significant emotions
                        fig.add_hline(
                            y=intensity,
                            line_dash="dot",
                            line_color=emotion_colors.get(emotion, 'gray'),
                            annotation_text=f"{emotion}: {intensity:.2f}",
                            row=3, col=1
                        )


class PerformanceVisualization:
    """Performance metrics visualization and analysis."""
    
    def __init__(self):
        self.logger = get_logger()
        self.performance_tracker = PerformanceTracker()
    
    def create_latency_breakdown_chart(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """Create module latency breakdown chart."""
        
        if not performance_data:
            return go.Figure().add_annotation(text="No performance data available")
        
        # Extract module performance data
        modules = []
        latencies = []
        memory_usage = []
        confidence_scores = []
        
        for data in performance_data:
            modules.append(data.get('module', 'Unknown'))
            latencies.append(data.get('duration_ms', 0))
            memory_usage.append(data.get('memory_delta_mb', 0))
            confidence_scores.append(data.get('confidence', 0))
        
        # Create subplot for multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Module Latency (ms)', 'Memory Usage (MB)', 
                          'Latency vs Confidence', 'Performance Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # 1. Latency bar chart
        fig.add_trace(
            go.Bar(
                x=modules,
                y=latencies,
                name='Latency (ms)',
                marker_color='steelblue',
                text=[f'{lat:.1f}ms' for lat in latencies],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Memory usage bar chart
        fig.add_trace(
            go.Bar(
                x=modules,
                y=memory_usage,
                name='Memory (MB)',
                marker_color='orange',
                text=[f'{mem:.1f}MB' for mem in memory_usage],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Latency vs Confidence scatter
        fig.add_trace(
            go.Scatter(
                x=latencies,
                y=confidence_scores,
                mode='markers+text',
                name='Modules',
                text=modules,
                textposition='top center',
                marker=dict(
                    size=10,
                    color=memory_usage,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Memory (MB)")
                )
            ),
            row=2, col=1
        )
        
        # 4. Performance timeline
        timestamps = [data.get('timestamp', datetime.now().isoformat()) for data in performance_data]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=latencies,
                mode='lines+markers',
                name='Latency Timeline',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        # Add confidence timeline on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidence_scores,
                mode='lines+markers',
                name='Confidence Timeline',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6),
                yaxis='y4'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Emergency AI Performance Analysis",
            height=700,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Modules", row=1, col=1)
        fig.update_xaxes(title_text="Modules", row=1, col=2)
        fig.update_xaxes(title_text="Latency (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=2)
        fig.update_yaxes(title_text="Confidence", secondary_y=True, row=2, col=2)
        
        return fig
    
    def create_confidence_heatmap(self, analysis_history: List[Dict[str, Any]]) -> go.Figure:
        """Create confidence heatmap over time and conditions."""
        
        if not analysis_history:
            return go.Figure().add_annotation(text="No analysis history available")
        
        # Extract data for heatmap
        timestamps = []
        audio_conditions = []
        confidence_scores = []
        
        for entry in analysis_history:
            timestamps.append(entry.get('timestamp', datetime.now().isoformat()))
            
            # Categorize audio conditions based on metadata
            metadata = entry.get('metadata', {})
            if metadata.get('noise_level', 'low') == 'high':
                condition = 'Noisy'
            elif metadata.get('volume_level') == 'whisper':
                condition = 'Whisper'
            elif metadata.get('volume_level') == 'loud':
                condition = 'Loud'
            elif metadata.get('speech_type') == 'rapid':
                condition = 'Rapid Speech'
            elif metadata.get('speech_type') == 'overlapping':
                condition = 'Overlapping'
            else:
                condition = 'Normal'
            
            audio_conditions.append(condition)
            confidence_scores.append(entry.get('confidence', 0))
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'condition': audio_conditions,
            'confidence': confidence_scores
        })
        
        # Create time bins (hourly)
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Pivot for heatmap
        heatmap_data = df.groupby(['hour', 'condition'])['confidence'].mean().unstack(fill_value=0)
        
        if heatmap_data.empty:
            return go.Figure().add_annotation(text="Insufficient data for heatmap")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=[str(idx) for idx in heatmap_data.index],
            colorscale='RdYlBu_r',
            zmid=0.5,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Confidence Score"),
            text=np.round(heatmap_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Confidence Heatmap by Time and Audio Conditions",
            xaxis_title="Audio Conditions",
            yaxis_title="Time (Hour)",
            height=500
        )
        
        return fig
    
    def create_system_resource_chart(self, resource_data: List[Dict[str, Any]]) -> go.Figure:
        """Create system resource utilization chart."""
        
        if not resource_data:
            return go.Figure().add_annotation(text="No resource data available")
        
        timestamps = [data.get('timestamp') for data in resource_data]
        cpu_usage = [data.get('cpu_percent', 0) for data in resource_data]
        memory_usage = [data.get('memory_mb', 0) for data in resource_data]
        processing_times = [data.get('duration_ms', 0) for data in resource_data]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (MB)', 'Processing Time (ms)')
        )
        
        # CPU usage
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cpu_usage,
                mode='lines+markers',
                name='CPU Usage',
                line=dict(color='red', width=2),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Memory usage
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_usage,
                mode='lines+markers',
                name='Memory Usage',
                line=dict(color='blue', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Processing time
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=processing_times,
                mode='lines+markers',
                name='Processing Time',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=3, col=1
        )
        
        # Add target lines
        fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                     annotation_text="CPU Warning (80%)", row=1, col=1)
        fig.add_hline(y=300, line_dash="dash", line_color="orange",
                     annotation_text="Latency Target (300ms)", row=3, col=1)
        
        fig.update_layout(
            title="System Resource Utilization",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return fig


class DeveloperDashboard:
    """Main developer dashboard for Emergency AI insights."""
    
    def __init__(self):
        self.waveform_viz = WaveformVisualizer()
        self.performance_viz = PerformanceVisualization()
        self.logger = get_logger()
    
    def render_dashboard(self):
        """Render the complete developer dashboard."""
        
        st.set_page_config(
            page_title="Emergency AI Developer Dashboard",
            page_icon="[EMERGENCY]",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("[EMERGENCY] Emergency AI Developer Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Upload an audio file to analyze"
            )
            
            # Analysis options
            st.subheader("Analysis Options")
            enable_waveform = st.checkbox("Waveform Visualization", value=True)
            enable_performance = st.checkbox("Performance Analysis", value=True)
            enable_confidence = st.checkbox("Confidence Analysis", value=True)
            
            # Real-time monitoring
            st.subheader("Real-time Monitoring")
            auto_refresh = st.checkbox("Auto-refresh data", value=False)
            if auto_refresh:
                refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        
        # Main dashboard content
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            try:
                # Process the audio file
                with st.spinner("Analyzing audio file..."):
                    analysis_results = process_audio_file(temp_path)
                
                # Display results summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Confidence Score",
                        f"{analysis_results.get('confidence', 0):.3f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Distress Level",
                        f"{analysis_results.get('distress_score', 0):.3f}",
                        delta=None,
                        delta_color="inverse"
                    )
                
                with col3:
                    transcript_length = len(analysis_results.get('transcript', ''))
                    st.metric(
                        "Transcript Length",
                        f"{transcript_length} chars",
                        delta=None
                    )
                
                with col4:
                    processing_time = analysis_results.get('processing_time_ms', 0)
                    st.metric(
                        "Processing Time",
                        f"{processing_time:.1f} ms",
                        delta=None
                    )
                
                st.markdown("---")
                
                # Waveform visualization
                if enable_waveform:
                    st.subheader("🎵 Waveform Analysis")
                    waveform_fig = self.waveform_viz.create_waveform_plot(temp_path, analysis_results)
                    st.plotly_chart(waveform_fig, use_container_width=True)
                
                # Performance analysis
                if enable_performance:
                    st.subheader("⚡ Performance Analysis")
                    
                    # Get performance data from the tracker
                    performance_data = self._get_recent_performance_data()
                    
                    if performance_data:
                        performance_fig = self.performance_viz.create_latency_breakdown_chart(performance_data)
                        st.plotly_chart(performance_fig, use_container_width=True)
                        
                        # System resources
                        resource_fig = self.performance_viz.create_system_resource_chart(performance_data)
                        st.plotly_chart(resource_fig, use_container_width=True)
                    else:
                        st.info("No performance data available. Process more files to see performance metrics.")
                
                # Confidence analysis
                if enable_confidence:
                    st.subheader("[TARGET] Confidence Analysis")
                    
                    # Get analysis history
                    analysis_history = self._get_analysis_history()
                    
                    if analysis_history:
                        confidence_fig = self.performance_viz.create_confidence_heatmap(analysis_history)
                        st.plotly_chart(confidence_fig, use_container_width=True)
                    else:
                        st.info("No analysis history available. Process more files to see confidence patterns.")
                
                # Detailed results
                with st.expander("[DASHBOARD] Detailed Analysis Results"):
                    st.json(analysis_results)
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                self.logger.error(f"Dashboard analysis error: {e}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        else:
            # Show sample dashboard with mock data
            st.info("👆 Upload an audio file to start analysis")
            self._render_sample_dashboard()
    
    def _get_recent_performance_data(self) -> List[Dict[str, Any]]:
        """Get recent performance data from logs or tracking."""
        # This would typically read from performance logs or database
        # For now, return sample data structure
        sample_data = [
            {
                'module': 'Speech-to-Text',
                'duration_ms': 250.5,
                'memory_delta_mb': 85.2,
                'confidence': 0.89,
                'timestamp': datetime.now().isoformat()
            },
            {
                'module': 'Emotion Detection',
                'duration_ms': 120.3,
                'memory_delta_mb': 42.1,
                'confidence': 0.76,
                'timestamp': datetime.now().isoformat()
            },
            {
                'module': 'Sound Classification',
                'duration_ms': 95.7,
                'memory_delta_mb': 38.5,
                'confidence': 0.82,
                'timestamp': datetime.now().isoformat()
            },
            {
                'module': 'Fusion Engine',
                'duration_ms': 45.2,
                'memory_delta_mb': 12.3,
                'confidence': 0.91,
                'timestamp': datetime.now().isoformat()
            }
        ]
        return sample_data
    
    def _get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history from logs."""
        # This would typically read from analysis logs
        # For now, return sample data
        base_time = datetime.now() - timedelta(hours=24)
        
        sample_history = []
        conditions = ['Normal', 'Noisy', 'Whisper', 'Loud', 'Rapid Speech', 'Overlapping']
        
        for i in range(50):  # Generate 50 sample entries
            timestamp = base_time + timedelta(minutes=i*30)
            condition = np.random.choice(conditions)
            
            # Simulate confidence based on condition
            if condition == 'Normal':
                confidence = np.random.normal(0.85, 0.1)
            elif condition == 'Noisy':
                confidence = np.random.normal(0.65, 0.15)
            elif condition == 'Whisper':
                confidence = np.random.normal(0.55, 0.2)
            else:
                confidence = np.random.normal(0.75, 0.12)
            
            confidence = np.clip(confidence, 0, 1)
            
            sample_history.append({
                'timestamp': timestamp.isoformat(),
                'confidence': confidence,
                'metadata': {
                    'noise_level': 'high' if condition == 'Noisy' else 'low',
                    'volume_level': condition.lower() if condition in ['Whisper', 'Loud'] else 'normal',
                    'speech_type': condition.lower().replace(' ', '_') if 'Speech' in condition or 'Overlapping' in condition else 'normal'
                }
            })
        
        return sample_history
    
    def _render_sample_dashboard(self):
        """Render sample dashboard with mock data."""
        st.subheader("[DASHBOARD] Sample Dashboard Preview")
        
        # Sample metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Confidence", "0.847", delta="0.023")
        
        with col2:
            st.metric("Avg Latency", "185.2 ms", delta="-12.5 ms")
        
        with col3:
            st.metric("Success Rate", "94.2%", delta="1.8%")
        
        with col4:
            st.metric("Memory Usage", "245 MB", delta="-18 MB")
        
        # Sample performance chart
        sample_perf_data = self._get_recent_performance_data()
        perf_fig = self.performance_viz.create_latency_breakdown_chart(sample_perf_data)
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # Sample confidence heatmap
        sample_history = self._get_analysis_history()
        conf_fig = self.performance_viz.create_confidence_heatmap(sample_history)
        st.plotly_chart(conf_fig, use_container_width=True)


def main():
    """Main entry point for the developer dashboard."""
    dashboard = DeveloperDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()