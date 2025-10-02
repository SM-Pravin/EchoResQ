import streamlit as st
import tempfile
import os
import time
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Configure Streamlit page
st.set_page_config(
    page_title="Emergency AI Analysis",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment optimization for performance
os.environ.setdefault("PARALLEL_MAX_WORKERS", "4")
os.environ.setdefault("ENABLE_BATCH_PROCESSING", "true")
os.environ.setdefault("AUDIO_BATCH_SIZE", "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")

# Import the processing functions only once with caching
@st.cache_resource
def load_processing_functions():
    """Load processing functions with caching to prevent duplicate model loading."""
    try:
        from analysis_pipeline import process_audio_file, process_audio_file_stream
        return process_audio_file, process_audio_file_stream
    except Exception as e:
        st.error(f"Failed to load processing functions: {e}")
        return None, None

# Load functions
process_audio_file, process_audio_file_stream = load_processing_functions()

# CSS for better styling
st.markdown("""
<style>
    .stMetric > div > div > div > div {
        font-size: 1.2rem;
        color: #262730;
    }
    .stAlert > div {
        padding: 1rem;
    }
    .chunk-info {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .performance-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0e7490;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def emotion_bar_chart(fused_scores):
    """Create emotion bar chart data."""
    try:
        df = pd.DataFrame([
            {"emotion": k.capitalize(), "score": float(v)}
            for k, v in (fused_scores or {}).items()
        ])
        return df if not df.empty else None
    except Exception:
        return None

def chunks_timeline_df(chunks):
    """Create timeline data for chunks."""
    if not chunks:
        return pd.DataFrame()

    severity_map = {
        "peak emergency distress": 4,
        "high distress": 3,
        "medium distress": 2,
        "low distress": 1
    }
    
    rows = []
    for c in chunks:
        rows.append({
            "index": c.get("index"),
            "start_s": c.get("start_s", 0.0),
            "end_s": c.get("end_s", 0.0),
            "mid_s": (c.get("start_s", 0.0) + c.get("end_s", 0.0)) / 2.0,
            "rms": c.get("rms", 0.0),
            "win_conf": c.get("win_conf", 0.0) or 0.0,
            "win_emotion": c.get("win_emotion") or "None",
            "win_distress": c.get("win_distress", "low distress"),
            "severity": severity_map.get(c.get("win_distress", "low distress"), 1)
        })
    return pd.DataFrame(rows)

def format_performance_info(processing_time, chunk_count, mode):
    """Format performance information."""
    chunks_per_sec = chunk_count / processing_time if processing_time > 0 else 0
    return f"""
    **Performance Metrics:**
    - Processing Time: {processing_time:.2f} seconds
    - Chunks Processed: {chunk_count}
    - Throughput: {chunks_per_sec:.1f} chunks/second
    - Mode: {mode}
    """

# Title and description
st.title("🚨 Emergency AI Analysis - Optimized")
st.markdown("""
**Enhanced with Performance Improvements:**
- ⚡ Parallel chunk processing
- 🔄 Batch audio inference  
- 🎯 Early keyword detection
- 💾 Optimized memory usage
""")

# Performance Settings in Sidebar
st.sidebar.header("⚙️ Performance Settings")

with st.sidebar.expander("🚀 Advanced Configuration", expanded=False):
    workers = st.slider("Parallel Workers", 1, 8, 4, help="Number of parallel workers for chunk processing")
    batch_size = st.slider("Batch Size", 1, 16, 8, help="Batch size for audio inference")
    enable_batch = st.checkbox("Enable Batch Processing", value=True, help="Use batch processing for better performance")
    
    # Update environment variables
    os.environ["PARALLEL_MAX_WORKERS"] = str(workers)
    os.environ["AUDIO_BATCH_SIZE"] = str(batch_size)
    os.environ["ENABLE_BATCH_PROCESSING"] = str(enable_batch).lower()

# Analysis Settings
st.sidebar.header("📊 Analysis Settings")
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
fast_mode = st.sidebar.checkbox("⚡ Fast Mode", help="Skip heavy analysis for quicker results")
use_streaming = st.sidebar.checkbox("🌊 Streaming Mode", help="Process chunks sequentially with real-time updates")

# Performance monitoring
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []

# Main processing area
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Performance info box
    with st.container():
        st.markdown('<div class="performance-info">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Workers", workers)
        col2.metric("Batch Size", batch_size)  
        col3.metric("Batch Mode", "ON" if enable_batch else "OFF")
        col4.metric("Fast Mode", "ON" if fast_mode else "OFF")
        st.markdown('</div>', unsafe_allow_html=True)

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filepath = tmp_file.name

    try:
        # Analysis buttons
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            analyze_btn = st.button("🔍 Analyze Audio", use_container_width=True)
            if use_streaming:
                stream_btn = st.button("🌊 Start Streaming Analysis", use_container_width=True)
            else:
                stream_btn = False
        
        with col_right:
            if st.button("📊 Benchmark", help="Run quick performance test"):
                st.info("Running benchmark... Check performance tab for results")

        # Standard Analysis
        if analyze_btn and process_audio_file:
            with st.spinner('🔄 Analyzing audio with performance optimizations...'):
                start_time = time.perf_counter()
                
                results = process_audio_file(
                    temp_filepath, 
                    fast_mode=fast_mode, 
                    return_chunks_details=True
                )
                
                processing_time = time.perf_counter() - start_time
                chunk_count = len(results.get("chunks", []))
                
                # Store performance data
                st.session_state.performance_history.append({
                    'time': processing_time,
                    'chunks': chunk_count,
                    'mode': 'Fast' if fast_mode else 'Full',
                    'workers': workers,
                    'batch_size': batch_size
                })

            if results.get("error"):
                st.error(f"❌ Analysis failed: {results['error']}")
            else:
                # Main results
                st.success(f"✅ Analysis completed in {processing_time:.2f} seconds!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🚨 Distress Level", results['distress'].replace('distress', '').strip().upper())
                col2.metric("😊 Emotion", results['emotion'].upper())
                col3.metric("🎯 Confidence", f"{results['confidence']:.2f}")
                col4.metric("⚡ Processing Time", f"{processing_time:.2f}s")
                
                # Performance info
                st.info(format_performance_info(processing_time, chunk_count, 'Standard Analysis'))
                
                # Tabs for organized results
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Timeline", "📜 Transcript", "⚙️ Details"])
                
                with tab1:
                    # Emotion scores chart
                    df_bar = emotion_bar_chart(results.get("fused_scores", {}))
                    if df_bar is not None:
                        st.subheader("Emotion Analysis")
                        try:
                            import altair as alt
                            chart = alt.Chart(df_bar).mark_bar().encode(
                                x=alt.X('emotion:N', sort='-y', title='Emotion'),
                                y=alt.Y('score:Q', title='Confidence Score'),
                                color=alt.Color('emotion:N', scale=alt.Scale(scheme='category10')),
                                tooltip=['emotion', 'score']
                            ).properties(height=300)
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            st.dataframe(df_bar, use_container_width=True)
                
                with tab2:
                    # Timeline chart
                    chunks = results.get("chunks", [])
                    df_timeline = chunks_timeline_df(chunks)
                    if not df_timeline.empty:
                        st.subheader("Distress Timeline")
                        try:
                            import altair as alt
                            
                            # Create timeline chart
                            line = alt.Chart(df_timeline).mark_line(
                                point=alt.OverlayMarkDef(filled=True, size=60)
                            ).encode(
                                x=alt.X('mid_s:Q', title='Time (seconds)'),
                                y=alt.Y('severity:Q', title='Distress Severity', scale=alt.Scale(domain=[0, 5])),
                                color=alt.Color('win_emotion:N', title='Emotion'),
                                tooltip=['index', 'start_s', 'end_s', 'win_emotion', 'win_conf', 'win_distress']
                            ).properties(height=400)
                            
                            st.altair_chart(line, use_container_width=True)
                            st.caption("📊 Severity Scale: 4=Peak Emergency, 3=High, 2=Medium, 1=Low")
                        except Exception as e:
                            st.dataframe(df_timeline, use_container_width=True)
                
                with tab3:
                    # Transcript
                    st.subheader("Speech Transcript")
                    transcript = results.get('transcript', '').strip()
                    if transcript:
                        st.write(transcript)
                        
                        # Word count and analysis
                        word_count = len(transcript.split())
                        st.caption(f"📝 Word count: {word_count} words")
                    else:
                        st.info("No speech detected in the audio")
                
                with tab4:
                    # Detailed chunk information
                    if chunks:
                        st.subheader("Chunk-Level Analysis")
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Chunks", len(chunks))
                        active_chunks = len([c for c in chunks if c.get('win_emotion')])
                        col2.metric("Active Chunks", active_chunks)
                        col3.metric("Processing Rate", f"{len(chunks)/processing_time:.1f} chunks/sec")
                        
                        # Detailed table
                        table_data = []
                        for c in chunks:
                            table_data.append({
                                "Chunk": c.get("index"),
                                "Start (s)": round(c.get("start_s", 0.0), 2),
                                "End (s)": round(c.get("end_s", 0.0), 2),
                                "Duration": round(c.get("end_s", 0.0) - c.get("start_s", 0.0), 2),
                                "RMS": round(c.get("rms", 0.0), 5),
                                "Emotion": c.get("win_emotion") or "Silent",
                                "Confidence": round(c.get("win_conf", 0.0), 3),
                                "Distress": c.get("win_distress", "low distress")
                            })
                        
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                
                # Additional insights
                if results.get('reason'):
                    st.info(f"💡 Analysis Insight: {results['reason']}")

        # Streaming Analysis
        elif stream_btn and process_audio_file_stream:
            st.subheader("🌊 Live Streaming Analysis")
            
            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Initialize session state for streaming
            if "live_chunks" not in st.session_state:
                st.session_state.live_chunks = []
            st.session_state.live_chunks.clear()
            
            def chunk_callback(chunk):
                """Callback for streaming chunk updates."""
                st.session_state.live_chunks.append(chunk)
                
                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Chunk", chunk.get("index", 0))
                    col2.metric("Emotion", chunk.get("win_emotion") or "Processing...")
                    col3.metric("Confidence", f"{chunk.get('win_conf', 0.0):.2f}")
                
                # Update progress (estimate based on chunk count)
                progress = min(len(st.session_state.live_chunks) / 10, 1.0)  # Assume ~10 chunks
                progress_bar.progress(progress)
                
                # Update timeline
                df_live = chunks_timeline_df(st.session_state.live_chunks)
                if not df_live.empty:
                    try:
                        import altair as alt
                        line = alt.Chart(df_live).mark_line(point=True).encode(
                            x=alt.X('mid_s:Q', title='Time (s)'),
                            y=alt.Y('severity:Q', title='Severity'),
                            tooltip=['index', 'win_emotion', 'win_distress']
                        ).properties(height=200)
                        chart_placeholder.altair_chart(line, use_container_width=True)
                    except Exception:
                        chart_placeholder.line_chart(df_live.set_index('mid_s')['severity'])
            
            # Run streaming analysis
            status_placeholder.info("🚀 Starting streaming analysis...")
            start_time = time.perf_counter()
            
            try:
                final_results = process_audio_file_stream(
                    temp_filepath,
                    fast_mode=fast_mode,
                    chunk_callback=chunk_callback,
                    simulate_realtime=True
                )
                
                processing_time = time.perf_counter() - start_time
                progress_bar.progress(1.0)
                status_placeholder.success(f"✅ Streaming analysis completed in {processing_time:.2f} seconds!")
                
                # Final results
                if not final_results.get("error"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🚨 Final Distress", final_results['distress'].upper())
                    col2.metric("😊 Final Emotion", final_results['emotion'].upper())
                    col3.metric("⏱️ Total Time", f"{processing_time:.2f}s")
                    
                    # Store performance data
                    st.session_state.performance_history.append({
                        'time': processing_time,
                        'chunks': len(st.session_state.live_chunks),
                        'mode': 'Streaming',
                        'workers': workers,
                        'batch_size': batch_size
                    })
                
            except Exception as e:
                status_placeholder.error(f"❌ Streaming failed: {e}")

    finally:
        # Cleanup
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass

# Performance History Tab
if st.session_state.performance_history:
    with st.expander("📈 Performance History", expanded=False):
        df_perf = pd.DataFrame(st.session_state.performance_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Processing Times")
            st.line_chart(df_perf[['time']])
        
        with col2:
            st.subheader("Throughput (chunks/sec)")
            df_perf['throughput'] = df_perf['chunks'] / df_perf['time']
            st.line_chart(df_perf[['throughput']])
        
        st.subheader("Performance Data")
        st.dataframe(df_perf, use_container_width=True)

else:
    st.info("📁 Please upload an audio file to begin analysis")

# Footer with tips
st.markdown("---")
st.markdown("""
**💡 Performance Tips:**
- Use **Fast Mode** for quick analysis
- Increase **Workers** for more parallel processing (if you have multiple CPU cores)
- Enable **Batch Processing** for better efficiency with multiple chunks
- **Streaming Mode** provides real-time updates but may be slower overall
""")
