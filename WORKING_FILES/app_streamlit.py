import streamlit as st
import tempfile
import os
import time
import pandas as pd
import numpy as np
import json
from modules.env_config import get_config_snapshot
from modules.logger import log_error

# Configure Streamlit page
st.set_page_config(page_title="Emergency AI", page_icon="🚨", layout="wide")

# Environment optimization for performance
os.environ.setdefault("PARALLEL_MAX_WORKERS", "4")
os.environ.setdefault("ENABLE_BATCH_PROCESSING", "true")
os.environ.setdefault("AUDIO_BATCH_SIZE", "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("VOSK_LOG_LEVEL", "-1")

# Import the processing functions only once with caching
# Cache processing function imports to avoid re-import and reloading models
@st.cache_resource
def load_processing_functions():
    """Load processing functions with caching to prevent duplicate model loading."""
    try:
        from analysis_pipeline import process_audio_file, process_audio_file_stream
        return process_audio_file, process_audio_file_stream
    except Exception as e:
        log_error("app_streamlit.load_processing_functions", e)
        st.error(f"Failed to load processing functions: {e}")
        return None, None

# Load functions
process_audio_file, process_audio_file_stream = load_processing_functions()

def _inject_base_css(theme: str = "dark"):
    """Inject base CSS with CSS variables supporting dark/light themes.

    We define two theme scopes using [data-theme="dark"] and [data-theme="light"] on the body.
    The Python function ensures the body tag receives the correct attribute via a small JS snippet.
    Existing elements now reference CSS variables instead of hard-coded colors so the theme toggle
    results in immediate visual changes without reloading heavy components.
    """
    # Core variable palettes (ensure adequate contrast)
    css = '''
    <style>
    :root {
        --radius-sm: .35rem;
        --radius-md: .65rem;
        --radius-pill: 999px;
        --font-xs: .60rem;
        --transition-fast: .25s ease;
    }
    body[data-theme="dark"] {
        --color-bg: #0f1115;
        --color-bg-alt: #1b2027;
        --color-border: #2c3540;
        --color-border-soft: #2a313a;
        --color-text: #f3f6fa;
        --color-text-dim: #9aa6b6;
        --color-accent: #ff4d4f;
        --gradient-metric: linear-gradient(145deg,#1e252d,#242d37);
        --metric-shadow: 0 2px 4px -2px #0008;
        --pill-low-bg:#1d3a26; --pill-low-fg:#62d992;
    --pill-medium-bg:#403614; --pill-medium-fg:#f7c948;
        --pill-high-bg:#3f1d1d; --pill-high-fg:#ff8d7e;
        --pill-peak-bg:#540000; --pill-peak-fg:#ffb3b3; --pill-peak-border:#ff7474;
    }
    body[data-theme="light"] {
        --color-bg: #f7f9fc;
        --color-bg-alt: #ffffff;
        --color-border: #d4dce4;
        --color-border-soft: #e2e8ef;
        --color-text: #1f2429;
        --color-text-dim: #5a6775;
        --color-accent: #c92a2f;
        --gradient-metric: linear-gradient(145deg,#f0f4f8,#ffffff);
        --metric-shadow: 0 2px 4px -2px #0001;
        --pill-low-bg:#e3f7e9; --pill-low-fg:#0f6d39;
        --pill-medium-bg:#fff4d6; --pill-medium-fg:#8a5b00;
        --pill-high-bg:#ffe2dd; --pill-high-fg:#b61010;
    --pill-peak-bg:#4d0a0a; --pill-peak-fg:#ffb3b3; --pill-peak-border:#d94c4c;
    }

    body { background: var(--color-bg); color: var(--color-text); transition: background var(--transition-fast), color var(--transition-fast); }
    section.main > div { padding-top: 1rem; }
    h1, h2, h3 { font-weight:600; }
    .metric-cluster { display:flex; gap:0.75rem; flex-wrap:wrap; margin:.85rem 0 1.2rem; }
    .metric-box { background: var(--gradient-metric); border:1px solid var(--color-border); padding:.65rem .9rem; border-radius:var(--radius-md); min-width:120px; box-shadow:var(--metric-shadow); }
    .metric-label { font-size:var(--font-xs); text-transform:uppercase; letter-spacing:1px; opacity:.66; }
    .metric-value { font-size:1.15rem; font-weight:600; line-height:1.1; }
    .distress-pill { display:inline-block; padding:0.40rem 0.95rem; border-radius:var(--radius-pill); font-size:.70rem; font-weight:600; letter-spacing:.5px; text-transform:uppercase; background:var(--pill-low-bg); color:var(--pill-low-fg); border:1px solid transparent; }
    .pill-low { background:var(--pill-low-bg); color:var(--pill-low-fg); }
    .pill-medium { background:var(--pill-medium-bg); color:var(--pill-medium-fg); }
    .pill-high { background:var(--pill-high-bg); color:var(--pill-high-fg); }
    .pill-peak { background:var(--pill-peak-bg); color:var(--pill-peak-fg); border:1px solid var(--pill-peak-border); }
    div[role="tablist"] button { padding:0.35rem 0.75rem !important; font-size:0.8rem !important; }
    div[data-testid="stDataFrame"] { border:1px solid var(--color-border-soft); border-radius:.5rem; }
    .subtle { font-size:0.65rem; opacity:.55; }
    #MainMenu, footer { visibility:hidden; }
    /* Skeleton */
    .skeleton-bar { position:relative; overflow:hidden; background:var(--color-border-soft); height:16px; border-radius:8px; }
    .skeleton-bar::after { content:""; position:absolute; inset:0; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent); animation: shimmer 1.2s infinite; }
    @keyframes shimmer { 0% { transform:translateX(-100%);} 100% { transform:translateX(100%);} }
        </style>
        <script>
            const desired = '__THEME__';
            const b = window.parent.document.body;
            if (b.getAttribute('data-theme') !== desired) { b.setAttribute('data-theme', desired); }
        </script>
    '''
    st.markdown(css, unsafe_allow_html=True)

# --- Early session state initialization & initial CSS inject ---
if 'ui_theme' not in st.session_state:
    st.session_state.ui_theme = 'dark'
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'call_history' not in st.session_state:
    st.session_state.call_history = []
_inject_base_css(st.session_state.ui_theme)


def render_metric_cluster(emotion: str, confidence: float, proc_time: float, chunks: int):
    html = f"""
    <div class='metric-cluster'>
      <div class='metric-box'><div class='metric-label'>Emotion</div><div class='metric-value'>{emotion}</div></div>
      <div class='metric-box'><div class='metric-label'>Confidence</div><div class='metric-value'>{confidence:.2f}</div></div>
      <div class='metric-box'><div class='metric-label'>Proc Time</div><div class='metric-value'>{proc_time:.2f}s</div></div>
      <div class='metric-box'><div class='metric-label'>Chunks</div><div class='metric-value'>{chunks}</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sticky_header(distress: str, emotion: str, confidence: float, proc_time: float, chunks: int, df_timeline=None):
    """Render a sticky header with distress pill, metrics, and optional sparkline severity."""
    pill_class = 'pill-low'
    d_lower = distress.lower()
    if 'medium' in d_lower:
        pill_class = 'pill-medium'
    if 'high' in d_lower:
        pill_class = 'pill-high'
    if 'peak' in d_lower:
        pill_class = 'pill-peak'

    spark_html = ''
    try:
        if df_timeline is not None and not df_timeline.empty:
            import altair as alt
            # Build minimal inline chart
            spark = alt.Chart(df_timeline).mark_area(line={'color':'#ff4d4f','strokeWidth':1}, color='rgba(255,77,79,0.25)').encode(
                x=alt.X('mid_s:Q', title=None),
                y=alt.Y('severity:Q', title=None, scale=alt.Scale(domain=[0,5]))
            ).properties(height=40, width=180, padding={'left':0,'right':0,'top':0,'bottom':0})
            spark_html = st._repr_html_(spark) if hasattr(st, '_repr_html_') else ''
    except Exception:
        pass

    container_html = f"""
    <div id='sticky-header' style='position:sticky; top:0; z-index:50; backdrop-filter:blur(8px); background:linear-gradient(var(--color-bg-alt), var(--color-bg-alt) 60%, rgba(0,0,0,0)); padding:.6rem .4rem .4rem; margin-bottom:.35rem; border-bottom:1px solid var(--color-border-soft);'>
       <div style='display:flex; align-items:center; gap:.85rem; flex-wrap:wrap;'>
          <span class='distress-pill {pill_class}'>DISTRESS: {distress.upper()}</span>
          <div class='metric-cluster' style='margin:.25rem 0 .25rem;'>
             <div class='metric-box'><div class='metric-label'>Emotion</div><div class='metric-value'>{emotion}</div></div>
             <div class='metric-box'><div class='metric-label'>Conf</div><div class='metric-value'>{confidence:.2f}</div></div>
             <div class='metric-box'><div class='metric-label'>Time</div><div class='metric-value'>{proc_time:.2f}s</div></div>
             <div class='metric-box'><div class='metric-label'>Chunks</div><div class='metric-value'>{chunks}</div></div>
          </div>
          {spark_html}
       </div>
    </div>
    """
    st.markdown(container_html, unsafe_allow_html=True)


# Theme toggle (stored in session)
if 'ui_theme' not in st.session_state:
    st.session_state.ui_theme = 'dark'
# Initial injection will occur after potential sidebar theme change to avoid double flash

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
st.title("🚨 Emergency AI")
st.caption("Real-time distress & emotion analysis for emergency audio calls")

# Performance Settings in Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload audio", type=["wav","mp3","ogg"], help="Provide a single spoken emergency call clip")
    fast_mode = st.toggle("Fast Mode", help="Skip deep per-chunk analysis; quick triage")
    use_streaming = st.toggle("Streaming Mode", help="Simulated realtime sequential chunk updates")
    with st.expander("Performance", expanded=False):
        workers = st.slider("Workers", 1, 8, 4)
        batch_size = st.slider("Batch Size", 1, 16, 8)
        enable_batch = st.checkbox("Batch Processing", value=True)
        os.environ["PARALLEL_MAX_WORKERS"] = str(workers)
        os.environ["AUDIO_BATCH_SIZE"] = str(batch_size)
        os.environ["ENABLE_BATCH_PROCESSING"] = str(enable_batch).lower()
    with st.expander("Appearance", expanded=False):
        new_theme = st.radio("Theme", ["dark","light"], horizontal=True, index=0 if st.session_state.ui_theme=='dark' else 1)
        if new_theme != st.session_state.ui_theme:
            st.session_state.ui_theme = new_theme
        # (Re)inject CSS reflecting current theme selection
        _inject_base_css(st.session_state.ui_theme)
    if os.environ.get("SHOW_CONFIG","0") == "1":
        with st.expander("Config", expanded=False):
            st.json(get_config_snapshot())
    with st.expander("History", expanded=False):
        if st.session_state.call_history:
            # Show only caller IDs (index) and distress summary
            ids = [f"#{i+1}: {c['distress'].upper()} ({c['emotion']} {c['confidence']:.2f})" for i,c in enumerate(st.session_state.call_history[-25:])]
            for item in reversed(ids):
                st.write(item)
            # Download buttons
            hist_json = json.dumps(st.session_state.call_history, indent=2)
            st.download_button("Download History JSON", hist_json, file_name="call_history.json", mime="application/json")
            # Attempt log file download
            log_path = os.path.join('logs','system.log')
            if os.path.exists(log_path):
                try:
                    with open(log_path,'r',encoding='utf-8', errors='ignore') as lf:
                        log_text = lf.read()
                    st.download_button("Download System Log", log_text, file_name="system.log", mime="text/plain")
                except Exception:
                    pass
        else:
            st.caption("No history yet")

# Performance monitoring (already initialized earlier)

# Main processing area
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Waveform preview (lightweight) - only decode small portion if large
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(uploaded_file)
        # Downsample for visualization speed
        if data.ndim > 1:
            data = data.mean(axis=1)
        max_points = 1200
        step = max(1, len(data)//max_points)
        ds = data[::step]
        import pandas as pd
        wf_df = pd.DataFrame({ 'x': np.arange(len(ds))/sr*step, 'amp': ds })
        try:
            import altair as alt
            wf_chart = alt.Chart(wf_df).mark_area(opacity=0.6).encode(
                x=alt.X('x:Q', title='Time (s)'),
                y=alt.Y('amp:Q', title='Amplitude')
            ).properties(height=120)
            st.altair_chart(wf_chart, use_container_width=True)
        except Exception:
            st.line_chart(wf_df.set_index('x'))
    except Exception:
        pass

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filepath = tmp_file.name

    try:
        # Analysis buttons
        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            analyze_btn = st.button("Run Analysis", use_container_width=True)
            stream_btn = st.button("Start Streaming", use_container_width=True) if use_streaming else False
        with col_right:
            pass

        # Standard Analysis
        if analyze_btn and process_audio_file:
            # Skeleton placeholder block
            sk_cols = st.columns(4)
            for c in sk_cols:
                c.markdown('<div class="skeleton-bar"></div>', unsafe_allow_html=True)
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
                st.error(f"Analysis failed: {results['error']}")
            else:
                chunks = results.get("chunks", [])
                df_for_spark = chunks_timeline_df(chunks)
                render_sticky_header(results['distress'], results['emotion'].upper(), results['confidence'], processing_time, len(chunks), df_for_spark)
                # Append call history entry
                st.session_state.call_history.append({
                    'distress': results['distress'],
                    'emotion': results['emotion'],
                    'confidence': float(results.get('confidence',0.0)),
                    'processing_time': processing_time,
                    'chunks': len(chunks),
                    'fast_mode': fast_mode,
                    'timestamp': time.time()
                })

                tabs = st.tabs(["Overview", "Timeline", "Chunks", "Export"])  # transcript moved below
                # Overview
                with tabs[0]:
                    df_bar = emotion_bar_chart(results.get("fused_scores", {}))
                    if df_bar is not None and not df_bar.empty:
                        try:
                            import altair as alt
                            chart = alt.Chart(df_bar).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                                x=alt.X('emotion:N', sort='-y'),
                                y=alt.Y('score:Q'),
                                color=alt.Color('emotion:N', legend=None)
                            ).properties(height=280)
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            st.dataframe(df_bar, use_container_width=True)
                    if results.get('reason'):
                        st.caption(f"Insight: {results['reason']}")
                # Timeline
                with tabs[1]:
                    df_timeline = chunks_timeline_df(chunks)
                    if not df_timeline.empty:
                        try:
                            import altair as alt
                            line = alt.Chart(df_timeline).mark_line(point=True).encode(
                                x=alt.X('mid_s:Q', title='Time (s)'),
                                y=alt.Y('severity:Q', title='Severity', scale=alt.Scale(domain=[0,5])),
                                tooltip=['index','win_emotion','win_distress','win_conf']
                            ).properties(height=260)
                            st.altair_chart(line, use_container_width=True)
                            st.caption("Severity: 4=Peak • 3=High • 2=Medium • 1=Low")
                        except Exception:
                            st.dataframe(df_timeline, use_container_width=True)
                    else:
                        st.caption("No active windows or silent audio")
                # Chunks table
                with tabs[2]:
                    if chunks:
                        table_data = []
                        for c in chunks:
                            table_data.append({
                                "#": c.get("index"),
                                "Start": round(c.get("start_s",0.0),2),
                                "End": round(c.get("end_s",0.0),2),
                                "Dur": round(c.get("end_s",0.0)-c.get("start_s",0.0),2),
                                "RMS": round(c.get("rms",0.0),4),
                                "Emotion": c.get("win_emotion") or "-",
                                "Conf": round(c.get("win_conf",0.0),2),
                                "Distress": c.get("win_distress","low distress")
                            })
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No chunks analyzed")
                # Export
                with tabs[3]:
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(results, indent=2),
                        file_name="analysis_results.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                # Transcript (outside tabs to prevent scroll snap when switching)
                tr = results.get('transcript','').strip()
                if tr:
                    with st.expander("Transcript", expanded=False):  # Streamlit doesn't let us add id directly
                        st.markdown(tr)
                        st.caption(f"Words: {len(tr.split())}")
                else:
                    st.caption("Transcript: (none detected)")

        # Streaming Analysis
        elif stream_btn and process_audio_file_stream:
            st.subheader("🌊 Live Streaming Analysis")
            
            # Create placeholders (used after processing completes)
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            # Skeletons
            sk_cols = st.columns(4)
            for c in sk_cols:
                c.markdown('<div class="skeleton-bar"></div>', unsafe_allow_html=True)
            
            # Initialize session state for streaming
            if "live_chunks" not in st.session_state:
                st.session_state.live_chunks = []
            st.session_state.live_chunks.clear()
            
            def chunk_callback(chunk):
                """Lightweight callback: only collect chunks (no Streamlit calls here)."""
                try:
                    st.session_state.live_chunks.append(chunk)
                except Exception:
                    pass
            
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
                status_placeholder.success(f"✅ Streaming analysis completed in {processing_time:.2f} seconds!")
                
                # Final results
                if not final_results.get("error"):
                    distress = final_results['distress']
                    df_live = chunks_timeline_df(st.session_state.live_chunks)
                    render_sticky_header(distress, final_results['emotion'].upper(), final_results.get('confidence',0.0), processing_time, len(st.session_state.live_chunks), df_live)

                    # Render final timeline and metrics after completion
                    df_live = df_live  # already computed
                    if not df_live.empty:
                        try:
                            import altair as alt
                            line = alt.Chart(df_live).mark_line(point=True).encode(
                                x=alt.X('mid_s:Q', title='Time (s)'),
                                y=alt.Y('severity:Q', title='Severity'),
                                tooltip=['index', 'win_emotion', 'win_distress']
                            ).properties(height=200)
                            chart_placeholder.altair_chart(line, width='stretch')
                        except Exception:
                            chart_placeholder.line_chart(df_live.set_index('mid_s')['severity'])
                    st.download_button(
                        label="⬇️ Download JSON Results",
                        data=json.dumps(final_results, indent=2),
                        file_name="streaming_results.json",
                        mime="application/json",
                        width='stretch',
                    )
                    # Append to call history
                    st.session_state.call_history.append({
                        'distress': final_results['distress'],
                        'emotion': final_results['emotion'],
                        'confidence': float(final_results.get('confidence',0.0)),
                        'processing_time': processing_time,
                        'chunks': len(st.session_state.live_chunks),
                        'fast_mode': fast_mode,
                        'streaming': True,
                        'timestamp': time.time()
                    })
                    
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

if not uploaded_file:
        st.info("Upload an audio file to begin analysis.")

# Scroll position persistence (avoid snapping back on rerun)
st.markdown("""
<script>
const key = 'scroll-pos';
window.addEventListener('load', () => {
    const y = sessionStorage.getItem(key);
    if (y) { window.scrollTo(0, parseInt(y)); }
});
document.addEventListener('scroll', () => {
    sessionStorage.setItem(key, window.scrollY.toString());
}, { passive: true });
// Transcript hotkey (press 't') - finds the details element containing Transcript label
document.addEventListener('keydown', (e) => {
  if (e.key === 't' || e.key === 'T') {
      const details = Array.from(document.querySelectorAll('details'))
          .find(d => d.innerText && d.innerText.trim().startsWith('Transcript'));
      if (details) { details.open = !details.open; }
  }
});
</script>
""", unsafe_allow_html=True)

st.markdown("<p class='subtle'>Emergency AI • minimal interface • v1 UI refactor</p>", unsafe_allow_html=True)
