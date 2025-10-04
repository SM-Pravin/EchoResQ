import streamlit as st
import os
import sys
import time
import json
import tempfile
from modules.logger import log_error
from modules.config_manager import get_config_manager

# Parse CLI arguments before Streamlit starts
config_manager = get_config_manager()

# Check for CLI arguments passed after -- in streamlit run command
if "--" in sys.argv:
    dash_index = sys.argv.index("--")
    streamlit_args = sys.argv[dash_index + 1:]
    
    # Create a temporary parser for streamlit-specific overrides
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], help='Detection sensitivity')
    parser.add_argument('--live-audio', action='store_true', help='Enable live audio (future use)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    try:
        parsed_args, _ = parser.parse_known_args(streamlit_args)
        config_manager.apply_cli_overrides(parsed_args)
    except:
        pass  # Ignore parsing errors for Streamlit

# Get configuration
config = config_manager.config

# Set page config using values from configuration
st.set_page_config(
    page_title=config.ui.streamlit.get('page_title', 'Emergency AI (Minimal)'),
    page_icon=config.ui.streamlit.get('page_icon', '[EMERGENCY]'),
    layout=config.ui.streamlit.get('layout', 'wide')
)

@st.cache_resource
def load_pipeline():
    try:
        from analysis_pipeline import process_audio_file
        return process_audio_file
    except Exception as e:
        log_error("app_minimal.load_pipeline", e)
        st.error(f"Failed to import pipeline: {e}")
        return None

process_audio_file = load_pipeline()

# Simple style
st.markdown(
    """
    <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    .pill { display:inline-block; padding:.35rem .8rem; border-radius:999px; font-size:.70rem; letter-spacing:.5px; font-weight:600; }
    .pill-low { background:#e3f7e9; color:#0f6d39; }
    .pill-medium { background:#fff4d6; color:#8a5b00; }
    .pill-high { background:#ffe2dd; color:#b61010; }
    .pill-peak { background:#2d0000; color:#ffb3b3; border:1px solid #ff6d6d; }
    .divider { height:1px; background:linear-gradient(90deg,#444,#222); margin:1.25rem 0; opacity:.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("[EMERGENCY] Emergency AI – Minimal UI")
st.caption("Lightweight placeholder interface. Upload audio, run analysis, view structured results.")

col_u, col_opts = st.columns([3,1])
with col_u:
    uploaded = st.file_uploader("Audio file", type=["wav","mp3","ogg"], label_visibility="collapsed")
with col_opts:
    auto_run = st.checkbox("Auto analyze", value=True, help="Run immediately after upload")

run_button = st.button("Analyze", disabled=uploaded is None) if not auto_run else False

if uploaded is not None and (auto_run or run_button):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name
    start = time.perf_counter()
    result = None  # Initialize result variable
    if not process_audio_file:
        st.error("Pipeline not available.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = process_audio_file(path, fast_mode=False, return_chunks_details=True)
            except Exception as e:
                log_error("app_minimal.analysis", e)
                st.error(f"Analysis failed: {e}")
                result = None
    dur = time.perf_counter() - start
    try:
        os.remove(path)
    except Exception:
        pass

    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            distress = result.get("distress", "unknown").lower()
            pill_cls = "pill-low"
            if "medium" in distress: pill_cls = "pill-medium"
            if "high" in distress: pill_cls = "pill-high"
            if "peak" in distress: pill_cls = "pill-peak"
            st.markdown(f"<span class='pill {pill_cls}'>DISTRESS: {distress.upper()}</span>", unsafe_allow_html=True)

            # High-level summary metrics (native components, no white cards)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Emotion", result.get('emotion','-'))
            m2.metric("Confidence", f"{result.get('confidence',0):.2f}")
            m3.metric("Processing (s)", f"{dur:.2f}")
            m4.metric("Chunks", len(result.get('chunks',[])))

            # Optional reasoning / insight
            if result.get('reason'):
                st.markdown(f"**Insight:** {result['reason']}")

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Fused scores (flat bar list)
            fused = result.get('fused_scores') or {}
            if fused:
                st.subheader("Emotion Scores")
                for k,v in sorted(fused.items(), key=lambda x: -x[1]):
                    st.progress(min(max(v,0.0),1.0), text=f"{k.capitalize()} {v:.2f}")

            # Transcript
            tr = (result.get('transcript') or '').strip()
            if tr:
                with st.expander("Transcript"):
                    st.text_area("Transcript", value=tr, height=200, label_visibility="collapsed")
                    st.caption(f"Words: {len(tr.split())}")

            # Chunks (compact)
            chunks = result.get('chunks') or []
            if chunks:
                st.subheader("Chunks (summary)")
                # Show only essential columns
                compact = []
                for c in chunks:
                    compact.append({
                        'i': c.get('index'),
                        'start': round(c.get('start_s',0),2),
                        'end': round(c.get('end_s',0),2),
                        'emotion': c.get('win_emotion') or '-',
                        'conf': round(c.get('win_conf',0),2),
                        'distress': c.get('win_distress','-')
                    })
                import pandas as pd  # local import to keep top lean
                st.dataframe(pd.DataFrame(compact), width='stretch', hide_index=True)

            # Raw JSON utilities
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.subheader("Result JSON")
            pretty = json.dumps(result, indent=2)
            st.code(pretty, language='json')
            st.download_button("Download JSON", pretty, file_name="analysis.json", mime="application/json")
            st.button("Copy JSON to clipboard", on_click=lambda t=pretty: st.session_state.update({'_copy_json':t}))
            if st.session_state.get('_copy_json'):
                st.toast("JSON copied (place in clipboard manually with CTRL+C from highlighted block)")

else:
    st.info("Upload an audio file to begin.")

st.caption("Minimal placeholder UI • No persistent history • Esc to cancel focus")
