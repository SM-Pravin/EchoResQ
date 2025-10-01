import streamlit as st
import tempfile
import os
import time
import pandas as pd

# Import the main processing function from your backend script
# Updated analysis_pipeline now exports process_audio_file and process_audio_file_stream
from analysis_pipeline import process_audio_file, process_audio_file_stream

st.set_page_config(
    page_title="Emergency AI Analysis",
    page_icon="🚨",
    layout="wide"
)

# --- Helper UI functions ---
def emotion_bar_chart(fused_scores):
    """
    Returns a simple Altair-friendly dataframe for emotion bar chart.
    fused_scores expected as {'angry':0.1,'happy':0.2,...}
    """
    try:
        df = pd.DataFrame([
            {"emotion": k.capitalize(), "score": float(v)}
            for k, v in (fused_scores or {}).items()
        ])
        if df.empty:
            return None
        return df
    except Exception:
        return None


def chunks_timeline_df(chunks):
    """
    Build a DataFrame suitable for a timeline/line chart.
    Each chunk should have: index, start_s, end_s, win_conf, win_distress
    We map distress tokens to numeric severity for plotting.
    """
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
    df = pd.DataFrame(rows)
    return df

# --- Sidebar controls ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
fast_mode = st.sidebar.checkbox("Enable Fast Mode", help="Skips heavy, chunked audio analysis for a quicker result.")
simulate_live = st.sidebar.checkbox("Enable 'Simulate Live Stream' (chunked updates)", value=False)

st.title("🚨 Emergency AI Analysis")
st.markdown("Upload an audio file (.wav, .mp3, .ogg). You can either run a full analysis or simulate live chunked processing.")

# --- Main UI ---
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Save it to a temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filepath = tmp_file.name

    try:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Actions")
            analyze_btn = st.button("Analyze Audio")
            simulate_btn = st.button("Simulate Live Stream") if not simulate_live else st.button("Start Simulated Live Stream")
        with col_right:
            st.subheader("Mode")
            st.write("Fast mode:" , "ON" if fast_mode else "OFF")
            st.write("Simulated Live Stream:", "ENABLED" if simulate_live else "DISABLED")

        # Run full analysis (one-shot)
        if analyze_btn:
            with st.spinner('Analyzing audio...'):
                results = process_audio_file(temp_filepath, fast_mode=fast_mode, return_chunks_details=True)

            if results.get("error"):
                st.error(f"An error occurred: {results['error']}")
            else:
                # show summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Distress Token", results['distress'].upper())
                col2.metric("Final Emotion", results['emotion'].upper())
                col3.metric("Confidence Score", f"{results['confidence']:.2f}")

                # Emotion bar chart
                df_bar = emotion_bar_chart(results.get("fused_scores", {}))
                if df_bar is not None and not df_bar.empty:
                    st.subheader("Fused Emotion Scores")
                    try:
                        import altair as alt
                        chart = alt.Chart(df_bar).mark_bar().encode(
                            x=alt.X('emotion:N', sort='-y'),
                            y=alt.Y('score:Q'),
                            tooltip=['emotion', 'score']
                        ).properties(height=200)
                        st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        st.write(df_bar.set_index('emotion'))

                # Chunk timeline
                chunks = results.get("chunks", [])
                df_timeline = chunks_timeline_df(chunks)
                if not df_timeline.empty:
                    st.subheader("Chunk-level Distress Timeline")
                    try:
                        import altair as alt
                        line = alt.Chart(df_timeline).mark_line(point=True).encode(
                            x='mid_s:Q',
                            y='severity:Q',
                            tooltip=['index', 'start_s', 'end_s', 'win_emotion', 'win_conf', 'win_distress']
                        ).properties(height=200)
                        st.altair_chart(line, use_container_width=True)

                        st.caption("Severity: 4=peak emergency, 3=high, 2=medium, 1=low")
                    except Exception:
                        st.write(df_timeline)

                # Transcript and chunk table
                with st.expander("📜 Full Transcript"):
                    st.write(results['transcript'] or '(No speech detected)')
                if chunks:
                    st.subheader("Chunk-level Details")
                    try:
                        # Flatten chunk info for a table
                        table_rows = []
                        for c in chunks:
                            table_rows.append({
                                "index": c.get("index"),
                                "start_s": round(c.get("start_s", 0.0), 2),
                                "end_s": round(c.get("end_s", 0.0), 2),
                                "rms": round(c.get("rms", 0.0), 5),
                                "emotion": (c.get("win_emotion") or "None"),
                                "conf": round(c.get("win_conf", 2) if isinstance(c.get("win_conf"), float) else 0.0, 2),
                                "distress": c.get("win_distress")
                            })
                        df_table = pd.DataFrame(table_rows)
                        st.dataframe(df_table)
                    except Exception:
                        st.write(chunks)

                if results.get('reason'):
                    st.info(f"💡 Reason: {results['reason']}")
                st.success("Analysis complete!")

        # Simulated live streaming processing: process chunks sequentially and update UI
        if simulate_btn:
            # placeholders
            status_ph = st.empty()
            metrics_ph = st.empty()
            chart_ph = st.empty()
            chunks_table_ph = st.empty()
            transcript_ph = st.empty()

            # session store for accumulating chunk results
            if "live_chunks" not in st.session_state:
                st.session_state["live_chunks"] = []

            st.session_state["live_chunks"].clear()

            def _on_chunk(chunk):
                """
                Callback executed for each chunk processed by process_audio_file_stream.
                Updates session_state and UI placeholders.
                """
                # append chunk
                st.session_state["live_chunks"].append(chunk)

                # update current metrics
                last = st.session_state["live_chunks"][-1]
                with metrics_ph.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Chunk Index", str(last.get("index")))
                    c2.metric("Chunk Emotion", (last.get("win_emotion") or "None"))
                    c3.metric("Chunk Confidence", f"{last.get('win_conf'):.2f}")

                # update timeline chart
                df_live = chunks_timeline_df(st.session_state["live_chunks"])
                if not df_live.empty:
                    try:
                        import altair as alt
                        line = alt.Chart(df_live).mark_line(point=True).encode(
                            x='mid_s:Q',
                            y='severity:Q',
                            tooltip=['index', 'start_s', 'end_s', 'win_emotion', 'win_conf', 'win_distress']
                        ).properties(height=240)
                        chart_ph.altair_chart(line, use_container_width=True)
                    except Exception:
                        chart_ph.write(df_live)

                # update chunk table
                try:
                    table_rows = []
                    for c in st.session_state["live_chunks"]:
                        table_rows.append({
                            "index": c.get("index"),
                            "start_s": round(c.get("start_s", 0.0), 2),
                            "end_s": round(c.get("end_s", 0.0), 2),
                            "rms": round(c.get("rms", 0.0), 5),
                            "emotion": (c.get("win_emotion") or "None"),
                            "conf": round(c.get("win_conf", 2) if isinstance(c.get("win_conf"), float) else 0.0, 2),
                            "distress": c.get("win_distress")
                        })
                    chunks_table_ph.dataframe(pd.DataFrame(table_rows))
                except Exception:
                    chunks_table_ph.write(st.session_state["live_chunks"])

                # transcript (only show once)
                if not transcript_ph:
                    transcript_ph.write("Transcript will appear at end of stream.")

            status_ph.info("Starting simulated live stream (chunked processing)...")

            # run the streaming processor (this runs synchronously while updating UI via callback)
            try:
                final = process_audio_file_stream(temp_filepath, fast_mode=fast_mode,
                                                 chunk_callback=_on_chunk, simulate_realtime=True)
                status_ph.success("Simulated stream complete.")
                # final summary
                if final.get("error"):
                    st.error(f"Error during streaming: {final['error']}")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Final Distress", final['distress'].upper())
                    col2.metric("Final Emotion", final['emotion'].upper())
                    col3.metric("Confidence", f"{final['confidence']:.2f}")

                    df_bar = emotion_bar_chart(final.get("fused_scores", {}))
                    if df_bar is not None and not df_bar.empty:
                        st.subheader("Final Fused Emotion Scores")
                        try:
                            import altair as alt
                            chart = alt.Chart(df_bar).mark_bar().encode(
                                x=alt.X('emotion:N', sort='-y'),
                                y=alt.Y('score:Q'),
                                tooltip=['emotion', 'score']
                            ).properties(height=200)
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            st.write(df_bar.set_index('emotion'))

                    with st.expander("📜 Full Transcript (final)"):
                        st.write(final.get("transcript") or "(No speech detected)")
                    if final.get("reason"):
                        st.info(f"💡 Reason: {final['reason']}")
            except Exception as e:
                status_ph.error(f"Streaming failed: {e}")

    finally:
        # Clean up temp file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
else:
    st.info("Please upload an audio file and click 'Analyze Audio' or 'Simulate Live Stream' to begin.")
