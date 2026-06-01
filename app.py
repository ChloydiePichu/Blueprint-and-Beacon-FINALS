import streamlit as st
import pandas as pd
from pypdf import PdfReader
from tracker import run_full_analysis
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is None:
        return ""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="B&B", page_icon="🎙️", layout="wide")

# --- Premium Enterprise CSS ---
st.markdown("""
    <style>
    /* Removed 'header {visibility: hidden}' so the sidebar toggle returns! */
    footer {visibility: hidden !important;}
    
    .stButton>button {
        background-color: #3B82F6 !important; 
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid #2563EB !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #2563EB !important;
        border-color: #1D4ED8 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }
    [data-testid="stFileUploadDropzone"] {
        border-radius: 8px !important;
        border: 1px dashed #4B5563 !important;
        background-color: #161B22 !important;
        padding: 2rem !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        background-color: #0E1117 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Blueprint & Beacon: AI Elevator Pitch Coach")
st.markdown("Practice your 60-second pitch. Our AI will track your timeline, cross-reference your resume, and generate actionable insights.")
st.divider()

current_run_data = None

# --- Main Layout ---
setup_container = st.container(border=True)
with setup_container:
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.subheader("📄 1. Provide Context")
        st.caption("Upload your resume to allow the AI to cross-reference your background.")
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"], label_visibility="collapsed")
        
        resume_text = ""
        if uploaded_resume is not None:
            resume_text = extract_text_from_pdf(uploaded_resume)
            st.success("✓ Resume parsed and loaded into AI context memory.")

    with col2:
        st.subheader("🎙️ 2. Execution")
        st.caption("The tracking window will open externally. Ensure your camera and microphone are clear before starting.")
        st.write("") 
        st.write("")
        
        if st.button("Start AI Pitch Analysis", use_container_width=True):
            with st.spinner("Initializing neural tracking models... Please interact with the camera window."):
                current_run_data = run_full_analysis(resume_text=resume_text)

# --- OUTSIDE THE COLUMNS: Full Width Rendering ---
if current_run_data:
    final_report, transcript, time_series = current_run_data
    
    if transcript == "":
        st.error(final_report)
    else:
        st.success("✓ Session successfully captured and analyzed.")
        
        st.session_state.history.append({
            "time": datetime.now().strftime("%I:%M %p"),
            "transcript": transcript,
            "report": final_report
        })
        
        st.divider()
        st.subheader("📈 Session Telemetry")
        st.caption("Real-time plotting of physical engagement metrics.")
        df = pd.DataFrame(time_series).set_index("Time (s)")
        st.line_chart(df)
        
        with st.expander("Show Raw Audio Transcript"):
            st.write(transcript)
            
        st.divider()
        st.subheader("📊 Executive Coaching Report")
        st.markdown(final_report)

# --- Sidebar: Pitch History ---
with st.sidebar:
    st.header("🕰️ Your Pitch History")
    
    if len(st.session_state.history) == 0:
        st.info("No previous pitches recorded yet.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Pitch from {entry['time']}"):
                st.write("**Transcript:**")
                st.caption(f'"{entry["transcript"]}"')
                st.download_button(
                    label="Download Report",
                    data=entry["report"],
                    file_name=f"pitch_report_{idx}.md",
                    mime="text/markdown",
                    key=f"download_{idx}"
                )