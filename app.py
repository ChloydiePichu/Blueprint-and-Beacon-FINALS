import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pypdf import PdfReader
from dotenv import load_dotenv

from tracker import run_full_analysis, get_sample_analysis
import workato_service

load_dotenv()

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is None:
        return ""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error parsing PDF file: {e}")
        return ""

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

if "current_run_data" not in st.session_state:
    st.session_state.current_run_data = None

if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {
        "name": "",
        "email": "",
        "target_role": "Full Stack Developer",
        "experience_level": "Mid-Level (3-5 yrs)",
        "skills": ""
    }

if "resume_context" not in st.session_state:
    st.session_state.resume_context = {
        "text": "",
        "filename": ""
    }

if "workato_dispatch_result" not in st.session_state:
    st.session_state.workato_dispatch_result = None

# --- Page Configuration ---
st.set_page_config(
    page_title="Blueprint | Talent Intake",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Professional Enterprise Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    footer {visibility: hidden !important;}
    
    /* Global Container Adjustments */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Clean Enterprise Buttons */
    .stButton>button {
        background-color: #2563EB !important; 
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: 1px solid #1D4ED8 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.1rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        border-color: #1E40AF !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    }
    
    /* File Uploader Container */
    [data-testid="stFileUploadDropzone"] {
        border-radius: 6px !important;
        border: 1px dashed #374151 !important;
        background-color: #111827 !important;
        padding: 1.25rem !important;
    }
    
    /* Expander Containers */
    [data-testid="stExpander"] {
        border: 1px solid #1F2937 !important;
        border-radius: 6px !important;
        background-color: #0F172A !important;
    }
    
    /* Metric Card Presentation */
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-top: 2px solid #2563EB;
        border-radius: 6px;
        padding: 1rem 0.75rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F3F4F6;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    /* Subheadings & Dividers */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: #F9FAFB !important;
    }
    h2, h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #E5E7EB !important;
    }
    
    .status-panel {
        background-color: #064E3B;
        border: 1px solid #059669;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        color: #D1FAE5;
        font-size: 0.88rem;
        line-height: 1.4;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.title("Blueprint: Candidate Intake & Evaluation")
st.markdown("Automated pitch analysis, physical telemetry tracking, and structured candidate intake for enterprise workflow orchestration.")
st.divider()

# --- Main Setup Layout ---
setup_container = st.container(border=True)
with setup_container:
    col1, col2 = st.columns([1.1, 1], gap="large")
    
    with col1:
        st.subheader("1. Candidate Profile & Background")
        st.caption("Structured identity parameters for downstream project allocation and task provisioning.")
        
        name_val = st.text_input(
            "Full Name *",
            value=st.session_state.candidate_info["name"],
            placeholder="e.g., Alex Johnson"
        )
        email_val = st.text_input(
            "Contact Email *",
            value=st.session_state.candidate_info["email"],
            placeholder="e.g., alex.johnson@example.com"
        )
        
        role_col, exp_col = st.columns(2)
        with role_col:
            role_options = [
                "Full Stack Developer", "Backend Engineer", "Frontend Engineer",
                "AI/ML Engineer", "DevOps / Cloud Engineer", "Project Manager / Scrum Master",
                "UI/UX Designer", "Data Analyst / Scientist", "Other"
            ]
            current_role_idx = role_options.index(st.session_state.candidate_info["target_role"]) if st.session_state.candidate_info["target_role"] in role_options else 0
            role_val = st.selectbox("Target Role", role_options, index=current_role_idx)
        
        with exp_col:
            exp_options = ["Junior (1-2 yrs)", "Mid-Level (3-5 yrs)", "Senior (5+ yrs)", "Lead / Principal"]
            current_exp_idx = exp_options.index(st.session_state.candidate_info["experience_level"]) if st.session_state.candidate_info["experience_level"] in exp_options else 1
            exp_val = st.selectbox("Experience Level", exp_options, index=current_exp_idx)
            
        skills_val = st.text_input(
            "Primary Skills & Technologies (comma separated)",
            value=st.session_state.candidate_info["skills"],
            placeholder="e.g., Python, React, FastAPI, PostgreSQL, Docker, AWS"
        )
        
        # Sync session state
        st.session_state.candidate_info["name"] = name_val
        st.session_state.candidate_info["email"] = email_val
        st.session_state.candidate_info["target_role"] = role_val
        st.session_state.candidate_info["experience_level"] = exp_val
        st.session_state.candidate_info["skills"] = skills_val

        st.markdown("**Resume Context (PDF)**")
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"], label_visibility="collapsed")
        
        if uploaded_resume is not None:
            parsed_text = extract_text_from_pdf(uploaded_resume)
            st.session_state.resume_context["text"] = parsed_text
            st.session_state.resume_context["filename"] = uploaded_resume.name
            st.success(f"Resume parsed: {uploaded_resume.name} ({len(parsed_text)} characters loaded).")

    with col2:
        st.subheader("2. Pitch Assessment")
        st.caption("Deliver a pitch up to 60 seconds. Assesses behavioral presence, speech clarity, and resume alignment.")
        
        st.info("Device Access: Ensure camera and microphone permissions are granted before initiating the assessment.")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Start Live Assessment", use_container_width=True):
                with st.spinner("Initializing camera capture and tracking models..."):
                    res = run_full_analysis(resume_text=st.session_state.resume_context["text"])
                    if len(res) == 4:
                        final_report, transcript, time_series, telemetry = res
                    else:
                        final_report, transcript, time_series = res[:3]
                        telemetry = {}
                        
                    if transcript == "" or "cancelled" in final_report.lower():
                        st.error(final_report)
                    else:
                        st.session_state.current_run_data = (final_report, transcript, time_series, telemetry)
                        st.session_state.workato_dispatch_result = None
                        st.session_state.history.append({
                            "time": datetime.now().strftime("%I:%M %p"),
                            "transcript": transcript,
                            "report": final_report
                        })
                        st.rerun()

        with btn_col2:
            if st.button("Load Sample Pitch (Simulation)", use_container_width=True):
                if not st.session_state.candidate_info["name"]:
                    st.session_state.candidate_info["name"] = "Alex Johnson"
                if not st.session_state.candidate_info["email"]:
                    st.session_state.candidate_info["email"] = "alex.johnson@techdemo.io"
                if not st.session_state.candidate_info["skills"]:
                    st.session_state.candidate_info["skills"] = "Python, React, FastAPI, Docker, PostgreSQL, Kubernetes"
                if not st.session_state.resume_context["text"]:
                    st.session_state.resume_context["text"] = (
                        "Alex Johnson - Full Stack Engineer\n"
                        "Experience: 4 years building web apps with Python, FastAPI, React, and PostgreSQL.\n"
                        "Key Projects: Monolith to Microservices migration on Kubernetes; Kafka streaming pipeline optimization.\n"
                        "Skills: Python, TypeScript, React, Docker, Kubernetes, AWS, PostgreSQL, Kafka."
                    )
                    st.session_state.resume_context["filename"] = "alex_johnson_resume.pdf"

                sample_report, sample_transcript, sample_time_series, sample_telemetry = get_sample_analysis()
                st.session_state.current_run_data = (sample_report, sample_transcript, sample_time_series, sample_telemetry)
                st.session_state.workato_dispatch_result = None
                st.session_state.history.append({
                    "time": datetime.now().strftime("%I:%M %p"),
                    "transcript": sample_transcript,
                    "report": sample_report
                })
                st.rerun()

# --- Render Pitch Telemetry & Coaching Report ---
if st.session_state.current_run_data:
    final_report, transcript, time_series, telemetry = st.session_state.current_run_data
    
    st.divider()
    st.success("Pitch session successfully captured and processed.")
    
    # Telemetry KPI Row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{telemetry.get('duration', 0)}s</div>
            <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        focus_pct = telemetry.get('focus_percentage', 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{focus_pct}%</div>
            <div class="metric-label">Eye Contact Focus</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        gesture_energy = telemetry.get('gesture_energy', 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{gesture_energy}</div>
            <div class="metric-label">Gesture Energy</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        profile = telemetry.get('gesture_profile', 'N/A').title()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="font-size:1.15rem; padding-top:0.35rem;">{profile}</div>
            <div class="metric-label">Behavior Profile</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Physical Telemetry Chart & Transcript
    chart_col, trans_col = st.columns([1.2, 1], gap="large")
    with chart_col:
        st.subheader("Physical Telemetry")
        st.caption("Second-by-second eye contact consistency and gesture movement metrics.")
        if time_series:
            df = pd.DataFrame(time_series).set_index("Time (s)")
            st.line_chart(df, height=260)
        else:
            st.info("No time-series data points available.")

    with trans_col:
        st.subheader("Spoken Transcript")
        st.caption("Automated speech recognition transcript via Whisper large model.")
        st.text_area("Transcript", value=transcript, height=260, disabled=True, label_visibility="collapsed")

    # Executive Coaching Report
    st.subheader("Executive Coaching Report")
    with st.container(border=True):
        st.markdown(final_report)

    # --- Workato Orchestration Gateway ---
    st.divider()
    workato_box = st.container(border=True)
    with workato_box:
        st.subheader("Workato Orchestration Gateway")
        st.caption("Package candidate credentials, resume context, and physical telemetry for automated qualification and downstream team allocation.")
        
        # Build payload
        payload = workato_service.build_candidate_payload(
            candidate_info=st.session_state.candidate_info,
            resume_text=st.session_state.resume_context["text"],
            spoken_text=transcript,
            telemetry_metrics=telemetry,
            coaching_report=final_report,
            resume_filename=st.session_state.resume_context["filename"]
        )

        # JSON preview expander
        with st.expander("Inspect Structured Payload (JSON)", expanded=False):
            st.json(payload)

        # Webhook configuration and trigger
        wcol1, wcol2 = st.columns([1.5, 1])
        with wcol1:
            configured_url = os.environ.get("WORKATO_WEBHOOK_URL", "mock")
            override_url = st.text_input(
                "Workato Webhook Endpoint",
                value=configured_url,
                help="Specify the target Workato recipe webhook URL or retain 'mock' for local simulation."
            )
        with wcol2:
            st.write("")
            st.write("")
            submit_btn = st.button("Submit Candidate to Workato", use_container_width=True)

        if submit_btn:
            if not st.session_state.candidate_info["name"]:
                st.warning("Candidate Full Name is required before submitting.")
            else:
                with st.spinner("Delivering candidate package to Workato Orchestrator..."):
                    success, msg, details = workato_service.send_to_workato(payload, webhook_url=override_url)
                    st.session_state.workato_dispatch_result = {
                        "success": success,
                        "message": msg,
                        "details": details,
                        "timestamp": datetime.now().strftime("%I:%M:%S %p")
                    }

        # Dispatch Result Banner
        if st.session_state.workato_dispatch_result:
            res_data = st.session_state.workato_dispatch_result
            if res_data["success"]:
                st.success(f"{res_data['message']} (Recorded at {res_data['timestamp']})")
                st.markdown("""
                <div class="status-panel">
                    <strong>Downstream Pipeline Status:</strong> Candidate intake package verified. Ready for Workato AI qualification, PongAI role allocation, and Jira task generation.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(res_data["message"])

# --- Sidebar: System Architecture & History ---
with st.sidebar:
    st.subheader("System Architecture")
    st.markdown("""
    - **Blueprint**: Candidate intake, physical telemetry, speech transcription.
    - **Workato**: Automated qualification and evaluation gateway.
    - **PongAI**: Project decomposition and candidate role matching.
    - **Jira**: Sprint ticket creation and responsibility assignment.
    """)
    st.divider()

    st.subheader("Session Archive")
    if len(st.session_state.history) == 0:
        st.info("No recorded sessions in current session.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Session {entry['time']}"):
                st.write("**Transcript Preview:**")
                st.caption(f'"{entry["transcript"][:140]}..."')
                st.download_button(
                    label="Download Report (Markdown)",
                    data=entry["report"],
                    file_name=f"pitch_report_{idx}.md",
                    mime="text/markdown",
                    key=f"download_{idx}"
                )
    
    st.divider()
    if st.button("Clear Current Session", use_container_width=True):
        st.session_state.current_run_data = None
        st.session_state.workato_dispatch_result = None
        st.rerun()