import cv2 
import mediapipe as mp 
import time 
import os 
import numpy as np 
import sounddevice as sd 
import soundfile as sf 
import pandas as pd 
import os
from dotenv import load_dotenv
from groq import Groq 
from crewai import Agent, Task, Crew, Process, LLM 

load_dotenv()

# ========================================== 
# 1. SETUP & CONFIGURATION 
# ========================================== 
MY_GROQ_KEY = os.environ.get("GROQ_API_KEY")  

groq_client = Groq(api_key=MY_GROQ_KEY) 

import crewai.llms.cache as _crewai_cache 
_crewai_cache.mark_cache_breakpoint = lambda msg: msg 

GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/groq/compound-mini")

free_agent_llm = LLM( 
    model=GROQ_MODEL, 
    api_key=MY_GROQ_KEY, 
    temperature=0.0 
)

# ========================================== 
# 2. INTERACTIVE UI & AUDIO STATE 
# ========================================== 
ui_state = { 
    "is_tracking": False, 
    "start_time": 0, 
    "should_stop": False 
} 

AUDIO_SAMPLE_RATE = 44100 
audio_data = None 

def on_mouse_click(event, x, y, flags, param): 
    if event == cv2.EVENT_LBUTTONDOWN: 
        if 30 <= x <= 230 and 400 <= y <= 450: 
            if not ui_state["is_tracking"]: 
                ui_state["is_tracking"] = True 
                ui_state["start_time"] = time.time() 
                print("▶️ Start signal received...") 
            else: 
                ui_state["should_stop"] = True 
                print("⏹️ Stop signal received...") 

# ========================================== 
# 3. LIVE HOLISTIC TRACKER 
# ========================================== 
def run_live_tracker(max_duration=60): 
    global audio_data 
    print("📷 System Standby. Click START on the video window to begin your pitch.") 
     
    mp_holistic = mp.solutions.holistic 
     
    cap = cv2.VideoCapture(1)  
    if not cap.isOpened(): 
        print("ℹ️ Camera index 1 not available. Falling back to default camera (index 0)...")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened(): 
        print("❌ Error: Could not access any webcam (tested indices 1 and 0).") 
        return None, 0, None, {} 
     
    cv2.namedWindow('Elevator Pitch Coach') 
    cv2.setMouseCallback('Elevator Pitch Coach', on_mouse_click) 
     
    total_frames = 0 
    distracted_frames = 0 
    total_hand_energy = 0.0 
     
    time_series_data = {"Time (s)": [], "Focus (100=Yes, 0=No)": [], "Hand Energy (Spikes)": []} 
    last_logged_second = -1 
    current_second_energy = 0.0 
     
    prev_right_wrist = None 
    prev_left_wrist = None 
    
    audio_started = False
     
    try: 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True: 
                ret, frame = cap.read() 
                if not ret: break 
                      
                # 1. ALWAYS RUN THE TRACKER (For Preview)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                results = holistic.process(rgb_frame) 
                 
                status_text = "Focused" 
                status_color = (0, 255, 0) 
                focus_score = 100 
                 
                # --- FACE TRACKING LOGIC (Runs continuously) --- 
                if results.face_landmarks: 
                    nose = results.face_landmarks.landmark[1] 
                    left_ear = results.face_landmarks.landmark[234] 
                    right_ear = results.face_landmarks.landmark[454] 
                     
                    dist_left = abs(nose.x - left_ear.x) 
                    dist_right = abs(nose.x - right_ear.x) 
                    ratio = dist_left / (dist_right + 0.0001) 
                     
                    if ratio > 3.0 or ratio < 0.33: 
                        status_text = "Distracted" 
                        status_color = (0, 0, 255) 
                        focus_score = 0 
                else: 
                    status_text = "Face Not Visible" 
                    status_color = (0, 0, 255) 
                    focus_score = 0 

                # 2. ONLY LOG DATA IF PITCH HAS STARTED
                if ui_state["is_tracking"]: 
                    
                    if not audio_started:
                        print("🎙️ Microphone Active - Recording Started.")
                        audio_data = sd.rec(int(max_duration * AUDIO_SAMPLE_RATE), samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='float32')
                        audio_started = True
                        
                    total_frames += 1 
                    
                    # Update distracted frames only during active recording
                    if focus_score == 0:
                        distracted_frames += 1
                     
                    # --- HAND TRACKING --- 
                    if results.right_hand_landmarks: 
                        curr_rw = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST] 
                        if prev_right_wrist: 
                            dist = np.sqrt((curr_rw.x - prev_right_wrist.x)**2 + (curr_rw.y - prev_right_wrist.y)**2) 
                            total_hand_energy += dist 
                            current_second_energy += dist 
                        prev_right_wrist = curr_rw 
                     
                    if results.left_hand_landmarks: 
                        curr_lw = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST] 
                        if prev_left_wrist: 
                            dist = np.sqrt((curr_lw.x - prev_left_wrist.x)**2 + (curr_lw.y - prev_left_wrist.y)**2) 
                            total_hand_energy += dist 
                            current_second_energy += dist 
                        prev_left_wrist = curr_lw 

                    # --- LOG DATA EVERY SECOND --- 
                    elapsed_time = time.time() - ui_state["start_time"] 
                    current_second = int(elapsed_time) 
                     
                    if current_second > last_logged_second: 
                        time_series_data["Time (s)"].append(current_second) 
                        time_series_data["Focus (100=Yes, 0=No)"].append(focus_score) 
                        time_series_data["Hand Energy (Spikes)"].append(int(current_second_energy * 100))  
                         
                        last_logged_second = current_second 
                        current_second_energy = 0.0 

                    # Draw Recording UI 
                    time_left = max(0, int(max_duration - elapsed_time)) 
                    cv2.putText(frame, f"Time: {time_left}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) 
                    cv2.rectangle(frame, (30, 400), (230, 450), (0, 0, 200), -1) 
                    cv2.putText(frame, "STOP PITCH", (50, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 
                     
                    if time_left == 0: ui_state["should_stop"] = True 

                else: 
                    # Draw Preview UI 
                    cv2.putText(frame, "SYSTEM STANDBY - PREVIEW", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2) 
                    cv2.rectangle(frame, (30, 400), (230, 450), (0, 200, 0), -1) 
                    cv2.putText(frame, "START PITCH", (45, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 

                # 3. ALWAYS DRAW THE FOCUS TEXT OVERLAY
                cv2.putText(frame, f"Focus: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2) 

                cv2.imshow('Elevator Pitch Coach', frame) 
                 
                if ui_state["should_stop"] or (cv2.waitKey(1) & 0xFF == ord('q')): 
                    break 
                    
                if cv2.getWindowProperty('Elevator Pitch Coach', cv2.WND_PROP_VISIBLE) < 1:
                    break
                     
    finally: 
        cap.release() 
        cv2.destroyAllWindows() 
        sd.stop()  
     
    actual_duration = time.time() - ui_state["start_time"] if ui_state["is_tracking"] else 0 
    if actual_duration < 2: return None, 0, None, {} 
     
    trimmed_audio = audio_data[:int(actual_duration * AUDIO_SAMPLE_RATE)] 
    sf.write('pitch_audio.wav', trimmed_audio, AUDIO_SAMPLE_RATE) 
     
    focus_percentage = 100 - ((distracted_frames / total_frames) * 100) if total_frames > 0 else 0 
    normalized_energy = (total_hand_energy / actual_duration) * 100  
     
    gesture_profile = "calm and controlled" 
    if normalized_energy > 40: gesture_profile = "highly erratic and distracting" 
    elif normalized_energy > 15: gesture_profile = "active and expressive" 
     
    summary = ( 
        f"The pitch lasted {int(actual_duration)} seconds. " 
        f"Visual Focus: Maintained eye contact {focus_percentage:.1f}% of the time. " 
        f"Overall Gesture Energy: {normalized_energy:.1f} ({gesture_profile})." 
    ) 
     
    ui_state["is_tracking"] = False 
    ui_state["should_stop"] = False 

    telemetry_metrics = {
        "duration": int(actual_duration),
        "focus_percentage": round(float(focus_percentage), 1),
        "gesture_energy": round(float(normalized_energy), 1),
        "gesture_profile": gesture_profile
    }
     
    return summary, actual_duration, time_series_data, telemetry_metrics

# ========================================== 
# 4. AUDIO TRANSCRIPTION 
# ========================================== 
def transcribe_audio(filename="pitch_audio.wav"): 
    try: 
        with open(filename, "rb") as file: 
            transcription = groq_client.audio.transcriptions.create( 
              file=(filename, file.read()), 
              model="whisper-large-v3" 
            ) 
        return transcription.text 
    except Exception as e: 
        return "No audio detected." 

# ========================================== 
# 5. MULTI-AGENT ANALYSIS LOOP 
# ========================================== 
def run_full_analysis(resume_text=""): 
    tracker_result = run_live_tracker(max_duration=60) 
     
    if not tracker_result or tracker_result[0] is None: 
        return "Pitch cancelled or too short.", "", None, {} 
         
    visual_metrics, duration, time_series, telemetry_metrics = tracker_result
    spoken_text = transcribe_audio() 
     
    # --- Token-Efficient Telemetry Distillation ---
    # Instead of sending a massive raw 60-element array that triggers Groq TPM limits,
    # extract exact seconds of interest so the LLM has timestamp precision with 90% fewer tokens.
    times = time_series.get("Time (s)", []) if time_series else []
    focus_vals = time_series.get("Focus (100=Yes, 0=No)", []) if time_series else []
    energy_vals = time_series.get("Hand Energy (Spikes)", []) if time_series else []

    focus_drops = [t for t, f in zip(times, focus_vals) if f == 0]
    hand_spikes = [t for t, e in zip(times, energy_vals) if e > 35]

    telemetry_summary = (
        f"Overall: {visual_metrics}\n"
        f"- Eye contact lost at seconds: {focus_drops if focus_drops else 'None (consistent focus)'}\n"
        f"- High hand movement spikes at seconds: {hand_spikes if hand_spikes else 'None (controlled gestures)'}"
    )

    clean_resume = resume_text[:2000] if resume_text else ""

    # --- Streamlined 2-Agent Crew Architecture ---
    # Eliminates redundant intermediate calls that exceed Groq's 8K TPM rate limit
    analyst_agent = Agent( 
        role='Blueprint (Multi-Modal Analyst)', 
        goal='Evaluate body language timeline data, spoken clarity, and resume alignment.', 
        backstory='You are the Blueprint analytics engine. You inspect timestamped behavioral telemetry, spoken pitch transcripts, and background resumes to uncover delivery strengths and missed opportunities.', 
        verbose=True, 
        llm=free_agent_llm 
    ) 

    executive_coach = Agent( 
        role='Blueprint (Executive Presentation Coach)', 
        goal='Combine all feedback into a master executive coaching report with required headers.', 
        backstory='You are the Blueprint executive presentation coach. You transform analytical data into actionable, inspiring advice with exact timestamps to illuminate mistakes and guide improvement.', 
        verbose=True, 
        llm=free_agent_llm 
    ) 

    if clean_resume:
        t1_desc = (
            f"Analyze this pitch:\n"
            f"TELEMETRY TIMELINE:\n{telemetry_summary}\n\n"
            f"SPOKEN PITCH ({int(duration)}s):\n\"{spoken_text}\"\n\n"
            f"RESUME CONTEXT:\n\"{clean_resume}\"\n\n"
            f"Provide a concise analytical breakdown of: (1) body language timestamps where focus dropped or hands spiked, (2) clarity and pacing of speech, and (3) 1-2 major resume strengths missed."
        )
        t2_desc = ( 
            'Take the analytical breakdown and produce a final "# Elevator Pitch Master Report". '
            'You MUST include these exact headers: ' 
            '"### Strengths", "### Weaknesses", "### Resume Missed Opportunities", "### Actionable Insights", and "### Top 3 Things to Fix". ' 
            'Include specific timestamps from the timeline data to back up your points.' 
        )
    else:
        t1_desc = (
            f"Analyze this pitch:\n"
            f"TELEMETRY TIMELINE:\n{telemetry_summary}\n\n"
            f"SPOKEN PITCH ({int(duration)}s):\n\"{spoken_text}\"\n\n"
            f"Provide a concise analytical breakdown of: (1) body language timestamps where focus dropped or hands spiked, and (2) clarity and pacing of speech."
        )
        t2_desc = ( 
            'Take the analytical breakdown and produce a final "# Elevator Pitch Master Report". '
            'You MUST include these exact headers: ' 
            '"### Strengths", "### Weaknesses", "### Actionable Insights", and "### Top 3 Things to Fix". ' 
            'Include specific timestamps from the timeline data to back up your points.' 
        )

    t1 = Task(description=t1_desc, expected_output='A consolidated multi-modal assessment.', agent=analyst_agent)
    t2 = Task(description=t2_desc, expected_output='A clean Markdown master report.', agent=executive_coach)

    crew = Crew(agents=[analyst_agent, executive_coach], tasks=[t1, t2], process=Process.sequential) 
    
    # Auto-retry handler for rate limits with backoff
    final_report = None
    import re
    for attempt in range(3):
        try:
            final_report = crew.kickoff()
            break
        except Exception as e:
            err_str = str(e)
            if ("rate_limit" in err_str.lower() or "ratelimit" in err_str.lower() or "429" in err_str) and attempt < 2:
                wait_sec = 4.5
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                if match:
                    wait_sec = float(match.group(1)) + 1.0
                print(f"⏳ Rate limit pause: waiting {wait_sec:.1f}s before retry ({attempt+1}/2)...")
                time.sleep(wait_sec)
            else:
                raise e
     
    if os.path.exists("pitch_audio.wav"): 
        os.remove("pitch_audio.wav") 
         
    return final_report.raw, spoken_text, time_series, telemetry_metrics

def get_sample_analysis():
    """
    Returns high-quality sample pitch analysis data for instant testing
    and demoing the Workato -> PongAI -> Jira workflow.
    """
    sample_report = """# Elevator Pitch Master Report

### Strengths
- **Clear Value Proposition**: Succinctly outlined 3+ years of experience building scalable backend microservices and React dashboards.
- **Strong Eye Contact**: Maintained steady visual focus during technical overview (seconds 10-38).
- **Concise Delivery**: Delivered a coherent narrative within a 45-second window without filler words.

### Weaknesses
- **Hand Gesture Spikes**: Noticeable sudden hand movements around 22s-26s when transitioning to cloud architecture topics.
- **Missed Metric Depth**: Spoke about scaling APIs but did not quantify throughput or latency improvements cited on the resume.

### Resume Missed Opportunities
- Failed to mention experience with **Kafka streaming** and **PostgreSQL optimization**, which are highlighted as major wins in the uploaded resume.

### Actionable Insights
1. Practice grounding your hands at waist height when transitioning between project stories.
2. Weave in quantifiable impacts (e.g., "reduced latency by 35%") directly into your 30-second hook.

### Top 3 Things to Fix
1. Anchor your hands during transitions (seconds 20-25).
2. State your target role immediately in the opening 5 seconds.
3. Reference your Kafka distributed systems experience.
"""
    sample_transcript = (
        "Hi, I'm Alex Johnson, a full-stack engineer with 4 years of experience building scalable web applications "
        "and distributed backend systems in Python, FastAPI, and React. Recently, I led the migration of a monolithic API "
        "to microservices deployed on Kubernetes, which cut infrastructure costs and improved system resilience. "
        "I thrive in collaborative agile teams, turning complex product specs into reliable, well-tested code, "
        "and I'm excited to contribute to high-impact software projects."
    )
    
    sample_time_series = {
        "Time (s)": list(range(1, 46)),
        "Focus (100=Yes, 0=No)": [100 if i not in (22, 23, 24) else 0 for i in range(1, 46)],
        "Hand Energy (Spikes)": [12 if i not in (21, 22, 23, 24, 25) else 55 for i in range(1, 46)]
    }
    
    sample_telemetry = {
        "duration": 45,
        "focus_percentage": 93.3,
        "gesture_energy": 18.5,
        "gesture_profile": "active and expressive"
    }
    
    return sample_report, sample_transcript, sample_time_series, sample_telemetry