import cv2 
import mediapipe as mp 
import time 
import os 
import numpy as np 
import sounddevice as sd 
import soundfile as sf 
import pandas as pd 
from groq import Groq 
from crewai import Agent, Task, Crew, Process, LLM 
import os
from dotenv import load_dotenv

load_dotenv()

# ========================================== 
# 1. SETUP & CONFIGURATION 
# ========================================== 
MY_GROQ_KEY = os.environ.get("GROQ_API_KEY")  

groq_client = Groq(api_key=MY_GROQ_KEY) 

import crewai.llms.cache as _crewai_cache 
_crewai_cache.mark_cache_breakpoint = lambda msg: msg 

free_agent_llm = LLM( 
    model="groq/meta-llama/llama-4-scout-17b-16e-instruct", 
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
        print("❌ Error: Could not access the webcam.") 
        return None, 0, None 
     
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
    if actual_duration < 2: return None, 0, None 
     
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
     
    return summary, actual_duration, time_series_data 

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
    visual_metrics, duration, time_series = run_live_tracker(max_duration=60) 
     
    if not visual_metrics: 
        return "Pitch cancelled or too short.", "", None 
         
    spoken_text = transcribe_audio() 
     
    body_language_agent = Agent( 
        role='Blueprint (Behavioral Analyst)', 
        goal='Evaluate the candidate\'s visual presence over time.', 
        backstory='You are the Blueprint analytics engine. You look at second-by-second timeline data to map the exact structural moments a speaker lost focus or became visibly nervous.', 
        verbose=True, 
        llm=free_agent_llm 
    ) 
     
    content_agent = Agent( 
        role='Blueprint (Speech Strategist)', 
        goal='Analyze the transcribed speech for clarity and pacing.', 
        backstory='You are the Blueprint language engine. You analyze transcripts to outline if the core message is clear, or if the speaker rambled.', 
        verbose=True, 
        llm=free_agent_llm 
    ) 
     
    executive_coach = Agent( 
        role='Beacon (Executive Presentation Coach)', 
        goal='Combine all feedback into a final master report.', 
        backstory='You are Beacon, the coaching engine. You synthesize the structural data from the Blueprint analysts into clear, actionable advice. You give precise timestamps to illuminate mistakes and guide improvement.', 
        verbose=True, 
        llm=free_agent_llm 
    ) 
     
    t1 = Task( 
        description=f'Analyze the overall metrics: "{visual_metrics}". Then, review this second-by-second data array: {time_series}. Point out specific timestamps (e.g., "At 15 seconds in...") where they dropped eye contact or had erratic hand spikes.',  
        expected_output='A timeline-based assessment of their body language.',  
        agent=body_language_agent 
    ) 

    t2 = Task(description=f'Analyze this transcribed speech: "{spoken_text}". Did they use their {int(duration)} seconds well?', expected_output='A critique of the spoken content.', agent=content_agent) 
     
    crew_agents = [body_language_agent, content_agent] 
    crew_tasks = [t1, t2] 
     
    if resume_text: 
        resume_agent = Agent( 
            role='Blueprint (Career Alignment Engine)', 
            goal='Cross-reference the resume with the spoken pitch.', 
            backstory='You are the Blueprint context engine. You analyze the foundational layout of a resume to find amazing achievements that the candidate forgot to mention.', 
            verbose=True, 
            llm=free_agent_llm 
        ) 
        t_resume = Task( 
            description=f'Compare this resume: "{resume_text}" with this pitch: "{spoken_text}". Identify 1-2 major strengths missed in the pitch.', 
            expected_output='A short critique detailing missed opportunities.', 
            agent=resume_agent 
        ) 
        crew_agents.append(resume_agent) 
        crew_tasks.append(t_resume) 
         
        t3_desc = ( 
            'Take the timeline visual assessment, the content critique, and the resume data. ' 
            'Create a final "Elevator Pitch Master Report". You MUST include these specific headers: ' 
            '"Strengths", "Weaknesses", "Resume Missed Opportunities", Actionable Insights, and "Top 3 Things to Fix". ' 
            'Include specific timestamps from the timeline data to back up your points.' 
        ) 
    else: 
        t3_desc = ( 
            'Take the timeline visual assessment and the content critique. ' 
            'Create a final "Elevator Pitch Master Report". You MUST include these specific headers: ' 
            '"Strengths", "Weaknesses", Actionable Insights, and "Top 3 Things to Fix". ' 
            'Include specific timestamps from the timeline data to back up your points.' 
        ) 

    t3 = Task(description=t3_desc, expected_output='A clean Markdown report.', agent=executive_coach) 
    crew_agents.append(executive_coach) 
    crew_tasks.append(t3) 
     
    crew = Crew(agents=crew_agents, tasks=crew_tasks, process=Process.sequential) 
    final_report = crew.kickoff() 
     
    if os.path.exists("pitch_audio.wav"): 
        os.remove("pitch_audio.wav") 
         
    return final_report.raw, spoken_text, time_series
