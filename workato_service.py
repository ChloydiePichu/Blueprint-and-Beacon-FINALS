import os
import uuid
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

def build_candidate_payload(
    candidate_info: Dict[str, Any],
    resume_text: str,
    spoken_text: str,
    telemetry_metrics: Dict[str, Any],
    coaching_report: str = "",
    resume_filename: str = ""
) -> Dict[str, Any]:
    """
    Constructs a standardized JSON payload representing candidate data and
    pitch analysis for downstream consumption by Workato, PongAI, and Jira.
    """
    candidate_id = candidate_info.get("candidate_id") or f"cand_{uuid.uuid4().hex[:8]}"
    
    # Process skills list
    raw_skills = candidate_info.get("skills", "")
    if isinstance(raw_skills, str):
        skills_list = [s.strip() for s in raw_skills.split(",") if s.strip()]
    elif isinstance(raw_skills, list):
        skills_list = raw_skills
    else:
        skills_list = []

    # Clean telemetry
    duration = telemetry_metrics.get("duration", 0)
    focus_pct = round(float(telemetry_metrics.get("focus_percentage", 0.0)), 1)
    gesture_energy = round(float(telemetry_metrics.get("gesture_energy", 0.0)), 1)
    gesture_profile = telemetry_metrics.get("gesture_profile", "unknown")

    payload = {
        "event": "candidate_pitch_submitted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "id": candidate_id,
            "name": candidate_info.get("name", "Anonymous Candidate").strip(),
            "email": candidate_info.get("email", "").strip(),
            "target_role": candidate_info.get("target_role", "Software Engineer"),
            "skills": skills_list,
            "experience_level": candidate_info.get("experience_level", "Mid-Level")
        },
        "resume": {
            "filename": resume_filename or "uploaded_resume.pdf",
            "character_count": len(resume_text),
            "text_preview": resume_text[:1000] if resume_text else "",
            "has_resume": bool(resume_text)
        },
        "pitch_evaluation": {
            "transcript": spoken_text,
            "duration_seconds": int(duration),
            "telemetry": {
                "focus_percentage": focus_pct,
                "gesture_energy": gesture_energy,
                "gesture_profile": gesture_profile
            },
            "executive_summary": coaching_report[:1200] if coaching_report else "",
            "full_report": coaching_report
        },
        "metadata": {
            "source": "Blueprint AI Coach",
            "version": "2.0-hackathon",
            "pipeline_target": "Workato -> PongAI -> Jira"
        }
    }
    return payload


def send_to_workato(
    payload: Dict[str, Any],
    webhook_url: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Dispatches the candidate payload to a Workato Webhook endpoint.
    If webhook_url is not provided or set to 'mock', returns a simulated response.
    
    Returns:
        (success: bool, message: str, details: dict)
    """
    url = webhook_url or os.environ.get("WORKATO_WEBHOOK_URL", "").strip()

    if not url or url.lower() in ("mock", "none", "demo", ""):
        # Simulated successful response for testing and hackathon demo flow
        return (
            True,
            "[Simulation Mode] Candidate payload successfully validated and prepared for Workato.",
            {
                "mode": "simulation",
                "status_code": 200,
                "message": "Payload ready for Workato recipe trigger.",
                "candidate_id": payload.get("candidate", {}).get("id"),
                "candidate_name": payload.get("candidate", {}).get("name"),
                "target_role": payload.get("candidate", {}).get("target_role")
            }
        )

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Blueprint-Coach/2.0"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=12)
        if 200 <= response.status_code < 300:
            try:
                data = response.json()
            except Exception:
                data = {"raw_response": response.text}
            return (
                True,
                f"Successfully delivered to Workato (HTTP {response.status_code})",
                {"status_code": response.status_code, "response": data}
            )
        else:
            return (
                False,
                f"Workato returned HTTP {response.status_code}: {response.text[:200]}",
                {"status_code": response.status_code, "error": response.text}
            )
    except requests.exceptions.Timeout:
        return False, "Request timed out while connecting to Workato webhook endpoint.", {"error": "timeout"}
    except requests.exceptions.RequestException as e:
        return False, f"Failed to reach Workato endpoint: {str(e)}", {"error": str(e)}
