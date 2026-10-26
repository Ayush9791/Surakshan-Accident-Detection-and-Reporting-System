"""HTTP alert client with optional shared detector token."""
import os
import requests

API_URL = os.environ.get("SURAKSHAN_ALERT_URL", "http://127.0.0.1:5000/alert")
DETECTOR_TOKEN = os.environ.get("SURAKSHAN_DETECTOR_TOKEN", "")


def send_alert(camera_id, location="Unknown", incident_id=None, confidence=None):
    payload = {
        "camera_id": camera_id,
        "location": location,
        "event": "POTENTIAL_ACCIDENT",
        "incident_id": incident_id,
        "confidence": confidence,
    }
    headers = {"X-Detector-Token": DETECTOR_TOKEN} if DETECTOR_TOKEN else {}
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=(2, 3))
        if 200 <= response.status_code < 300:
            print(f"[API] Alert accepted for {camera_id} (HTTP {response.status_code})")
            return True
        print(f"[API] Alert failed: HTTP {response.status_code}: {response.text[:200]}")
    except requests.RequestException as exc:
        print(f"[API] Alert delivery failed: {exc}")
    return False
