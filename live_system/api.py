import requests

# ----------------------------------------------
# Change this to your own backend URL
# Example: "https://surakshan-server.onrender.com/alert"
# ----------------------------------------------
API_URL = "http://localhost:5000/alert"   # placeholder URL


def send_alert(camera_id, location="Unknown"):
    """
    Sends accident alert to backend.
    You can integrate with:
      - FastAPI backend
      - Node.js server
      - Discord webhook
      - Telegram bot
      - Firebase
    """

    data = {
        "camera_id": camera_id,
        "location": location,
        "event": "ACCIDENT"
    }

    try:
        response = requests.post(API_URL, json=data, timeout=3)

        if response.status_code == 200:
            print(f"[API] Alert sent for {camera_id}")
        else:
            print(f"[API] Server responded with status: {response.status_code}")

    except Exception as e:
        print(f"[API] Failed to send alert → {e}")
