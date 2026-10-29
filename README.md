# Surakshan — Traffic Incident Detection and Response Platform

Surakshan is a college-project prototype that combines webcam-based accident classification, incident persistence, role-based web dashboards, hospital assignment, audit logging, and simulated emergency notifications. It is designed to be demonstrated at zero cost on a Windows PC. The current CNN is a baseline and can be retrained/replaced later.

> **Important:** this is not a certified emergency system. The AI can miss incidents or raise false alerts. Dispatch is simulated only; no police, ambulance, hospital, or emergency service is contacted. The API and demo accounts are not production-hardened. Do not expose the service publicly as-is.

## Features in this branch

- PyTorch frame-classification baseline and existing live webcam inference pipeline.
- Temporal incident gate to avoid triggering on a single positive frame.
- Incident event records in SQLite.
- Next.js/React dashboard replacing Streamlit for the operator workflow.
- Login for **admin**, **traffic authority**, and **hospital** roles.
- Role-based permissions: admins create authority/hospital accounts; authorities review incidents, update status, assign verified cases to a hospital, and create simulated dispatch records; hospitals only see verified/resolved cases assigned to their own organisation.
- Password hashing using PBKDF2-HMAC-SHA256, expiring signed sessions, password change, and an admin audit trail.
- Dashboard buttons to start/stop the Python detector on the same host as the API.
- Optional shared detector-ingestion token for a separately configured detector client.
- Automated Python API tests and Next.js production build workflow.

## Free local setup (Windows PowerShell)

No paid service, paid API key, or cloud account is required for the college demo. Install Python 3.10 or 3.11, Node.js 20 LTS, and Git.

From the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.10 is not installed, use a version listed by `py -0p`, such as `py -3.11`.

### Start the API

Open Terminal 1 in the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

API documentation: http://127.0.0.1:5000/docs

The SQLite database is created automatically at `data/surakshan.db`. Do not delete it while the API is running.

### Start the web dashboard

Open Terminal 2:

```powershell
cd dashboard-web
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. Keep both terminals running.

### Demo logins

These accounts are created automatically the first time the database starts, unless matching usernames already exist.

| Role | Username | Demo password |
|---|---|---|
| Administrator | `admin` | `Admin@12345` |
| Traffic authority | `authority` | `Authority@12345` |
| Hospital | `hospital` | `Hospital@12345` |

Change each password after logging in. The admin can create additional hospital and authority accounts in **Account management**. Hospital access is limited to verified/resolved incidents assigned to that hospital organisation. To assign a case, use the exact organisation name shown in the admin account list (the seeded hospital is `Demo General Hospital`).

To use different seeded passwords on a fresh database, set `SURAKSHAN_ADMIN_PASSWORD`, `SURAKSHAN_AUTHORITY_PASSWORD`, and `SURAKSHAN_HOSPITAL_PASSWORD` before first starting the API. Existing accounts are not reset when these variables change. Set `SURAKSHAN_AUTH_SECRET` to a long random value before sharing the app. Without it, the API generates a temporary random secret at startup, so existing sessions will need to sign in again after an API restart.

### Test the end-to-end workflow

1. Sign in as `admin`.
2. Click **Create test incident**.
3. Select the new incident, verify it, and assign it to `Demo General Hospital`.
4. Click **Simulate dispatch** and verify the notification appears in the dispatch log.
5. Sign out and sign in as `hospital`. The verified incident should appear in the hospital queue.
6. Use **Account management** to create additional authority/hospital users.
7. Use **Start AI detector** to launch the Python inference process on the computer running the API. It opens an OpenCV window and writes logs to `logs/detector.log`. Stop the browser preview before starting the Python detector.
8. By default, the detector uses the local webcam (`SURAKSHAN_CAMERA_URL=0`). To use Android IP Webcam, start its server on the phone, open the displayed IP address from the laptop, then set the stream URL in the same PowerShell session **before starting the API**:
   ```powershell
   $env:SURAKSHAN_CAMERA_URL = "http://192.168.1.105:8080/video"
   uvicorn live_system.server:app --host 127.0.0.1 --port 5000
   ```
   Replace `192.168.1.105` with the phone's actual IP. Use `http`, not `https`, and include `/video`. Keep both devices on a network that permits device-to-device traffic. The API's detector subprocess inherits this setting. If the stream opens but has no frames, check `logs/detector.log`.
9. When a positive event passes the existing incident gate, the detector submits an incident to the API. Review it in the dashboard. Model output remains a potential incident until a human verifies it.

## Roles and permissions

- **Admin:** view all incidents; manage authority/hospital accounts; change incident status; assign incidents; view audit events; start/stop local detector; create simulated dispatch records.
- **Traffic authority:** view incident queue; review/verify/dismiss/resolve; assign verified cases to hospitals; start/stop local detector; record simulated dispatch.
- **Hospital:** view only verified/resolved incidents assigned to its own organisation; no global incident access or dispatch privileges.

The current detector ingestion endpoint `POST /alert` is intended for the local detector. For a trusted network or cloud deployment, set `SURAKSHAN_DETECTOR_TOKEN` and configure the same environment variable for the detector process; the API rejects detector alerts without the matching `X-Detector-Token`. Do not expose the API publicly without HTTPS, firewalling, rate limiting, secret management, and additional security review.

## Run automated tests

```powershell
python -m unittest discover -s tests -v
```

Build the frontend locally:

```powershell
cd dashboard-web
npm run build
```

## AI model and dataset plan

The current frame-level CNN is a baseline, not a validated real-world crash detector. Training a stronger model is a separate stage.

Start by benchmarking the CCTV-focused **ACCIDENT** dataset:
- Dataset and benchmark: https://accidentbench.github.io/
- Paper: https://openaccess.thecvf.com/content/CVPR2026W/AUTOPILOT/html/Picek_ACCIDENT_A_Benchmark_Dataset_for_Vehicle_Accident_Detection_from_Traffic_CVPRW_2026_paper.html

Recommended future approach: pretrained vehicle detector + multi-object tracker + temporal accident-recognition model. Split train/validation/test by source video or camera, include hard negative traffic events, and report event recall, false alerts per camera-hour, detection delay, and performance on unseen cameras. Check dataset licensing and annotation formats before use.

## Zero-cost deployment guidance

For the submission deadline, **run the full demo locally**. This is the most reliable option and costs nothing. The Python detector uses the camera attached to the machine running the API; a cloud server cannot access your laptop webcam just because the browser has a preview.

Optional free-tier hosting can be explored later, but providers change limits, may sleep or require billing details, and the detector still needs a reachable authorised video source. Do not make a public deployment a deadline dependency. A free cloud host is not a substitute for securing the API.

## Current limitations

- The existing model is frame-based; temporal crash recognition has not been trained yet.
- The browser webcam preview is only a preview. AI inference runs in the Python detector process on the API host.
- The detector's OpenCV window is local to the API host; it is not streamed into the web dashboard.
- The dashboard evidence panel does not yet play saved incident clips.
- Simulated dispatch does not contact real authorities.
- This prototype has no production-grade HTTPS deployment, multi-factor authentication, formal password reset, or agency integration.


## Incident evidence photos and manual review

When the temporal gate opens a potential accident, the detector saves the annotated trigger frame as a JPEG in the local `evidence/` folder, then includes that image with the alert sent to the API. The API stores the image, associates it with the incident record, and exposes it through the authenticated `/incidents/{incident_id}/evidence` endpoint. Operators can view the photo in **Incident details → Captured evidence** before marking the incident verified or dismissed. Existing incidents created before this change will show “No photo”; their original camera frames cannot be reconstructed.

### Use a camera connected to your own Windows PC with the hosted dashboard

A cloud host such as Render does not have access to your laptop's webcam. Run the Python detector on the PC physically connected to the webcam, and send incident alerts to the hosted API. From the repository root, in PowerShell:

```powershell
$env:SURAKSHAN_ALERT_URL = "https://surakshan-accident-detection-and.onrender.com/alert"
$env:SURAKSHAN_CAMERA_URL = "0"
# Only set this if SURAKSHAN_DETECTOR_TOKEN is configured on the hosted API:
# $env:SURAKSHAN_DETECTOR_TOKEN = "the-same-token-configured-on-the-API"
python -m live_system.advanced_live
```

Keep the detector process running while the camera is monitored. If the API has a detector token configured, set the same value in the local environment before launching it. New incidents and their photos should then appear in the hosted dashboard after the alert is accepted.

**Storage note:** the default `evidence/` directory is local filesystem storage. On a free Render web service, files can be lost when the service is restarted or redeployed. Use a persistent disk or external object storage for evidence that must survive deployments. This prototype's stored photos are review aids, not forensic-grade evidence.
