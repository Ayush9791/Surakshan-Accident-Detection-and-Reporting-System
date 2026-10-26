# Surakshan — Real-Time Traffic Incident Monitoring Prototype

Surakshan is an academic prototype for CCTV-style traffic incident detection and operator response. The current repository combines a PyTorch frame classifier, webcam monitoring, temporal incident gating, evidence recording, a FastAPI incident API, SQLite persistence, and a Next.js/React operator dashboard.

> **Safety and deployment warning:** This is a prototype, not a certified emergency detection or dispatch system. Model output can be wrong. The dashboard's dispatch workflow is simulated and does not contact emergency services. Do not use this system as the sole basis for emergency response.

## Current architecture

- `model.py`, `train.py`: existing PyTorch frame classifier and baseline training pipeline.
- `live_system/basic_live.py`: simple webcam inference preview.
- `live_system/advanced_live.py`: webcam inference, repeated-positive-frame gating, event logging, video evidence, and HTTP alert submission.
- `live_system/server.py`: FastAPI API, SQLite incidents, operator status updates, and simulated dispatch log.
- `dashboard-web/`: Next.js/React dashboard replacing Streamlit as the recommended operator UI.
- `dashboard/dashboard.py`: legacy Streamlit dashboard retained during migration.
- `tests/`: unit and API tests.

## Run the Python backend (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API in one terminal:

```powershell
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

Useful endpoints:

- `GET /health` — service health.
- `POST /alert` — persist a potential incident.
- `GET /incidents?limit=50` — list recent incidents.
- `PATCH /incidents/{incident_id}/status` — update operator status (`New`, `Under Review`, `Verified`, `Dismissed`, `Resolved`).
- `POST /dispatch/simulate` — record a simulated notification. Never contacts an authority.
- `GET /dispatch/notifications` — list simulated notifications.
- `GET /docs` — interactive API documentation.

The SQLite database defaults to `data/surakshan.db`. Set `SURAKSHAN_DB_PATH` before starting the API to change its location.

## Run the Next.js dashboard

Requires Node.js 20 LTS recommended. In a second terminal from the repository root:

```powershell
cd dashboard-web
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. The dashboard polls the API every five seconds. Use **Create test incident** to create a persisted sample incident, then select it to test status changes and simulated dispatch logging.

### Current dashboard scope

- Operator overview and incident queue with search/status filters.
- Incident detail, confidence display, status changes and evidence placeholder.
- Browser-local webcam preview.
- Simulated dispatch log and API connection indicator.

**Important:** the webcam preview uses browser `getUserMedia` and is preview-only. It does not send frames to the backend or run the model. The existing `python -m live_system.advanced_live` workflow is still the webcam-to-model inference path. The incident evidence graphic is illustrative, not an actual recorded clip. The dashboard is an initial UI migration, not a completed real-time video streaming service.

## Run the existing webcam detector

Start the API, then in another terminal from the repository root:

```powershell
python -m live_system.advanced_live
```

Press Esc to stop. This existing process reads camera 0, runs the current frame classifier, gates repeated positive predictions, records clips/logs and sends incident metadata to the API. It remains a baseline and is not yet the recommended final accident-recognition model.

For a simple prediction-only preview:

```powershell
python -m live_system.basic_live
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions runs Python tests and builds the Next.js dashboard on pushes and pull requests. The tests cover incident gating, evidence recording, API validation, incident status changes and simulated dispatch persistence. Passing unit tests does not validate real-world model accuracy, camera compatibility or emergency response suitability.

## Dataset and model direction

The existing two-class CNN is useful as a baseline, but classifying individual frames is not sufficient for reliable crash recognition. The recommended benchmark to start with is **ACCIDENT**, a traffic-surveillance dataset with real and synthetic clips annotated for accident time, spatial location and collision type:

- Dataset and benchmark: https://accidentbench.github.io/
- CVPR Workshops paper: https://openaccess.thecvf.com/content/CVPR2026W/AUTOPILOT/html/Picek_ACCIDENT_A_Benchmark_Dataset_for_Vehicle_Accident_Detection_from_Traffic_CVPRW_2026_paper.html

Recommended experimental approach:

1. Benchmark on ACCIDENT with splits grouped by source video and test on unseen cameras/scenes.
2. Use a real-time vehicle detector plus multi-object tracker as the perception layer.
3. Evaluate a temporal video model for crash recognition and accident-time localisation; tracking alone is not crash detection.
4. Include normal traffic, near misses, abrupt braking, congestion, occlusion, low-light and weather variation.
5. Report accident recall, false alerts per camera-hour, detection delay and out-of-distribution performance—not only image-level accuracy.
6. Validate on separately collected, appropriately licensed Indian CCTV footage before claiming generalisation.

## Cloud and camera architecture notes

The selected initial demo uses a local webcam, while the intended deployment target is a server/cloud. A cloud backend cannot directly open a webcam attached to an operator's computer. For the production architecture, run an authorised camera-ingestion worker near the camera (RTSP/ONVIF where permitted) or securely stream media to an inference service. Use a media gateway for browser playback (for example WebRTC/HLS); browsers do not play RTSP directly.

Before deployment beyond localhost, add authentication and role-based access, HTTPS, secrets management, rate limits, audit logging, secure camera credentials, video retention controls and monitoring. The current API has no authentication; keep it bound to localhost. Simulated dispatch must remain clearly labelled until an authorised emergency-service integration is available.

## Future roadmap

1. Connect a browser camera preview or authorised RTSP source to an explicit ingestion/inference worker.
2. Replace frame-only accident classification with a benchmarked temporal model and tracking.
3. Add authenticated users, camera inventory, region/jurisdiction mapping and role permissions.
4. Store incident evidence with retention policies and link clips to incident records.
5. Add real-time event delivery, multi-camera incident correlation, camera health and response metrics.
6. Pilot with human verification and simulated notifications before any approved external dispatch integration.
