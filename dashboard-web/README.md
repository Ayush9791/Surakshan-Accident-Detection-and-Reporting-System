# Surakshan Web Dashboard

Next.js/React replacement for the former Streamlit dashboard. This first phase provides an operator-oriented incident queue, incident detail view, status actions, local browser webcam preview, and explicitly simulated dispatch notifications.

## Requirements

- Node.js 20 LTS recommended
- The Surakshan FastAPI service running locally or at a trusted server
- Browser camera access is available on localhost or HTTPS

## Run locally (Windows PowerShell)

From the repository root:

```powershell
cd dashboard-web
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. The default API URL is http://127.0.0.1:5000.

In a separate terminal from the repository root, start the backend:

```powershell
..venv\Scripts\Activate.ps1
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

The dashboard polls the API every five seconds. Use **Create test incident** to create a persisted test record, then select it to verify/dismiss it or record a simulated dispatch notification.

## What is and is not connected

- Incident queue, status changes, and simulated notification log use the FastAPI backend and SQLite.
- The webcam button uses browser `getUserMedia` for local preview only. It does **not** transmit frames to the server and does not run AI inference.
- The existing Python `live_system.advanced_live` process remains the webcam-to-model inference path for now.
- The illustrated evidence panel is a UI placeholder, not recorded incident footage.
- Simulated dispatch is stored locally; it does not contact emergency services.

## Deployment note

A cloud-hosted backend cannot open a webcam physically attached to an operator's PC. For a real deployment, run an authorised camera/edge ingestion worker near the camera or stream video securely to a GPU inference service. Use HTTPS, authentication, access controls, secrets management, and retention policies before exposing the dashboard beyond localhost. Do not expose the current unauthenticated API to the public internet.

## Next AI phase

Use the existing CNN only as a baseline. Benchmark the CCTV-focused ACCIDENT dataset (https://accidentbench.github.io/) and evaluate grouped-by-video / unseen-camera performance. A practical system should combine vehicle detection and tracking with temporal accident recognition; a detector alone is not an accident classifier. See the repository's architecture plan for rollout details.
