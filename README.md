# Surakshan: CCTV Accident Detection and Incident Response

Surakshan is an academic prototype for CCTV accident screening and incident handling. It combines a Keras frame classifier, temporal alert gates, a FastAPI backend, SQLite persistence, and a Next.js operator dashboard with admin, traffic-authority, and hospital roles.

> **Safety notice:** This is not a certified emergency system. The classifier can miss accidents or generate false alerts. Human review is required. Dispatch is simulated and does not contact police, ambulance, hospitals, or other emergency services. Do not use this prototype as the sole basis for real-world emergency response.

## Current architecture

- **Model:** TensorFlow/Keras CNN binary frame classifier.
- **Input:** 128 × 128 grayscale frames, float32 normalized to [0, 1].
- **Class indices:** Accident = 0; Non_Accident = 1.
- **Weights:** `models/keras/car_accident_model.weights.h5`.
- **Training metadata:** `models/keras/model_metadata.json`, written after training. It contains the validation-selected threshold and model preprocessing details.
- **Inference:** `live_system/predict.py`, shared by the live detector.
- **Temporal alerting:** per-camera consecutive-positive logic in the predictor, followed by the incident gate in `live_system/incident.py`.
- **Backend:** FastAPI with SQLite, role-based access, evidence-photo storage, audit events, and simulated dispatch.
- **Frontend:** Next.js/React in `dashboard-web/`.

The old PyTorch/ResNet18 trainer, old Streamlit dashboard, and experimental video-model code have been removed from the active source tree. The current classifier is still a frame-based CNN, not a temporal video model.

## Requirements

- Python 3.10 or 3.11
- Node.js 20 LTS
- Git

Install Python dependencies in PowerShell from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.10 is unavailable, use Python 3.11.

### GPU note for Windows

The standard TensorFlow pip package on native Windows does not provide CUDA GPU support for modern TensorFlow versions. On native Windows, this setup may train on CPU even when an NVIDIA GPU is installed. For TensorFlow CUDA acceleration with an RTX GPU, use a supported WSL2/Linux environment with the NVIDIA WSL driver and a compatible TensorFlow installation. Always verify with:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

An empty list means TensorFlow will use CPU. Do not assume that selecting the RTX GPU in NVIDIA Control Panel makes TensorFlow use it.

## Prepare the video dataset

Place the source video collections under `vid_dataset/` using the expected folder structure and metadata files described in `dataset/README.md`.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe preprocess_video_dataset.py --workers 8
```

The preprocessor reads `configs/keras_training.json`, audits source videos, assigns source groups to splits, and writes the extracted frame dataset and manifests under `dataset/resnet18_frames/`. That directory name is retained to avoid invalidating an existing local extraction; the frames are not ResNet-specific.

Do not randomly split individual frames. The source-video grouping in the manifest is required to prevent adjacent frames from the same source appearing in multiple splits.

## Train the Keras classifier

```powershell
.\.venv\Scripts\python.exe train.py
```

Optional overrides:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 20 --batch-size 32
```

Training reads `frames_manifest.csv`, validates paths and split integrity, decodes frame paths before OpenCV preprocessing, and uses group/class-balanced sample weights. It aggregates frame probabilities by annotated sequence window for validation threshold selection and held-out test evaluation.

Outputs:

- `models/keras/car_accident_model.weights.h5`: best validation-selected weights.
- `models/keras/model_metadata.json`: model details and validation-selected accident threshold used by live inference.
- `results/keras/training_history.json`: epoch history.
- `results/keras/test_metrics.json`: held-out test metrics.
- `results/keras/model_metadata.json`: copy of model metadata.

The predictor reads the threshold from the metadata next to the weights. Set `SURAKSHAN_ACCIDENT_THRESHOLD` to override it explicitly. If metadata is absent or invalid, inference falls back to 0.93. Do not report test metrics until a complete training run has produced them.

## Run the API

Terminal 1:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

API documentation: http://127.0.0.1:5000/docs

The database is created at `data/surakshan.db`. Do not delete it while the API is running.

## Run the web dashboard

Terminal 2:

```powershell
cd dashboard-web
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. Keep both the API and frontend running.

## Demo accounts

The API seeds these accounts on a fresh database:

| Role | Username | Demo password |
|---|---|---|
| Administrator | `admin` | `Admin@12345` |
| Traffic authority | `authority` | `Authority@12345` |
| Hospital | `hospital` | `Hospital@12345` |

Change demo passwords before sharing the application. Set `SURAKSHAN_ADMIN_PASSWORD`, `SURAKSHAN_AUTHORITY_PASSWORD`, and `SURAKSHAN_HOSPITAL_PASSWORD` before the database is initialized to use different initial passwords. Set a stable, random `SURAKSHAN_AUTH_SECRET` for shared deployments. Existing accounts are not reset when environment variables change.

## Run live camera inference

The dashboard can start the detector process on the same machine that runs the API. The detector uses the configured local camera or stream source. The default is the Android IP Webcam URL currently configured in `live_system/advanced_live.py`.

Override it in the same PowerShell session used to start the API:

```powershell
$env:SURAKSHAN_CAMERA_URL = "0"
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

Use `0` for the default webcam or provide an accessible camera stream URL. The detector runs on the API host, not inside the browser. Logs are written to `logs/detector.log`.

The predictor requires consecutive positive frame predictions. The incident gate applies an additional temporal rule before an incident is recorded. This can add detection delay; evaluate both false alerts and time-to-detection on representative footage before changing these settings.

## Incident workflow

1. The detector submits a potential incident and optional JPEG evidence.
2. The incident appears in the operator queue as unverified.
3. An admin or traffic authority reviews the incident.
4. A verified incident can be assigned to a hospital.
5. Dispatch records are simulated and visible in the dashboard. No external emergency service is contacted.

Evidence photos are stored under `evidence/` by default. Local filesystem storage can be lost on ephemeral cloud hosting. Use persistent storage or object storage if evidence must survive restarts and deployments. Photos are review aids, not forensic-grade evidence.

## Tests and frontend build

Python tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Frontend build:

```powershell
cd dashboard-web
npm run build
```

## Known limitations

- The model is a frame classifier and does not directly learn vehicle motion across a video clip.
- Performance on unseen cameras, locations, weather, lighting, and unusual crash types is not established.
- False alerts and missed accidents remain possible.
- The browser camera preview is not the inference pipeline.
- The detector's OpenCV window is local to the API host and is not streamed into the dashboard.
- The evidence panel stores photos, not playable incident clips.
- Dispatch is simulated only.
- Production deployment still requires security hardening, HTTPS, secret management, retention controls, monitoring, and approved agency integrations.
