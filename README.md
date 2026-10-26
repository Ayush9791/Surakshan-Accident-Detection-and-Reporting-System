# Surakshan: Accident Detection and Reporting System

Surakshan is an academic Python prototype that classifies video frames as **Accident** or **Non Accident**, presents live predictions, groups repeated positive predictions into potential incidents, records video evidence, and sends an HTTP alert to a local backend.

> **Safety limitation:** This is a demonstration prototype, not a certified emergency-detection system. Predictions can be wrong. Treat detections as potential incidents requiring human verification. Do not use the application as the sole basis for emergency dispatch.

## Project modules

- `model.py`: PyTorch convolutional neural network.
- `train.py`: training, validation, best-checkpoint selection, and held-out test metrics.
- `live_system/basic_live.py`: basic webcam prediction preview.
- `live_system/advanced_live.py`: temporal incident gating, event logs, video evidence, and HTTP alerts.
- `live_system/server.py`: local FastAPI incident API backed by SQLite.
- `dashboard/dashboard.py`: Streamlit monitoring interface.
- `tests/`: unit tests for incident gating and evidence recording, with API tests.

## Requirements and installation

Use Python 3.10 or newer. From the project root, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly with `.venv\Scripts\python.exe`.

The checked-in dataset is expected to use this structure:

```text
dataset/
  Accident/
  Non Accident/
```

If you replace or extend the dataset, keep the two class folder names exactly as shown.

## Run the project

Run commands from the repository root. The checked-in `surakshan_model.pth` checkpoint is used by default. Set `SURAKSHAN_MODEL_PATH` to use a different checkpoint.

### 1. Basic webcam preview

```powershell
python -m live_system.basic_live
```

Press **Esc** to exit. This mode only displays per-frame predictions.

### 2. Advanced incident monitoring

Start the local API first in one terminal:

```powershell
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

In another terminal, run:

```powershell
python -m live_system.advanced_live
```

Press **Esc** to stop. The default alert endpoint is `http://127.0.0.1:5000/alert`. Override it with:

```powershell
$env:SURAKSHAN_ALERT_URL = "http://127.0.0.1:5000/alert"
```

To change the SQLite database location, set `SURAKSHAN_DB_PATH` before launching the API. For example:

```powershell
$env:SURAKSHAN_DB_PATH = "data/demo.db"
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

API endpoints:
- `GET /health`: service health.
- `POST /alert`: record a potential incident.
- `GET /incidents?limit=50`: list recent incidents.
- `GET /docs`: interactive local API documentation.

### 3. Streamlit dashboard

```powershell
streamlit run dashboard/dashboard.py
```

Use **Start monitoring** to begin webcam inference and **Stop monitoring** to release the camera. The dashboard uses camera 0 by default. Source configuration is in `dashboard/dashboard.py`.

### 4. Train and evaluate the model

```powershell
python train.py
```

Training uses a seeded 70/15/15 image-level split and writes `surakshan_model.pth`. The script reports accuracy, precision, recall, F1, and loss.

**Evaluation warning:** an image-level random split can leak near-duplicate frames from the same video across training and test data, producing overly optimistic metrics. The repository currently does not preserve source-video IDs needed for a reliable grouped split. Do not claim the test metrics establish real-world accident-detection accuracy. For a defensible evaluation, split by source video or scene, then test on footage not used for training or tuning.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions runs the unit suite on pushes to `main` and on pull requests. The suite checks incident-gate behaviour, video recorder frame selection, and API validation/incident persistence. Unit tests do not verify model accuracy or camera compatibility.

## Outputs and local data

- Potential incidents are appended to `logs/accidents.log`.
- Recorded video clips are written to `clips/`.
- API incident records are written to `data/surakshan.db` by default.
- Generated local logs, clips, databases, bytecode, and virtual environments should not be committed.

## Limitations and security

- The classifier labels the full frame; it does not localise a collision or identify individual vehicles.
- Temporal gating suppresses repeated frame-level alerts. It is not a physical collision-understanding model.
- The camera location is not inferred automatically; the API defaults location to `Unknown`.
- The local API has **no authentication**. Keep it bound to `127.0.0.1`. Do not expose it to a local network or the public internet without adding authentication, access controls, and transport security.
- Alert acceptance by the local API does not notify police, ambulance services, or other emergency responders.
- Camera availability, performance, and MP4 encoding depend on the host hardware and its OpenCV build.
- The live webcam/RTSP workflow and real-world model performance must be tested on the actual submission/demo machine.

## Academic project summary

Surakshan demonstrates a computer-vision pipeline that classifies frames, applies temporal confirmation to reduce repeated alerts, preserves a clip around a potential incident, records metadata in SQLite, and exposes incidents through a local API and monitoring dashboard. Its results should be presented as a prototype demonstration rather than as a validated safety-critical system.
