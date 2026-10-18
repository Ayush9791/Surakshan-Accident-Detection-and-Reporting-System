# Surakshan: Accident Detection and Reporting System

Surakshan is an academic Python prototype that classifies video frames as **Accident** or **Non Accident**, displays live predictions, groups repeated positive predictions into potential incidents, records video evidence, and can send an HTTP alert to a separately running backend.

> **Safety limitation:** This is not a certified emergency-detection system. Predictions can be wrong. Treat detections as potential incidents requiring human verification, and do not use this application as the sole basis for emergency dispatch.

## Features

- PyTorch CNN frame classifier.
- OpenCV webcam capture, with configurable video-file and RTSP sources.
- Shared inference code with predicted-class confidence and accident probability.
- Temporal confirmation and duplicate-event suppression.
- Rolling pre-event buffer and actual post-event frame recording.
- Local incident log and configurable HTTP alert client.
- Streamlit monitoring dashboard.
- Training, validation, and held-out test metrics.

## Setup

Python 3.10 or newer is recommended. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset must have these exact folder names:

```text
dataset/
  Accident/
  Non Accident/
```

## Train and evaluate

```bash
python train.py
```

The script uses a 70/15/15 image-level train/validation/test split, selects the checkpoint with the best validation loss, and prints test metrics. The model checkpoint is written to `surakshan_model.pth`.

**Evaluation caveat:** images from the same source video can be near-duplicates. The repository does not preserve source-video IDs for a grouped split, so this random image-level split may overestimate real-world performance. For credible reported metrics, split by source video or scene and test on footage not used during training.

## Run live mode

Basic camera preview:

```bash
python -m live_system.basic_live
```

Advanced monitoring with incident logging, evidence recording, and alert attempts:

```bash
python -m live_system.advanced_live
```

Press **Esc** to exit the advanced preview.

Dashboard:

```bash
streamlit run dashboard/dashboard.py
```

## Run the local incident API

Start the included SQLite-backed API in a separate terminal:

```bash
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

Check `http://127.0.0.1:5000/health` and list recent incidents at `http://127.0.0.1:5000/incidents`. Records are stored in `data/surakshan.db` by default. Set `SURAKSHAN_DB_PATH` to change the database location.

**Security:** The API has no authentication. Keep it bound to localhost for local demos. Do not expose it to a network or the public internet without authentication, access controls, and transport security.

## Alert endpoint

The client defaults to `http://127.0.0.1:5000/alert`. Start the included API in a separate terminal before using alert delivery.

Set another endpoint using PowerShell:

```powershell
$env:SURAKSHAN_ALERT_URL = "http://127.0.0.1:5000/alert"
```

A local detection log is not proof that a remote alert was delivered. Check the API response logs.

## Evidence recording

The recorder keeps a rolling buffer of recent frames and, after an incident is triggered, captures actual subsequent frames before writing a uniquely named MP4 clip. Output duration depends on the real camera frame rate and the configured recorder FPS. Verify clips on the target camera before relying on them as evidence.

## Tests

```bash
python -m unittest discover -s tests -v
```

The included unit tests cover temporal incident confirmation and duplicate suppression. They do not measure model accuracy or prove real-world accident detection performance.

## Known limitations

- The classifier predicts a class for the entire frame; it does not localise vehicles or produce bounding boxes.
- Temporal gating is a debounce mechanism, not a collision-understanding model.
- Evaluation must be repeated with source-video-separated test data.
- Camera location is not inferred automatically.
- Alert delivery needs a separately deployed backend.
- Camera and video encoding behaviour depends on the local hardware and OpenCV build.
