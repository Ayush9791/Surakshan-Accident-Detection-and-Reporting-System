# Surakshan: Real-Time AI and Dataset Plan

## Goal

Build a multi-camera traffic incident operations platform. The first deliverable is a safe local demo with a webcam, persistent incidents, human review, and simulated notifications. It is not an automatic emergency-dispatch system.

## Recommended model strategy

Do not replace the existing PyTorch CNN until there is a measured baseline. The current CNN predicts a class from one resized frame; it cannot reason directly about collision motion over time.

### Stage A — Baseline and data audit

- Inventory the existing Accident/Non Accident images and recover source video identifiers where possible.
- Group train/validation/test splits by source video or scene; do not randomly split adjacent frames.
- Record preprocessing, class mapping, checkpoint version and reproducible evaluation outputs.
- Measure precision, recall, F1, false positives per camera-hour and detection delay.

### Stage B — CCTV dataset benchmark

Start with ACCIDENT: https://accidentbench.github.io/

The benchmark provides 2,027 real surveillance clips and 2,211 synthetic clips, with accident time, spatial location and high-level collision type annotations. Use the real subset for realism and the synthetic subset as additional training/controlled variation, while reporting results separately. Check dataset licence and access terms before redistribution.

Paper: https://openaccess.thecvf.com/content/CVPR2026W/AUTOPILOT/html/Picek_ACCIDENT_A_Benchmark_Dataset_for_Vehicle_Accident_Detection_from_Traffic_CVPRW_2026_paper.html

### Stage C — Perception plus temporal recognition

Recommended baseline architecture:

1. **Vehicle detector:** use a lightweight pretrained detector suitable for the available GPU/CPU budget.
2. **Tracker:** start with ByteTrack or BoT-SORT to maintain vehicle tracks across frames.
3. **Temporal crash model:** evaluate a compact video model or temporal head over sampled frame features and track/motion features. Predict a potential-crash score over a time window, rather than treating one frame as definitive.
4. **Event state machine:** combine model scores across time, cooldown and incident state; tune thresholds on validation footage.
5. **Evidence capture:** preserve a rolling pre-event buffer and post-event footage, and attach timestamps and camera ID to the incident.
6. **Human review:** require operator confirmation before any external dispatch integration.

YOLO tracking documentation: https://docs.ultralytics.com/modes/track/

The detector/tracker is not the accident model. Bounding-box overlap, sudden deceleration or a single high-confidence frame should not alone trigger a confirmed-crash label.

### Stage D — Evaluation

- Keep all clips from a source video in one split.
- Hold out camera locations or scenes to measure out-of-distribution performance.
- Include normal traffic, congestion, sudden braking, near misses, occlusions, night, rain and low-resolution streams.
- Report event-level recall, false alerts per camera-hour, precision, time-to-detection, localisation quality and performance by lighting/weather.
- Calibrate confidence on held-out data and use an explicit “uncertain / operator review” state.
- Test latency and throughput on the intended deployment hardware.

## Target runtime architecture

- **Camera ingestion worker:** authorised RTSP/ONVIF sources or local test camera; reconnect and monitor frozen/offline streams.
- **Inference workers:** one or more isolated workers for detection/tracking and temporal classification; do not put heavy inference inside HTTP request handlers.
- **Incident API:** FastAPI, validation, event state transitions and WebSocket/SSE updates.
- **Storage:** PostgreSQL/PostGIS for incidents, camera inventory, jurisdictions and users; object storage for video clips; Redis/queue only when job volume warrants it.
- **Web app:** Next.js operator dashboard. Use a media gateway for browser playback (WebRTC/HLS), not raw RTSP in the browser.
- **Notification adapter:** simulated during the prototype. Future real dispatch must use an approved agency integration, jurisdiction mapping, acknowledgement and audit trail.

## Deployment constraints

A cloud service cannot directly open a webcam attached to a browser user's PC. A local camera preview is not the same as server-side inference. For a cloud-hosted model, securely send the video to an inference service or run an edge worker close to the camera. Production deployment requires authentication, HTTPS, role-based access, secure camera credentials, retention rules, monitoring and incident audit logs.

## Current milestone status

- [x] Next.js/React dashboard scaffold with incident queue and filters.
- [x] FastAPI status transition endpoint.
- [x] Persisted simulated-dispatch endpoint and log.
- [x] Browser-local webcam preview (preview only).
- [x] CI workflow for dashboard build.
- [ ] Connect video frames to the inference service.
- [ ] Replace/benchmark the frame-only classifier with a temporal accident model.
- [ ] Add authentication, roles, camera inventory and WebSocket/SSE updates.
- [ ] Add media gateway and stored evidence playback.
- [ ] Validate on unseen cameras before any real-world pilot.
