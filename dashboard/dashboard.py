import sys
import os
import cv2
import time

# allow parent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from live_system.camera import CameraStream
from live_system.predict import predict_frame
from live_system.recorder import Recorder
from live_system.logger import AccidentLogger


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Surakshan Dashboard",
    layout="wide"
)

st.title("🚨 Surakshan Accident Detection Dashboard (Live)")
st.markdown("Monitoring CCTV feeds in real-time")


# ----------------------------------------------------------
# Camera Sources
# ----------------------------------------------------------
CAMERA_SOURCES = {
    "Camera 1": 0,       # webcam  
    # "Highway": "test_video.mp4",
    # "StreetCam": "rtsp://user:pass@IP/stream"
}

# Initialize components ONLY ONCE
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.streams = {cam: CameraStream(src) for cam, src in CAMERA_SOURCES.items()}
    st.session_state.recorders = {cam: Recorder() for cam in CAMERA_SOURCES}
    st.session_state.logger = AccidentLogger()
    st.session_state.running = False

# ----------------------------------------------------------
# Start / Stop
# ----------------------------------------------------------
start_btn = st.button("▶ Start Monitoring")
stop_btn = st.button("⏹ Stop Monitoring")

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False


# ----------------------------------------------------------
# Create camera placeholders
# ----------------------------------------------------------
cam_cols = st.columns(len(CAMERA_SOURCES))
cam_placeholders = {cam: cam_cols[i].empty() for i, cam in enumerate(CAMERA_SOURCES)}


# ----------------------------------------------------------
# MAIN LIVE LOOP (NO rerun needed)
# ----------------------------------------------------------
while st.session_state.running:

    for i, (cam_id, stream) in enumerate(st.session_state.streams.items()):

        ret, frame = stream.read()
        if not ret:
            cam_placeholders[cam_id].error(f"{cam_id}: No signal")
            continue

        # prediction
        result = predict_frame(frame)
        color = (0, 255, 0) if result == "NORMAL" else (0, 0, 255)
        cv2.putText(frame, f"{cam_id}: {result}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # accident event
        if result == "ACCIDENT":
            st.session_state.logger.log(cam_id)
            clip_path = st.session_state.recorders[cam_id].save_clip(frame, cam_id)

            st.sidebar.error(f"🚨 Accident detected on {cam_id}")

            if clip_path:
                st.sidebar.video(clip_path)

        # store frame in buffer
        st.session_state.recorders[cam_id].add_frame(frame)

        # display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_placeholders[cam_id].image(rgb, caption=cam_id, use_container_width=True)

    # maintain 30 FPS
    time.sleep(0.03)
# ----------------------------------------------------------