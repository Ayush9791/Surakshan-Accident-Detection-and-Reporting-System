"""Responsive Streamlit dashboard for Surakshan."""
import sys
from pathlib import Path

import cv2
import streamlit as st
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_system.api import send_alert
from live_system.camera import CameraStream
from live_system.incident import IncidentGate
from live_system.logger import AccidentLogger
from live_system.predict import predict_frame_details
from live_system.recorder import Recorder

st.set_page_config(page_title="Surakshan Dashboard", layout="wide")
st.title("Surakshan Accident Detection Dashboard")
st.caption("Detections are potential incidents and require human verification.")

SOURCES = {"Camera 1": 0}
for key, default in (
    ("running", False), ("streams", {}), ("recorders", {}),
    ("gates", {}), ("logger", None),
):
    if key not in st.session_state:
        st.session_state[key] = AccidentLogger() if key == "logger" else default


def stop_monitoring():
    st.session_state.running = False
    for stream in st.session_state.streams.values():
        stream.stop()
    for recorder in st.session_state.recorders.values():
        recorder.close()
    st.session_state.streams = {}
    st.session_state.recorders = {}
    st.session_state.gates = {}


left, right = st.columns(2)
if left.button("Start monitoring", disabled=st.session_state.running):
    st.session_state.streams = {name: CameraStream(src) for name, src in SOURCES.items()}
    st.session_state.recorders = {name: Recorder() for name in SOURCES}
    st.session_state.gates = {name: IncidentGate() for name in SOURCES}
    st.session_state.running = True
if right.button("Stop monitoring", disabled=not st.session_state.running):
    stop_monitoring()


@st.fragment(run_every=0.5)
def show_cameras():
    if not st.session_state.running:
        st.info("Monitoring is stopped.")
        return
    columns = st.columns(max(1, len(st.session_state.streams)))
    for index, (camera_id, stream) in enumerate(st.session_state.streams.items()):
        with columns[index]:
            st.subheader(camera_id)
            ok, frame = stream.read()
            if not ok or frame is None:
                st.warning("No camera signal.")
                continue
            details = predict_frame_details(frame)
            label = details["label"]
            color = (0, 0, 255) if label == "ACCIDENT" else (0, 255, 0)
            cv2.putText(frame, f"{label} ({details['confidence']:.0%})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            recorder = st.session_state.recorders[camera_id]
            recorder.add_frame(frame)
            if st.session_state.gates[camera_id].update(label):
                incident_id = str(uuid.uuid4())
                st.session_state.logger.log(camera_id, incident_id=incident_id)
                recorder.start_incident(camera_id)
                send_alert(
                    camera_id,
                    incident_id=incident_id,
                    confidence=details["accident_probability"],
                )
                st.sidebar.warning(f"Potential accident: {camera_id} ({incident_id[:8]})")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"{label} | model confidence {details['confidence']:.1%}")


show_cameras()

st.subheader("Recent event log")
log_file = ROOT / "logs" / "accidents.log"
if log_file.exists():
    entries = log_file.read_text(encoding="utf-8").splitlines()
    st.code("\n".join(entries[-20:]) if entries else "No events logged.")
else:
    st.caption("No events logged yet.")
