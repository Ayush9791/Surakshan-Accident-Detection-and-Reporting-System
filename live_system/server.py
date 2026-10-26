"""Incident API for the Surakshan prototype.

The API stores potential incidents and simulated dispatch records. It does not
contact emergency services. Keep it on localhost unless authentication and
network protections are added.
"""
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("SURAKSHAN_DB_PATH", "data/surakshan.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Surakshan Incident API", version="1.1.0")

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "SURAKSHAN_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class AlertPayload(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128)
    location: str = Field(default="Unknown", max_length=256)
    event: str = Field(default="POTENTIAL_ACCIDENT", max_length=64)
    incident_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class IncidentStatusUpdate(BaseModel):
    status: Literal["New", "Under Review", "Verified", "Dismissed", "Resolved"]


class SimulatedDispatchPayload(BaseModel):
    incident_id: str = Field(min_length=1, max_length=128)
    destination: str = Field(default="DEMO-CONTROL-ROOM", min_length=1, max_length=128)
    channel: str = Field(default="dashboard-demo", min_length=1, max_length=64)


def connect():
    connection = sqlite3.connect(DB_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db():
    connection = connect()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                location TEXT NOT NULL,
                event TEXT NOT NULL,
                confidence REAL,
                status TEXT NOT NULL DEFAULT 'New'
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS simulated_dispatches (
                notification_id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                destination TEXT NOT NULL,
                channel TEXT NOT NULL,
                result TEXT NOT NULL DEFAULT 'SIMULATED'
            )
        """)
        connection.commit()
    finally:
        connection.close()


initialize_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/alert", status_code=202)
def receive_alert(payload: AlertPayload):
    incident_id = payload.incident_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    connection = connect()
    try:
        connection.execute(
            """INSERT OR IGNORE INTO incidents
               (incident_id, timestamp, camera_id, location, event, confidence, status)
               VALUES (?, ?, ?, ?, ?, ?, 'New')""",
            (incident_id, timestamp, payload.camera_id, payload.location,
             payload.event, payload.confidence),
        )
        connection.commit()
    finally:
        connection.close()
    return {"accepted": True, "incident_id": incident_id}


@app.get("/incidents")
def list_incidents(limit: int = Query(default=50, ge=1, le=200)):
    connection = connect()
    try:
        rows = connection.execute(
            "SELECT incident_id, timestamp, camera_id, location, event, confidence, status "
            "FROM incidents ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    return {"incidents": [dict(row) for row in rows]}


@app.patch("/incidents/{incident_id}/status")
def update_incident_status(incident_id: str, payload: IncidentStatusUpdate):
    connection = connect()
    try:
        cursor = connection.execute(
            "UPDATE incidents SET status = ? WHERE incident_id = ?",
            (payload.status, incident_id),
        )
        connection.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
        row = connection.execute(
            "SELECT incident_id, status FROM incidents WHERE incident_id = ?",
            (incident_id,),
        ).fetchone()
        return dict(row)
    finally:
        connection.close()


@app.post("/dispatch/simulate", status_code=201)
def simulate_dispatch(payload: SimulatedDispatchPayload):
    """Record a simulated dispatch event; never contacts a real authority."""
    connection = connect()
    try:
        incident = connection.execute(
            "SELECT incident_id FROM incidents WHERE incident_id = ?",
            (payload.incident_id,),
        ).fetchone()
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")
        notification_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        connection.execute(
            """INSERT INTO simulated_dispatches
               (notification_id, incident_id, timestamp, destination, channel, result)
               VALUES (?, ?, ?, ?, ?, 'SIMULATED')""",
            (notification_id, payload.incident_id, timestamp,
             payload.destination, payload.channel),
        )
        connection.commit()
        return {
            "notification_id": notification_id,
            "incident_id": payload.incident_id,
            "timestamp": timestamp,
            "destination": payload.destination,
            "channel": payload.channel,
            "result": "SIMULATED",
            "message": "Recorded locally. No emergency service was contacted.",
        }
    finally:
        connection.close()


@app.get("/dispatch/notifications")
def list_simulated_dispatches(limit: int = Query(default=50, ge=1, le=200)):
    connection = connect()
    try:
        rows = connection.execute(
            """SELECT notification_id, incident_id, timestamp, destination, channel, result
               FROM simulated_dispatches ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    return {"notifications": [dict(row) for row in rows]}
