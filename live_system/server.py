"""Local FastAPI receiver for potential-accident alerts.

Keep this server on localhost unless authentication and network protections are added.
"""
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("SURAKSHAN_DB_PATH", "data/surakshan.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Surakshan Incident API", version="1.0.0")


class AlertPayload(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128)
    location: str = Field(default="Unknown", max_length=256)
    event: str = Field(default="POTENTIAL_ACCIDENT", max_length=64)
    incident_id: Optional[str] = Field(default=None, max_length=128)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


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
def list_incidents(limit: int = 50):
    limit = max(1, min(limit, 200))
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
