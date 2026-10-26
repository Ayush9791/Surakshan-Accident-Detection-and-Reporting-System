"""Surakshan incident API with demo authentication and role-based access.

Demo credentials are seeded for academic evaluation. Change all demo passwords and
SURAKSHAN_AUTH_SECRET before any shared deployment. This is not a certified
emergency dispatch service; dispatch events are simulations only.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("SURAKSHAN_DB_PATH", "data/surakshan.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
AUTH_SECRET = os.environ.get("SURAKSHAN_AUTH_SECRET", "local-academic-demo-secret-change-before-deployment")
TOKEN_HOURS = 8
DETECTOR_PROCESS = None
DETECTOR_LOG = Path("logs/detector.log")
DETECTOR_LOG.parent.mkdir(parents=True, exist_ok=True)
DETECTOR_LOG_HANDLE = None
STATUSES = {"New", "Under Review", "Verified", "Dismissed", "Resolved"}
app = FastAPI(title="Surakshan Incident API", version="2.0.0")
origins = [x.strip() for x in os.environ.get("SURAKSHAN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
                   allow_headers=["Content-Type", "Authorization"])


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


class LoginPayload(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class UserCreatePayload(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=256)
    role: Literal["authority", "hospital"]
    organization: str = Field(min_length=1, max_length=160)


class PasswordChangePayload(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=10, max_length=256)


def connect():
    db = sqlite3.connect(DB_PATH, timeout=5)
    db.row_factory = sqlite3.Row
    return db


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return base64.urlsafe_b64encode(salt).decode() + "$" + base64.urlsafe_b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_s, digest_s = stored.split("$", 1)
        salt = base64.urlsafe_b64decode(salt_s.encode())
        expected = base64.urlsafe_b64decode(digest_s.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def issue_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user["username"], "role": user["role"], "organization": user["organization"],
               "iat": int(now.timestamp()), "exp": int((now + timedelta(hours=TOKEN_HOURS)).timestamp())}
    encoded = b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = b64(hmac.new(AUTH_SECRET.encode(), encoded.encode(), hashlib.sha256).digest())
    return encoded + "." + signature


def decode_token(token: str) -> dict:
    try:
        encoded, signature = token.split(".", 1)
        expected = b64(hmac.new(AUTH_SECRET.encode(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if int(payload["exp"]) <= int(datetime.now(timezone.utc).timestamp()):
            raise ValueError("expired")
        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in again.") from exc


def audit(db, username: str, action: str, details: str = ""):
    db.execute("INSERT INTO audit_log (audit_id, timestamp, username, action, details) VALUES (?, ?, ?, ?, ?)",
               (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(timespec="seconds"), username, action, details[:500]))


def initialize_db():
    db = connect()
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, camera_id TEXT NOT NULL,
            location TEXT NOT NULL, event TEXT NOT NULL, confidence REAL,
            status TEXT NOT NULL DEFAULT 'New', assigned_to TEXT NOT NULL DEFAULT '')""")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(incidents)")}
        if "assigned_to" not in columns:
            db.execute("ALTER TABLE incidents ADD COLUMN assigned_to TEXT NOT NULL DEFAULT ''")
        db.execute("""CREATE TABLE IF NOT EXISTS simulated_dispatches (
            notification_id TEXT PRIMARY KEY, incident_id TEXT NOT NULL, timestamp TEXT NOT NULL,
            destination TEXT NOT NULL, channel TEXT NOT NULL, result TEXT NOT NULL DEFAULT 'SIMULATED',
            created_by TEXT NOT NULL DEFAULT '')""")
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL,
            organization TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS audit_log (
            audit_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, username TEXT NOT NULL,
            action TEXT NOT NULL, details TEXT NOT NULL)""")
        # Seed demo accounts only if their usernames do not already exist.
        demos = [
            ("admin", os.environ.get("SURAKSHAN_ADMIN_PASSWORD", "Admin@12345"), "admin", "Surakshan Administration"),
            ("authority", os.environ.get("SURAKSHAN_AUTHORITY_PASSWORD", "Authority@12345"), "authority", "Demo Traffic Control Room"),
            ("hospital", os.environ.get("SURAKSHAN_HOSPITAL_PASSWORD", "Hospital@12345"), "hospital", "Demo General Hospital"),
        ]
        for username, password, role, org in demos:
            db.execute("INSERT OR IGNORE INTO users (username,password_hash,role,organization,active,created_at) VALUES (?,?,?,?,1,?)",
                       (username, hash_password(password), role, org, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        db.commit()
    finally:
        db.close()


initialize_db()


def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign-in required")
    token = decode_token(authorization.split(" ", 1)[1].strip())
    db = connect()
    try:
        row = db.execute("SELECT username,role,organization,active FROM users WHERE username=?", (token.get("sub"),)).fetchone()
    finally:
        db.close()
    if not row or not row["active"] or row["role"] != token.get("role"):
        raise HTTPException(status_code=401, detail="Account is inactive or session is no longer valid")
    return dict(row)


def require_roles(*roles):
    def check(user: dict = Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Your role cannot perform this action")
        return user
    return check


@app.get("/health")
def health():
    return {"status": "ok", "service": "surakshan-api"}


@app.post("/auth/login")
def login(payload: LoginPayload):
    db = connect()
    try:
        row = db.execute("SELECT * FROM users WHERE username=?", (payload.username,)).fetchone()
        if not row or not row["active"] or not verify_password(payload.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        user = {"username": row["username"], "role": row["role"], "organization": row["organization"]}
        audit(db, user["username"], "login", "Successful login")
        db.commit()
        return {"access_token": issue_token(user), "token_type": "bearer", "expires_in": TOKEN_HOURS * 3600, "user": user}
    finally:
        db.close()


@app.get("/auth/me")
def who_am_i(user: dict = Depends(current_user)):
    return user


@app.post("/auth/change-password")
def change_password(payload: PasswordChangePayload, user: dict = Depends(current_user)):
    db = connect()
    try:
        row = db.execute("SELECT password_hash FROM users WHERE username=?", (user["username"],)).fetchone()
        if not row or not verify_password(payload.current_password, row["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        db.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(payload.new_password), user["username"]))
        audit(db, user["username"], "password_changed")
        db.commit()
        return {"updated": True}
    finally:
        db.close()


@app.post("/admin/users", status_code=201)
def create_user(payload: UserCreatePayload, admin: dict = Depends(require_roles("admin"))):
    db = connect()
    try:
        try:
            db.execute("INSERT INTO users (username,password_hash,role,organization,active,created_at) VALUES (?,?,?,?,1,?)",
                       (payload.username, hash_password(payload.password), payload.role, payload.organization,
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Username already exists") from exc
        audit(db, admin["username"], "user_created", f"{payload.username} | {payload.role} | {payload.organization}")
        db.commit()
        return {"username": payload.username, "role": payload.role, "organization": payload.organization, "active": True}
    finally:
        db.close()


@app.get("/admin/users")
def list_users(admin: dict = Depends(require_roles("admin"))):
    db = connect()
    try:
        rows = db.execute("SELECT username,role,organization,active,created_at FROM users ORDER BY role,username").fetchall()
        return {"users": [dict(row) for row in rows]}
    finally:
        db.close()


@app.post("/alert", status_code=202)
def receive_alert(payload: AlertPayload, x_detector_token: Optional[str] = Header(default=None)):
    expected_detector_token = os.environ.get("SURAKSHAN_DETECTOR_TOKEN")
    if expected_detector_token and not hmac.compare_digest(x_detector_token or "", expected_detector_token):
        raise HTTPException(status_code=401, detail="Invalid detector token")
    incident_id = payload.incident_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db = connect()
    try:
        db.execute("""INSERT OR IGNORE INTO incidents
            (incident_id,timestamp,camera_id,location,event,confidence,status)
            VALUES (?,?,?,?,?,?,'New')""",
            (incident_id,timestamp,payload.camera_id,payload.location,payload.event,payload.confidence))
        audit(db, "detector", "incident_received", f"{incident_id} | {payload.camera_id}")
        db.commit()
    finally:
        db.close()
    return {"accepted": True, "incident_id": incident_id}


@app.get("/incidents")
def list_incidents(limit: int = Query(default=100, ge=1, le=200), user: dict = Depends(current_user)):
    db = connect()
    try:
        if user["role"] == "hospital":
            rows = db.execute("""SELECT incident_id,timestamp,camera_id,location,event,confidence,status,assigned_to
                FROM incidents WHERE status IN ('Verified','Resolved') AND assigned_to=?
                ORDER BY timestamp DESC LIMIT ?""", (user["organization"], limit)).fetchall()
        else:
            rows = db.execute("""SELECT incident_id,timestamp,camera_id,location,event,confidence,status,assigned_to
                FROM incidents ORDER BY timestamp DESC LIMIT ?""", (limit,)).fetchall()
        return {"incidents": [dict(row) for row in rows]}
    finally:
        db.close()


@app.patch("/incidents/{incident_id}/status")
def update_incident_status(incident_id: str, payload: IncidentStatusUpdate,
                           user: dict = Depends(require_roles("admin", "authority"))):
    db = connect()
    try:
        cursor = db.execute("UPDATE incidents SET status=? WHERE incident_id=?", (payload.status, incident_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
        audit(db, user["username"], "incident_status_updated", f"{incident_id} -> {payload.status}")
        db.commit()
        row = db.execute("SELECT incident_id,status FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
        return dict(row)
    finally:
        db.close()


@app.post("/incidents/{incident_id}/assign")
def assign_incident(incident_id: str, organization: str = Query(min_length=1, max_length=160),
                    user: dict = Depends(require_roles("admin", "authority"))):
    db = connect()
    try:
        target = db.execute("SELECT username FROM users WHERE organization=? AND role='hospital' AND active=1 LIMIT 1", (organization,)).fetchone()
        if not target:
            raise HTTPException(status_code=404, detail="No active hospital account found for that organization")
        cursor = db.execute("UPDATE incidents SET assigned_to=? WHERE incident_id=?", (organization, incident_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Incident not found")
        audit(db, user["username"], "incident_assigned", f"{incident_id} -> {organization}")
        db.commit()
        return {"incident_id": incident_id, "assigned_to": organization}
    finally:
        db.close()


@app.post("/dispatch/simulate", status_code=201)
def simulate_dispatch(payload: SimulatedDispatchPayload, user: dict = Depends(require_roles("admin", "authority"))):
    db = connect()
    try:
        incident = db.execute("SELECT incident_id FROM incidents WHERE incident_id=?", (payload.incident_id,)).fetchone()
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")
        notification_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        db.execute("""INSERT INTO simulated_dispatches
            (notification_id,incident_id,timestamp,destination,channel,result,created_by)
            VALUES (?,?,?,?,?,'SIMULATED',?)""",
            (notification_id,payload.incident_id,timestamp,payload.destination,payload.channel,user["username"]))
        audit(db, user["username"], "simulated_dispatch", f"{payload.incident_id} -> {payload.destination}")
        db.commit()
        return {"notification_id":notification_id,"incident_id":payload.incident_id,"timestamp":timestamp,
                "destination":payload.destination,"channel":payload.channel,"result":"SIMULATED",
                "message":"Recorded locally. No emergency service was contacted."}
    finally:
        db.close()


@app.get("/dispatch/notifications")
def list_simulated_dispatches(limit: int = Query(default=50, ge=1, le=200),
                              user: dict = Depends(require_roles("admin", "authority"))):
    db = connect()
    try:
        rows = db.execute("""SELECT notification_id,incident_id,timestamp,destination,channel,result,created_by
            FROM simulated_dispatches ORDER BY timestamp DESC LIMIT ?""", (limit,)).fetchall()
        return {"notifications": [dict(row) for row in rows]}
    finally:
        db.close()



@app.get("/detector/status")
def detector_status(user: dict = Depends(require_roles("admin", "authority"))):
    running = DETECTOR_PROCESS is not None and DETECTOR_PROCESS.poll() is None
    return {"running": running, "mode": "server-local-webcam", "note": "Inference uses the webcam attached to the API host."}


@app.post("/detector/start")
def detector_start(user: dict = Depends(require_roles("admin", "authority"))):
    global DETECTOR_PROCESS, DETECTOR_LOG_HANDLE
    if DETECTOR_PROCESS is not None and DETECTOR_PROCESS.poll() is None:
        return {"running": True, "message": "Detector is already running"}
    try:
        DETECTOR_LOG_HANDLE = open(DETECTOR_LOG, "a", encoding="utf-8")
        DETECTOR_PROCESS = subprocess.Popen(
            [sys.executable, "-m", "live_system.advanced_live"],
            stdin=subprocess.DEVNULL, stdout=DETECTOR_LOG_HANDLE,
            stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parent.parent),
        )
        db = connect()
        try:
            audit(db, user["username"], "detector_started", "Local host webcam inference")
            db.commit()
        finally:
            db.close()
        return {"running": True, "pid": DETECTOR_PROCESS.pid, "message": "Detector started on the API host"}
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not start detector: {exc}") from exc


@app.post("/detector/stop")
def detector_stop(user: dict = Depends(require_roles("admin", "authority"))):
    global DETECTOR_PROCESS, DETECTOR_LOG_HANDLE
    if DETECTOR_PROCESS is None or DETECTOR_PROCESS.poll() is not None:
        return {"running": False, "message": "Detector is not running"}
    DETECTOR_PROCESS.terminate()
    try:
        DETECTOR_PROCESS.wait(timeout=5)
    except subprocess.TimeoutExpired:
        DETECTOR_PROCESS.kill()
        DETECTOR_PROCESS.wait(timeout=3)
    if DETECTOR_LOG_HANDLE:
        DETECTOR_LOG_HANDLE.close()
        DETECTOR_LOG_HANDLE = None
    db = connect()
    try:
        audit(db, user["username"], "detector_stopped", "Local host webcam inference")
        db.commit()
    finally:
        db.close()
    return {"running": False, "message": "Detector stopped"}


@app.get("/admin/audit")
def list_audit(limit: int = Query(default=100, ge=1, le=500), admin: dict = Depends(require_roles("admin"))):
    db = connect()
    try:
        rows = db.execute("SELECT audit_id,timestamp,username,action,details FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return {"events": [dict(row) for row in rows]}
    finally:
        db.close()
