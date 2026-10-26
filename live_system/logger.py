"""Append-only local incident logger."""
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AccidentLogger:
    def __init__(self, log_file="logs/accidents.log"):
        self.log_file = Path(log_file)
        if not self.log_file.is_absolute():
            self.log_file = PROJECT_ROOT / self.log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, camera_id, incident_id=None):
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        incident = incident_id or "unassigned"
        entry = f"{timestamp} - POTENTIAL_ACCIDENT camera={camera_id} incident={incident}\n"
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        print(f"[LOGGER] {entry.strip()}")
        return entry.strip()
