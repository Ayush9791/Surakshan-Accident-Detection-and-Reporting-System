"""Tests for incident status management and simulated dispatch records."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class IncidentWorkflowApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "workflow.db"
        self.env_patch = patch.dict(os.environ, {"SURAKSHAN_DB_PATH": str(self.db_path)})
        self.env_patch.start()
        import live_system.server as server_module
        self.server = importlib.reload(server_module)
        self.client = TestClient(self.server.app)
        self.incident_id = "workflow-test-001"
        response = self.client.post("/alert", json={
            "camera_id": "LOCAL-WEBCAM-01",
            "location": "Local demo",
            "event": "SIMULATED_POTENTIAL_ACCIDENT",
            "incident_id": self.incident_id,
            "confidence": 0.93,
        })
        self.assertEqual(response.status_code, 202)

    def tearDown(self):
        self.client.close()
        self.env_patch.stop()
        self.temp_dir.cleanup()
        import live_system.server as server_module
        importlib.reload(server_module)

    def test_status_can_be_updated(self):
        response = self.client.patch(
            f"/incidents/{self.incident_id}/status",
            json={"status": "Verified"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "Verified")
        incidents = self.client.get("/incidents").json()["incidents"]
        self.assertEqual(incidents[0]["status"], "Verified")

    def test_invalid_status_is_rejected(self):
        response = self.client.patch(
            f"/incidents/{self.incident_id}/status",
            json={"status": "Send Police"},
        )
        self.assertEqual(response.status_code, 422)

    def test_simulated_dispatch_is_recorded_without_external_side_effect(self):
        response = self.client.post("/dispatch/simulate", json={
            "incident_id": self.incident_id,
            "destination": "DEMO-CONTROL-ROOM",
            "channel": "dashboard-demo",
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["result"], "SIMULATED")
        self.assertIn("No emergency service was contacted", response.json()["message"])
        records = self.client.get("/dispatch/notifications").json()["notifications"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["incident_id"], self.incident_id)

    def test_dispatch_for_unknown_incident_is_rejected(self):
        response = self.client.post("/dispatch/simulate", json={
            "incident_id": "missing-incident",
        })
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
