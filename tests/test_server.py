"""Integration tests for the local incident API without a running server."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError:  # Dependency installation errors should be clearly surfaced at test setup.
    TestClient = None


@unittest.skipIf(TestClient is None, "FastAPI test client dependencies are not installed")
class IncidentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.env_patch = patch.dict(os.environ, {"SURAKSHAN_DB_PATH": str(self.db_path)})
        self.env_patch.start()
        import live_system.server as server_module
        self.server = importlib.reload(server_module)
        self.client = TestClient(self.server.app)

    def tearDown(self):
        self.client.close()
        self.env_patch.stop()
        self.temp_dir.cleanup()
        # Reload into the default configuration for any other code importing this module.
        import live_system.server as server_module
        importlib.reload(server_module)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_alert_is_persisted_and_listed(self):
        payload = {
            "camera_id": "Camera 1",
            "location": "Demo location",
            "event": "POTENTIAL_ACCIDENT",
            "incident_id": "incident-test-001",
            "confidence": 0.91,
        }
        response = self.client.post("/alert", json=payload)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["incident_id"], "incident-test-001")
        listed = self.client.get("/incidents")
        self.assertEqual(listed.status_code, 200)
        incidents = listed.json()["incidents"]
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["camera_id"], "Camera 1")
        self.assertEqual(incidents[0]["confidence"], 0.91)

    def test_duplicate_incident_id_is_idempotent(self):
        payload = {
            "camera_id": "Camera 1",
            "incident_id": "duplicate-001",
            "confidence": 0.5,
        }
        self.assertEqual(self.client.post("/alert", json=payload).status_code, 202)
        self.assertEqual(self.client.post("/alert", json=payload).status_code, 202)
        incidents = self.client.get("/incidents").json()["incidents"]
        self.assertEqual(len(incidents), 1)

    def test_confidence_out_of_range_is_rejected(self):
        response = self.client.post("/alert", json={
            "camera_id": "Camera 1", "confidence": 1.4
        })
        self.assertEqual(response.status_code, 422)

    def test_incident_limit_must_be_in_range(self):
        self.assertEqual(self.client.get("/incidents?limit=201").status_code, 422)


if __name__ == "__main__":
    unittest.main()
