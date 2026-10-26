"""API tests for authentication, incidents, roles and simulated dispatch."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class IncidentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.env_patch = patch.dict(os.environ, {"SURAKSHAN_DB_PATH": str(self.db_path)})
        self.env_patch.start()
        import live_system.server as server_module
        self.server = importlib.reload(server_module)
        self.client = TestClient(self.server.app)
        self.admin = self.login("admin", "Admin@12345")
        self.authority = self.login("authority", "Authority@12345")
        self.hospital = self.login("hospital", "Hospital@12345")
        self.headers = {"Authorization": f"Bearer {self.admin}"}
        self.authority_headers = {"Authorization": f"Bearer {self.authority}"}
        self.hospital_headers = {"Authorization": f"Bearer {self.hospital}"}

    def login(self, username, password):
        response = self.client.post("/auth/login", json={"username": username, "password": password})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["access_token"]

    def create_incident(self, incident_id="incident-test-001", status=None):
        response = self.client.post("/alert", json={
            "camera_id": "Camera 1", "location": "Demo location", "event": "POTENTIAL_ACCIDENT",
            "incident_id": incident_id, "confidence": 0.91,
        })
        self.assertEqual(response.status_code, 202, response.text)
        if status:
            update = self.client.patch(f"/incidents/{incident_id}/status", json={"status": status}, headers=self.headers)
            self.assertEqual(update.status_code, 200, update.text)
        return incident_id

    def tearDown(self):
        self.client.close()
        self.env_patch.stop()
        self.temp_dir.cleanup()
        import live_system.server as server_module
        importlib.reload(server_module)

    def test_health_endpoint(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_incidents_require_authentication(self):
        self.create_incident()
        self.assertEqual(self.client.get("/incidents").status_code, 401)

    def test_alert_is_persisted_and_listed_for_admin(self):
        self.create_incident()
        response = self.client.get("/incidents", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        incidents = response.json()["incidents"]
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["camera_id"], "Camera 1")
        self.assertEqual(incidents[0]["confidence"], 0.91)

    def test_duplicate_incident_id_is_idempotent(self):
        self.create_incident("duplicate-001")
        self.create_incident("duplicate-001")
        self.assertEqual(len(self.client.get("/incidents", headers=self.headers).json()["incidents"]), 1)

    def test_confidence_out_of_range_is_rejected(self):
        response = self.client.post("/alert", json={"camera_id": "Camera 1", "confidence": 1.4})
        self.assertEqual(response.status_code, 422)

    def test_incident_limit_must_be_in_range(self):
        self.assertEqual(self.client.get("/incidents?limit=201", headers=self.headers).status_code, 422)

    def test_invalid_login_rejected(self):
        response = self.client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        self.assertEqual(response.status_code, 401)

    def test_admin_can_create_authority_account(self):
        response = self.client.post("/admin/users", headers=self.headers, json={
            "username": "new_authority", "password": "LongPassword!1", "role": "authority",
            "organization": "City Traffic Control",
        })
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["role"], "authority")

    def test_hospital_cannot_create_users_or_update_status(self):
        self.create_incident()
        create = self.client.post("/admin/users", headers=self.hospital_headers, json={
            "username": "blocked_user", "password": "LongPassword!1", "role": "hospital", "organization": "Other",
        })
        self.assertEqual(create.status_code, 403)
        status = self.client.patch("/incidents/incident-test-001/status", headers=self.hospital_headers, json={"status": "Verified"})
        self.assertEqual(status.status_code, 403)

    def test_hospital_only_sees_verified_incidents_assigned_to_its_organization(self):
        incident_id = self.create_incident(status="Verified")
        unassigned = self.client.get("/incidents", headers=self.hospital_headers).json()["incidents"]
        self.assertEqual(unassigned, [])
        assignment = self.client.post(
            f"/incidents/{incident_id}/assign?organization=Demo+General+Hospital",
            headers=self.authority_headers,
        )
        self.assertEqual(assignment.status_code, 200, assignment.text)
        visible = self.client.get("/incidents", headers=self.hospital_headers).json()["incidents"]
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["incident_id"], incident_id)

    def test_simulated_dispatch_is_persisted_and_role_protected(self):
        incident_id = self.create_incident()
        payload = {"incident_id": incident_id, "destination": "DEMO-CONTROL-ROOM", "channel": "dashboard-demo"}
        response = self.client.post("/dispatch/simulate", headers=self.authority_headers, json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["result"], "SIMULATED")
        self.assertIn("No emergency service was contacted", response.json()["message"])
        records = self.client.get("/dispatch/notifications", headers=self.authority_headers).json()["notifications"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["incident_id"], incident_id)
        forbidden = self.client.post("/dispatch/simulate", headers=self.hospital_headers, json=payload)
        self.assertEqual(forbidden.status_code, 403)


if __name__ == "__main__":
    unittest.main()
