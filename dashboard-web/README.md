# Surakshan Web Dashboard

Next.js/React role-based incident operations dashboard.

## Free local setup (Windows PowerShell)

Requirements: Node.js 20 LTS, Python environment and dependencies installed from the repository root.

Terminal 1, from repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn live_system.server:app --host 127.0.0.1 --port 5000
```

Terminal 2:

```powershell
cd dashboard-web
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000.

## Demo accounts

- Admin: `admin` / `Admin@12345`
- Traffic authority: `authority` / `Authority@12345`
- Hospital: `hospital` / `Hospital@12345`

These are local demo credentials. Change the passwords after signing in; do not use these defaults on a public server. Admins can create additional authority/hospital accounts.

## Workflows

- Admin: all incidents, account creation, audit trail, detector controls, status updates, hospital assignment and simulated dispatch.
- Traffic authority: incident review/status updates, assignment, detector controls and simulated dispatch.
- Hospital: only verified/resolved incidents assigned to its organisation.
- Incident records, assignments, users and simulated dispatches persist in SQLite.
- Browser webcam preview is only a preview. **Start AI detector** launches `python -m live_system.advanced_live` on the same host as the API, where it uses that machine's camera and opens an OpenCV window.
- Simulated dispatch only records a local notification. It never contacts real authorities.

## Configuration

The API reads `.env.backend.example` as documentation; copy its values into your shell environment if you need custom secrets. Important variables:

- `SURAKSHAN_AUTH_SECRET`: session-signing secret.
- `SURAKSHAN_ADMIN_PASSWORD`, `SURAKSHAN_AUTHORITY_PASSWORD`, `SURAKSHAN_HOSPITAL_PASSWORD`: seeded passwords on first database creation.
- `SURAKSHAN_DETECTOR_TOKEN`: optional shared secret required by `POST /alert` when configured; set the same value for the detector process.
- `SURAKSHAN_CORS_ORIGINS`: allowed dashboard origins.

Keep the API bound to localhost for the college demo. A public/cloud deployment needs HTTPS, a firewall, secure secret configuration, and further security hardening.

## Build

```powershell
npm install
npm run build
```
