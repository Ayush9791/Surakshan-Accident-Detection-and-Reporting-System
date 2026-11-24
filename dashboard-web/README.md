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

## Deploy dashboard to Vercel

1. Import this GitHub repository into Vercel.
2. Set **Root Directory** to `dashboard-web`.
3. Keep the Framework Preset as **Next.js** and use the default build command (`npm run build`).
4. Add this environment variable in **Project Settings → Environment Variables**:

   `NEXT_PUBLIC_SURAKSHAN_API_URL=https://YOUR-PUBLIC-BACKEND-URL`

   Replace the example value with the public HTTPS URL of the deployed FastAPI service. Do not use `127.0.0.1` or `localhost` for a cloud deployment.
5. Redeploy after adding or changing the environment variable.

The Python FastAPI service is separate from the Vercel frontend and must be hosted on a service that supports the project's Python dependencies. Set `SURAKSHAN_CORS_ORIGINS` on the backend to the exact Vercel dashboard origin, for example `https://your-project.vercel.app`. Configure `SURAKSHAN_AUTH_SECRET` and strong demo-account passwords as backend environment variables before sharing the deployment.

Vercel hosts the dashboard only. Continuous CCTV capture and OpenCV inference require a persistent backend machine with access to the camera; a cloud dashboard cannot access a webcam attached to your laptop. The default SQLite database is local to the backend filesystem and may not persist on hosts with ephemeral storage, so use persistent storage or a managed database for data that must survive restarts.

## Workflows

- Admin: all incidents, account creation, audit trail, detector controls, status updates, hospital assignment and simulated dispatch.
- Traffic authority: incident review/status updates, assignment, detector controls and simulated dispatch.
- Hospital: only verified/resolved incidents assigned to its organisation.
- Incident records, assignments, users and simulated dispatches persist in SQLite.
- **Start AI detector** launches `python -m live_system.advanced_live` on the same machine as the Python API, using the selected input.
- Camera inputs in the dashboard:
  - **PC Webcam:** opens camera index 0 on the API host.
  - **Bundled accident test video:** runs `Accident_Detection.mp4` through the existing model.
  - **PC Webcam:** opens camera index 0 on the API host.
- The browser video player is only a preview. The Python API separately reads the bundled clip for inference.
- Run the Python API locally to use your laptop GPU. A Vercel frontend connected to Render runs inference on Render, not on your PC. Detector logs are written to `logs/detector.log`.
- Demo incident creation uses an authenticated operator endpoint. Hospital assignment and simulated dispatch are available only after an incident is verified.
- Simulated dispatch only records a local notification. It never contacts real authorities.

## Configuration

`.env.backend.example` documents optional variables; the API does not automatically load this file. Set values in your PowerShell session before starting Uvicorn if you need custom secrets. Important variables:

- `SURAKSHAN_AUTH_SECRET`: session-signing secret.
- `SURAKSHAN_ADMIN_PASSWORD`, `SURAKSHAN_AUTHORITY_PASSWORD`, `SURAKSHAN_HOSPITAL_PASSWORD`: seeded passwords on first database creation.
- `SURAKSHAN_DETECTOR_TOKEN`: optional shared secret required by `POST /alert` when configured; set the same value for the detector process.
- `SURAKSHAN_CORS_ORIGINS`: allowed dashboard origins.

Keep the API bound to localhost for the college demo. A public/cloud deployment needs HTTPS, a firewall, secure secret configuration, and further security hardening.

## Checks

```powershell
npm install
npm run lint
npm run build
```

## Bundled accident video: local AI test

The repository root contains `Accident_Detection.mp4`. Sign in with the local admin account, open **AI video test**, leave **Bundled accident test video** selected, and click **Start AI detector**. The Python API reads the clip and passes frames through the existing Keras model. The browser video player is only a preview; inference runs in the Python process.

The local dashboard must call the local API. Ensure `dashboard-web/.env.local` contains:

```env
NEXT_PUBLIC_SURAKSHAN_API_URL=http://127.0.0.1:5000
```

After editing the environment file, restart `npm run dev`.

Check TensorFlow GPU visibility in the same Python environment used to launch Uvicorn:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow is not using a GPU in that environment. Enabling CUDA in Windows or NVIDIA Control Panel alone does not establish TensorFlow GPU support. A Vercel dashboard connected to Render runs inference on Render, not on your laptop GPU.
