"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertOctagon, ArrowDownRight, ArrowUpRight, Bell, Camera,
  Check, ChevronDown, Clock3, Crosshair, ExternalLink, Filter, MapPin,
  Play, Radio, RefreshCw, Search, ShieldCheck, Siren, Video, Wifi,
  X, CircleHelp, Menu
} from "lucide-react";

const API = (process.env.NEXT_PUBLIC_SURAKSHAN_API_URL || "http://127.0.0.1:5000").replace(/\/$/, "");
const DEMO_INCIDENTS = [
  { incident_id: "demo-incident-001", timestamp: new Date(Date.now() - 1000 * 60 * 4).toISOString(), camera_id: "CAM-014", location: "Ring Road · Dhaula Kuan", event: "POTENTIAL_ACCIDENT", confidence: 0.94, status: "New", demo: true },
  { incident_id: "demo-incident-002", timestamp: new Date(Date.now() - 1000 * 60 * 18).toISOString(), camera_id: "CAM-008", location: "NH-48 · Mahipalpur", event: "POTENTIAL_ACCIDENT", confidence: 0.87, status: "Under Review", demo: true },
  { incident_id: "demo-incident-003", timestamp: new Date(Date.now() - 1000 * 60 * 46).toISOString(), camera_id: "CAM-021", location: "Outer Ring Road · Munirka", event: "POTENTIAL_ACCIDENT", confidence: 0.78, status: "Resolved", demo: true },
];
const STATUS = {
  "New": "status-new",
  "Under Review": "status-review",
  "Verified": "status-verified",
  "Dismissed": "status-dismissed",
  "Resolved": "status-resolved",
};

function timeAgo(value) {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return "Just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
function shortId(id) { return String(id || "").slice(0, 8).toUpperCase(); }
function confidence(value) { return typeof value === "number" ? `${Math.round(value * 100)}%` : "—"; }

export default function Dashboard() {
  const [incidents, setIncidents] = useState(DEMO_INCIDENTS);
  const [selected, setSelected] = useState(DEMO_INCIDENTS[0]);
  const [apiState, setApiState] = useState("checking");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All incidents");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [dispatches, setDispatches] = useState([]);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const loadData = useCallback(async () => {
    try {
      const [healthResponse, incidentsResponse, dispatchResponse] = await Promise.all([
        fetch(`${API}/health`, { cache: "no-store" }),
        fetch(`${API}/incidents?limit=100`, { cache: "no-store" }),
        fetch(`${API}/dispatch/notifications`, { cache: "no-store" }),
      ]);
      if (!healthResponse.ok || !incidentsResponse.ok) throw new Error("API unavailable");
      const payload = await incidentsResponse.json();
      setIncidents(payload.incidents?.length ? payload.incidents : []);
      if (dispatchResponse.ok) setDispatches((await dispatchResponse.json()).notifications || []);
      setApiState("online");
    } catch {
      setApiState("offline");
    }
  }, []);

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 5000);
    return () => clearInterval(timer);
  }, [loadData]);

  useEffect(() => {
    if (!selected) return;
    const fresh = incidents.find((item) => item.incident_id === selected.incident_id);
    if (fresh) setSelected(fresh);
    else if (incidents.length) setSelected(incidents[0]);
  }, [incidents, selected]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  const filtered = useMemo(() => incidents.filter((item) => {
    const matchesQuery = `${item.incident_id} ${item.camera_id} ${item.location} ${item.event}`.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === "All incidents" || (filter === "Active" ? !["Resolved", "Dismissed"].includes(item.status) : item.status === filter);
    return matchesQuery && matchesFilter;
  }), [incidents, query, filter]);

  const activeCount = incidents.filter((item) => !["Resolved", "Dismissed"].includes(item.status)).length;
  const verifiedCount = incidents.filter((item) => item.status === "Verified").length;
  const highConfidence = incidents.filter((item) => (item.confidence || 0) >= 0.9 && !["Dismissed", "Resolved"].includes(item.status)).length;

  async function updateStatus(incident, status) {
    if (!incident || incident.demo) {
      setNotice("Demo-only example. Connect the API and create a test alert to change a persisted incident.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API}/incidents/${encodeURIComponent(incident.incident_id)}/status`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
      });
      if (!response.ok) throw new Error((await response.text()) || "Status update failed");
      setNotice(`Incident ${shortId(incident.incident_id)} marked ${status.toLowerCase()}.`);
      await loadData();
    } catch (error) { setNotice(`Could not update incident: ${error.message}`); }
    finally { setBusy(false); }
  }

  async function simulateAlert() {
    setBusy(true);
    try {
      const id = `demo-${crypto.randomUUID()}`;
      const response = await fetch(`${API}/alert`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: "LOCAL-WEBCAM-01", location: "Local demo · Operator workstation", event: "SIMULATED_POTENTIAL_ACCIDENT", incident_id: id, confidence: 0.93 }),
      });
      if (!response.ok) throw new Error((await response.text()) || "Alert could not be created");
      setNotice("Simulated incident created. No emergency service has been contacted.");
      await loadData();
    } catch (error) { setNotice(`Cannot create test incident: ${error.message}. Start the FastAPI backend first.`); }
    finally { setBusy(false); }
  }

  async function simulateDispatch(incident) {
    if (!incident || incident.demo) {
      setNotice("Dispatch simulation requires a persisted incident. Create a test incident first.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API}/dispatch/simulate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incident_id: incident.incident_id, destination: "DEMO-CONTROL-ROOM", channel: "dashboard-demo" }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Simulation failed");
      setNotice("Simulated notification recorded. This is not a real dispatch.");
      await loadData();
    } catch (error) { setNotice(`Dispatch simulation failed: ${error.message}`); }
    finally { setBusy(false); }
  }

  async function startCamera() {
    setCameraError("");
    if (cameraOn) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraOn(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch (error) {
      setCameraError(error.name === "NotAllowedError" ? "Camera permission was denied. Allow access in your browser settings." : "No accessible webcam found. Check that another app is not using it.");
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Surakshan home">
          <span className="brand-mark"><Crosshair size={22} strokeWidth={2.2} /></span>
          <span><strong>SURAKSHAN</strong><small>TRAFFIC INTELLIGENCE</small></span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <nav className="side-nav">
          <a className="nav-item active" href="#overview"><Activity size={17} /> Overview <span className="nav-dot" /></a>
          <a className="nav-item" href="#incidents"><AlertOctagon size={17} /> Incidents <span className="nav-count">{activeCount}</span></a>
          <a className="nav-item" href="#cameras"><Video size={17} /> Camera feeds</a>
          <a className="nav-item" href="#dispatch"><Siren size={17} /> Dispatch log</a>
        </nav>
        <div className="sidebar-spacer" />
        <div className="environment-card">
          <div className="env-title"><span className={apiState === "online" ? "live-dot" : "offline-dot"} /> API {apiState === "online" ? "connected" : apiState === "checking" ? "checking…" : "disconnected"}</div>
          <p>{apiState === "online" ? "Incident updates every 5 seconds." : "Start FastAPI to load real incidents."}</p>
          <button className="button button-quiet full-width" onClick={loadData}><RefreshCw size={14} /> Refresh data</button>
        </div>
        <div className="sidebar-foot"><ShieldCheck size={15} /> Operator workspace <span>v0.1</span></div>
      </aside>

      <section className="main-area" id="overview">
        <header className="topbar">
          <div className="breadcrumb"><span>Operations</span><span className="crumb-sep">/</span><strong>Overview</strong></div>
          <div className="topbar-right">
            <span className="system-state"><span className={apiState === "online" ? "live-dot" : "offline-dot"} /> {apiState === "online" ? "SYSTEM ONLINE" : "API OFFLINE"}</span>
            <button className="icon-button" aria-label="Refresh" onClick={loadData}><RefreshCw size={17} /></button>
            <div className="avatar">OP</div>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div><div className="eyebrow">TRAFFIC CONTROL · DEMO ENVIRONMENT</div><h1>Operations overview</h1><p>Monitor potential incidents, review evidence and coordinate simulated responses.</p></div>
            <div className="heading-actions"><span className="updated-label"><span className="live-dot" /> Refreshes every 5s</span><button className="button button-primary" onClick={simulateAlert} disabled={busy || apiState !== "online"}><AlertOctagon size={16} /> Create test incident</button></div>
          </div>

          {notice && <div className="notice" role="status"><span>{notice}</span><button onClick={() => setNotice("")} aria-label="Dismiss notice"><X size={16} /></button></div>}

          <div className="metrics-grid">
            <Metric label="Active incidents" value={activeCount} delta="Requires review" icon={<AlertOctagon size={18} />} tone="red" />
            <Metric label="High-confidence alerts" value={highConfidence} delta="≥ 90% model score" icon={<Crosshair size={18} />} tone="amber" />
            <Metric label="Verified incidents" value={verifiedCount} delta="Operator-confirmed" icon={<ShieldCheck size={18} />} tone="green" />
            <Metric label="API status" value={apiState === "online" ? "Online" : "Offline"} delta={apiState === "online" ? "Backend responding" : "Waiting for backend"} icon={<Activity size={18} />} tone={apiState === "online" ? "green" : "muted"} />
          </div>

          <div className="content-grid">
            <section className="panel incidents-panel" id="incidents">
              <div className="panel-heading"><div><h2>Incident queue</h2><p>Latest reports from connected detectors</p></div><span className="panel-pill">{filtered.length} RECORDS</span></div>
              <div className="queue-toolbar">
                <label className="search-box"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search camera, location, ID…" /></label>
                <label className="select-wrap"><Filter size={14} /><select value={filter} onChange={(event) => setFilter(event.target.value)}><option>All incidents</option><option>Active</option><option>New</option><option>Under Review</option><option>Verified</option><option>Dismissed</option><option>Resolved</option></select><ChevronDown size={13} /></label>
              </div>
              <div className="incident-list">
                {filtered.length === 0 && <div className="empty-state"><AlertOctagon size={24} /><strong>No incidents found</strong><span>Start the API and create a test incident, or adjust your filters.</span></div>}
                {filtered.map((incident) => <button className={selected?.incident_id === incident.incident_id ? "incident-row selected" : "incident-row"} key={incident.incident_id} onClick={() => setSelected(incident)}>
                  <span className="incident-indicator"><AlertOctagon size={17} /></span>
                  <span className="incident-copy"><strong>{incident.location || incident.camera_id}</strong><small>{incident.camera_id} · {timeAgo(incident.timestamp)}</small><span className={`status-badge ${STATUS[incident.status] || "status-new"}`}>{incident.status || "New"}</span></span>
                  <span className="incident-confidence">{confidence(incident.confidence)}<small>confidence</small></span>
                </button>)}
              </div>
              <div className="panel-footer"><span>Live data from Surakshan API</span><button className="text-button" onClick={loadData}>Refresh queue <ArrowUpRight size={14} /></button></div>
            </section>

            <section className="panel detail-panel">
              <div className="panel-heading"><div><h2>Incident detail</h2><p>Evidence and operator actions</p></div><span className="detail-id">#{shortId(selected?.incident_id || "—")}</span></div>
              {selected ? <div className="detail-content">
                <div className="evidence-frame">
                  <div className="evidence-top"><span><span className="record-dot" /> EVENT EVIDENCE</span><span>SIMULATED VIEW</span></div>
                  <div className="evidence-art">
                    <div className="road-lines" /><div className="road-car car-one" /><div className="road-car car-two" /><div className="detection-box box-one"><span>TRACK 08</span></div><div className="detection-box box-two"><span>TRACK 12</span></div>
                    <div className="evidence-watermark"><Crosshair size={14} /> SURAKSHAN VISION</div>
                  </div>
                  <div className="evidence-bottom"><span><Radio size={13} /> {selected.camera_id}</span><span>{new Date(selected.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></div>
                </div>
                <div className="detail-location"><span className="location-icon"><MapPin size={17} /></span><div><strong>{selected.location || "Location not configured"}</strong><small>{selected.camera_id} · {new Date(selected.timestamp).toLocaleString()}</small></div></div>
                <div className="detail-stats"><div><span>AI confidence</span><strong>{confidence(selected.confidence)}</strong><div className="confidence-track"><span style={{ width: confidence(selected.confidence) }} /></div></div><div><span>Incident status</span><strong><span className={`status-badge ${STATUS[selected.status] || "status-new"}`}>{selected.status || "New"}</span></strong></div></div>
                <div className="detail-actions"><button className="button button-primary" disabled={busy || selected.demo || selected.status === "Verified"} onClick={() => updateStatus(selected, "Verified")}><Check size={15} /> Verify incident</button><button className="button button-secondary" disabled={busy || selected.demo} onClick={() => simulateDispatch(selected)}><Bell size={15} /> Simulate dispatch</button><button className="button button-quiet" disabled={busy || selected.demo} onClick={() => updateStatus(selected, "Dismissed")}><X size={15} /> Dismiss</button></div>
                {selected.demo && <div className="inline-note">Illustrative sample only. Create a persisted test incident to enable workflow actions.</div>}
                <div className="detail-footnote"><ShieldCheck size={14} /> AI output is a potential incident, not a verified emergency.</div>
              </div> : <div className="empty-state"><CircleHelp size={24} /><strong>Select an incident</strong><span>Choose a record from the queue to review details.</span></div>}
            </section>
          </div>

          <div className="lower-grid">
            <section className="panel camera-panel" id="cameras">
              <div className="panel-heading"><div><h2>Local camera preview</h2><p>Browser webcam · preview only, not yet connected to AI inference</p></div><span className={cameraOn ? "stream-pill stream-live" : "stream-pill"}><span className={cameraOn ? "live-dot" : "offline-dot"} /> {cameraOn ? "PREVIEW LIVE" : "NOT CONNECTED"}</span></div>
              <div className="camera-preview">
                <video ref={videoRef} autoPlay playsInline muted className={cameraOn ? "webcam-video visible" : "webcam-video"} />
                {!cameraOn && <div className="camera-placeholder"><Camera size={28} /><strong>Camera preview is stopped</strong><span>Allow browser access to preview your local webcam.</span></div>}
                {cameraOn && <div className="camera-overlay"><span><span className="record-dot" /> LOCAL PREVIEW</span><span>AI inference not connected</span></div>}
              </div>
              {cameraError && <p className="error-text">{cameraError}</p>}
              <div className="camera-controls"><div><span className={cameraOn ? "live-dot" : "offline-dot"} /> {cameraOn ? "Camera permission granted" : "No active browser stream"}</div><button className={cameraOn ? "button button-secondary" : "button button-primary"} onClick={startCamera}>{cameraOn ? <X size={15} /> : <Play size={15} />}{cameraOn ? "Stop preview" : "Start webcam preview"}</button></div>
              <p className="camera-disclaimer">This preview does not send frames to the server or run the accident model. Use the existing Python camera pipeline for inference.</p>
            </section>

            <section className="panel dispatch-panel" id="dispatch">
              <div className="panel-heading"><div><h2>Simulated dispatch log</h2><p>Test notifications only · no real authorities contacted</p></div><span className="panel-pill">{dispatches.length} EVENTS</span></div>
              <div className="dispatch-list">
                {dispatches.slice(0, 5).map((item) => <div className="dispatch-row" key={item.notification_id}><span className="dispatch-icon"><Bell size={15} /></span><div><strong>{item.destination || "DEMO-CONTROL-ROOM"}</strong><small>Incident #{shortId(item.incident_id)} · {timeAgo(item.timestamp)}</small></div><span className="simulated-tag">SIMULATED</span></div>)}
                {dispatches.length === 0 && <div className="empty-state compact"><Bell size={22} /><strong>No simulated notifications</strong><span>Create a test incident, then select “Simulate dispatch”.</span></div>}
              </div>
              <div className="dispatch-note"><ShieldCheck size={15} /><span>All outbound notifications are simulated and recorded locally. No emergency service is contacted.</span></div>
            </section>
          </div>

          <footer className="page-footer"><span>Surakshan prototype · Incident decisions require human review.</span><span><Wifi size={13} /> {apiState === "online" ? "Connected to API" : "Waiting for API"} <span className="footer-sep">·</span> Local preview build</span></footer>
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value, delta, icon, tone }) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><span className={`metric-icon tone-${tone}`}>{icon}</span></div><div className="metric-value">{value}</div><div className="metric-bottom"><span className="metric-trend"><ArrowDownRight size={13} /> {delta}</span></div></div>;
}
