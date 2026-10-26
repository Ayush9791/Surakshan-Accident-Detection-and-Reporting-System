"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, AlertTriangle, BellRing, Camera, CheckCircle2, ChevronRight, CircleUserRound, ClipboardList, Hospital, LogOut, MapPin, RefreshCw, ShieldCheck, Siren, UserPlus, Video, XCircle } from "lucide-react";

const API = (process.env.NEXT_PUBLIC_SURAKSHAN_API_URL || "http://127.0.0.1:5000").replace(/\/$/, "");
const STATUSES = ["New", "Under Review", "Verified", "Dismissed", "Resolved"];

export default function Dashboard() {
  const [token, setToken] = useState("");
  const [user, setUser] = useState(null);
  const [login, setLogin] = useState({ username: "admin", password: "Admin@12345" });
  const [loginError, setLoginError] = useState("");
  const [incidents, setIncidents] = useState([]);
  const [dispatches, setDispatches] = useState([]);
  const [users, setUsers] = useState([]);
  const [audit, setAudit] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [newUser, setNewUser] = useState({ username: "", password: "", role: "hospital", organization: "" });
  const [assignOrg, setAssignOrg] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [detectorRunning, setDetectorRunning] = useState(false);
  const [changePassword, setChangePassword] = useState({ current_password: "", new_password: "" });
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    const savedToken = window.localStorage.getItem("surakshan_token");
    if (savedToken) {
      setToken(savedToken);
      fetch(`${API}/auth/me`, { headers: { Authorization: `Bearer ${savedToken}` } })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(setUser)
        .catch(() => { window.localStorage.removeItem("surakshan_token"); setToken(""); });
    }
  }, []);

  const authFetch = useCallback((path, options = {}) => fetch(`${API}${path}`, {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${token}`, ...(options.body ? { "Content-Type": "application/json" } : {}) },
  }), [token]);

  const loadData = useCallback(async () => {
    if (!token || !user) return;
    try {
      const response = await authFetch("/incidents?limit=200", { cache: "no-store" });
      if (response.status === 401) { window.localStorage.removeItem("surakshan_token"); setToken(""); setUser(null); return; }
      if (!response.ok) throw new Error("Could not load incidents");
      setIncidents((await response.json()).incidents || []);
      setApiOnline(true);
      if (user.role !== "hospital") {
        const detectorResponse = await authFetch("/detector/status", { cache: "no-store" });
        if (detectorResponse.ok) setDetectorRunning((await detectorResponse.json()).running);
      }
      if (user.role !== "hospital") {
        const dispatchResponse = await authFetch("/dispatch/notifications?limit=50", { cache: "no-store" });
        if (dispatchResponse.ok) setDispatches((await dispatchResponse.json()).notifications || []);
      }
      if (user.role === "admin") {
        const [userResponse, auditResponse] = await Promise.all([authFetch("/admin/users"), authFetch("/admin/audit?limit=50")]);
        if (userResponse.ok) setUsers((await userResponse.json()).users || []);
        if (auditResponse.ok) setAudit((await auditResponse.json()).events || []);
      }
    } catch { setApiOnline(false); }
  }, [token, user, authFetch]);

  useEffect(() => {
    if (!token || !user) return;
    loadData();
    const timer = setInterval(loadData, 5000);
    return () => clearInterval(timer);
  }, [token, user, loadData]);

  useEffect(() => () => streamRef.current?.getTracks().forEach(track => track.stop()), []);

  const selected = incidents.find(item => item.incident_id === selectedId) || incidents[0] || null;
  const filtered = useMemo(() => incidents.filter(item => {
    const text = [item.incident_id, item.camera_id, item.location, item.event, item.status].join(" ").toLowerCase();
    return text.includes(query.toLowerCase()) && (filter === "All" || (filter === "Active" ? !["Resolved", "Dismissed"].includes(item.status) : item.status === filter));
  }), [incidents, query, filter]);
  const activeCount = incidents.filter(item => !["Resolved", "Dismissed"].includes(item.status)).length;
  const verifiedCount = incidents.filter(item => item.status === "Verified").length;

  function logout() {
    window.localStorage.removeItem("surakshan_token");
    setToken(""); setUser(null); setIncidents([]); setDispatches([]); setUsers([]); setAudit([]);
  }

  async function signIn(event) {
    event.preventDefault(); setLoginError(""); setBusy(true);
    try {
      const response = await fetch(`${API}/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(login) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Sign-in failed");
      window.localStorage.setItem("surakshan_token", payload.access_token);
      setToken(payload.access_token); setUser(payload.user); setNotice(`Signed in as ${payload.user.role}.`);
    } catch (error) { setLoginError(error.message); }
    finally { setBusy(false); }
  }

  async function createTestIncident() {
    setBusy(true);
    try {
      const response = await fetch(`${API}/alert`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        camera_id: "LOCAL-WEBCAM-01", location: "College demonstration site", event: "SIMULATED_POTENTIAL_ACCIDENT",
        incident_id: `test-${crypto.randomUUID()}`, confidence: 0.93,
      }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not create incident");
      setNotice("Test incident recorded. It is labelled as simulated; no emergency service was contacted.");
      await loadData();
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }

  async function statusAction(status) {
    if (!selected) return;
    setBusy(true);
    try {
      const response = await authFetch(`/incidents/${encodeURIComponent(selected.incident_id)}/status`, { method: "PATCH", body: JSON.stringify({ status }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Status update failed");
      setNotice(`Incident updated to ${status}.`); await loadData();
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }

  async function dispatchAction() {
    if (!selected) return;
    setBusy(true);
    try {
      const response = await authFetch("/dispatch/simulate", { method: "POST", body: JSON.stringify({ incident_id: selected.incident_id, destination: assignOrg || "DEMO-CONTROL-ROOM", channel: "dashboard-demo" }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Dispatch simulation failed");
      setNotice("Simulated notification recorded. No real authority or hospital was contacted."); await loadData();
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }

  async function assignIncident() {
    if (!selected || !assignOrg) { setNotice("Enter the hospital organisation name created by an administrator."); return; }
    setBusy(true);
    try {
      const response = await authFetch(`/incidents/${encodeURIComponent(selected.incident_id)}/assign?organization=${encodeURIComponent(assignOrg)}`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Assignment failed");
      setNotice(`Incident assigned to ${payload.assigned_to}.`); await loadData();
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }

  async function createAccount(event) {
    event.preventDefault(); setBusy(true);
    try {
      const response = await authFetch("/admin/users", { method: "POST", body: JSON.stringify(newUser) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Account creation failed");
      setNotice(`Created ${payload.role} account: ${payload.username}`);
      setNewUser({ username: "", password: "", role: "hospital", organization: "" }); await loadData();
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }

  async function changeMyPassword(event) {
    event.preventDefault(); setBusy(true);
    try {
      const response = await authFetch("/auth/change-password", { method: "POST", body: JSON.stringify(changePassword) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Password update failed");
      setNotice("Password updated."); setChangePassword({ current_password: "", new_password: "" });
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }


  async function detectorAction(action) {
    setBusy(true);
    try {
      const response = await authFetch(`/detector/${action}`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Detector action failed");
      setDetectorRunning(Boolean(payload.running));
      setNotice(payload.message || (payload.running ? "Detector started." : "Detector stopped."));
    } catch (error) { setNotice(error.message); }
    finally { setBusy(false); }
  }

  async function toggleCamera() {
    setCameraError("");
    if (cameraOn) { streamRef.current?.getTracks().forEach(track => track.stop()); streamRef.current = null; setCameraOn(false); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch (error) { setCameraError(error.name === "NotAllowedError" ? "Camera permission denied. Allow access in browser settings." : "No accessible webcam found."); }
  }

  if (!user) return <main className="login-shell">
    <form className="login-card" onSubmit={signIn}>
      <div className="brand-mark login-mark"><ShieldCheck size={25} /></div>
      <div className="eyebrow">TRAFFIC INCIDENT RESPONSE PLATFORM</div>
      <h1>Surakshan</h1>
      <p>Secure operator sign-in</p>
      <label>Username<input required autoComplete="username" value={login.username} onChange={e => setLogin({ ...login, username: e.target.value })} /></label>
      <label>Password<input required type="password" autoComplete="current-password" value={login.password} onChange={e => setLogin({ ...login, password: e.target.value })} /></label>
      {loginError && <div className="error-box">{loginError}</div>}
      <button className="button button-primary full-width" disabled={busy}>{busy ? "Signing in…" : "Sign in securely"}</button>
      <div className="login-help"><strong>Local demo accounts</strong><br />Admin: admin / Admin@12345<br />Authority: authority / Authority@12345<br />Hospital: hospital / Hospital@12345<br /><small>Change demo passwords before any shared deployment.</small></div>
      <small className="login-foot">Demo environment · Dispatch is simulated</small>
    </form>
  </main>;

  return <main className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview"><span className="brand-mark"><ShieldCheck size={22}/></span><span><strong>SURAKSHAN</strong><small>INCIDENT OPERATIONS</small></span></a>
      <div className="sidebar-label">WORKSPACE</div>
      <nav className="side-nav">
        <a className="nav-item active" href="#overview"><Activity size={17}/> Overview</a>
        <a className="nav-item" href="#incidents"><AlertTriangle size={17}/> Incident queue <span className="nav-count">{activeCount}</span></a>
        <a className="nav-item" href="#camera"><Video size={17}/> Local camera</a>
        {user.role === "admin" && <a className="nav-item" href="#accounts"><UserPlus size={17}/> Accounts</a>}
        {user.role !== "hospital" && <a className="nav-item" href="#dispatch"><BellRing size={17}/> Dispatch log</a>}
      </nav>
      <div className="sidebar-spacer"/>
      <div className="environment-card"><div className="env-title"><span className={apiOnline ? "live-dot" : "offline-dot"}/> API {apiOnline ? "connected" : "unavailable"}</div><p>Refreshes incident records every five seconds.</p><button className="button button-quiet full-width" onClick={loadData}><RefreshCw size={14}/> Refresh data</button></div>
      <div className="sidebar-foot"><CircleUserRound size={15}/><span>{user.username} · {user.role}</span><button className="icon-button" aria-label="Sign out" onClick={logout}><LogOut size={15}/></button></div>
    </aside>
    <section className="main-area" id="overview">
      <header className="topbar"><div className="breadcrumb"><span>Operations</span><span>/</span><strong>{user.role === "hospital" ? "Hospital desk" : user.role === "authority" ? "Authority control room" : "Administration"}</strong></div><div className="topbar-right"><span className="system-state"><span className={apiOnline ? "live-dot" : "offline-dot"}/>{apiOnline ? "API ONLINE" : "API OFFLINE"}</span><span className="avatar">{user.username.slice(0,2).toUpperCase()}</span></div></header>
      <div className="page-content">
        <div className="page-heading"><div><div className="eyebrow">{user.organization.toUpperCase()}</div><h1>{user.role === "hospital" ? "Hospital incident desk" : user.role === "authority" ? "Authority operations" : "Operations overview"}</h1><p>{user.role === "hospital" ? "Review incidents verified by the control room and assigned to your organisation." : "Monitor potential accidents, review incident status and coordinate simulated responses."}</p></div><div className="heading-actions"><span className="updated-label"><span className={apiOnline ? "live-dot" : "offline-dot"}/> {apiOnline ? "Connected" : "Disconnected"}</span>{user.role !== "hospital" && <button className="button button-primary" onClick={createTestIncident} disabled={busy}><AlertTriangle size={15}/> Create test incident</button>}</div></div>
        {notice && <div className="notice" role="status"><span>{notice}</span><button onClick={() => setNotice("")} aria-label="Dismiss"><XCircle size={16}/></button></div>}
        <div className="metrics-grid">
          <Metric label="Incidents visible" value={incidents.length} icon={<ClipboardList size={18}/>} tone="muted"/>
          <Metric label="Active incidents" value={activeCount} icon={<AlertTriangle size={18}/>} tone="red"/>
          <Metric label="Verified" value={verifiedCount} icon={<CheckCircle2 size={18}/>} tone="green"/>
          <Metric label="Account role" value={user.role.toUpperCase()} icon={user.role === "hospital" ? <Hospital size={18}/> : <ShieldCheck size={18}/>} tone="amber"/>
        </div>
        <div className="content-grid">
          <section className="panel incidents-panel" id="incidents">
            <div className="panel-heading"><div><h2>Incident queue</h2><p>{user.role === "hospital" ? "Verified or resolved cases available to your hospital" : "Recent detector reports"}</p></div><span className="panel-pill">{filtered.length} RECORDS</span></div>
            <div className="queue-toolbar"><label className="search-box"><Activity size={15}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search camera, location, ID…"/></label><label className="select-wrap"><select value={filter} onChange={e => setFilter(e.target.value)}><option>All</option><option>Active</option>{STATUSES.map(status => <option key={status}>{status}</option>)}</select></label></div>
            <div className="incident-list">{filtered.length === 0 && <div className="empty-state"><AlertTriangle size={24}/><strong>No incidents to display</strong><span>New detector alerts appear here when the API is connected.</span></div>}
              {filtered.map(item => <button className={selected?.incident_id === item.incident_id ? "incident-row selected" : "incident-row"} key={item.incident_id} onClick={() => setSelectedId(item.incident_id)}><span className="incident-indicator"><AlertTriangle size={17}/></span><span className="incident-copy"><strong>{item.location || item.camera_id}</strong><small>{item.camera_id} · {new Date(item.timestamp).toLocaleString()}</small><span className={`status-badge ${statusClass(item.status)}`}>{item.status}</span></span><span className="incident-confidence">{typeof item.confidence === "number" ? Math.round(item.confidence * 100) + "%" : "—"}<small>score</small></span></button>)}
            </div>
          </section>
          <section className="panel detail-panel"><div className="panel-heading"><div><h2>Incident details</h2><p>Review and response actions</p></div><span className="detail-id">{selected ? selected.incident_id.slice(0,8).toUpperCase() : "—"}</span></div>
            {selected ? <div className="detail-content"><div className="detail-location"><span className="location-icon"><MapPin size={17}/></span><div><strong>{selected.location || "Location not configured"}</strong><small>{selected.camera_id} · {new Date(selected.timestamp).toLocaleString()}</small></div></div>
              <div className="detail-stats"><div><span>Model confidence</span><strong>{typeof selected.confidence === "number" ? Math.round(selected.confidence*100)+"%" : "Not supplied"}</strong><div className="confidence-track"><span style={{width: typeof selected.confidence === "number" ? `${selected.confidence*100}%` : "0%"}}/></div></div><div><span>Incident status</span><strong><span className={`status-badge ${statusClass(selected.status)}`}>{selected.status}</span></strong></div></div>
              {user.role !== "hospital" && <div className="detail-actions"><button className="button button-primary" disabled={busy || selected.status === "Verified"} onClick={() => statusAction("Verified")}><CheckCircle2 size={15}/> Verify</button><button className="button button-secondary" disabled={busy} onClick={() => statusAction("Under Review")}>Review</button><button className="button button-quiet" disabled={busy} onClick={() => statusAction("Dismissed")}>Dismiss</button><button className="button button-quiet" disabled={busy} onClick={() => statusAction("Resolved")}>Resolve</button></div>}
              {user.role !== "hospital" && <><label className="field-label">Hospital organisation<input value={assignOrg} onChange={e => setAssignOrg(e.target.value)} placeholder="Exact organisation name from accounts"/></label><div className="detail-actions"><button className="button button-secondary" disabled={busy} onClick={assignIncident}><Hospital size={15}/> Assign to hospital</button><button className="button button-quiet" disabled={busy} onClick={dispatchAction}><BellRing size={15}/> Simulate dispatch</button></div></>}
              {user.role === "hospital" && <div className="inline-note">This view is restricted to verified/resolved incidents available to your organisation. Dispatch actions are controlled by the authority.</div>}
              <div className="detail-footnote"><ShieldCheck size={14}/> AI detection is a potential incident, not a verified emergency.</div>
            </div> : <div className="empty-state"><AlertTriangle size={24}/><strong>Select an incident</strong><span>Incident detail and permitted actions will appear here.</span></div>}
          </section>
        </div>
        <div className="lower-grid">
          <section className="panel camera-panel" id="camera"><div className="panel-heading"><div><h2>Local webcam preview</h2><p>Browser preview only; model inference remains in the Python process</p></div><span className="stream-pill"><span className={cameraOn ? "live-dot" : "offline-dot"}/>{cameraOn ? "PREVIEW LIVE" : "STOPPED"}</span></div>
            <div className="camera-preview"><video ref={videoRef} autoPlay playsInline muted className={cameraOn ? "webcam-video visible" : "webcam-video"}/>{!cameraOn && <div className="camera-placeholder"><Camera size={28}/><strong>Camera preview stopped</strong><span>Allow browser camera access to start.</span></div>}</div>
            {cameraError && <p className="error-text">{cameraError}</p>}<div className="camera-controls"><div><span className={detectorRunning ? "live-dot" : "offline-dot"}/>{detectorRunning ? "Python detector running on API host" : "Python detector stopped"}</div><div className="camera-control-buttons"><button className={cameraOn ? "button button-secondary" : "button button-quiet"} onClick={toggleCamera}>{cameraOn ? "Stop browser preview" : "Preview webcam"}</button>{user.role !== "hospital" && <button className={detectorRunning ? "button button-secondary" : "button button-primary"} disabled={busy || cameraOn} onClick={() => detectorAction(detectorRunning ? "stop" : "start")}>{detectorRunning ? "Stop AI detector" : "Start AI detector"}</button>}</div></div>{cameraOn && detectorRunning && <p className="error-text">Stop one camera client before starting the other; both may compete for the same webcam.</p>}<p className="camera-disclaimer">The browser preview is not AI. Start AI detector to launch the existing Python inference process on the API host; it opens an OpenCV window on that machine and writes output to logs/detector.log. This works when the API runs on the computer connected to the webcam, not on a cloud host without a camera.</p>
          </section>
          {user.role !== "hospital" && <section className="panel dispatch-panel" id="dispatch"><div className="panel-heading"><div><h2>Simulated dispatch log</h2><p>No external notifications are sent</p></div><span className="panel-pill">{dispatches.length} EVENTS</span></div><div className="dispatch-list">{dispatches.length === 0 ? <div className="empty-state compact"><BellRing size={22}/><strong>No simulated dispatches</strong><span>Select an incident and use Simulate dispatch.</span></div> : dispatches.slice(0,8).map(item => <div className="dispatch-row" key={item.notification_id}><span className="dispatch-icon"><BellRing size={15}/></span><div><strong>{item.destination}</strong><small>{item.incident_id} · {new Date(item.timestamp).toLocaleString()}</small></div><span className="simulated-tag">SIMULATED</span></div>)}</div><div className="dispatch-note"><ShieldCheck size={15}/>Simulated records only; no authority has been contacted.</div></section>}
        </div>
        {user.role === "admin" && <div className="admin-grid" id="accounts">
          <section className="panel"><div className="panel-heading"><div><h2>Account management</h2><p>Create authority and hospital logins</p></div><UserPlus size={18}/></div><form className="admin-form" onSubmit={createAccount}><label>Username<input required minLength={3} value={newUser.username} onChange={e => setNewUser({...newUser, username:e.target.value})} placeholder="e.g. city_control"/></label><label>Temporary password<input required type="password" minLength={10} value={newUser.password} onChange={e => setNewUser({...newUser,password:e.target.value})} placeholder="At least 10 characters"/></label><label>Role<select value={newUser.role} onChange={e => setNewUser({...newUser,role:e.target.value})}><option value="authority">Traffic authority</option><option value="hospital">Hospital</option></select></label><label>Organisation<input required value={newUser.organization} onChange={e => setNewUser({...newUser,organization:e.target.value})} placeholder="Official organisation name"/></label><button className="button button-primary" disabled={busy}><UserPlus size={15}/> Create account</button></form><div className="user-list">{users.map(account => <div className="user-row" key={account.username}><CircleUserRound size={16}/><span><strong>{account.username}</strong><small>{account.organization}</small></span><span className="status-badge status-review">{account.role}</span></div>)}</div></section>
          <section className="panel"><div className="panel-heading"><div><h2>Audit trail</h2><p>Recent sign-ins and operator actions</p></div><ClipboardList size={18}/></div><div className="dispatch-list">{audit.length === 0 ? <div className="empty-state compact"><strong>No audit events</strong></div> : audit.map(item => <div className="dispatch-row" key={item.audit_id}><div><strong>{item.action}</strong><small>{item.username} · {new Date(item.timestamp).toLocaleString()}</small><small>{item.details}</small></div></div>)}</div></section>
        </div>}
        <section className="panel password-panel"><div className="panel-heading"><div><h2>Change your password</h2><p>Use a unique password for each account</p></div></div><form className="password-form" onSubmit={changeMyPassword}><label>Current password<input type="password" required value={changePassword.current_password} onChange={e => setChangePassword({...changePassword,current_password:e.target.value})}/></label><label>New password<input type="password" required minLength={10} value={changePassword.new_password} onChange={e => setChangePassword({...changePassword,new_password:e.target.value})}/></label><button className="button button-secondary" disabled={busy}>Update password</button></form></section>
        <footer className="page-footer"><span>Surakshan academic project · Operator review required.</span><span><Siren size={13}/> Dispatch simulation only</span></footer>
      </div>
    </section>
  </main>;
}

function statusClass(status) {
  return ({New:"status-new","Under Review":"status-review",Verified:"status-verified",Dismissed:"status-dismissed",Resolved:"status-resolved"})[status] || "status-new";
}
function Metric({label,value,icon,tone}) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><span className={`metric-icon tone-${tone}`}>{icon}</span></div><div className="metric-value metric-role">{value}</div><div className="metric-bottom"><span className="metric-trend">Updated from incident API</span></div></div>;
}
