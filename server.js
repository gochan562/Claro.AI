require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const helmet  = require('helmet');
const session = require('express-session');
const rateLimit = require('express-rate-limit');
const { ipKeyGenerator } = require('express-rate-limit');
const bcrypt  = require('bcrypt');
const fs      = require('fs');
const path    = require('path');
const { Readable } = require('stream');
const { execFile } = require('child_process');
const { promisify } = require('util');

const app  = express();
const PORT = process.env.PORT || 5000;
const execFileAsync = promisify(execFile);

// ── Firebase Admin (for verifying Firebase ID tokens) ───────────────────
if (process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON) {
  try {
    const credPath = '/tmp/firebase-service-account.json';
    fs.writeFileSync(credPath, process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON);
    process.env.GOOGLE_APPLICATION_CREDENTIALS = credPath;
  } catch (e) {
    console.error('Failed to write GOOGLE_APPLICATION_CREDENTIALS_JSON to tmp file', e);
  }
}
let admin;
let getAuth;
try {
  admin = require('firebase-admin');
  ({ getAuth } = require('firebase-admin/auth'));
  // Only initialize if credentials are available; otherwise lazy-init will fail gracefully on first use
  if (process.env.GOOGLE_APPLICATION_CREDENTIALS || process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON || process.env.FIREBASE_CONFIG) {
    admin.initializeApp({ credential: admin.credential.applicationDefault() });
  } else {
    // Initialize without explicit credential for environments where ADC is not set;
    // verifyIdToken will fail until proper credentials are provided, which is expected in tests.
    try { admin.initializeApp(); } catch (_) {}
  }
} catch (e) {
  console.warn('firebase-admin not available or failed to init:', e.message);
  admin = null;
  getAuth = null;
}

// GPU provider abstraction.  GPU_PROVIDER controls which backend resolves:
//   "zerogpu" (default) → Hugging Face ZeroGPU Space (Gradio API)
//   "modal"             → existing Modal endpoints (forwarded unchanged)
const {
  resolveBackend,
  ZeroGPUBackend,
  GpuError,
} = require('./gpu_backends');
const gpuBackend = resolveBackend(process.env);

// Training backend — generic engine (inference vs training are independent)
const trainingBackend = require('./training_backend');

// ── Security headers ───────────────────────────────────────────────────
app.use(helmet({
  crossOriginOpenerPolicy: { policy: "same-origin-allow-popups" },
  contentSecurityPolicy: {
    directives: {
      ...helmet.contentSecurityPolicy.getDefaultDirectives(),
      "script-src": [
        "'self'",
        "https://www.gstatic.com",       // Firebase SDK
        "https://cdn.jsdelivr.net",      // marked, dompurify, xterm, chart.js
        "https://cdnjs.cloudflare.com",  // codemirror
      ],
      "img-src": [
        "'self'",
        "data:",
        "https://api.dicebear.com",      // dashboard avatar
      ],
      "connect-src": [
        "'self'",
        "https://identitytoolkit.googleapis.com",  // Firebase Auth REST calls
        "https://securetoken.googleapis.com",       // Firebase token refresh
      ],
      "script-src-attr": ["'unsafe-inline'"],
    },
  },
}));

// ── CORS allow-list ────────────────────────────────────────────────────
const allowedOrigins = (process.env.FRONTEND_ORIGIN || 'http://localhost:5000')
  .split(',').map(s=>s.trim()).filter(Boolean);
app.use(cors({
  origin: function(origin, cb){
    // allow non-browser requests (no origin) and allow-listed origins
    if (!origin) return cb(null, true);
    if (allowedOrigins.includes(origin) || allowedOrigins.includes('*')) return cb(null, true);
    // For development, allow localhost any port
    if (origin.startsWith('http://localhost:') || origin.startsWith('http://127.0.0.1:')) return cb(null, true);
    return cb(null, false); // reject cleanly — no thrown Error, no 500
  },
  credentials: true
}));

// ── Body parsing ───────────────────────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ── Session ────────────────────────────────────────────────────────────
const SESSION_SECRET = process.env.SESSION_SECRET || 'change-me-to-a-long-random-string-please-set-SESSION_SECRET';
app.use(session({
  secret: SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production', // true only over https in prod
    sameSite: 'lax',
    maxAge: 1000*60*60*24 // 1 day
  }
}));

// ── Rate limiting ───────────────────────────────────────────────────────
const globalLimiter = rateLimit({
  windowMs: 15*60*1000,
  max: 300,
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(globalLimiter);

const strictLimiter = rateLimit({
  windowMs: 60*1000,
  max: 10,
  keyGenerator: (req) => req.session?.userId || ipKeyGenerator(req.ip),
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req,res)=> res.status(429).json({ error: 'too many requests', code: 'rate_limited' })
});

// ── User store (file-based JSON outside web root) ─────────────────────
const USERS_FILE = path.join(__dirname, 'users.json');
function loadUsers(){
  try {
    if (!fs.existsSync(USERS_FILE)) return [];
    const raw = fs.readFileSync(USERS_FILE,'utf8');
    return JSON.parse(raw);
  } catch { return []; }
}
function saveUsers(users){
  try { fs.writeFileSync(USERS_FILE, JSON.stringify(users,null,2)); } catch(e){ console.error('Failed to save users', e); }
}
function findUserByEmail(email){
  const users = loadUsers();
  return users.find(u => u.email.toLowerCase() === String(email).toLowerCase());
}
function findUserById(id){
  const users = loadUsers();
  return users.find(u => u.id === id);
}
function validateEmail(email){
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email).trim());
}

// ── Auth middleware ────────────────────────────────────────────────────
function requireAuth(req,res,next){
  if (!req.session?.userId) return res.status(401).json({ error: 'unauthorized', code: 'unauthorized' });
  const user = findUserById(req.session.userId);
  if (!user) return res.status(401).json({ error: 'unauthorized', code: 'unauthorized' });
  req.user = user;
  next();
}
function requireAdmin(req,res,next){
  if (!req.session?.userId) return res.status(401).json({ error: 'unauthorized', code: 'unauthorized' });
  const user = findUserById(req.session.userId);
  if (!user) return res.status(401).json({ error: 'unauthorized', code: 'unauthorized' });
  const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
  const isAdmin = user.role === 'admin' || adminEmails.includes(user.email.toLowerCase());
  if (!isAdmin) return res.status(403).json({ error: 'forbidden', code: 'forbidden' });
  req.user = user;
  next();
}
function checkOwnershipOr404(job, userId, res){
  if (!trainingBackend.isOwner(job, userId)) {
    // Check admin override
    const user = findUserById(userId);
    const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
    const isAdmin = user && (user.role === 'admin' || adminEmails.includes(user.email.toLowerCase()));
    if (isAdmin) return true;
    // Return 404 to avoid confirming existence
    res.status(404).json({ error: 'Job not found', code: 'not_found' });
    return false;
  }
  return true;
}

// ── Static serving (restricted to public/) ─────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));
// Explicitly handle root
app.get('/', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Auth endpoints ─────────────────────────────────────────────────────
app.post('/api/auth/signup', async (req,res)=>{
  try {
    const { email, password, username, name } = req.body || {};
    const cleanEmail = String(email||'').trim().toLowerCase();
    const cleanUser = String(username || name || '').trim();
    const pwd = String(password||'');
    if (!validateEmail(cleanEmail)) return res.status(400).json({ error: 'Invalid email format', code: 'bad_request' });
    if (!pwd || pwd.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters', code: 'bad_request' });
    if (findUserByEmail(cleanEmail)) return res.status(409).json({ error: 'Email already registered', code: 'conflict' });
    const hash = await bcrypt.hash(pwd, 12);
    const users = loadUsers();
    const id = 'user_' + require('crypto').randomBytes(6).toString('hex');
    const role = users.length === 0 ? 'admin' : 'user'; // first user is admin for bootstrapping
    const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
    const finalRole = adminEmails.includes(cleanEmail) ? 'admin' : role;
    const user = { id, email: cleanEmail, username: cleanUser || cleanEmail.split('@')[0], passwordHash: hash, role: finalRole, createdAt: new Date().toISOString() };
    users.push(user);
    saveUsers(users);
    // Auto-login after signup
    req.session.userId = user.id;
    req.session.email = user.email;
    req.session.role = user.role;
    return res.json({ ok:true, user: { id: user.id, email: user.email, username: user.username, role: user.role } });
  } catch(e){
    console.error('signup error', e);
    return res.status(500).json({ error: 'signup failed', code: 'server_error' });
  }
});

app.post('/api/auth/login', async (req,res)=>{
  try {
    const { email, password } = req.body || {};
    const cleanEmail = String(email||'').trim().toLowerCase();
    const pwd = String(password||'');
    if (!cleanEmail || !pwd) return res.status(400).json({ error: 'Email and password required', code: 'bad_request' });
    const user = findUserByEmail(cleanEmail);
    if (!user) return res.status(401).json({ error: 'Invalid credentials', code: 'unauthorized' });
    const ok = await bcrypt.compare(pwd, user.passwordHash);
    if (!ok) return res.status(401).json({ error: 'Invalid credentials', code: 'unauthorized' });
    req.session.userId = user.id;
    req.session.email = user.email;
    req.session.role = user.role;
    return res.json({ ok:true, user: { id: user.id, email: user.email, username: user.username, role: user.role } });
  } catch(e){
    console.error('login error', e);
    return res.status(500).json({ error: 'login failed', code: 'server_error' });
  }
});

app.post('/api/auth/logout', (req,res)=>{
  req.session.destroy(err=>{
    if (err) return res.status(500).json({ error: 'logout failed' });
    res.clearCookie('connect.sid');
    return res.json({ ok:true });
  });
});

app.get('/api/auth/me', (req,res)=>{
  if (!req.session?.userId) return res.status(401).json({ error: 'unauthorized', code: 'unauthorized' });
  const user = findUserById(req.session.userId);
  if (!user) return res.status(401).json({ error: 'unauthorized' });
  return res.json({ user: { id: user.id, email: user.email, username: user.username, role: user.role } });
});

app.post('/api/auth/firebase-session', async (req, res) => {
  try {
    if (!admin || !getAuth) return res.status(500).json({ error: 'firebase admin not configured', code: 'server_error' });
    const { idToken, username } = req.body || {};
    if (!idToken) return res.status(400).json({ error: 'missing idToken', code: 'bad_request' });
    const decoded = await getAuth().verifyIdToken(idToken);
    const cleanEmail = String(decoded.email || '').trim().toLowerCase();
    if (!cleanEmail) return res.status(400).json({ error: 'token has no email', code: 'bad_request' });

    let user = findUserByEmail(cleanEmail);
    if (!user) {
      const users = loadUsers();
      const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '')
        .split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
      const role = adminEmails.includes(cleanEmail) ? 'admin'
                  : (users.length === 0 ? 'admin' : 'user');
      user = {
        id: 'user_' + require('crypto').randomBytes(6).toString('hex'),
        email: cleanEmail,
        username: (username || decoded.name || cleanEmail.split('@')[0]),
        passwordHash: null,
        provider: 'firebase',
        createdAt: new Date().toISOString()
      };
      users.push(user);
      saveUsers(users);
    }
    req.session.userId = user.id;
    req.session.email = user.email;
    req.session.role = user.role;
    return res.json({ ok: true, user: { id: user.id, email: user.email, username: user.username, role: user.role } });
  } catch (e) {
    console.error('firebase-session error', e);
    return res.status(401).json({ error: 'invalid or expired token', code: 'unauthorized' });
  }
});

// ── GET /api/hf-loader (public, used for model picker) ─────────────────
app.get('/api/hf-loader', async (req, res) => {
  const modelId = String(req.query.modelId || '').trim();
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(modelId)) {
    return res.status(400).json({ error: 'A valid Hugging Face model id is required.' });
  }

  const script = [
    'import json, sys',
    'from hf_loader.notebook_builder import build_loader_cell',
    'try:',
    '    print(json.dumps({"loader": build_loader_cell(sys.argv[1])}))',
    'except Exception as exc:',
    '    print(json.dumps({"error": str(exc)}))',
    '    raise',
  ].join('\n');

  try {
    const { stdout } = await execFileAsync(
      process.env.PYTHON_BIN || 'python3',
      ['-c', script, modelId],
      { cwd: __dirname, timeout: 10000, maxBuffer: 256 * 1024 }
    );
    const payload = JSON.parse(stdout.trim());
    if (payload.error || !payload.loader) {
      return res.status(500).json({ error: payload.error || 'Could not generate loader cell.' });
    }
    return res.json(payload);
  } catch (err) {
    const detail = (err.stderr || err.message || 'Python loader generation failed').trim();
    console.error('HF loader generation error:', detail);
    return res.status(500).json({ error: 'Could not generate the Hugging Face loader cell.' });
  }
});

// ── Helper: forward a request to a Modal endpoint ────────────────────────────
async function callModal(url, body) {
  const { MODAL_AUTH_SECRET } = process.env;
  const res  = await fetch(url, {
    method:  'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${MODAL_AUTH_SECRET || ''}`
    },
    body: JSON.stringify(body)
  });

  const text = await res.text();          // always safe to read as text
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    // Modal returned a non-JSON body (auth error, cold-start page, etc.)
    data = { error: text.slice(0, 500) }; // surface the raw message
  }
  return { status: res.status, data };
}

// ── GET /api/gpu-status ────────────────────────────────────────────────────────
app.get('/api/gpu-status', async (req, res) => {
  try {
    const info = await gpuBackend.status();
    return res.json(info);
  } catch (err) {
    return res.status(502).json({
      state: 'disconnected',
      provider: gpuBackend.kind,
      error: err && err.message ? err.message : String(err),
    });
  }
});

// ── GET /api/gpu-config ──────────────────────────────────────────────────────
app.get('/api/gpu-config', (req, res) => {
  res.json({
    provider: gpuBackend.kind,
    zerogpu: {
      space: gpuBackend.kind === 'zerogpu' ? gpuBackend.spaceId : (process.env.ZEROGPU_SPACE || 'Gochan562/claro_ai_gpu'),
    },
    modal: {
      configured: !!(process.env.MODAL_RUN_URL),
    },
  });
});

// ── GET /api/stream-logs ─────────────────────────────────────────────────────
// NOTE: This endpoint was previously open to RCE when GPU_PROVIDER=modal.
// Now gated behind admin auth.
app.get('/api/stream-logs', requireAuth, requireAdmin, async (req, res) => {
  const { MODAL_STREAM_URL, MODAL_AUTH_SECRET } = process.env;
  const code = req.query.code || '';

  res.set({
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    Connection:           'keep-alive',
    'X-Accel-Buffering': 'no'
  });
  res.flushHeaders();

  const sendLine = (line) => res.write(`data: ${JSON.stringify({ line })}\n\n`);
  const sendDone = (ok, message) => {
    res.write(`event: done\ndata: ${JSON.stringify({ ok, message: message || '' })}\n\n`);
    res.end();
  };

  if (!MODAL_STREAM_URL) {
    sendLine('❌ MODAL_STREAM_URL is not configured. Deploy trainer.py to Modal then set the env var.');
    return sendDone(false);
  }
  if (!code.trim()) {
    sendLine('❌ No code provided.');
    return sendDone(false);
  }

  const controller = new AbortController();
  req.on('close', () => controller.abort());

  try {
    const url = `${MODAL_STREAM_URL}?code=${encodeURIComponent(code)}`;
    const modalRes = await fetch(url, {
      method:  'GET',
      headers: { Authorization: `Bearer ${MODAL_AUTH_SECRET || ''}` },
      signal:  controller.signal
    });

    if (!modalRes.ok || !modalRes.body) {
      const text = await modalRes.text();
      sendLine(`❌ Modal error (${modalRes.status}): ${text.slice(0, 400)}`);
      return sendDone(false);
    }

    const nodeStream = Readable.fromWeb(modalRes.body);
    let buffer = '';

    for await (const chunk of nodeStream) {
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        sendLine(buffer.slice(0, idx));
        buffer = buffer.slice(idx + 1);
      }
    }
    if (buffer) sendLine(buffer);

    sendDone(true);
  } catch (err) {
    if (err.name !== 'AbortError') {
      sendLine(`❌ Connection error: ${err.message}`);
      sendDone(false);
    }
  }
});

// ── POST /api/zerogpu-run ─────────────────────────────────────────────────────
app.post('/api/zerogpu-run', requireAuth, strictLimiter, async (req, res) => {
  if (gpuBackend.kind !== 'zerogpu') {
    return res.status(400).json({
      error: `Not on ZeroGPU backend (current provider: ${gpuBackend.kind}). Set GPU_PROVIDER=zerogpu to use this endpoint.`,
    });
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(process.env.ZEROGPU_TIMEOUT_MS || 120000));
  const authToken = req.headers.authorization?.replace(/^Bearer\s+/i, '') || '';
  try {
    const result = await gpuBackend.run(req.body || {}, controller.signal, null, authToken);
    return res.json(result);
  } catch (err) {
    const code = err && err.code ? err.code : 'zerogpu_runtime';
    const status = err && err.status ? err.status : 502;
    if (err instanceof GpuError) {
      return res.status(status).json({ error: err.message, code });
    }
    return res.status(502).json({ error: `ZeroGPU call failed: ${err.message || err}`, code });
  } finally {
    clearTimeout(timer);
  }
});

// ── GET /api/zerogpu-stream ──────────────────────────────────────────────────
app.get('/api/zerogpu-stream', requireAuth, async (req, res) => {
  res.set({
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    Connection:           'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders();

  const sendLine = (line) => res.write(`data: ${JSON.stringify({ line })}\n\n`);
  const sendDone = (ok, message) => {
    res.write(`event: done\ndata: ${JSON.stringify({ ok, message: message || '' })}\n\n`);
    res.end();
  };

  if (gpuBackend.kind !== 'zerogpu') {
    sendLine(`❌ Not on ZeroGPU backend (current provider: ${gpuBackend.kind}).`);
    return sendDone(false);
  }
  let body;
  try {
    const raw = req.query.req || '{}';
    body = JSON.parse(decodeURIComponent(raw));
  } catch (err) {
    sendLine(`❌ Malformed ?req= JSON: ${err.message}`);
    return sendDone(false);
  }

  const controller = new AbortController();
  req.on('close', () => controller.abort());

  try {
    await gpuBackend.run(body, controller.signal, sendLine);
    sendDone(true);
  } catch (err) {
    const code = err && err.code ? err.code : 'zerogpu_runtime';
    sendLine(`❌ ${err.message || err}  (code: ${code})`);
    sendDone(false);
  }
});

// ── POST /api/run-cell ────────────────────────────────────────────────────────
// Disabled by default for security. Only admin may use it when explicitly enabled.
// Even admin is rate-limited.
app.post('/api/run-cell', requireAuth, requireAdmin, strictLimiter, async (req, res) => {
  const { code } = req.body || {};
  if (!code || !String(code).trim()) {
    return res.status(400).json({ error: 'No code provided.' });
  }

  if (gpuBackend.kind === 'zerogpu') {
    return res.status(400).json({
      error:
        'ZeroGPU backend only accepts structured { model_id, task, inputs } ' +
        'requests via POST /api/zerogpu-run (or GET /api/zerogpu-stream). ' +
        'Plain notebook-cell Python is not executed on ZeroGPU.',
      code: 'zerogpu_no_python_cells',
    });
  }

  try {
    const { status, data } = await gpuBackend.run({ code });
    return res.status(status).json(data);
  } catch (err) {
    const status = err && err.status ? err.status : 502;
    return res.status(status).json({ error: err.message || String(err) });
  }
});

// ── Training Engine ────────────────────────────────────────────────────────
app.post('/api/train/start', requireAuth, strictLimiter, async (req, res) => {
  try {
    // Per-user quota: 1 concurrent job
    const active = trainingBackend.getActiveJobCountForUser(req.session.userId);
    const limit = parseInt(process.env.TRAIN_MAX_CONCURRENT_PER_USER || '1',10);
    if (active >= limit) {
      return res.status(429).json({ error: `Concurrent job limit reached (${limit}). Wait for your current job to finish.`, code: 'quota_exceeded' });
    }
    const config = trainingBackend.validateTrainingRequest(req.body);
    const job = trainingBackend.createJob(config, req.session.userId);
    trainingBackend.startJob(job);
    return res.json({
      job_id: job.job_id,
      status: job.status,
      config: job.config,
      message: 'Training job queued',
    });
  } catch (err) {
    const status = err.status || 400;
    const code = err.code || 'bad_request';
    return res.status(status).json({ error: err.message, code });
  }
});

app.get('/api/train/status/:job_id', requireAuth, (req, res) => {
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  if (!checkOwnershipOr404(job, req.session.userId, res)) return;
  return res.json({
    job_id: job.job_id,
    status: job.status,
    progress: job.progress,
    config: job.config,
    error: job.error,
    created_at: job.created_at,
    metrics_count: job.metrics.length,
    artifacts_dir: job.artifacts_dir,
  });
});

app.get('/api/train/metrics/:job_id', requireAuth, (req, res) => {
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  if (!checkOwnershipOr404(job, req.session.userId, res)) return;

  const wantsSSE = (req.headers.accept || '').includes('text/event-stream') || req.query.stream === '1' || req.query.stream === 'true';
  if (wantsSSE) {
    res.set({
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    res.flushHeaders();
    trainingBackend.attachSSE(job.job_id, res);
    const ping = setInterval(() => {
      try { res.write(': ping\n\n'); } catch (_) { clearInterval(ping); }
    }, 15000);
    req.on('close', () => {
      clearInterval(ping);
    });
    return;
  }

  const limit = Math.min(parseInt(req.query.limit || '100', 10) || 100, 500);
  const slice = job.metrics.slice(-limit);
  return res.json({
    job_id: job.job_id,
    status: job.status,
    progress: job.progress,
    metrics: slice,
  });
});

app.get('/api/train/stream/:job_id', requireAuth, (req, res) => {
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  if (!checkOwnershipOr404(job, req.session.userId, res)) return;
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders();
  trainingBackend.attachSSE(job.job_id, res);
  const ping = setInterval(() => {
    try { res.write(': ping\n\n'); } catch (_) { clearInterval(ping); }
  }, 15000);
  req.on('close', () => clearInterval(ping));
});

app.get('/api/train/list', requireAuth, requireAdmin, (req, res) => {
  return res.json({ jobs: trainingBackend.listJobs() });
});

app.post('/api/train/stop', requireAuth, (req, res) => {
  const job_id = String(req.body.job_id || req.body.jobId || req.query.job_id || '').trim();
  const diagCaller = req.body._diag_caller || req.headers['x-diag-caller'] || 'unknown';
  const diagCellId = req.body._diag_cellId || req.body.cellId || 'unknown';
  const diagUserInitiated = req.body._diag_userInitiated;
  const diagStack = req.body._diag_stack || '';
  console.log(`[TRAIN-DIAG] POST /api/train/stop job_id=${job_id} caller=${diagCaller} cellId=${diagCellId} userInitiated=${diagUserInitiated} ts=${new Date().toISOString()} ip=${req.ip} stack=${String(diagStack).slice(0,500)}`);
  if (!job_id) return res.status(400).json({ error: 'job_id is required', code: 'bad_request' });
  const job = trainingBackend.getJob(job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  if (!checkOwnershipOr404(job, req.session.userId, res)) return;
  const stopped = trainingBackend.stopJob(job_id, { caller: diagCaller, cellId: diagCellId, userInitiated: diagUserInitiated, stack: diagStack });
  if (!stopped) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({ job_id, status: stopped.status, message: 'Job stopped' });
});

app.post('/api/train/stop/:job_id', requireAuth, (req, res) => {
  console.log(`[TRAIN-DIAG] POST /api/train/stop/:job_id job_id=${req.params.job_id} ts=${new Date().toISOString()} ip=${req.ip}`);
  const job = trainingBackend.getJob(req.params.job_id);
  if (!job) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  if (!checkOwnershipOr404(job, req.session.userId, res)) return;
  const stopped = trainingBackend.stopJob(req.params.job_id, { caller: 'stop/:job_id', userInitiated: false });
  if (!stopped) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
  return res.json({ job_id: stopped.job_id, status: stopped.status, message: 'Job stopped' });
});

app.get('/api/train/artifacts/:job_id', requireAuth, (req, res) => {
  // Need to check ownership before revealing metadata; getArtifactMetadata will throw if job not found,
  // but we need to ensure 404 for non-owner. We can check via getJob first.
  const job = trainingBackend.getJob(req.params.job_id);
  // Also handle case where job is on disk but not in memory: need to check diskMeta ownership
  // We try to get metadata and then check, but if not in memory we need to read disk job.json
  // For simplicity, if job exists in memory, check ownership; if not, try to load disk and check.
  if (job) {
    if (!checkOwnershipOr404(job, req.session.userId, res)) return;
  } else {
    // Fallback: try to read disk job.json to check owner without leaking existence
    try {
      const dir = trainingBackend._safeArtifactDir(req.params.job_id);
      const jobJsonPath = path.join(dir, 'job.json');
      if (fs.existsSync(jobJsonPath)) {
        const meta = JSON.parse(fs.readFileSync(jobJsonPath,'utf8'));
        if (meta.user_id && meta.user_id !== req.session.userId) {
          // Check admin
          const user = findUserById(req.session.userId);
          const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
          const isAdmin = user && (user.role === 'admin' || adminEmails.includes(user.email.toLowerCase()));
          if (!isAdmin) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
        } else if (!meta.user_id) {
          // Legacy job without owner: only admin may access
          const user = findUserById(req.session.userId);
          const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
          const isAdmin = user && (user.role === 'admin' || adminEmails.includes(user.email.toLowerCase()));
          if (!isAdmin) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
        }
      }
    } catch {}
  }
  try {
    const meta = trainingBackend.getArtifactMetadata(req.params.job_id);
    return res.json(meta);
  } catch (err) {
    const status = err.status || 500;
    const code = err.code || 'artifact_error';
    return res.status(status).json({ error: err.message, code });
  }
});

app.post('/api/inference/trained', requireAuth, strictLimiter, async (req, res) => {
  const { job_id, prompt, max_new_tokens, task, image, image_base64, imageBase64 } = req.body || {};
  if (!job_id) return res.status(400).json({ error: 'job_id is required', code: 'bad_request' });
  // Ownership check before inference
  const jobCheck = trainingBackend.getJob(String(job_id).trim());
  if (jobCheck) {
    if (!checkOwnershipOr404(jobCheck, req.session.userId, res)) return;
  } else {
    // Fallback disk check as in artifacts
    try {
      const dir = trainingBackend._safeArtifactDir(String(job_id).trim());
      const jobJsonPath = path.join(dir, 'job.json');
      if (fs.existsSync(jobJsonPath)) {
        const meta = JSON.parse(fs.readFileSync(jobJsonPath,'utf8'));
        if (meta.user_id && meta.user_id !== req.session.userId) {
          const user = findUserById(req.session.userId);
          const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
          const isAdmin = user && (user.role === 'admin' || adminEmails.includes(user.email.toLowerCase()));
          if (!isAdmin) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
        } else if (!meta.user_id) {
          const user = findUserById(req.session.userId);
          const adminEmails = (process.env.ADMIN_EMAILS || process.env.ADMIN_EMAIL || '').split(',').map(s=>s.trim().toLowerCase()).filter(Boolean);
          const isAdmin = user && (user.role === 'admin' || adminEmails.includes(user.email.toLowerCase()));
          if (!isAdmin) return res.status(404).json({ error: 'Job not found', code: 'not_found' });
        }
      }
    } catch {}
  }
  let meta;
  try {
    meta = trainingBackend.getArtifactMetadata(String(job_id).trim());
  } catch (err) {
    const status = err.status || 500;
    const code = err.code || 'artifact_error';
    return res.status(status).json({ error: err.message, code });
  }

  // Use artifact metadata to derive paths — never use client-provided paths
  const jobId = meta.job_id;
  const taskType = String(task || meta.task_type || 'text-generation').toLowerCase();
  const maxTokens = Math.min(parseInt(max_new_tokens, 10) || 100, 2048);
  const isImageTask = taskType === 'image-classification';
  const imagePayload = image_base64 || image || imageBase64 || null;
  let promptStr = null;
  if (!isImageTask) {
    if (!prompt || typeof prompt !== 'string' || !prompt.trim()) return res.status(400).json({ error: 'prompt is required', code: 'bad_request' });
    promptStr = String(prompt).slice(0, 4000);
  } else {
    if (imagePayload) {
      promptStr = String(prompt || '').slice(0, 4000);
    } else if (prompt && typeof prompt === 'string' && prompt.startsWith('data:image')) {
      promptStr = prompt.slice(0, 5000000);
    } else if (prompt && typeof prompt === 'string' && prompt.trim()) {
      promptStr = String(prompt).slice(0, 4000);
    } else {
      return res.status(400).json({ error: 'image is required for image-classification (send image_base64)', code: 'bad_request' });
    }
  }

  const pythonBin = process.env.PYTHON_BIN || 'python3';
  const runnerPath = path.join(__dirname, 'inference_runner.py');
  if (!require('fs').existsSync(runnerPath)) {
    return res.status(500).json({ error: 'inference_runner.py not found', code: 'inference_error' });
  }

  const args = [
    runnerPath,
    '--job_id', jobId,
    '--task_type', taskType,
    '--max_new_tokens', String(maxTokens),
  ];
  if (isImageTask) {
    if (imagePayload) {
      args.push('--image_base64', String(imagePayload).slice(0, 8000000));
    } else if (promptStr && promptStr.startsWith('data:image')) {
      args.push('--image_base64', promptStr);
    } else if (promptStr) {
      args.push('--prompt', promptStr);
    }
  } else {
    args.push('--prompt', promptStr);
  }

  const { spawn } = require('child_process');
  let proc;
  try {
    proc = spawn(pythonBin, args, { cwd: __dirname, env: process.env });
  } catch (e) {
    return res.status(500).json({ error: `Failed to spawn inference: ${e.message}`, code: 'inference_error' });
  }

  let stdout = '';
  let stderr = '';
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    try { proc.kill('SIGKILL'); } catch (_) {}
  }, 60000);

  proc.stdout.on('data', (chunk) => { stdout += chunk.toString('utf8'); });
  proc.stderr.on('data', (chunk) => { stderr += chunk.toString('utf8'); });

  proc.on('close', (code) => {
    clearTimeout(timer);
    if (timedOut) {
      return res.status(504).json({ error: 'Inference timed out', code: 'inference_timeout' });
    }
    if (code !== 0) {
      try {
        const parsed = JSON.parse(stdout.trim().split('\n').pop());
        if (parsed && parsed.error) {
          return res.status(500).json({ error: parsed.error, code: 'inference_error', stderr: stderr.slice(0, 1000) });
        }
      } catch (_) {}
      return res.status(500).json({ error: `Inference failed (exit ${code}): ${stderr.slice(0, 1000) || stdout.slice(0, 1000)}`, code: 'inference_error' });
    }
    try {
      const lines = stdout.trim().split('\n');
      let data = null;
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith('{') && line.endsWith('}')) {
          try { data = JSON.parse(line); break; } catch (_) {}
        }
      }
      if (!data || (data.output === undefined && data.label === undefined && data.scores === undefined)) {
        return res.status(500).json({ error: `No output from inference: ${stdout.slice(0, 1000)}`, code: 'inference_error', stderr: stderr.slice(0, 500) });
      }
      return res.json({
        task: data.task || taskType,
        output: data.output !== undefined ? data.output : (data.label || JSON.stringify(data)),
        label: data.label,
        confidence: data.confidence,
        prediction: data.prediction,
        scores: data.scores,
        tokens: data.tokens,
        entities: data.entities,
        raw: data
      });
    } catch (e) {
      return res.status(500).json({ error: `Failed to parse inference output: ${e.message}`, code: 'inference_error', stdout: stdout.slice(0, 1000), stderr: stderr.slice(0, 500) });
    }
  });

  proc.on('error', (err) => {
    clearTimeout(timer);
    return res.status(500).json({ error: `Inference spawn error: ${err.message}`, code: 'inference_error' });
  });
});

// ── POST /api/train (legacy) ──────────────────────────────────────────────────
app.post('/api/train', requireAuth, strictLimiter, async (req, res) => {
  const { MODAL_URL } = process.env;

  if (!MODAL_URL) {
    return res.status(500).json({ error: 'MODAL_URL is not configured.' });
  }

  const { modelId, datasetId, epochs, lr } = req.body;
  if (!modelId || !datasetId) {
    return res.status(400).json({ error: 'modelId and datasetId are required.' });
  }

  try {
    const { status, data } = await callModal(MODAL_URL, {
      model_id:      modelId,
      dataset_id:    datasetId,
      epochs:        parseInt(epochs, 10) || 3,
      learning_rate: parseFloat(lr)       || 2e-4
    });
    return res.status(status).json(data);
  } catch (err) {
    return res.status(502).json({ error: `Could not reach Modal: ${err.message}` });
  }
});

// Fallback for client-side routing: serve index.html for unknown non-API routes
app.get('/*splat', (req,res,next)=>{
  if (req.path.startsWith('/api/')) return next();
  // Let static handle if file exists, otherwise 404
  return res.status(404).send('Not found');
});

// ── Generic error handler (must be last) ────────────────────────────────
app.use((err, req, res, next) => {
  console.error(err); // full detail stays in server logs
  if (res.headersSent) return next(err);
  res.status(err.status || 500).json({
    error: process.env.NODE_ENV === 'production' ? 'internal server error' : err.message,
    code: 'internal_error'
  });
});

if (require.main === module) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Claro.AI server listening on port ${PORT}`);
  });
}
module.exports = app;

