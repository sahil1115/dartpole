// src/js/main.js
// DartPole — local RAG UI controller.
// Drives the 3-stage single screen (fresh -> indexing -> ready), theme + model
// pickers, live system stats, and the chat/citation flow, wired to the Flask API.

const API_URL = '/api';

/* ----------------------------- Themes ----------------------------- */
const THEMES = {
  'Light':      { bg:'#fdfdfd', sf:'#ffffff', s2:'#f2f2f2', bd:'#e8e8e8', tx:'#1c1c1e', tm:'#6e6e73', tf:'#a8a8ad', ac:'#1c1c1e', as:'#ececec', at:'#3a3a3c', oa:'#ffffff', dg:'#c0503c' },
  'Dark':       { bg:'#0e0e10', sf:'#18181b', s2:'#232327', bd:'#2c2c31', tx:'#ededf0', tm:'#98989f', tf:'#5f5f66', ac:'#ededf0', as:'#26262b', at:'#c8c8ce', oa:'#141416', dg:'#e07a63' },
  'Linen':      { bg:'#faf7f2', sf:'#fffdf9', s2:'#f4ede2', bd:'#eee5d8', tx:'#3d3630', tm:'#8a7d6c', tf:'#c4b8a6', ac:'oklch(0.72 0.14 40)',  as:'#f6e8dc', at:'#9a5c3e', oa:'#ffffff', dg:'#b05c4a' },
  'Matcha':     { bg:'#f4f8f5', sf:'#fbfdfb', s2:'#eef5ef', bd:'#e2ece4', tx:'#2e3a33', tm:'#6b8272', tf:'#a9c0ae', ac:'oklch(0.65 0.11 165)', as:'#e3f0e6', at:'#41705a', oa:'#ffffff', dg:'#b05c4a' },
  'Periwinkle': { bg:'#f6f6fb', sf:'#fcfcfe', s2:'#efeffa', bd:'#e7e7f2', tx:'#35364a', tm:'#75769a', tf:'#b3b4d6', ac:'oklch(0.6 0.12 285)',  as:'#e9e9f7', at:'#54558a', oa:'#ffffff', dg:'#b05c4a' },
  'Oat':        { bg:'#f8f6f4', sf:'#fffefd', s2:'#f0ebe6', bd:'#ece6df', tx:'#38322c', tm:'#7d7268', tf:'#a89d92', ac:'oklch(0.55 0.09 60)',  as:'#efe6dc', at:'#6e5138', oa:'#ffffff', dg:'#b05c4a' },
  'Harbor':     { bg:'#f3f7fa', sf:'#fcfdfe', s2:'#eaf1f6', bd:'#e0eaf1', tx:'#2c3a44', tm:'#5b7181', tf:'#8aa0af', ac:'oklch(0.6 0.12 240)',  as:'#e2edf5', at:'#3c607c', oa:'#ffffff', dg:'#b05c4a' },
  'Petal':      { bg:'#faf5f6', sf:'#fefcfc', s2:'#f5eaec', bd:'#f0e2e5', tx:'#3e3236', tm:'#8a6c73', tf:'#b298a0', ac:'oklch(0.62 0.12 0)',   as:'#f5e3e7', at:'#8c4b5a', oa:'#ffffff', dg:'#b05c4a' },
  'Fern':       { bg:'#f7f8f3', sf:'#fdfefb', s2:'#eef1e6', bd:'#e8ebdd', tx:'#333a2c', tm:'#6e7861', tf:'#9aa48c', ac:'oklch(0.6 0.11 120)',  as:'#e8efdb', at:'#556b3a', oa:'#ffffff', dg:'#b05c4a' }
};
const VARS = { bg:'--bg', sf:'--sf', s2:'--s2', bd:'--bd', tx:'--tx', tm:'--tm', tf:'--tf', ac:'--ac', as:'--as', at:'--at', oa:'--oa', dg:'--dg' };

const DOT = { pdf: 'var(--pdf)', doc: 'var(--docdot)', md: 'var(--mddot)', txt: 'var(--mddot)' };

const SUGGESTIONS = [
  'What are the biggest cost drivers?',
  'Summarize the key points',
  'What are the next steps?'
];

/* ----------------------------- State ----------------------------- */
const state = {
  stage: 'fresh',            // fresh | indexing | ready
  folder: '',
  model: null,               // currently selected model name
  models: [],                // [{name, size}]
  themeName: 'Light',
  statsOpen: true,
  cpu: 0, ram: 0, disk: 0,
  embedder: '—',
  chunks: 0,
  docCount: 0,
  messages: [],              // { role:'user'|'ai', text, cites?, error? }
  thinking: false,
  activeCites: null,         // array reference of currently-shown citations
  activeIndex: 0,
  needModel: false,          // docs processed but no model loaded yet
  modelLoaded: false,
  busy: false,
  centerTab: 'chat',         // chat | insights | bills
  insights: null,            // { summary, entities, entity_source, document_count, documents, model, cached }
  insightsLoading: false,
  insightsError: null,
  billsDetected: false,      // does this corpus look bill-like? (set from /initialize or /status)
  billsJobStatus: 'idle',    // idle | queued | running | done | error
  billsDone: 0,
  billsTotal: 0,
  billsCurrentFile: null,
  billsError: null,
  billsRecords: [],
  billsForecast: null,
  billsCategories: [],
  billsPollTimer: null
};

let statsTimer = null;

/* ----------------------------- DOM refs ----------------------------- */
const $ = (id) => document.getElementById(id);
let app;

/* ----------------------------- Helpers ----------------------------- */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function baseName(p) { return String(p || '').split(/[/\\]/).pop(); }

/* ----------------------------- Theming ----------------------------- */
function applyTheme(name) {
  const t = THEMES[name] || THEMES['Light'];
  for (const k in VARS) document.documentElement.style.setProperty(VARS[k], t[k]);
  try { localStorage.setItem('ida-theme', name); } catch (e) {}
  state.themeName = name;
  $('themePillName').textContent = name;
  renderThemeMenu();
}

function renderThemeMenu() {
  const wrap = $('themeMenuRows');
  wrap.innerHTML = '';
  Object.keys(THEMES).forEach(name => {
    const t = THEMES[name];
    const row = el('div', 'picker-row theme-row' + (name === state.themeName ? ' active' : ''));
    const sw = el('span', 'sw');
    sw.style.background = `linear-gradient(135deg, ${t.bg} 50%, ${t.ac} 50%)`;
    row.appendChild(sw);
    row.appendChild(el('span', 'grow', name));
    row.appendChild(el('span', 'check', name === state.themeName ? '✓' : ''));
    row.addEventListener('click', () => { applyTheme(name); closeMenus(); });
    wrap.appendChild(row);
  });
}

/* ----------------------------- Model picker ----------------------------- */
function renderModelPill() {
  $('modelPillName').textContent = state.model || 'No model';
}
function renderModelMenu() {
  const wrap = $('modelMenuRows');
  wrap.innerHTML = '';
  if (!state.models.length) {
    const row = el('div', 'picker-row');
    row.appendChild(el('span', 'grow', 'No models found'));
    wrap.appendChild(row);
    return;
  }
  state.models.forEach(m => {
    const row = el('div', 'picker-row' + (m.name === state.model ? ' active' : ''));
    row.appendChild(el('span', 'check', m.name === state.model ? '✓' : ''));
    row.appendChild(el('span', 'grow', m.name));
    if (m.size) row.appendChild(el('span', 'size', m.size));
    row.addEventListener('click', () => { pickModel(m.name); closeMenus(); });
    wrap.appendChild(row);
  });
}

async function pickModel(name) {
  if (name === state.model && state.modelLoaded) return;
  state.model = name;
  renderModelPill();
  renderModelMenu();
  renderSteps();
  // If documents are already indexed, (re)load the model in the backend.
  if (state.stage === 'ready') {
    await loadModel(name);
  }
}

/* ----------------------------- Menus ----------------------------- */
function closeMenus() {
  $('modelMenu').classList.add('hidden');
  $('themeMenu').classList.add('hidden');
}
function toggleMenu(which) {
  const model = $('modelMenu'), theme = $('themeMenu');
  if (which === 'model') {
    const open = model.classList.contains('hidden');
    theme.classList.add('hidden');
    model.classList.toggle('hidden', !open);
  } else {
    const open = theme.classList.contains('hidden');
    model.classList.add('hidden');
    theme.classList.toggle('hidden', !open);
  }
}

/* ----------------------------- System stats ----------------------------- */
function renderStats() {
  const setBar = (fillId, pctId, val, hot) => {
    const v = Math.max(0, Math.min(100, Number(val) || 0));
    const fill = $(fillId);
    fill.style.width = v + '%';
    fill.style.background = hot && v > 85 ? 'var(--cpu-hot)' : 'var(--ac)';
    $(pctId).textContent = Math.round(v) + '%';
  };
  setBar('cpuFill', 'cpuPct', state.cpu, true);
  setBar('ramFill', 'ramPct', state.ram, false);
  setBar('diskFill', 'diskPct', state.disk, false);
  $('sysSummary').textContent =
    `CPU ${Math.round(state.cpu)}% · RAM ${Math.round(state.ram)}% · DISK ${Math.round(state.disk)}%`;
  $('sysEmbedder').textContent = state.embedder;
  $('sysChunks').textContent = Number(state.chunks).toLocaleString();
}

function renderStatsCollapse() {
  const open = state.statsOpen;
  $('sysBody').classList.toggle('hidden', !open);
  $('sysSummary').classList.toggle('hidden', open);
  $('sysRefresh').classList.toggle('hidden', !open);
  $('sysCaret').textContent = open ? '⌄' : '⌃';
  // Extra rows (embedder/chunks) are only relevant once there is an index.
  $('sysExtra').classList.toggle('hidden', !(open && (state.stage === 'indexing' || state.stage === 'ready')));
}

async function pollStats() {
  try {
    const [cpu, ram, disk] = await Promise.all([
      fetch(`${API_URL}/stats/cpu`).then(r => r.ok ? r.json() : { cpu_usage: 0 }),
      fetch(`${API_URL}/stats/ram`).then(r => r.ok ? r.json() : { ram_usage_percent: 0 }),
      fetch(`${API_URL}/stats/disk`).then(r => r.ok ? r.json() : { disk_usage_percent: 0 })
    ]);
    state.cpu = cpu.cpu_usage ?? 0;
    state.ram = ram.ram_usage_percent ?? 0;
    state.disk = disk.disk_usage_percent ?? 0;
  } catch (e) {
    // leave last known values
  }
  renderStats();
}
function startStats() {
  if (statsTimer) clearInterval(statsTimer);
  pollStats();
  statsTimer = setInterval(pollStats, 2000);
}

/* ----------------------------- Stage rendering ----------------------------- */
function showStage(stage) {
  state.stage = stage;
  app.dataset.stage = stage;
  renderDocsBadge();
  renderSteps();
  renderStatsCollapse();
  renderFooter();
}

function renderDocsBadge() {
  const b = $('docsBadge');
  if (state.stage === 'ready') b.textContent = state.docCount ? `${state.docCount} indexed` : '';
  else if (state.stage === 'indexing') b.textContent = 'indexing…';
  else b.textContent = '';
}

function renderSteps() {
  const hasFolder = !!state.folder;
  $('step1Path').textContent = hasFolder ? state.folder : 'No folder chosen';
  $('step1Num').textContent = hasFolder ? '✓' : '1';
  $('step1').classList.toggle('done', hasFolder);

  $('step2').classList.toggle('active', hasFolder);
  $('step2').classList.toggle('dim', !hasFolder);
  $('initBtn').classList.toggle('hidden', !hasFolder);
  $('browseBtn').classList.toggle('hidden', hasFolder);

  $('step3Sub').textContent = state.model
    ? `Any Ollama model you've pulled — currently ${state.model}`
    : "Any Ollama model you've pulled";
}

function renderFooter() {
  const dot = $('statusDot'), txt = $('statusText'), right = $('statusRight');
  if (state.stage === 'indexing') {
    dot.style.background = 'var(--ac)';
    txt.textContent = 'Building your local index…';
    right.textContent = 'building vector index locally';
  } else if (state.stage === 'ready') {
    dot.style.background = 'var(--ready)';
    if (state.needModel || !state.modelLoaded) {
      txt.textContent = state.models.length
        ? 'Documents ready — select a model to start'
        : 'Documents ready — no Ollama models found. Check Ollama.';
    } else {
      txt.textContent = `Ready · ${state.docCount} document${state.docCount === 1 ? '' : 's'} indexed · ${state.model}`;
    }
    right.textContent = state.chunks
      ? `${Number(state.chunks).toLocaleString()} chunks · ${state.embedder}`
      : state.embedder;
  } else {
    dot.style.background = 'var(--tf)';
    txt.textContent = state.folder
      ? 'Folder selected — click Initialize to build the index'
      : 'Not initialized — choose a folder to begin';
    right.textContent = 'PDF · DOCX · TXT · MD';
  }
}

/* ----------------------------- Indexing view ----------------------------- */
async function renderIndexFiles() {
  const rows = $('idxRows');
  rows.innerHTML = '';
  let files = [];
  try {
    const res = await fetch(`${API_URL}/list-folder?path=${encodeURIComponent(state.folder)}`);
    const data = await res.json();
    if (res.ok && data.status === 'success') files = data.files || [];
  } catch (e) {}
  $('docsBadge').textContent = files.length ? `${files.length} files` : 'indexing…';
  files.forEach(f => {
    const row = el('div', 'idx-row');
    const spin = el('span', 'idx-spin');
    row.appendChild(spin);
    row.appendChild(el('span', 'name', f.name));
    row.appendChild(el('span', 'note', 'embedding…'));
    rows.appendChild(row);
  });
}

/* ----------------------------- Ready: documents ----------------------------- */
async function loadDocuments() {
  try {
    const res = await fetch(`${API_URL}/documents`);
    const data = await res.json();
    if (!res.ok || data.status !== 'success') return;
    state.docCount = data.unique_source_count || 0;
    state.chunks = data.total_chunk_count || 0;
    $('docsPath').textContent = data.active_directory || state.folder || '';
    const wrap = $('docRows');
    wrap.innerHTML = '';
    (data.documents || []).forEach(d => {
      const ext = (d.source.split('.').pop() || '').toLowerCase();
      const type = ext === 'pdf' ? 'pdf' : (ext === 'docx' || ext === 'doc') ? 'doc' : ext === 'md' ? 'md' : 'txt';
      const row = el('div', 'doc-row');
      const dot = el('span', 'dot'); dot.style.background = DOT[type];
      row.appendChild(dot);
      row.appendChild(el('span', 'name', d.source));
      const pages = (d.pages && d.pages !== 'N/A') ? d.pages.split(',').length : (d.chunk_count || 0);
      row.appendChild(el('span', 'pages', `${pages} p`));
      row.title = d.full_path || d.source;
      row.addEventListener('click', () => switchCenterTab('insights'));
      wrap.appendChild(row);
    });
    renderDocsBadge();
    renderFooter();
  } catch (e) {}
}

/* ----------------------------- Chat ----------------------------- */
function citeLabel(cite) {
  const name = baseName(cite.metadata && cite.metadata.source);
  const page = cite.metadata && cite.metadata.page;
  if (page && String(page) !== 'N/A') return `${name} · p.${page}`;
  return name || 'source';
}

function renderChat() {
  const scroll = $('chatScroll');
  scroll.innerHTML = '';
  if (state.messages.length === 0 && !state.thinking) {
    const empty = el('div', 'chat-empty');
    empty.appendChild(el('div', 'h', state.needModel
      ? 'Select a model to begin.'
      : `Your ${state.docCount} document${state.docCount === 1 ? '' : 's'} are ready.`));
    empty.appendChild(el('div', 'sub', state.needModel
      ? 'Pick an Ollama model from the header, then ask anything.'
      : 'Ask anything — answers cite the exact passages they came from.'));
    if (!state.needModel) {
      const sug = el('div', 'suggestions');
      SUGGESTIONS.forEach(s => {
        const b = el('button', 'suggestion', s);
        b.addEventListener('click', () => ask(s));
        sug.appendChild(b);
      });
      empty.appendChild(sug);
    }
    scroll.appendChild(empty);
    return;
  }

  state.messages.forEach(m => {
    if (m.role === 'user') {
      scroll.appendChild(el('div', 'msg-user', m.text));
    } else {
      const ai = el('div', 'msg-ai');
      ai.appendChild(el('div', 'bubble' + (m.error ? ' error' : ''), m.text));
      if (m.cites && m.cites.length) {
        const chips = el('div', 'msg-chips');
        m.cites.forEach((c, i) => {
          const isActive = state.activeCites === m.cites && state.activeIndex === i;
          const chip = el('button', 'chip' + (isActive ? ' active' : ''), citeLabel(c));
          chip.addEventListener('click', () => {
            state.activeCites = m.cites;
            state.activeIndex = i;
            renderChat();
            renderSources();
          });
          chips.appendChild(chip);
        });
        ai.appendChild(chips);
      }
      scroll.appendChild(ai);
    }
  });

  if (state.thinking) {
    const t = el('div', 'thinking');
    t.appendChild(el('span', 'dot'));
    t.appendChild(el('span', null, `Searching ${state.docCount} document${state.docCount === 1 ? '' : 's'}…`));
    scroll.appendChild(t);
  }
  scroll.scrollTop = scroll.scrollHeight;
}

function renderSources() {
  const cites = state.activeCites;
  const empty = $('sourcesEmpty'), list = $('sourcesList'), note = $('sourcesNote');
  if (!cites || !cites.length) {
    empty.classList.remove('hidden');
    list.classList.add('hidden');
    note.classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  list.classList.remove('hidden');
  note.classList.remove('hidden');
  list.innerHTML = '';
  cites.forEach((c, i) => {
    const card = el('div', 'cite-card' + (i === state.activeIndex ? ' active' : ''));
    card.appendChild(el('div', 'cite-src', citeLabel(c)));
    card.appendChild(el('div', 'cite-quote', c.quote ? `“${c.quote.trim()}”` : 'Passage text unavailable.'));
    card.addEventListener('click', () => { state.activeIndex = i; renderChat(); renderSources(); });
    list.appendChild(card);
  });
}

async function ask(text) {
  text = (text || '').trim();
  if (!text || state.thinking) return;
  if (state.needModel || !state.modelLoaded) {
    setFooter('Select a model from the header first.');
    return;
  }
  state.messages.push({ role: 'user', text });
  state.thinking = true;
  renderChat();
  try {
    const res = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text })
    });
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      const cites = data.sources || [];
      state.messages.push({ role: 'ai', text: data.answer || 'Received an empty answer.', cites });
      state.activeCites = cites;
      state.activeIndex = 0;
      $('exportBtn').disabled = false;
    } else {
      state.messages.push({ role: 'ai', text: `Error: ${data.message || 'Query failed.'}`, error: true });
    }
  } catch (e) {
    state.messages.push({ role: 'ai', text: `Network error: ${e.message}`, error: true });
  } finally {
    state.thinking = false;
    renderChat();
    renderSources();
    renderFooter();
  }
}

/* ----------------------------- Insights ----------------------------- */
function renderCenterTabs() {
  $('tabChat').classList.toggle('active', state.centerTab === 'chat');
  $('tabInsights').classList.toggle('active', state.centerTab === 'insights');
  $('tabBills').classList.toggle('active', state.centerTab === 'bills');
  $('tabBills').classList.toggle('hidden', !state.billsDetected);
  const onInsights = state.centerTab === 'insights';
  const onBills = state.centerTab === 'bills';
  $('chatScroll').classList.toggle('hidden', onInsights || onBills);
  $('chatInputRow').classList.toggle('hidden', onInsights || onBills);
  $('insightsView').classList.toggle('hidden', !onInsights);
  $('billsView').classList.toggle('hidden', !onBills);
}

function switchCenterTab(tab) {
  if (tab === 'bills' && !state.billsDetected) return;
  state.centerTab = tab;
  renderCenterTabs();
  if (tab === 'insights' && !state.insights && !state.insightsLoading) {
    fetchInsights(false);
  }
  if (tab === 'bills') {
    // Render whatever state is already known immediately (e.g. a scan that
    // finished via the enterReady() prefetch, before this tab was ever
    // opened) — fetchBillsStatus() only re-renders as a side effect if this
    // tab is already active, so switching to it must render on its own too.
    renderBills();
    if (state.billsJobStatus === 'idle' || state.billsJobStatus === 'running' || state.billsJobStatus === 'queued') {
      fetchBillsStatus();
    }
  }
}

function renderInsights() {
  const body = $('insightsBody');
  body.innerHTML = '';
  $('insightsDocCount').textContent = state.docCount
    ? `Across ${state.docCount} document${state.docCount === 1 ? '' : 's'}`
    : '';

  if (state.insightsLoading) {
    const loading = el('div', 'insights-loading');
    loading.appendChild(el('span', 'dot'));
    loading.appendChild(el('span', null, 'Generating executive summary…'));
    body.appendChild(loading);
    return;
  }
  if (state.insightsError) {
    body.appendChild(el('div', 'insights-error', state.insightsError));
    return;
  }
  if (!state.insights) {
    body.appendChild(el('div', 'insights-error', 'No insights yet.'));
    return;
  }

  body.appendChild(el('div', 'insights-summary', state.insights.summary || 'No summary available.'));

  const entities = state.insights.entities || [];
  if (entities.length) {
    body.appendChild(el('div', 'insights-entities-title', 'KEY ENTITIES'));
    const chips = el('div', 'ent-chips');
    entities.forEach(en => {
      const chip = el('span', 'ent-chip');
      chip.appendChild(document.createTextNode(en.text));
      chip.appendChild(el('span', 'ent-type', en.type));
      chips.appendChild(chip);
    });
    body.appendChild(chips);
  }
}

async function fetchInsights(refresh) {
  if (state.insightsLoading) return;
  state.insightsLoading = true;
  state.insightsError = null;
  renderInsights();
  try {
    const res = await fetch(`${API_URL}/insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh: !!refresh })
    });
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      state.insights = data;
    } else {
      state.insightsError = data.message || 'Failed to generate insights.';
    }
  } catch (e) {
    state.insightsError = `Network error: ${e.message}`;
  } finally {
    state.insightsLoading = false;
    renderInsights();
  }
}

/* ----------------------------- Bills ----------------------------- */
function fmtAmount(n) {
  if (n == null) return '—';
  const sign = n < 0 ? '-' : '';
  return `${sign}$${Math.abs(n).toFixed(2)}`;
}
function fmtDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso + 'T00:00:00').toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  } catch (e) { return iso; }
}

function stopBillsPolling() {
  if (state.billsPollTimer) { clearInterval(state.billsPollTimer); state.billsPollTimer = null; }
}
function startBillsPolling() {
  stopBillsPolling();
  state.billsPollTimer = setInterval(fetchBillsStatus, 2000);
}

async function fetchBillsStatus() {
  try {
    const res = await fetch(`${API_URL}/bills`);
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      state.billsJobStatus = data.job_status;
      state.billsDone = data.done;
      state.billsTotal = data.total;
      state.billsCurrentFile = data.current_file;
      state.billsError = data.error;
      state.billsRecords = data.records || [];
      state.billsForecast = data.forecast;
      state.billsCategories = data.categories || [];
      if (data.job_status === 'running' || data.job_status === 'queued') {
        if (!state.billsPollTimer) startBillsPolling();
      } else {
        stopBillsPolling();
      }
    }
  } catch (e) {
    // leave last known state; the next poll or manual action will retry
  }
  if (state.centerTab === 'bills') renderBills();
}

async function startBillsScan() {
  $('billsScanBtn').disabled = true;
  try {
    const res = await fetch(`${API_URL}/bills/scan`, { method: 'POST' });
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      state.billsJobStatus = 'queued';
      state.billsDone = 0;
      state.billsTotal = data.total || 0;
      state.billsError = null;
      renderBills();
      startBillsPolling();
    } else {
      state.billsError = data.message || 'Failed to start bills scan.';
      renderBills();
    }
  } catch (e) {
    state.billsError = `Network error: ${e.message}`;
    renderBills();
  } finally {
    $('billsScanBtn').disabled = false;
  }
}

async function patchBillField(id, field, value) {
  try {
    const res = await fetch(`${API_URL}/bills/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [field]: value })
    });
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      const idx = state.billsRecords.findIndex(r => r.id === id);
      if (idx !== -1) state.billsRecords[idx] = data.record;
      renderBills();
    } else {
      setFooter(`Could not save change: ${data.message || 'unknown error'}`);
    }
  } catch (e) {
    setFooter(`Network error: ${e.message}`);
  }
}

async function deleteBillRow(id) {
  try {
    const res = await fetch(`${API_URL}/bills/${id}`, { method: 'DELETE' });
    if (res.ok) {
      state.billsRecords = state.billsRecords.filter(r => r.id !== id);
      state.billsForecast = null;
      renderBills();
      fetchBillsStatus();
    }
  } catch (e) {
    setFooter(`Network error: ${e.message}`);
  }
}

function billsEditableCell(record, field, display, inputType, extra) {
  const td = el('td', 'bills-cell-editable');
  td.textContent = display;
  td.title = 'Click to edit';
  td.addEventListener('click', function onClick() {
    td.removeEventListener('click', onClick);
    td.textContent = '';
    let input;
    if (inputType === 'select') {
      input = document.createElement('select');
      (extra || []).forEach(opt => {
        const o = document.createElement('option');
        o.value = opt; o.textContent = opt;
        if (opt === record[field]) o.selected = true;
        input.appendChild(o);
      });
    } else {
      input = document.createElement('input');
      input.type = inputType || 'text';
      input.value = inputType === 'number' ? (record[field] != null ? record[field] : '') : (record[field] || '');
    }
    td.appendChild(input);
    input.focus();
    if (input.select) input.select();
    const commit = () => {
      const raw = input.value;
      const value = inputType === 'number' ? parseFloat(raw) : raw;
      if (inputType === 'number' && Number.isNaN(value)) { renderBills(); return; }
      patchBillField(record.id, field, value);
    };
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { input.removeEventListener('blur', commit); renderBills(); }
    });
    if (inputType === 'select') input.addEventListener('change', commit);
  });
  return td;
}

function renderBills() {
  const body = $('billsBody');
  body.innerHTML = '';
  $('billsHeadTitle').textContent = state.billsRecords.length
    ? `${state.billsRecords.length} bill${state.billsRecords.length === 1 ? '' : 's'} found`
    : '';
  $('billsScanBtn').textContent = state.billsJobStatus === 'done' ? 'Rescan' : 'Scan for bills';
  $('billsScanBtn').classList.toggle('hidden', state.billsJobStatus === 'running' || state.billsJobStatus === 'queued');

  if (state.billsJobStatus === 'idle') {
    const empty = el('div', 'bills-empty');
    empty.appendChild(el('div', 'h', 'Find your bills automatically.'));
    empty.appendChild(el('div', 'sub', 'DartPole will read every indexed document, identify bills and invoices, and pull out vendor, category, amount, and due date — entirely on this machine.'));
    empty.appendChild(el('div', 'bills-note', 'Extracted figures are assistive, not authoritative — please verify anything before paying it.'));
    body.appendChild(empty);
    return;
  }

  if (state.billsJobStatus === 'error' && !state.billsRecords.length) {
    body.appendChild(el('div', 'bills-error', state.billsError || 'Bills scan failed.'));
    return;
  }

  if (state.billsJobStatus === 'running' || state.billsJobStatus === 'queued') {
    const progress = el('div', 'bills-progress');
    progress.appendChild(el('span', 'dot'));
    const label = state.billsCurrentFile
      ? `${state.billsDone} / ${state.billsTotal} · reading ${state.billsCurrentFile}…`
      : `Starting scan of ${state.billsTotal} document${state.billsTotal === 1 ? '' : 's'}…`;
    progress.appendChild(el('span', null, label));
    body.appendChild(progress);
    if (state.billsRecords.length) {
      body.appendChild(el('div', 'bills-note', `${state.billsRecords.length} bill${state.billsRecords.length === 1 ? '' : 's'} found so far — editing unlocks once the scan finishes.`));
    }
    return;
  }

  // done
  if (!state.billsRecords.length) {
    const empty = el('div', 'bills-empty');
    empty.appendChild(el('div', 'h', 'No bills found.'));
    empty.appendChild(el('div', 'sub', "This folder didn't contain anything DartPole could confidently identify as a bill or invoice."));
    body.appendChild(empty);
    return;
  }

  const forecast = state.billsForecast || { category_totals: {}, upcoming: [], past_due: [], window_days: 30 };
  const summary = el('div', 'bills-summary');
  const upcomingTotal = forecast.upcoming.reduce((s, r) => s + (r.amount || 0), 0);
  const pastDueTotal = forecast.past_due.reduce((s, r) => s + (r.amount || 0), 0);

  const dueStat = el('div', 'bills-stat');
  dueStat.appendChild(el('div', 'lbl', `Due in ${forecast.window_days} days`));
  dueStat.appendChild(el('div', 'val', `${fmtAmount(upcomingTotal)} · ${forecast.upcoming.length}`));
  summary.appendChild(dueStat);

  if (forecast.past_due.length) {
    const pastStat = el('div', 'bills-stat past-due');
    pastStat.appendChild(el('div', 'lbl', 'Past due'));
    pastStat.appendChild(el('div', 'val', `${fmtAmount(pastDueTotal)} · ${forecast.past_due.length}`));
    summary.appendChild(pastStat);
  }

  const catChips = el('div', 'bills-cat-chips');
  Object.keys(forecast.category_totals).sort().forEach(cat => {
    const chip = el('span', 'ent-chip');
    chip.appendChild(document.createTextNode(cat));
    chip.appendChild(el('span', 'ent-type', fmtAmount(forecast.category_totals[cat])));
    catChips.appendChild(chip);
  });
  if (Object.keys(forecast.category_totals).length) {
    const catWrap = el('div', 'bills-stat');
    catWrap.style.flex = '1';
    catWrap.appendChild(el('div', 'lbl', 'By category'));
    catWrap.appendChild(catChips);
    summary.appendChild(catWrap);
  }
  body.appendChild(summary);

  const tableWrap = el('div', 'bills-table-wrap');
  const table = document.createElement('table');
  table.className = 'bills-table';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  ['Vendor', 'Category', 'Amount', 'Due Date', 'Source', ''].forEach(h => {
    headRow.appendChild(el('th', null, h));
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  const sorted = [...state.billsRecords].sort((a, b) => (a.due_date || '9999') < (b.due_date || '9999') ? -1 : 1);
  sorted.forEach(r => {
    const row = document.createElement('tr');
    row.appendChild(billsEditableCell(r, 'vendor', r.vendor || 'Unknown', 'text'));
    row.appendChild(billsEditableCell(r, 'category', r.category || 'Other', 'select', state.billsCategories));

    const amountTd = billsEditableCell(r, 'amount', '', 'number');
    amountTd.textContent = '';
    if (!r.amount_confident) amountTd.appendChild(el('span', 'bills-low-confidence'));
    const amountSpan = el('span', 'bills-amount' + (r.amount < 0 ? ' negative' : ''), fmtAmount(r.amount));
    amountTd.appendChild(amountSpan);
    if (!r.amount_confident) amountTd.title = 'Not matched to text found in the source — please verify. Click to edit.';
    row.appendChild(amountTd);

    const dueTd = billsEditableCell(r, 'due_date', '', 'date');
    dueTd.textContent = '';
    if (!r.due_date_confident) dueTd.appendChild(el('span', 'bills-low-confidence'));
    dueTd.appendChild(document.createTextNode(fmtDate(r.due_date)));
    if (!r.due_date_confident) dueTd.title = 'Not matched to text found in the source — please verify. Click to edit.';
    row.appendChild(dueTd);

    const sourceTd = el('td');
    const sourceSpan = el('span', 'bills-source', baseName(r.source));
    sourceSpan.title = r.source;
    sourceTd.appendChild(sourceSpan);
    row.appendChild(sourceTd);

    const delTd = el('td');
    const delBtn = el('button', 'bills-delete-btn', '×');
    delBtn.title = 'Remove this bill';
    delBtn.addEventListener('click', () => deleteBillRow(r.id));
    delTd.appendChild(delBtn);
    row.appendChild(delTd);

    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  tableWrap.appendChild(table);
  body.appendChild(tableWrap);
}

function setFooter(msg) { $('statusText').textContent = msg; }

/* ----------------------------- Flow: initialize / model ----------------------------- */
async function fetchModels() {
  try {
    const res = await fetch(`${API_URL}/models`);
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      state.models = data.models || [];
      if (!state.model && state.models.length) state.model = state.models[0].name;
    }
  } catch (e) {}
  renderModelPill();
  renderModelMenu();
  renderSteps();
}

async function doInitialize() {
  if (!state.folder || state.busy) return;
  state.busy = true;
  state.centerTab = 'chat';
  state.insights = null;
  state.insightsLoading = false;
  state.insightsError = null;
  state.billsDetected = false;
  state.billsJobStatus = 'idle';
  state.billsRecords = [];
  state.billsForecast = null;
  stopBillsPolling();
  showStage('indexing');
  renderIndexFiles();
  try {
    const res = await fetch(`${API_URL}/initialize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ directory_path: state.folder })
    });
    const data = await res.json();
    if (!res.ok || (data.status !== 'success' && data.status !== 'warning')) {
      throw new Error(data.message || 'Initialization failed.');
    }
    state.billsDetected = !!data.bills_detected;
    // Docs are indexed. Now load the selected model (or the first available).
    await fetchModels();
    const target = state.model || (state.models[0] && state.models[0].name);
    state.busy = false;
    await enterReady();
    if (target) {
      await loadModel(target);
    } else {
      state.needModel = true;
      state.modelLoaded = false;
      renderChat();
      renderFooter();
    }
  } catch (e) {
    state.busy = false;
    showStage('fresh');
    setFooter(`Error: ${e.message}`);
  }
}

async function enterReady() {
  await loadDocuments();
  showStage('ready');
  renderChat();
  renderSources();
  renderCenterTabs();
  if (state.billsDetected) fetchBillsStatus();
}

async function loadModel(name) {
  state.model = name;
  renderModelPill();
  renderModelMenu();
  state.busy = true;
  setFooter(`Loading model ${name}…`);
  $('indexH').textContent = `Loading ${name}…`;
  try {
    const res = await fetch(`${API_URL}/select_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: name })
    });
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      state.needModel = false;
      state.modelLoaded = true;
    } else {
      state.needModel = true;
      state.modelLoaded = false;
      setFooter(`Model error: ${data.message || 'failed to load'}`);
    }
  } catch (e) {
    state.needModel = true;
    state.modelLoaded = false;
    setFooter(`Model error: ${e.message}`);
  } finally {
    state.busy = false;
    updateInputEnabled();
    renderChat();
    renderFooter();
    renderSteps();
    $('indexH').textContent = 'Building your local index…';
  }
}

function updateInputEnabled() {
  const enabled = state.stage === 'ready' && state.modelLoaded && !state.busy;
  $('queryInput').disabled = !enabled;
  $('sendBtn').classList.toggle('disabled', !enabled);
}

/* ----------------------------- Header actions ----------------------------- */
async function doBrowse() {
  $('browseBtn').disabled = true;
  setFooter('Opening folder selection dialog…');
  try {
    const res = await fetch(`${API_URL}/browse-folder`);
    const data = await res.json();
    if (res.ok && data.status === 'success' && data.path) {
      state.folder = data.path;
      renderSteps();
      renderFooter();
    } else if (data.status === 'cancelled') {
      renderFooter();
    } else {
      throw new Error(data.message || 'Could not open folder dialog.');
    }
  } catch (e) {
    setFooter(`Folder error: ${e.message}`);
  } finally {
    $('browseBtn').disabled = false;
  }
}

async function doCleanup(toFresh) {
  state.busy = true;
  setFooter('Cleaning up…');
  try {
    await fetch(`${API_URL}/cleanup`, { method: 'POST' });
  } catch (e) {}
  state.busy = false;
  // reset session state
  state.folder = '';
  state.messages = [];
  state.activeCites = null;
  state.activeIndex = 0;
  state.docCount = 0;
  state.chunks = 0;
  state.needModel = false;
  state.modelLoaded = false;
  state.thinking = false;
  state.centerTab = 'chat';
  state.insights = null;
  state.insightsLoading = false;
  state.insightsError = null;
  state.billsDetected = false;
  state.billsJobStatus = 'idle';
  state.billsRecords = [];
  state.billsForecast = null;
  stopBillsPolling();
  $('exportBtn').disabled = true;
  $('docRows').innerHTML = '';
  renderChat();
  renderSources();
  renderCenterTabs();
  updateInputEnabled();
  showStage('fresh');
  if (!toFresh) setFooter('Cleaned up — choose a folder to begin.');
}

async function doReindex() {
  if (!state.folder) return;
  state.messages = [];
  state.activeCites = null;
  state.modelLoaded = false;
  $('exportBtn').disabled = true;
  await doInitialize();
}

async function doShutdown() {
  if (!confirm('Shut down the DartPole server? This terminates the application.')) return;
  setFooter('Shutting down…');
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 3000);
    await fetch(`${API_URL}/shutdown`, { method: 'POST', signal: ctrl.signal });
    clearTimeout(t);
  } catch (e) {}
  document.body.innerHTML =
    '<div style="height:100vh;display:flex;align-items:center;justify-content:center;font-family:\'Nunito Sans\',system-ui,sans-serif;color:#6e6e73;font-size:16px">DartPole is shutting down. You can close this tab.</div>';
}

/* ----------------------------- Export chat ----------------------------- */
function doExport() {
  if (!state.messages.length) return;
  let out = 'DartPole Chat History\n';
  out += `Downloaded: ${new Date().toLocaleString()}\n`;
  out += `Model: ${state.model || 'N/A'}\n`;
  out += `Documents: ${state.folder || 'N/A'}\n`;
  out += '='.repeat(50) + '\n\n';
  for (const m of state.messages) {
    if (m.role === 'user') { out += `User: ${m.text}\n`; continue; }
    out += `Assistant: ${m.text}\n`;
    if (m.cites && m.cites.length) {
      out += 'Citations:\n';
      m.cites.forEach(c => {
        out += `  - ${citeLabel(c)}`;
        if (c.quote) out += `: "${c.quote.trim()}"`;
        out += '\n';
      });
    }
    out += '-'.repeat(50) + '\n\n';
  }
  const blob = new Blob([out], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'DartPole_Chat_History.txt';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ----------------------------- Startup ----------------------------- */
async function pageLoad() {
  try {
    const res = await fetch(`${API_URL}/status`);
    const data = await res.json();
    if (res.ok && data.status === 'success') {
      const s = data.system_status;
      state.embedder = s.embedding_model || '—';
      state.folder = s.active_documents_directory || '';
      state.billsDetected = !!s.bills_detected;
      if (s.selected_ollama_model) state.model = s.selected_ollama_model;
      await fetchModels();

      if (s.llm_initialized) {
        state.modelLoaded = true;
        state.needModel = false;
        await enterReady();
        updateInputEnabled();
      } else if (s.docs_processed) {
        state.needModel = true;
        state.modelLoaded = false;
        await enterReady();
        updateInputEnabled();
      } else {
        state.folder = '';
        showStage('fresh');
      }
    } else {
      showStage('fresh');
    }
  } catch (e) {
    showStage('fresh');
    setFooter('Could not reach backend. Is the server running?');
  }
  renderStats();
}

/* ----------------------------- Wire up ----------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  app = $('app');

  // theme
  let theme = null;
  try { theme = localStorage.getItem('ida-theme'); } catch (e) {}
  if (!theme || !THEMES[theme]) theme = 'Light';
  applyTheme(theme);

  // pickers
  $('modelPill').addEventListener('click', (e) => { e.stopPropagation(); toggleMenu('model'); });
  $('themePill').addEventListener('click', (e) => { e.stopPropagation(); toggleMenu('theme'); });
  $('modelMenu').addEventListener('click', (e) => e.stopPropagation());
  $('themeMenu').addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', closeMenus);

  // system stats collapse
  $('sysHead').addEventListener('click', () => { state.statsOpen = !state.statsOpen; renderStatsCollapse(); });

  // fresh actions
  $('browseBtn').addEventListener('click', doBrowse);
  $('initBtn').addEventListener('click', doInitialize);

  // header actions
  $('cleanupBtn').addEventListener('click', () => doCleanup(false));
  $('shutdownBtn').addEventListener('click', doShutdown);

  // ready actions
  $('reindexBtn').addEventListener('click', doReindex);
  $('changeFolderBtn').addEventListener('click', () => doCleanup(true));
  const submitQuery = () => { const v = $('queryInput').value; $('queryInput').value = ''; ask(v); };
  $('sendBtn').addEventListener('click', submitQuery);
  $('queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submitQuery(); }
  });
  $('exportBtn').addEventListener('click', doExport);

  // insights
  $('tabChat').addEventListener('click', () => switchCenterTab('chat'));
  $('tabInsights').addEventListener('click', () => switchCenterTab('insights'));
  $('insightsRefresh').addEventListener('click', () => fetchInsights(true));
  $('tabBills').addEventListener('click', () => switchCenterTab('bills'));
  $('billsScanBtn').addEventListener('click', startBillsScan);

  renderStatsCollapse();
  renderModelPill();
  renderModelMenu();
  updateInputEnabled();
  startStats();
  pageLoad();
});
