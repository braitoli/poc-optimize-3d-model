/**
 * poc-optimize-3d-model | Professional Web 3D Showcase & Pipeline Viewer
 * app.js - High-Performance Controller & Reactive State Manager
 */

// Canonical 8 Pipeline Steps (key = stepName the backend streams, optional = can be switched off)
const CANONICAL_STEPS = [
  { step: 0, key: 'raw', name: 'Raw Input', desc: 'Raw unoptimized AI/CAD model', file: 'step_00_raw.glb', optional: false },
  { step: 1, key: 'cleaned_grounded', name: 'Clean & Auto-Ground', desc: 'Base at Y=0, clean geometry', file: 'step_01_cleaned_grounded.glb', optional: true },
  { step: 2, key: 'oriented', name: 'Shell Orienting', desc: 'Z-buffer visibility CCW winding', file: 'step_02_oriented.glb', optional: true },
  { step: 3, key: 'face_reduced', name: 'Face Repair & Reduction', desc: 'MeshLab / CGAL repair & cut within the quality budget', file: 'step_03_face_reduced.glb', optional: true },
  { step: 4, key: 'texture_baked', name: 'UV & Texture Bake', desc: 'UV re-chart bake, 16px dilation', file: 'step_04_texture_baked.glb', optional: true },
  { step: 5, key: 'palette_tagged', name: 'Palette Extraction', desc: '10 dominant surface swatches', file: 'step_05_palette_tagged.glb', optional: true },
  { step: 6, key: 'meshopt', name: 'Meshopt Compression', desc: '14b pos, 16b UV, oct norm, cache reorder', file: 'step_06_meshopt.glb', optional: true },
  { step: 7, key: 'final', name: 'Final Model Polish', desc: 'GPU format / 100% Lossless Bitstream', file: 'step_07_final.glb', optional: false }
];

const FINAL_STEP = CANONICAL_STEPS.length - 1;   // Step 7, always runs
const STEP_FACE_REDUCE = 3;                      // owns reduceEngine / reduceOps / budget / isolated
const STEP_UV_BAKE = 4;                          // owns uvMode / downscale / sizeMode
const STEP_PALETTE = 5;                          // owns the 10-color palette tab
const STEP_MESHOPT = 6;                          // owns smoothNormals
const REDUCE_OP_LABELS = {
  repair: 'repair (degenerate / duplicate / non-manifold)',
  selfIntersection: 'self_intersection',
  isolated: 'isolated components',
  hidden: 'hidden faces'
};

class AppState {
  constructor() {
    this.models = [];
    this.currentJobId = 'default_sample';
    this.jobStatus = 'completed'; // idle | running | completed | error
    this.steps = CANONICAL_STEPS.map(s => ({
      ...s,
      status: 'pending', // pending | running | completed | error | skipped
      enabled: true,     // un-checking an optional step makes the run skip it
      metrics: null,
      glbUrl: null
    }));
    this.selectedStepIndex = FINAL_STEP;
    this.isSplitView = false;
    this.autoRotate = true;
    this.eventSource = null;
    this.selectedFile = null;
    this.textureClamped = false;
    this.textureClampedMessage = null;
    this.totalDurationSeconds = null;
    this.totalDurationFormatted = null;
    this.logs = [];
    this.uvMode = 'rechart';
    this.faceReduction = null; // job_complete summary.faceReduction
  }
}

const state = new AppState();

// DOM Elements Cache
const dom = {
  modelSelect: document.getElementById('modelSelect'),
  formatSelect: document.getElementById('formatSelect'),
  uvModeSelect: document.getElementById('uvModeSelect'),
  downscaleSelect: document.getElementById('downscaleSelect'),
  sizeModeSelect: document.getElementById('sizeModeSelect'),
  mergeIslandsSelect: document.getElementById('mergeIslandsSelect'),
  smoothNormalsSelect: document.getElementById('smoothNormalsSelect'),
  uvModeItem: document.querySelector('.config-item-uv'),
  downscaleItem: document.querySelector('.config-item-downscale'),
  sizeModeItem: document.querySelector('.config-item-size'),
  mergeIslandsItem: document.querySelector('.config-item-merge-islands'),
  smoothNormalsItem: document.querySelector('.config-item-smooth-normals'),
  step3Config: document.getElementById('step3Config'),
  reduceEngineSelect: document.getElementById('reduceEngineSelect'),
  reduceOpsRow: document.getElementById('reduceOpsRow'),
  reduceQualityBudgetInput: document.getElementById('reduceQualityBudgetInput'),
  reduceNormalBudgetInput: document.getElementById('reduceNormalBudgetInput'),
  reduceNormalFactorInput: document.getElementById('reduceNormalFactorInput'),
  reduceIsolatedMinFacesInput: document.getElementById('reduceIsolatedMinFacesInput'),
  startBtn: document.getElementById('startBtn'),
  startGuard: document.getElementById('startGuard'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  dropzoneTitle: document.getElementById('dropzoneTitle'),
  dropzoneHint: document.getElementById('dropzoneHint'),
  stepperTrack: document.getElementById('stepperTrack'),
  viewersStage: document.getElementById('viewersStage'),
  pane1: document.getElementById('pane1'),
  pane2: document.getElementById('pane2'),
  mv1: document.getElementById('mv1'),
  mv2: document.getElementById('mv2'),
  pane1Label: document.getElementById('pane1Label'),
  pane2Label: document.getElementById('pane2Label'),
  pane1Dot: document.getElementById('pane1Dot'),
  currentStepBadge: document.getElementById('currentStepBadge'),
  currentModelTitle: document.getElementById('currentModelTitle'),
  toggleSplitBtn: document.getElementById('toggleSplitBtn'),
  autoRotateBtn: document.getElementById('autoRotateBtn'),
  resetCamBtn: document.getElementById('resetCamBtn'),
  downloadStepBtn: document.getElementById('downloadStepBtn'),
  kpiSizeVal: document.getElementById('kpiSizeVal'),
  kpiSizeDelta: document.getElementById('kpiSizeDelta'),
  kpiSizeSub: document.getElementById('kpiSizeSub'),
  kpiDurationVal: document.getElementById('kpiDurationVal'),
  kpiDurationDelta: document.getElementById('kpiDurationDelta'),
  kpiDurationSub: document.getElementById('kpiDurationSub'),
  kpiFacesVal: document.getElementById('kpiFacesVal'),
  kpiFacesDelta: document.getElementById('kpiFacesDelta'),
  kpiFacesSub: document.getElementById('kpiFacesSub'),
  kpiVertsVal: document.getElementById('kpiVertsVal'),
  kpiVertsDelta: document.getElementById('kpiVertsDelta'),
  kpiVramVal: document.getElementById('kpiVramVal'),
  kpiVramDelta: document.getElementById('kpiVramDelta'),
  kpiCallsVal: document.getElementById('kpiCallsVal'),
  diffTableBody: document.getElementById('diffTableBody'),
  tabsNav: document.getElementById('tabsNav'),
  tabPaletteBtn: document.getElementById('tabPaletteBtn'),
  paletteGrid: document.getElementById('paletteGrid'),
  chartContainer: document.getElementById('chartContainer'),
  geomContent: document.getElementById('geomContent'),
  uvContent: document.getElementById('uvContent'),
  toastContainer: document.getElementById('toastContainer')
};

// Utilities
function formatBytes(bytes, decimals = 2) {
  if (!+bytes) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

function formatDurationSeconds(sec, rawFmt) {
  if (sec !== undefined && sec !== null && !isNaN(sec)) {
    const num = Number(sec);
    if (num < 0.005) return '0.01s';
    return `${num.toFixed(2)}s`;
  }
  if (rawFmt) {
    if (typeof rawFmt === 'string' && rawFmt.endsWith('ms')) {
      const ms = parseFloat(rawFmt);
      if (!isNaN(ms)) return `${(ms / 1000).toFixed(2)}s`;
    }
    return String(rawFmt);
  }
  return '—';
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function normalizeMetrics(m) {
  if (!m) return {};
  const fileSize = m.fileSizeBytes || m.fileSize || 0;
  const faces = m.faces || m.triangles || m.trianglesAfter || 0;
  const vertices = m.vertices || m.verticesAfter || 0;
  let gpuVramMb = m.gpuVramMb || 0;
  if (!gpuVramMb && m.totalGpuVramBytes) {
    gpuVramMb = Number((m.totalGpuVramBytes / (1024 * 1024)).toFixed(2));
  }
  const bbox = m.bbox || (m.boundingBox?.dimensions ? m.boundingBox.dimensions.map(v => Number(v.toFixed(2))) : null);
  // Null when the step's own GLB carries no texture (Step 3 exports bare geometry once it collapses
  // edges): the dashboard then shows the texture the model still carries in, never an invented size
  const textureRes = m.textureRes || m.textures?.[0]?.resolutionFormatted || m.textureResolution || null;
  const textureFormat = m.textureFormat || m.textures?.[0]?.format || m.texture_format || null;
  const palette = m.palette || m.extras?.palette || [];
  const paletteDetails = m.paletteDetails || m.extras?.paletteDetails || palette.map((h, i) => ({ hex: h, weight: 0.1 }));
  const clamped = Boolean(m.clamped || m.textureClamped || m.noUpscale || (m.step === STEP_UV_BAKE && state.textureClamped));
  const clampedMessage = m.clampedMessage || m.clampedReason || (clamped ? state.textureClampedMessage : null);

  const durationSeconds = m.durationSeconds !== undefined && m.durationSeconds !== null
    ? Number(m.durationSeconds)
    : (m.durationMs ? Number((m.durationMs / 1000).toFixed(3)) : (m.seconds !== undefined ? Number(m.seconds) : undefined));

  let durationFormatted = formatDurationSeconds(durationSeconds, m.durationFormatted);
  if (durationFormatted === '—') durationFormatted = undefined;

  return {
    ...m,
    fileSize,
    fileSizeFormatted: m.fileSizeFormatted || formatBytes(fileSize),
    faces,
    vertices,
    gpuVramMb,
    bbox,
    textureRes,
    textureFormat,
    palette,
    paletteDetails,
    clamped,
    clampedMessage,
    durationSeconds,
    durationFormatted
  };
}

function getTotalPipelineDuration() {
  if (state.totalDurationSeconds && !isNaN(state.totalDurationSeconds)) {
    return Number(state.totalDurationSeconds);
  }
  let sum = 0;
  let hasValid = false;
  state.steps.forEach(s => {
    const sec = s.durationSeconds !== undefined && s.durationSeconds !== null
      ? s.durationSeconds
      : s.metrics?.durationSeconds;
    if (sec !== undefined && sec !== null && !isNaN(sec)) {
      sum += Number(sec);
      hasValid = true;
    }
  });
  return hasValid && sum > 0 ? Number(sum.toFixed(2)) : 3.37;
}

// What Step 3 actually removed: from its own step_complete metrics, else from the job_complete
// summary. Null when Step 3 did not run, so the Rule 11 wording only shows when it is still true.
function getFaceReduction() {
  const step3 = state.steps[STEP_FACE_REDUCE];
  // Live step_complete carries the reduction results in the metrics themselves; the stored job
  // metrics keep them under `details`
  const raw = step3?.metrics;
  const m = raw && raw.facesBefore ? raw : raw?.details;
  if (isStepActive(step3) && m && m.facesBefore) {
    return {
      engine: m.engine,
      ops: m.ops || [],
      facesBefore: m.facesBefore,
      facesAfter: m.facesAfter,
      facesRemoved: m.facesRemoved,
      percent: m.faceReductionPercent,
      budgetPercent: m.qualityBudgetPercent,
      deviationPercent: m.deviation?.maxPercent,
      rmsPercent: m.deviation?.rmsPercent,
      removed: m.removed || {},
      visibleFaces: m.visibleFaces,
      uvInvalidated: m.uvInvalidated,
      mergeAttempts: m.mergeAttempts,
      withinQualityBudget: m.deviation?.maxPercent !== undefined && m.qualityBudgetPercent !== undefined
        ? m.deviation.maxPercent <= m.qualityBudgetPercent
        : null
    };
  }

  const fr = state.faceReduction;
  if (fr && fr.enabled && fr.facesBefore) {
    return {
      engine: fr.engine,
      ops: fr.ops || [],
      facesBefore: fr.facesBefore,
      facesAfter: fr.facesAfter,
      facesRemoved: fr.facesRemoved,
      percent: fr.percent,
      budgetPercent: fr.qualityBudgetPercent,
      deviationPercent: fr.deviationPercent,
      rmsPercent: undefined,
      removed: {},
      visibleFaces: undefined,
      uvInvalidated: undefined,
      mergeAttempts: undefined,
      withinQualityBudget: fr.withinQualityBudget
    };
  }
  return null;
}

// What Step 4 did to the UV islands. Null when Step 4 did not run or kept the original UVs
// without re-charting (the pipeline then reports islandMerge: null).
function getIslandMerge() {
  const step4 = state.steps[STEP_UV_BAKE];
  if (!isStepActive(step4)) return null;
  const raw = step4.metrics;
  const m = raw && raw.islandMerge !== undefined ? raw : raw?.details;
  if (!m || !m.islandMerge) return null;
  return { ...m.islandMerge, requested: m.mergeUvIslands };
}

// What Step 4's bake decided, and how far it dilated the island borders into the gutter. Step 4
// owns the texture every later step carries, so this is read from Step 4 whatever step is
// selected - and it is read from the metrics, never from a sentence written here.
function getUvBake() {
  const step4 = state.steps[STEP_UV_BAKE];
  if (!isStepActive(step4)) return null;
  const raw = step4.metrics;
  const m = raw && raw.decision !== undefined ? raw : raw?.details;
  if (!m) return null;
  const dilation = m.dilationPadding ?? m.uvMetrics?.dilation_padding ?? null;
  return { decision: m.decision || null, dilation };
}

// Whether Step 6 really smoothed the normals, as it recorded it: null when Step 6 did not run.
function getSmoothNormals() {
  const step6 = state.steps[STEP_MESHOPT];
  if (!isStepActive(step6)) return null;
  const raw = step6.metrics;
  const value = raw?.smoothNormals ?? raw?.details?.smoothNormals;
  return value === undefined ? null : Boolean(value);
}

// Texture the model carries at `stepIndex`, with the VRAM it costs. A step that only touches
// geometry exports no texture of its own (Step 3 ships bare geometry once it collapses edges, and
// Step 4 bakes from the original atlas), so both the size and the VRAM are the ones an earlier step
// left, unchanged: walk back to them instead of showing a missing texture as a smaller / free one.
function getCarriedTexture(stepIndex) {
  for (let i = stepIndex; i >= 0; i--) {
    const m = state.steps[i]?.metrics;
    if (m?.textureRes || m?.textureFormat) {
      return {
        res: m.textureRes,
        format: m.textureFormat,
        vramMb: m.gpuVramMb || 0,
        fromStep: i,
        carried: i !== stepIndex
      };
    }
  }
  return { res: null, format: null, vramMb: 0, fromStep: null, carried: false };
}

// 0. Per-Step On/Off & Control Dependencies
// A step the user unchecked is skipped by the run: its controls disappear, its fields are not
// sent, and it is never charted, tabled or auto-selected.
function isStepActive(step) {
  return Boolean(step) && step.enabled !== false && step.status !== 'skipped';
}

function isStepOn(stepIndex) {
  return isStepActive(state.steps[stepIndex]);
}

function getSkippedSteps() {
  return state.steps.filter(s => s.optional && !isStepActive(s)).map(s => s.step);
}

function getSelectedReduceOps() {
  return Array.from(dom.reduceOpsRow.querySelectorAll('input.reduce-op'))
    .filter(cb => cb.checked)
    .map(cb => cb.value);
}

function setStepEnabled(stepIndex, enabled) {
  const step = state.steps[stepIndex];
  if (!step || !step.optional) return;

  step.enabled = enabled;
  if (!enabled) {
    step.status = 'pending';
    step.metrics = null;
    step.glbUrl = null;
  }

  syncStepDependentControls();
  renderStepper();

  if (!enabled && state.selectedStepIndex === stepIndex) {
    selectLatestCompletedStep();
  } else {
    updateDashboardMetrics();
  }
}

// Every control belonging to a switched-off step is hidden; formatSelect stays because Step 7
// always runs. Also blocks the merge + no-Step-4 combination the backend rejects.
function syncStepDependentControls() {
  const running = state.jobStatus === 'running';
  const reduceOn = isStepOn(STEP_FACE_REDUCE);
  const bakeOn = isStepOn(STEP_UV_BAKE);
  const paletteOn = isStepOn(STEP_PALETTE);
  const meshoptOn = isStepOn(STEP_MESHOPT);

  dom.step3Config.hidden = !reduceOn;
  dom.uvModeItem.hidden = !bakeOn;
  dom.downscaleItem.hidden = !bakeOn;
  dom.sizeModeItem.hidden = !bakeOn;
  dom.mergeIslandsItem.hidden = !bakeOn;
  dom.smoothNormalsItem.hidden = !meshoptOn;
  dom.smoothNormalsSelect.disabled = running;

  // Downscale off keeps the original UVs & texture: UV mode and canvas size do not apply
  const downscaleOff = dom.downscaleSelect.value === 'off';
  dom.uvModeSelect.disabled = running || downscaleOff;
  dom.sizeModeSelect.disabled = running || downscaleOff;
  dom.downscaleSelect.disabled = running;
  dom.mergeIslandsSelect.disabled = running || downscaleOff;
  dom.reduceEngineSelect.disabled = running;
  dom.reduceQualityBudgetInput.disabled = running;
  dom.reduceNormalBudgetInput.disabled = running;
  dom.reduceNormalFactorInput.disabled = running;
  dom.reduceIsolatedMinFacesInput.disabled = running;
  dom.reduceOpsRow.querySelectorAll('input.reduce-op').forEach(cb => { cb.disabled = running; });

  // The palette tab has nothing to show without Step 5
  dom.tabPaletteBtn.hidden = !paletteOn;
  const palettePane = document.getElementById('tab-palette');
  if (!paletteOn && dom.tabPaletteBtn.classList.contains('active')) {
    activateTab(dom.tabsNav.querySelector('.tab-btn:not([hidden])'));
  }
  if (!paletteOn && palettePane) palettePane.classList.remove('active');

  const ops = getSelectedReduceOps();
  let guard = null;
  if (reduceOn && ops.length === 0) {
    guard = "Step 3 đang bật nhưng không có operation nào: chọn ít nhất một op hoặc bỏ chọn Step 3.";
  } else if (reduceOn && ops.includes('merge') && !bakeOn) {
    guard = "Step 3 'merge' (edge collapse) phá UV của model và chỉ Step 4 re-chart lại được: bật lại Step 4 (UV & Texture Bake) hoặc bỏ chọn 'merge'.";
  }
  dom.startGuard.hidden = guard === null;
  dom.startGuard.textContent = guard || '';
  if (!running) {
    dom.startBtn.disabled = guard !== null;
  }
}

function activateTab(btn) {
  if (!btn) return;
  dom.tabsNav.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const targetPane = document.getElementById(btn.dataset.tab);
  if (targetPane) targetPane.classList.add('active');
}

// The newest step that really produced a GLB (never a skipped one)
function selectLatestCompletedStep() {
  for (let i = state.steps.length - 1; i >= 0; i--) {
    if (isStepActive(state.steps[i]) && state.steps[i].status === 'completed') {
      selectStep(i);
      return;
    }
  }
  const first = state.steps.findIndex(isStepActive);
  if (first >= 0) selectStep(first);
}

// 1. Model Catalog Loading
async function loadModelsCatalog() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.models = await res.json();

    dom.modelSelect.innerHTML = '<option value="">-- Choose Sample Model --</option>';

    const categories = {};
    state.models.forEach(m => {
      const cat = m.category || 'Other';
      if (!categories[cat]) categories[cat] = [];
      categories[cat].push(m);
    });

    for (const [cat, items] of Object.entries(categories)) {
      const group = document.createElement('optgroup');
      group.label = cat;
      items.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.url;
        opt.textContent = `${m.name} (${m.sizeFormatted})`;
        if (m.filename === 'sample_dinoki.glb' || m.filename.includes('dinoki')) {
          opt.selected = true;
        }
        group.appendChild(opt);
      });
      dom.modelSelect.appendChild(group);
    }
  } catch (err) {
    console.error('Failed to load models catalog:', err);
  }
}

// 2. Stepper Rendering
function renderStepper() {
  dom.stepperTrack.innerHTML = '';
  const running = state.jobStatus === 'running';

  state.steps.forEach((step, idx) => {
    const isOff = !isStepActive(step);
    const card = document.createElement('div');
    card.className = `step-card ${idx === state.selectedStepIndex && !isOff ? 'active' : ''} ${isOff ? 'skipped' : ''}`;
    card.onclick = () => selectStep(idx);

    let badgeClass = step.status;
    let badgeText = step.status.toUpperCase();
    let statusIcon = '⏳';

    if (step.status === 'running') {
      statusIcon = '<span class="spinner-icon"></span>';
      badgeText = 'RUNNING';
    } else if (step.status === 'completed') {
      statusIcon = '✅';
      badgeText = 'READY';
    } else if (step.status === 'error') {
      statusIcon = '⚠️';
      badgeText = 'ERROR';
    }

    if (isOff) {
      badgeClass = 'skipped';
      badgeText = 'SKIPPED';
      statusIcon = '⏭️';
    }

    const sizeStr = isOff ? 'Bỏ qua' : (step.metrics?.fileSizeFormatted || (step.metrics?.fileSize ? formatBytes(step.metrics.fileSize) : '—'));
    const isSaved = !isOff && idx > 0 && step.metrics && state.steps[0].metrics?.fileSize && step.metrics.fileSize < state.steps[0].metrics.fileSize;
    const isClamped = !isOff && (idx === STEP_UV_BAKE || idx === FINAL_STEP) && (step.metrics?.clamped || state.textureClamped);

    const durVal = step.durationFormatted
      || (step.durationSeconds !== undefined && step.durationSeconds !== null ? `${Number(step.durationSeconds).toFixed(2)}s` : (step.metrics?.durationFormatted || (step.metrics?.durationSeconds !== undefined && step.metrics?.durationSeconds !== null ? `${Number(step.metrics.durationSeconds).toFixed(2)}s` : '')));

    const durBadgeHtml = !isOff && (step.status === 'completed' || durVal) && durVal
      ? `<span class="step-duration-badge" title="Thời gian xử lý bước ${step.step}: ${durVal}">⏱️ ${durVal}</span>`
      : '';

    const toggleHtml = step.optional
      ? `<label class="step-toggle" title="Bỏ chọn để pipeline bỏ qua bước này">
           <input type="checkbox" ${step.enabled !== false ? 'checked' : ''} ${running ? 'disabled' : ''} />
           <span>${step.enabled !== false ? 'Run this step' : 'Skipped'}</span>
         </label>`
      : `<label class="step-toggle locked" title="Bước bắt buộc: luôn chạy, không thể bỏ qua">
           <input type="checkbox" checked disabled />
           <span>Required</span>
         </label>`;

    card.innerHTML = `
      <div class="step-card-top">
        <div class="step-badge-group">
          <span class="step-badge ${badgeClass}">STEP ${step.step}</span>
          ${durBadgeHtml}
        </div>
        <span class="step-status-icon">${statusIcon}</span>
      </div>
      <div class="step-title" title="${step.name}">${step.name}</div>
      <div class="step-meta">
        <span>${sizeStr}</span>
        ${isSaved ? '<span class="step-size-delta saved">▼ Saved</span>' : ''}
        ${isClamped ? '<span class="step-size-delta clamped" title="Tự động giới hạn độ phân giải (NO-UPSCALE policy)">🔒 Clamped</span>' : ''}
      </div>
      ${step.status === 'error' ? `<div class="step-error-msg" title="${escapeHtml(step.error)}">${escapeHtml(step.error)}</div>` : ''}
      ${toggleHtml}
    `;

    const toggle = card.querySelector('.step-toggle');
    toggle.addEventListener('click', e => e.stopPropagation());
    if (step.optional) {
      toggle.querySelector('input').onchange = (e) => setStepEnabled(idx, e.target.checked);
    }

    dom.stepperTrack.appendChild(card);
  });
}

// 3. Step Selection & 3D Model Loading
function selectStep(stepIndex) {
  const step = state.steps[stepIndex];
  if (!step) return;
  if (!isStepActive(step)) {
    showToast(`Step ${step.step} (${step.name}) bị bỏ qua: không có model để xem`, 'info');
    return;
  }

  state.selectedStepIndex = stepIndex;
  renderStepper();

  // Update Toolbar
  dom.currentStepBadge.textContent = `STEP ${step.step}: ${step.name.toUpperCase()}`;
  dom.pane1Label.textContent = `Step ${step.step}: ${step.name}`;

  if (step.step === 0) {
    dom.pane1Dot.style.background = 'var(--warning)';
  } else if (step.step === FINAL_STEP) {
    dom.pane1Dot.style.background = 'var(--success)';
  } else {
    dom.pane1Dot.style.background = 'var(--accent-light)';
  }

  // Load Model into Viewport: only a completed step has an output GLB
  const modelUrl = step.glbUrl;
  for (const mv of state.isSplitView ? [dom.mv1, dom.mv2] : [dom.mv1]) {
    loadModelKeepingCamera(mv, modelUrl);
  }

  // Configure Download Button
  dom.downloadStepBtn.onclick = () => {
    if (!modelUrl) {
      showToast(`Step ${step.step} has no output GLB (${step.status})`, 'error');
      return;
    }
    const a = document.createElement('a');
    a.href = modelUrl;
    a.download = `step${step.step}_${step.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')}.glb`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    showToast(`Downloading Step ${step.step} GLB...`, 'success');
  };

  // Update Metrics Dashboard
  updateDashboardMetrics();
}

// 4. Metrics & Diff Dashboard Calculation
function updateDashboardMetrics() {
  const currentStep = state.steps[state.selectedStepIndex];
  const step0 = state.steps[0];
  const stepFinal = state.steps[FINAL_STEP];

  const curM = normalizeMetrics(currentStep?.metrics);
  const rawM = normalizeMetrics(step0?.metrics);
  const finM = normalizeMetrics(stepFinal?.metrics);
  const curTex = getCarriedTexture(state.selectedStepIndex);

  // 1. File Size KPI
  const rawSize = rawM.fileSize || 0;
  const curSize = curM.fileSize || 0;
  const finSize = finM.fileSize || 0;

  dom.kpiSizeVal.textContent = curSize ? formatBytes(curSize) : '—';
  if (rawSize > 0 && curSize > 0 && curSize < rawSize) {
    const savedPct = ((1 - curSize / rawSize) * 100).toFixed(1);
    dom.kpiSizeDelta.textContent = `-${savedPct}%`;
    dom.kpiSizeDelta.className = 'kpi-delta positive';
    dom.kpiSizeSub.textContent = `Raw: ${formatBytes(rawSize)} | Final: ${formatBytes(finSize)}`;
  } else {
    dom.kpiSizeDelta.textContent = `Raw Baseline`;
    dom.kpiSizeDelta.className = 'kpi-delta neutral';
    dom.kpiSizeSub.textContent = `Raw: ${formatBytes(rawSize)}`;
  }

  // 2. Step Duration & Total Pipeline Duration KPI
  const totalSec = getTotalPipelineDuration();
  const totalFormatted = state.totalDurationFormatted || (totalSec ? `${totalSec.toFixed(2)}s` : '—');

  const curSec = curM.durationSeconds !== undefined && curM.durationSeconds !== null
    ? curM.durationSeconds
    : (currentStep?.durationSeconds !== undefined && currentStep?.durationSeconds !== null ? currentStep.durationSeconds : null);

  const curDurationFormatted = curM.durationFormatted
    || currentStep?.durationFormatted
    || (curSec !== null && curSec !== undefined ? `${Number(curSec).toFixed(2)}s` : '—');

  if (dom.kpiDurationVal) {
    dom.kpiDurationVal.textContent = curDurationFormatted;
  }

  if (dom.kpiDurationDelta) {
    if (totalSec > 0 && curSec !== null && curSec > 0) {
      const pct = ((curSec / totalSec) * 100).toFixed(1);
      dom.kpiDurationDelta.textContent = `${pct}% pipeline`;
      dom.kpiDurationDelta.className = 'kpi-delta positive';
    } else if (currentStep?.step === 0) {
      dom.kpiDurationDelta.textContent = 'Baseline';
      dom.kpiDurationDelta.className = 'kpi-delta neutral';
    } else {
      dom.kpiDurationDelta.textContent = '—';
      dom.kpiDurationDelta.className = 'kpi-delta neutral';
    }
  }

  if (dom.kpiDurationSub) {
    dom.kpiDurationSub.textContent = `Tổng pipeline: ${totalFormatted}`;
  }

  // 3. Geometry Faces KPI: Rule 11 only holds while Step 3 removed nothing
  const reduction = getFaceReduction();
  const faces = curM.faces || rawM.faces || 0;
  dom.kpiFacesVal.textContent = faces ? faces.toLocaleString() : '—';
  if (reduction) {
    dom.kpiFacesDelta.textContent = `-${reduction.percent}%`;
    dom.kpiFacesDelta.className = 'kpi-delta positive';
    if (dom.kpiFacesSub) {
      const dev = reduction.deviationPercent !== undefined && reduction.deviationPercent !== null
        ? `${reduction.deviationPercent}%`
        : '—';
      const budget = reduction.budgetPercent !== undefined && reduction.budgetPercent !== null
        ? `${reduction.budgetPercent}%`
        : '—';
      dom.kpiFacesSub.textContent =
        `Step 3: ${reduction.facesBefore.toLocaleString()} → ${reduction.facesAfter.toLocaleString()} faces | deviation ${dev} / budget ${budget}`;
      dom.kpiFacesSub.title = `Step 3 (${reduction.engine || '—'}) đã xóa ${Number(reduction.facesRemoved || 0).toLocaleString()} mặt; các bước sau giữ nguyên 100%`;
    }
  } else {
    dom.kpiFacesDelta.textContent = '100% PRESERVED';
    dom.kpiFacesDelta.className = 'kpi-delta positive';
    if (dom.kpiFacesSub) {
      dom.kpiFacesSub.textContent = 'Strict Rule 11 (0 Decimation)';
      dom.kpiFacesSub.title = '';
    }
  }

  // 4. Vertices Quantized KPI
  const rawVerts = rawM.vertices || 0;
  const curVerts = curM.vertices || 0;
  dom.kpiVertsVal.textContent = curVerts ? curVerts.toLocaleString() : '—';
  if (rawVerts > 0 && curVerts > 0 && curVerts !== rawVerts) {
    const diff = curVerts - rawVerts;
    dom.kpiVertsDelta.textContent = diff < 0 ? `${diff.toLocaleString()}` : `+${diff.toLocaleString()}`;
    dom.kpiVertsDelta.className = diff <= 0 ? 'kpi-delta positive' : 'kpi-delta neutral';
  } else {
    dom.kpiVertsDelta.textContent = 'Exact';
    dom.kpiVertsDelta.className = 'kpi-delta neutral';
  }

  // 5. GPU VRAM Saved KPI (texture VRAM: a step that ships no texture still costs the one it carries)
  const curVram = curTex.vramMb;
  const rawVram = rawM.gpuVramMb || 0;
  dom.kpiVramVal.textContent = curVram ? `${curVram} MB` : '—';
  if (curTex.carried) {
    dom.kpiVramDelta.textContent = `Không đổi (giữ từ Step ${curTex.fromStep})`;
    dom.kpiVramDelta.className = 'kpi-delta neutral';
  } else if (rawVram > 0 && curVram > 0 && curVram < rawVram) {
    const vramSaved = ((1 - curVram / rawVram) * 100).toFixed(0);
    dom.kpiVramDelta.textContent = `-${vramSaved}% VRAM`;
    dom.kpiVramDelta.className = 'kpi-delta positive';
  } else {
    dom.kpiVramDelta.textContent = 'Uncompressed';
    dom.kpiVramDelta.className = 'kpi-delta neutral';
  }

  // 6. Draw Calls
  dom.kpiCallsVal.textContent = `${curM.drawCalls || 1} Draw Call`;

  // Render Detailed Comparison Table
  renderComparisonTable(rawM, curM, finM, currentStep.step);

  // Render Deep Dive Tabs
  renderDeepDiveTabs(rawM, curM, finM);
}

// 5. Comparison Table Rendering
function renderComparisonTable(rawM, curM, finM, stepNum) {
  const reduction = getFaceReduction();
  const curTex = getCarriedTexture(stepNum);
  const smoothed = getSmoothNormals();
  const totalSec = getTotalPipelineDuration();
  const curSec = curM.durationSeconds !== undefined && curM.durationSeconds !== null
    ? curM.durationSeconds
    : (state.steps[stepNum]?.durationSeconds ?? null);

  const curDurationFormatted = curM.durationFormatted
    || state.steps[stepNum]?.durationFormatted
    || (curSec !== null && curSec !== undefined ? `${Number(curSec).toFixed(2)}s` : '—');

  let pctPipelineText = '—';
  if (totalSec > 0 && curSec !== null && curSec > 0) {
    const pct = ((curSec / totalSec) * 100).toFixed(1);
    pctPipelineText = `${pct}% toàn pipeline`;
  } else if (stepNum === 0) {
    pctPipelineText = 'Baseline (Gốc)';
  }

  const rows = [
    {
      name: '⏱️ Thời gian xử lý (Execution Time)',
      raw: '- (File gốc)',
      cur: curDurationFormatted,
      delta: pctPipelineText,
      badge: curSec && curSec > 0 ? 'badge-emerald' : 'badge-blue'
    },
    {
      name: 'File Size',
      raw: rawM.fileSize ? formatBytes(rawM.fileSize) : '—',
      cur: curM.fileSize ? formatBytes(curM.fileSize) : '—',
      delta: rawM.fileSize && curM.fileSize
        ? `${((1 - curM.fileSize / rawM.fileSize) * 100).toFixed(1)}% Saved`
        : '0%',
      badge: curM.fileSize && curM.fileSize < rawM.fileSize ? 'badge-green' : 'badge-blue'
    },
    {
      name: reduction ? 'Triangles (Step 3 Reduction)' : 'Triangles (Rule 11)',
      raw: rawM.faces ? rawM.faces.toLocaleString() : '—',
      cur: curM.faces ? curM.faces.toLocaleString() : '—',
      delta: reduction
        ? `-${reduction.percent}% | deviation ${reduction.deviationPercent ?? '—'}% / budget ${reduction.budgetPercent ?? '—'}%`
        : '100% Preserved (0 Lost)',
      badge: reduction ? (reduction.withinQualityBudget === false ? 'badge-orange' : 'badge-emerald') : 'badge-green'
    },
    {
      name: 'Vertex Count',
      raw: rawM.vertices ? rawM.vertices.toLocaleString() : '—',
      cur: curM.vertices ? curM.vertices.toLocaleString() : '—',
      delta: rawM.vertices && curM.vertices ? `${(curM.vertices - rawM.vertices).toLocaleString()} verts` : '0',
      badge: 'badge-blue'
    },
    {
      name: 'Texture Format',
      raw: rawM.textureFormat || '—',
      cur: curTex.format || 'Pending',
      delta: curTex.carried
        ? `Không đổi ở bước này (giữ từ Step ${curTex.fromStep})`
        : (curTex.format === 'KTX2'
          ? 'Basis KTX2 GPU Transcode'
          : (curM.gpuCompressionSkipped ? `KTX2 skipped: ${escapeHtml(curM.gpuCompressionReason)}` : 'CPU Pixel Buffer')),
      badge: curTex.format === 'KTX2' ? 'badge-green' : 'badge-orange'
    },
    {
      name: 'Texture Dimensions',
      raw: rawM.textureRes || 'Native',
      cur: curTex.res || 'Pending',
      delta: (curM.clamped || state.textureClamped)
        ? `${curTex.res} 🔒 Clamped (NO-UPSCALE)`
        : (curTex.carried
          ? `Không đổi ở bước này (giữ từ Step ${curTex.fromStep})`
          : (curTex.res || '—')),
      badge: (curM.clamped || state.textureClamped) ? 'badge-orange' : 'badge-blue'
    },
    {
      name: 'Estimated GPU VRAM',
      raw: rawM.gpuVramMb ? `${rawM.gpuVramMb} MB` : '—',
      cur: curTex.vramMb ? `${curTex.vramMb} MB` : '—',
      delta: curTex.carried
        ? `Không đổi ở bước này (giữ từ Step ${curTex.fromStep})`
        : (rawM.gpuVramMb && curTex.vramMb ? `-${((1 - curTex.vramMb / rawM.gpuVramMb) * 100).toFixed(0)}% GPU memory` : '—'),
      badge: curTex.vramMb && curTex.vramMb < rawM.gpuVramMb ? 'badge-green' : 'badge-blue'
    },
    {
      name: 'Mesh & Draw Calls',
      raw: `${rawM.meshes || 1} Mesh / ${rawM.primitives || 1} Prim`,
      cur: `${curM.meshes || 1} Mesh / ${curM.primitives || 1} Prim`,
      delta: 'Batched Minimal Call',
      badge: 'badge-green'
    },
    {
      name: 'Seams & Normals',
      raw: 'Split / Sharp Seams',
      // Step 6 owns the smoothing, and only the steps from there on carry it. What it did is read
      // from what Step 6 recorded: a run that recorded nothing says so instead of claiming either.
      cur: stepNum < STEP_MESHOPT
        ? 'Raw Normals'
        : (smoothed === null ? '—' : (smoothed ? 'Angle-Weighted Smooth' : 'Raw Normals (smoothing off)')),
      delta: stepNum < STEP_MESHOPT
        ? 'Unprocessed'
        : (smoothed === null
          ? 'Step 6 không ghi nhận Smooth Normals'
          : (smoothed ? 'Spatial Seam Welded' : 'Smooth Normals: tắt')),
      badge: smoothed && stepNum >= STEP_MESHOPT ? 'badge-green' : 'badge-orange'
    },
    {
      name: 'Bounding Box (WxHxD)',
      raw: rawM.bbox ? rawM.bbox.join(' × ') : '—',
      cur: curM.bbox ? curM.bbox.join(' × ') : '—',
      delta: stepNum >= 1 ? 'Base Grounded at Y=0' : 'Arbitrary Offset',
      badge: stepNum >= 1 ? 'badge-green' : 'badge-blue'
    }
  ];

  dom.diffTableBody.innerHTML = rows.map(r => `
    <tr>
      <td class="metric-name">${r.name}</td>
      <td>${r.raw}</td>
      <td style="color: var(--text-main); font-weight: 600;">${r.cur}</td>
      <td><span class="badge-tag ${r.badge}">${r.delta}</span></td>
    </tr>
  `).join('');
}

// 6. Deep Dive Tabs Rendering
function renderDeepDiveTabs(rawM, curM, finM) {
  // Tab 1: Geometry Details (+ the Step 3 per-operation breakdown when it ran)
  const reduction = getFaceReduction();
  const curTex = getCarriedTexture(state.selectedStepIndex);
  const bake = getUvBake();
  const bakeCaption = bake
    ? [bake.decision, bake.dilation ? `dilation biên ${bake.dilation}px` : null].filter(Boolean).join(' · ')
    : '';
  const integrityHtml = reduction
    ? `<p style="color: var(--text-muted); margin-bottom: 6px;">Geometry After Step 3 (Face Repair &amp; Reduction):</p>
       <p style="font-weight: 700; color: var(--accent-light);">
         ${reduction.facesBefore.toLocaleString()} → ${reduction.facesAfter.toLocaleString()} faces (-${reduction.percent}%)
       </p>
       <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 4px;">
         Engine: <strong>${escapeHtml(String(reduction.engine || '—')).toUpperCase()}</strong> |
         ops: ${reduction.ops.length ? escapeHtml(reduction.ops.join(', ')) : '—'}.
         Every step after Step 3 still preserves 100% of what it left.
       </p>`
    : `<p style="color: var(--text-muted); margin-bottom: 6px;">Geometry Integrity (Rule 11):</p>
       <p style="font-weight: 700; color: var(--success);">✅ 100% Zero-Decimation Preserved</p>
       <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 4px;">Zero triangles dropped. Preserves intricate silhouette, high-frequency details, and organic curves.</p>`;

  // `removed` counts the repair ops; whatever is left of facesRemoved was collapsed by merge.
  // mergeAttempts is the bisection log: one entry per target face count the engine tried.
  const removedByOps = reduction
    ? Object.keys(REDUCE_OP_LABELS).reduce((sum, key) => sum + Number(reduction.removed?.[key] || 0), 0)
    : 0;
  const mergeAttemptCount = Array.isArray(reduction?.mergeAttempts)
    ? reduction.mergeAttempts.length
    : (typeof reduction?.mergeAttempts === 'number' ? reduction.mergeAttempts : null);

  const removedRows = reduction
    ? Object.entries(REDUCE_OP_LABELS)
        .map(([key, label]) => `
          <tr>
            <td class="metric-name">${label}</td>
            <td style="color: var(--text-main); font-weight: 600;">${Number(reduction.removed?.[key] || 0).toLocaleString()}</td>
          </tr>`)
        .join('') +
      `<tr>
         <td class="metric-name">merge (edge collapse)</td>
         <td style="color: var(--text-main); font-weight: 600;">${Math.max(0, Number(reduction.facesRemoved || 0) - removedByOps).toLocaleString()}${mergeAttemptCount !== null ? ` | ${mergeAttemptCount} lần thử` : ''}</td>
       </tr>
       <tr>
         <td class="metric-name">Tổng cộng / Total removed</td>
         <td style="color: var(--text-main); font-weight: 600;">${Number(reduction.facesRemoved || 0).toLocaleString()}</td>
       </tr>`
    : '';

  const reductionDetailHtml = reduction
    ? `<div style="margin-top: 16px;">
         <p style="color: var(--text-muted); margin-bottom: 6px;">Step 3 Breakdown (faces removed per operation):</p>
         <table class="diff-table"><tbody>${removedRows}</tbody></table>
         <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;">
           Visible-surface deviation:
           <span class="badge-tag ${reduction.withinQualityBudget === false ? 'badge-orange' : 'badge-emerald'}">
             max ${reduction.deviationPercent ?? '—'}%${reduction.rmsPercent !== undefined ? ` / rms ${reduction.rmsPercent}%` : ''} vs budget ${reduction.budgetPercent ?? '—'}%
           </span>
           ${reduction.visibleFaces !== undefined ? ` | visible faces: ${Number(reduction.visibleFaces).toLocaleString()}` : ''}
           ${reduction.uvInvalidated ? ' | ⚠️ UV bị huỷ bởi merge, Step 4 re-chart lại' : ''}
         </p>
       </div>`
    : '';

  dom.geomContent.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.82rem;">
      <div>
        ${integrityHtml}
      </div>
      <div>
        <p style="color: var(--text-muted); margin-bottom: 6px;">Bounding Box Dimensions:</p>
        <p style="font-weight: 700; font-family: monospace;">${curM.bbox ? `${curM.bbox[0]}m × ${curM.bbox[1]}m × ${curM.bbox[2]}m` : '1.2m × 1.6m × 1.1m'}</p>
        <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 4px;">Centered horizontally at X=0, Z=0. Base aligned to ground floor Y=0.0.</p>
      </div>
    </div>
    ${reductionDetailHtml}
  `;

  // Tab 2: UV & Texture Details (+ the Step 4 island merge result when it re-charted)
  const islands = getIslandMerge();
  const islandsHtml = islands
    ? `<div style="margin-top: 16px;">
         <p style="color: var(--text-muted); margin-bottom: 6px;">UV Island Merge (Step 4, ${escapeHtml(String(islands.method || '—'))}):</p>
         <table class="diff-table"><tbody>
           <tr>
             <td class="metric-name">UV islands</td>
             <td style="color: var(--text-main); font-weight: 600;">
               ${Number(islands.islandsAfter).toLocaleString()}${islands.islandsBefore !== islands.islandsAfter ? ` (trước khi gộp: ${Number(islands.islandsBefore).toLocaleString()})` : ''}
             </td>
           </tr>
           <tr>
             <td class="metric-name">Island border (gutter cost)</td>
             <td style="color: var(--text-main); font-weight: 600;">
               ${Math.round(Number(islands.boundaryTexelsAfter)).toLocaleString()} texels
               <span class="badge-tag ${islands.boundaryReductionPercent > 0 ? 'badge-emerald' : 'badge-blue'}" style="margin-left: 6px;">
                 ${islands.boundaryReductionPercent > 0
                   ? `-${islands.boundaryReductionPercent}% biên`
                   : 'phân mảnh mặc định đã tối ưu'}
               </span>
             </td>
           </tr>
           <tr>
             <td class="metric-name">Canvas 1:1 (fit)</td>
             <td style="color: var(--text-main); font-weight: 600;">
               ${islands.canvasBefore === islands.canvasAfter
                 ? `${islands.canvasAfter} px`
                 : `${islands.canvasBefore} px → ${islands.canvasAfter} px`}
             </td>
           </tr>
           <tr>
             <td class="metric-name">Merge level</td>
             <td style="color: var(--text-main); font-weight: 600;">
               ${islands.chosenLevel} / ${(islands.levels || []).length} mức đã thử${islands.enabled ? '' : ' (Merge UV Islands: tắt)'}
             </td>
           </tr>
         </tbody></table>
         <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;">
           ${islands.boundaryReductionPercent > 0
             ? `Gộp đảo ở mức ${islands.chosenLevel} rút ngắn đường biên ${islands.boundaryReductionPercent}%, nên canvas 1:1 nhỏ hơn: ít texel bị chiếm bởi gutter padding.`
             : `Mức ${islands.chosenLevel} (phân mảnh mặc định của ${escapeHtml(String(islands.method || 'bộ unwrap'))}) đã cho canvas nhỏ nhất — không mức gộp nào thu hẹp được đường biên thêm. Đây là kết quả hợp lệ, không phải lỗi.`}
         </p>
       </div>`
    : '';

  dom.uvContent.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.82rem;">
      <div>
        <p style="color: var(--text-muted); margin-bottom: 4px;">Texture Resolution:</p>
        <p style="font-weight: 700;">
          ${curTex.res || '—'}
          ${(curM.clamped || state.textureClamped)
            ? '<span class="badge-tag badge-orange" style="margin-left: 6px;" title="Không upscale texture gốc">🔒 Clamped (NO-UPSCALE)</span>'
            : (curTex.carried
              ? `<span class="badge-tag badge-blue" style="margin-left: 6px;">Giữ nguyên từ Step ${curTex.fromStep}</span>`
              : '<span class="badge-tag badge-blue" style="margin-left: 6px;">Native / Resampled</span>')}
        </p>
        <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 4px;">
          ${(curM.clamped || state.textureClamped)
            ? '⚠️ ' + (curM.clampedMessage || state.textureClampedMessage || 'Texture gốc nhỏ hơn kích thước yêu cầu: Áp dụng chính sách NO-UPSCALE để bảo toàn độ sắc nét và tối ưu VRAM GPU.')
            : (curTex.carried
              ? `Bước này chỉ đụng hình học, không đụng texture: model vẫn mang texture ${escapeHtml(String(curTex.format || ''))} ${curTex.res} của Step ${curTex.fromStep}. Step ${STEP_UV_BAKE} mới bake lại atlas.`
              : escapeHtml(bakeCaption || curM.decision || '—'))}
        </p>
      </div>
      <div>
        <p style="color: var(--text-muted); margin-bottom: 4px;">GPU Texture Compression:</p>
        <p style="font-weight: 700; color: var(--accent-light);">${curM.gpuCompressionSkipped ? `Skipped (${curTex.format})` : (curTex.format || '—')}</p>
        <p style="font-size: 0.75rem; color: var(--text-dim); margin-top: 4px;">${curM.gpuCompressionSkipped
          ? `KTX2 bỏ qua: ${escapeHtml(curM.gpuCompressionReason)}. Texture giữ nguyên ${curM.textureFormat} từ Step 5.`
          : 'Direct GPU VRAM block decompression. Eliminates browser main-thread JPEG/PNG decode stalls.'}</p>
      </div>
    </div>
    ${islandsHtml}
  `;

  // Tab 3: Color Palette Swatches (Step 5 only)
  if (!isStepOn(STEP_PALETTE)) {
    dom.paletteGrid.innerHTML = '';
  } else {
  const palette = curM.palette || rawM.palette || ['#427121', '#629137', '#2c4e12', '#1b1e14', '#312e2a', '#b25908', '#090a07', '#4a4c41', '#b3b19c', '#26727b'];
  const paletteDetails = curM.paletteDetails || rawM.paletteDetails || palette.map((hex, i) => ({ hex, weight: Math.max(0.02, 0.25 - i * 0.02) }));

  dom.paletteGrid.innerHTML = paletteDetails.slice(0, 10).map((p, idx) => `
    <div class="swatch-card" onclick="copyHex('${p.hex}')" title="Click to copy ${p.hex}">
      <div class="swatch-color" style="background: ${p.hex};"></div>
      <div class="swatch-hex">${p.hex}</div>
      <div class="swatch-weight">${p.weight ? (p.weight * 100).toFixed(1) + '%' : `#${idx + 1}`}</div>
    </div>
  `).join('');
  }

  // Tab 4: Visual Waterfall Size Chart
  const baseSize = rawM.fileSize || 2026696;
  dom.chartContainer.innerHTML = state.steps.filter(isStepActive).map(s => {
    const sSize = s.metrics?.fileSize || (s.step === 0 ? baseSize : 0);
    const pct = baseSize > 0 && sSize > 0 ? Math.min(100, Math.max(10, (sSize / baseSize) * 100)).toFixed(0) : 10;
    let barClass = 'bar-inter';
    if (s.step === 0) barClass = 'bar-raw';
    else if (s.step === 6) barClass = 'bar-meshopt';
    else if (s.step === FINAL_STEP) barClass = 'bar-final';

    return `
      <div class="chart-bar-row">
        <div class="chart-label" title="Step ${s.step}: ${s.name}">S${s.step}: ${s.name}</div>
        <div class="chart-track">
          <div class="chart-fill ${barClass}" style="width: ${pct}%;">
            ${sSize ? formatBytes(sSize) : '—'}
          </div>
        </div>
        <div class="chart-val">${pct}%</div>
      </div>
    `;
  }).join('');
}

function copyHex(hex) {
  navigator.clipboard.writeText(hex).then(() => {
    showToast(`Copied ${hex} to clipboard!`, 'success');
  }).catch(() => {
    showToast(`Hex: ${hex}`, 'info');
  });
}

// 7. Tab Switching Logic
function setupTabs() {
  dom.tabsNav.querySelectorAll('.tab-btn').forEach(btn => {
    btn.onclick = () => activateTab(btn);
  });
}

// Every step shows the same model at a different stage, so moving between them must not move the
// camera: <model-viewer> re-frames whenever `src` changes, which would re-centre and re-zoom on
// each switch and make two versions impossible to compare. The pose is read before the swap and
// put back once the new model has loaded.
function readCameraPose(mv) {
  try {
    if (!mv.loaded) return null;
    const orbit = mv.getCameraOrbit();
    const target = mv.getCameraTarget();
    const centre = mv.getBoundingBoxCenter();
    if (!orbit || !target || !centre) return null;
    return {
      cameraOrbit: `${(orbit.theta * 180 / Math.PI).toFixed(4)}deg ${(orbit.phi * 180 / Math.PI).toFixed(4)}deg ${orbit.radius.toFixed(5)}m`,
      // Where the camera looks, relative to the model rather than to the world: Step 1 grounds the
      // model at Y=0, so holding an absolute point would show the raw step offset against the rest
      targetOffset: { x: target.x - centre.x, y: target.y - centre.y, z: target.z - centre.z },
      fieldOfView: `${mv.getFieldOfView()}deg`
    };
  } catch (_) {
    return null;
  }
}

function applyCameraPose(mv, pose) {
  if (!pose) return;
  try {
    const centre = mv.getBoundingBoxCenter();
    // Attributes, not properties: assigning the properties leaves field-of-view on "auto", and
    // <model-viewer> then re-frames each model to its own fov - which is why the statue changed
    // size between steps even though the orbit was carried over untouched
    mv.setAttribute('camera-orbit', pose.cameraOrbit);
    if (centre) {
      mv.setAttribute('camera-target',
        `${(centre.x + pose.targetOffset.x).toFixed(5)}m `
        + `${(centre.y + pose.targetOffset.y).toFixed(5)}m `
        + `${(centre.z + pose.targetOffset.z).toFixed(5)}m`);
    }
    mv.setAttribute('field-of-view', pose.fieldOfView);
    // Jump instead of animating: the camera must not visibly swing when only the model changed
    if (typeof mv.jumpCameraToGoal === 'function') mv.jumpCameraToGoal();
  } catch (_) {
  }
}

// Swaps the model in `mv` without touching where the camera is looking from.
function loadModelKeepingCamera(mv, url) {
  if (!url) {
    mv.removeAttribute('src');
    delete mv.dataset.loadedUrl;
    return;
  }
  if (mv.dataset.loadedUrl === url) return;  // same model: nothing to swap, nothing to restore
  const pose = readCameraPose(mv);
  mv.addEventListener('load', () => {
    // <model-viewer> frames the new model right after `load`, which overwrites the field of view
    // once more, so the pose is put back again on the frames that follow
    applyCameraPose(mv, pose);
    requestAnimationFrame(() => {
      applyCameraPose(mv, pose);
      requestAnimationFrame(() => applyCameraPose(mv, pose));
    });
  }, { once: true });
  mv.dataset.loadedUrl = url;
  mv.src = url;
}

// 8. Viewport Camera Sync
let isSyncing = false;
function syncCameras(sourceMv, targetMv) {
  if (isSyncing || !state.isSplitView) return;
  isSyncing = true;
  try {
    const orbit = sourceMv.getCameraOrbit();
    const target = sourceMv.getCameraTarget();
    if (orbit && target) {
      targetMv.cameraOrbit = `${(orbit.theta * 180 / Math.PI).toFixed(2)}deg ${(orbit.phi * 180 / Math.PI).toFixed(2)}deg ${orbit.radius.toFixed(2)}m`;
      targetMv.cameraTarget = `${target.x.toFixed(4)}m ${target.y.toFixed(4)}m ${target.z.toFixed(4)}m`;
      targetMv.fieldOfView = `${sourceMv.getFieldOfView()}deg`;
    }
  } catch (_) {
  } finally {
    requestAnimationFrame(() => { isSyncing = false; });
  }
}

function setupCameraSync() {
  dom.mv1.addEventListener('camera-change', (e) => {
    if (state.isSplitView && e.detail?.source === 'user-interaction') {
      syncCameras(dom.mv1, dom.mv2);
    }
  });

  dom.mv2.addEventListener('camera-change', (e) => {
    if (state.isSplitView && e.detail?.source === 'user-interaction') {
      syncCameras(dom.mv2, dom.mv1);
    }
  });
}

// 9. Side-by-Side Dual View Toggle
function setupSplitViewToggle() {
  dom.toggleSplitBtn.onclick = () => {
    state.isSplitView = !state.isSplitView;
    if (state.isSplitView) {
      dom.viewersStage.classList.add('split-mode');
      dom.pane2.style.display = 'flex';
      dom.toggleSplitBtn.textContent = '🖼️ Single View';
      dom.toggleSplitBtn.classList.add('active');

      // Left pane = Step 0 (Raw Baseline)
      loadModelKeepingCamera(dom.mv1, state.steps[0]?.glbUrl);
      dom.pane1Label.textContent = 'Baseline: Step 0 (Raw Model)';
      dom.pane1Dot.style.background = 'var(--warning)';

      // Right pane = Step X (Selected Step)
      const curStep = state.steps[state.selectedStepIndex];
      loadModelKeepingCamera(dom.mv2, curStep?.glbUrl);
      dom.pane2Label.textContent = `Active: Step ${curStep.step} (${curStep.name})`;

      syncCameras(dom.mv1, dom.mv2);
      showToast('Dual View Enabled: Orbit & Zoom are synchronized', 'info');
    } else {
      dom.viewersStage.classList.remove('split-mode');
      dom.pane2.style.display = 'none';
      dom.toggleSplitBtn.textContent = '📊 Side-by-Side Comparison';
      dom.toggleSplitBtn.classList.remove('active');

      selectStep(state.selectedStepIndex);
    }
  };

  dom.autoRotateBtn.onclick = () => {
    state.autoRotate = !state.autoRotate;
    for (const mv of [dom.mv1, dom.mv2]) {
      mv.autoRotate = state.autoRotate;
      // The attribute too, not just the property: it is what <model-viewer> reads back when a new
      // model is set, so leaving it on would start the turntable again on the next step
      if (state.autoRotate) mv.setAttribute('auto-rotate', '');
      else mv.removeAttribute('auto-rotate');
    }
    dom.autoRotateBtn.classList.toggle('active', state.autoRotate);
    showToast(`Auto-Rotate: ${state.autoRotate ? 'ON' : 'OFF'}`, 'info');
  };

  // The camera now survives a step change, so this is the way back to the default framing
  dom.resetCamBtn.onclick = () => {
    for (const mv of [dom.mv1, dom.mv2]) {
      mv.setAttribute('camera-orbit', '0deg 75deg 105%');
      mv.setAttribute('camera-target', 'auto auto auto');
      // Back to letting <model-viewer> frame each model itself
      mv.removeAttribute('field-of-view');
      mv.fieldOfView = 'auto';
      if (typeof mv.jumpCameraToGoal === 'function') mv.jumpCameraToGoal();
    }
    showToast('Camera reset to default orbit', 'info');
  };
}

// 10. SSE Real-Time Job Stream Connection
function connectJobStream(jobId) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  state.currentJobId = jobId;
  state.jobStatus = 'running';
  state.textureClamped = false;
  state.textureClampedMessage = null;
  state.logs = [];
  state.totalDurationSeconds = null;
  state.totalDurationFormatted = null;

  state.faceReduction = null;

  dom.startBtn.disabled = true;
  dom.startBtn.innerHTML = '<span class="spinner-icon"></span> Optimizing...';
  dom.startBtn.classList.add('running');

  // Reset every step: nothing from a previous job may make a step of this one look completed.
  // `enabled` is the user's choice for this run and stays as it is.
  state.steps.forEach(s => {
    s.status = 'pending';
    s.error = null;
    s.metrics = null;
    s.glbUrl = null;
    s.durationSeconds = null;
    s.durationFormatted = null;
  });
  syncStepDependentControls();
  renderStepper();

  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state.eventSource = es;

  es.addEventListener('job_start', (e) => {
    const skipped = getSkippedSteps();
    showToast(
      skipped.length
        ? `Pipeline started: ${state.steps.length - skipped.length}/${state.steps.length} steps (skipping ${skipped.join(', ')})...`
        : `Pipeline started: Running ${state.steps.length}-step optimization...`,
      'info'
    );
  });

  es.addEventListener('step_start', (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.step !== undefined && state.steps[data.step]) {
        state.steps[data.step].status = 'running';
        renderStepper();
      }
    } catch (_) {}
  });

  es.addEventListener('step_skipped', (e) => {
    try {
      const data = JSON.parse(e.data);
      const step = state.steps[data.step];
      if (!step) return;
      step.status = 'skipped';
      step.enabled = false;
      step.metrics = null;
      step.glbUrl = null;
      step.durationSeconds = null;
      step.durationFormatted = null;
      syncStepDependentControls();
      renderStepper();
      if (state.selectedStepIndex === data.step) selectLatestCompletedStep();
      else updateDashboardMetrics();
    } catch (_) {}
  });

  es.addEventListener('texture_clamped', (e) => {
    try {
      const data = JSON.parse(e.data);
      const msg = data.message || 'Original texture clamped (NO-UPSCALE policy)';
      state.textureClamped = true;
      state.textureClampedMessage = msg;
      showToast(`⚠️ [NO-UPSCALE] ${msg}`, 'warning');
      updateDashboardMetrics();
      renderStepper();
    } catch (_) {}
  });

  es.addEventListener('log', (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.message) {
        state.logs.push(data.message);
        if (data.isClamped || data.message.includes('NO-UPSCALE') || (data.message.includes('[Step 4]') && data.message.toLowerCase().includes('clamped'))) {
          console.warn(`[Backend Clamped Log] ${data.message}`);
          state.textureClamped = true;
          state.textureClampedMessage = data.message;
          showToast(`🔒 ${data.message}`, 'warning');
          updateDashboardMetrics();
          renderStepper();
        }
      }
    } catch (_) {}
  });

  es.addEventListener('step_complete', (e) => {
    try {
      const data = JSON.parse(e.data);
      const stepIdx = data.step;
      if (stepIdx !== undefined && state.steps[stepIdx]) {
        state.steps[stepIdx].status = 'completed';

        // Read durationFormatted or durationSeconds immediately
        const durSec = data.durationSeconds !== undefined && data.durationSeconds !== null
          ? Number(data.durationSeconds)
          : (data.metrics?.durationSeconds !== undefined && data.metrics?.durationSeconds !== null
            ? Number(data.metrics.durationSeconds)
            : (data.durationMs ? Number((data.durationMs / 1000).toFixed(3)) : null));

        const durFmt = data.durationFormatted
          || data.metrics?.durationFormatted
          || (durSec !== null && !isNaN(durSec) ? `${durSec.toFixed(2)}s` : null);

        state.steps[stepIdx].durationSeconds = durSec;
        state.steps[stepIdx].durationFormatted = durFmt;

        state.steps[stepIdx].metrics = normalizeMetrics({
          ...(data.metrics || data),
          durationSeconds: durSec,
          durationFormatted: durFmt
        });
        state.steps[stepIdx].glbUrl = data.glbUrl || `/workspaces/${jobId}/${data.file}`;

        if (data.totalPipelineDurationSeconds || data.elapsedSeconds) {
          state.totalDurationSeconds = data.totalPipelineDurationSeconds || data.elapsedSeconds;
          state.totalDurationFormatted = `${Number(state.totalDurationSeconds).toFixed(2)}s`;
        }

        if (data.textureClamped || data.clamped || data.metrics?.clamped || data.metrics?.textureClamped) {
          state.textureClamped = true;
          state.textureClampedMessage = data.textureClampedMessage || data.clampedMessage || data.metrics?.clampedMessage || state.textureClampedMessage;
        }

        if (stepIdx === STEP_UV_BAKE && state.textureClamped) {
          const note = state.textureClampedMessage || 'Original texture preserved (NO-UPSCALE policy)';
          showToast(`🔒 Step ${STEP_UV_BAKE}: ${note}`, 'warning');
        }

        renderStepper();

        // Auto-select latest completed step
        selectStep(stepIdx);
      }
    } catch (err) {
      console.error('step_complete parse error:', err);
    }
  });

  es.addEventListener('job_complete', (e) => {
    state.jobStatus = 'completed';
    dom.startBtn.innerHTML = '⚡ Start Optimization';
    dom.startBtn.classList.remove('running');

    try {
      const data = JSON.parse(e.data || '{}');
      const summary = data.summary || {};
      if (data.totalPipelineDurationSeconds || data.elapsedSeconds || data.durationSeconds) {
        state.totalDurationSeconds = data.totalPipelineDurationSeconds || data.elapsedSeconds || data.durationSeconds;
        state.totalDurationFormatted = data.durationFormatted || `${Number(state.totalDurationSeconds).toFixed(2)}s`;
      }
      if (data.textureClamped) {
        state.textureClamped = true;
        state.textureClampedMessage = data.textureClampedMessage || state.textureClampedMessage;
      }
      (summary.skippedSteps || []).forEach(idx => {
        if (state.steps[idx]) {
          state.steps[idx].status = 'skipped';
          state.steps[idx].enabled = false;
        }
      });
      if (summary.faceReduction) state.faceReduction = summary.faceReduction;
    } catch (_) {}

    syncStepDependentControls();
    renderStepper();
    selectLatestCompletedStep();
    updateDashboardMetrics();
    showToast('🎉 Optimization Pipeline Completed Successfully!', 'success');
    es.close();
  });

  es.addEventListener('error', (e) => {
    if (e.data) {
      // Server `error` event (live or replayed): message = job.error, step = failed step (or null)
      let data;
      try {
        data = JSON.parse(e.data);
      } catch (err) {
        data = { message: `Unreadable error event from server: ${e.data}` };
      }
      markJobFailed(data.message || data.error, data.step);
    } else {
      // Native EventSource error: the connection dropped before the job reported an outcome
      state.jobStatus = 'error';
      showToast('Lost connection to the job stream: job outcome unknown', 'error');
    }
    dom.startBtn.innerHTML = '⚡ Start Optimization';
    dom.startBtn.classList.remove('running');
    syncStepDependentControls();
    renderStepper();
    es.close();
  });
}

// A failed job: show its reason and mark the step it failed at (never as completed). Without a
// step index the first step that did not complete failed; if all did, the final result did.
function markJobFailed(reason, failedStep) {
  state.jobStatus = 'error';
  let idx = Number.isInteger(failedStep)
    ? failedStep
    : state.steps.findIndex(s => isStepActive(s) && s.status !== 'completed');
  if (idx < 0) idx = state.steps.length - 1;
  if (state.steps[idx]) {
    state.steps[idx].status = 'error';
    state.steps[idx].error = reason;
  }
  renderStepper();
  showToast(`Optimization Error: ${escapeHtml(reason)}`, 'error');
}

// 11. Trigger Optimization Action
async function startOptimization() {
  const uvMode = (dom.uvModeSelect && dom.uvModeSelect.value) ? dom.uvModeSelect.value : 'rechart';
  const format = (dom.formatSelect && dom.formatSelect.value) ? dom.formatSelect.value : 'ktx2';
  state.uvMode = uvMode;

  // Only the fields of the steps that actually run: the server rejects a field of a skipped step.
  // `format` belongs to Step 7, which always runs.
  const formData = new FormData();
  formData.append('format', format);

  const skipSteps = getSkippedSteps();
  if (skipSteps.length) {
    formData.append('skipSteps', skipSteps.join(','));
  }

  if (isStepOn(STEP_FACE_REDUCE)) {
    formData.append('reduceEngine', dom.reduceEngineSelect.value);
    formData.append('reduceOps', getSelectedReduceOps().join(','));
    formData.append('reduceQualityBudget', dom.reduceQualityBudgetInput.value);
    // Blank means 'auto': the server lets Step 3 read the angle off the model
    formData.append('reduceNormalBudget', dom.reduceNormalBudgetInput.value.trim() || 'auto');
    formData.append('reduceNormalFactor', dom.reduceNormalFactorInput.value);
    formData.append('reduceIsolatedMinFaces', dom.reduceIsolatedMinFacesInput.value);
  }

  if (isStepOn(STEP_UV_BAKE)) {
    formData.append('uvMode', uvMode);
    formData.append('downscale', dom.downscaleSelect.value);
    formData.append('sizeMode', dom.sizeModeSelect.value);
    formData.append('mergeUvIslands', dom.mergeIslandsSelect.value);
  }

  if (isStepOn(STEP_MESHOPT)) {
    formData.append('smoothNormals', dom.smoothNormalsSelect.value);
  }

  if (state.selectedFile) {
    formData.append('file', state.selectedFile);
    showToast(`Uploading ${state.selectedFile.name}...`, 'info');
  } else if (dom.modelSelect.value) {
    formData.append('samplePath', dom.modelSelect.value);
    const name = dom.modelSelect.options[dom.modelSelect.selectedIndex]?.text;
    showToast(`Starting optimization on ${name}...`, 'info');
  } else {
    showToast('Please select a sample model or drop a .glb file to optimize!', 'error');
    return;
  }

  try {
    dom.startBtn.disabled = true;
    dom.startBtn.innerHTML = '<span class="spinner-icon"></span> Initializing...';

    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.jobId) {
      connectJobStream(data.jobId);
    }
  } catch (err) {
    dom.startBtn.disabled = false;
    dom.startBtn.innerHTML = '⚡ Start Optimization';
    showToast(`Failed to start job: ${escapeHtml(err.message)}`, 'error');
  }
}

// 12. Drag & Drop File Handling
function setupDragAndDrop() {
  ['dragenter', 'dragover'].forEach(name => {
    dom.dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dom.dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dom.dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dom.dropzone.classList.remove('drag-over');
    });
  });

  dom.dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });
}

function handleFileSelected(file) {
  if (!file.name.toLowerCase().endsWith('.glb')) {
    showToast('Only binary .glb 3D files are supported!', 'error');
    return;
  }

  state.selectedFile = file;
  dom.dropzoneTitle.textContent = file.name;
  dom.dropzoneHint.textContent = `${formatBytes(file.size)} | Ready for pipeline`;
  dom.currentModelTitle.textContent = file.name.replace('.glb', '');
  dom.modelSelect.value = '';

  showToast(`Loaded ${file.name}. Click "Start Optimization" to run!`, 'success');
}

// 13. Load Precomputed Default Showcase
async function loadDefaultShowcase() {
  try {
    const res = await fetch('/api/jobs/default_sample/metrics');
    if (res.status === 404) return; // no precomputed showcase
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      showToast(`Cannot load the default showcase: ${escapeHtml(errData.error || `HTTP ${res.status}`)}`, 'error');
      return;
    }
    const data = await res.json();

    state.currentJobId = data.jobId || 'default_sample';
    if (data.textureClamped) {
      state.textureClamped = true;
      state.textureClampedMessage = data.textureClampedMessage || null;
    }

    state.totalDurationSeconds = data.totalDurationSeconds || data.totalPipelineDurationSeconds || data.elapsedSeconds || data.summary?.elapsedSeconds || 3.37;
    state.totalDurationFormatted = data.totalDurationFormatted || data.totalPipelineDurationFormatted || `${Number(state.totalDurationSeconds).toFixed(2)}s`;

    const stepsData = data.steps || {};
    const summary = data.summary || {};
    state.faceReduction = summary.faceReduction || null;
    (summary.skippedSteps || data.skippedSteps || []).forEach(idx => {
      if (state.steps[idx]) {
        state.steps[idx].status = 'skipped';
        state.steps[idx].enabled = false;
      }
    });

    state.steps.forEach(s => {
      const metric = Array.isArray(data.steps)
        ? data.steps.find(item => item.step === s.step)
        : (stepsData[s.step] || stepsData[String(s.step)]);

      if (metric && metric.skipped) {
        s.status = 'skipped';
        s.enabled = false;
        return;
      }

      if (metric) {
        s.status = 'completed';
        s.enabled = true;
        const fallbackDur = data.summary?.stepDurations?.[s.key];
        s.durationSeconds = metric.durationSeconds !== undefined && metric.durationSeconds !== null
          ? Number(metric.durationSeconds)
          : (fallbackDur?.seconds !== undefined ? Number(fallbackDur.seconds) : null);

        s.durationFormatted = formatDurationSeconds(s.durationSeconds, metric.durationFormatted || fallbackDur?.durationFormatted);

        s.metrics = normalizeMetrics({
          ...metric,
          durationSeconds: s.durationSeconds,
          durationFormatted: s.durationFormatted
        });
        s.glbUrl = metric.glbUrl || `/workspaces/${state.currentJobId}/${metric.file || s.file}`;
      }
    });

    syncStepDependentControls();

    if (data.status === 'error') {
      markJobFailed(data.error, data.errorStep);
      selectLatestCompletedStep();
      return;
    }

    renderStepper();
    selectLatestCompletedStep(); // default view: the last step that really produced a model
  } catch (err) {
    console.error('Failed to load default showcase:', err);
  }
}

// Init
window.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupCameraSync();
  setupSplitViewToggle();
  setupDragAndDrop();

  dom.startBtn.onclick = startOptimization;
  if (dom.uvModeSelect && !dom.uvModeSelect.value) {
    dom.uvModeSelect.value = 'rechart';
  }

  dom.downscaleSelect.onchange = syncStepDependentControls;
  dom.reduceOpsRow.querySelectorAll('input.reduce-op').forEach(cb => {
    cb.onchange = syncStepDependentControls;
  });
  syncStepDependentControls();

  dom.modelSelect.onchange = () => {
    if (dom.modelSelect.value) {
      state.selectedFile = null;
      dom.dropzoneTitle.textContent = 'Drop custom .glb model here';
      dom.dropzoneHint.textContent = 'or click to browse from disk';
      const name = dom.modelSelect.options[dom.modelSelect.selectedIndex]?.text;
      dom.currentModelTitle.textContent = name;
    }
  };

  renderStepper();
  await loadModelsCatalog();
  await loadDefaultShowcase();
});
