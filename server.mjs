import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { Readable } from 'node:stream';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// PORT unset -> 3000; any other value must be an integer 0-65535 (0 = any free port)
function parsePort(raw) {
  if (raw === undefined) return 3000;
  if (!/^\d+$/.test(raw) || Number(raw) > 65535) {
    console.error(`❌ Invalid PORT '${raw}': expected an integer 0-65535`);
    process.exit(1);
  }
  return Number(raw);
}

const PORT = parsePort(process.env.PORT);
const WORKSPACES_DIR = path.join(__dirname, 'workspaces');
const PYTHON_BIN = path.join(__dirname, '.venv', 'bin', 'python');
// Output files of the 8 pipeline steps (optimizer/step_pipeline.py StepPipeline.STEP_DEFINITIONS)
const STEP_FILES = [
  'step_00_raw.glb',
  'step_01_cleaned_grounded.glb',
  'step_02_oriented.glb',
  'step_03_face_reduced.glb',
  'step_04_texture_baked.glb',
  'step_05_palette_tagged.glb',
  'step_06_meshopt.glb',
  'step_07_final.glb'
];
const TOTAL_STEPS = STEP_FILES.length;
const INTERRUPTED_REASON = 'interrupted (server restarted)';
const STDERR_TAIL_CHARS = 4000;

// Step files a run must have produced: a skipped step writes none (optimizer/step_pipeline.py)
function missingStepFiles(wsDir, skipSteps) {
  const skipped = new Set(skipSteps);
  return STEP_FILES.filter((file, step) => !skipped.has(step) && !fs.existsSync(path.join(wsDir, file)));
}

// The steps a stored run skipped: a pipeline-written metrics.json lists them in `skippedSteps`, and
// every `steps` entry of a skipped step carries `skipped: true` with `file: null`
function storedSkippedSteps(data) {
  if (Array.isArray(data.skippedSteps)) return data.skippedSteps;
  const steps = Array.isArray(data.steps) ? data.steps : (data.steps ? Object.values(data.steps) : []);
  return steps.filter(s => s && s.skipped === true).map(s => s.step);
}

// Ensure workspaces directory and gitignore
if (!fs.existsSync(WORKSPACES_DIR)) {
  fs.mkdirSync(WORKSPACES_DIR, { recursive: true });
}
const gitignorePath = path.join(WORKSPACES_DIR, '.gitignore');
if (!fs.existsSync(gitignorePath)) {
  fs.writeFileSync(gitignorePath, '*\n!.gitignore\n!.gitkeep\n', 'utf-8');
}
const gitkeepPath = path.join(WORKSPACES_DIR, '.gitkeep');
if (!fs.existsSync(gitkeepPath)) {
  fs.writeFileSync(gitkeepPath, '', 'utf-8');
}

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json',
  '.bin': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

function formatBytes(bytes, decimals = 2) {
  if (!+bytes) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

// An already optimized GLB carries one of these; the pipeline refuses it (no decompression), so the
// model picker does not list it
const COMPRESSED_GLB_EXTENSIONS = ['EXT_meshopt_compression', 'KHR_draco_mesh_compression', 'KHR_texture_basisu'];
const loggedSkippedModels = new Set();

// Compression extensions a GLB lists in extensionsRequired / extensionsUsed (throws if unreadable)
function compressedExtensionsOf(filePath) {
  let gltf;
  try {
    gltf = readGlbJson(filePath);
  } catch (err) {
    throw new Error(`${filePath}: ${err.message}`);
  }
  const listed = [];
  for (const key of ['extensionsRequired', 'extensionsUsed']) {
    if (gltf[key] === undefined) continue;
    if (!Array.isArray(gltf[key])) throw new Error(`${filePath}: invalid GLB JSON chunk: "${key}" is not an array`);
    listed.push(...gltf[key]);
  }
  return COMPRESSED_GLB_EXTENSIONS.filter(ext => listed.includes(ext));
}

function scanModels() {
  const models = [];
  const skipped = [];
  const searchDirs = [
    { dir: path.join(__dirname, 'examples'), prefix: '/examples/' },
    { dir: path.join(__dirname, 'examples', 'models'), prefix: '/examples/models/' }
  ];

  for (const { dir, prefix } of searchDirs) {
    if (!fs.existsSync(dir)) continue;
    const files = fs.readdirSync(dir);
    for (const file of files) {
      if (file.endsWith('.glb')) {
        const fullPath = path.join(dir, file);
        const compressed = compressedExtensionsOf(fullPath);
        if (compressed.length > 0) {
          skipped.push({ url: `${prefix}${file}`, compressed });
          continue;
        }
        const stat = fs.statSync(fullPath);
        const isOptimized = file.includes('opt') || file.includes('baseline');
        models.push({
          name: file.replace('.glb', ''),
          filename: file,
          url: `${prefix}${file}`,
          sizeBytes: stat.size,
          sizeFormatted: formatBytes(stat.size),
          isOptimized: isOptimized,
          category: dir.includes('models') ? 'Representative Models' : 'Sample Showcase'
        });
      }
    }
  }
  const newlySkipped = skipped.filter(s => !loggedSkippedModels.has(s.url));
  if (newlySkipped.length > 0) {
    newlySkipped.forEach(s => loggedSkippedModels.add(s.url));
    console.log(
      `[models] Not listing ${newlySkipped.length} already-compressed GLB(s): ` +
      newlySkipped.map(s => `${s.url} (${s.compressed.join(', ')})`).join('; ')
    );
  }
  return models;
}

// Reads `length` bytes at `position` (readSync may return fewer than asked); returns the count read
function readFully(fd, length, position) {
  const buffer = Buffer.alloc(length);
  let total = 0;
  while (total < length) {
    const n = fs.readSync(fd, buffer, total, length - total, position + total);
    if (n === 0) break;
    total += n;
  }
  return { buffer, bytesRead: total };
}

// The glTF JSON of a GLB file. Reads the whole JSON chunk; throws when the file is not a readable
// GLB with a valid JSON chunk.
function readGlbJson(filePath) {
  const fd = fs.openSync(filePath, 'r');
  try {
    const fileSize = fs.fstatSync(fd).size;
    const { buffer: header, bytesRead } = readFully(fd, 20, 0);
    if (bytesRead !== 20) {
      throw new Error(`not a GLB: ${fileSize} bytes, shorter than the 20-byte GLB header`);
    }
    // GLB magic: 0x46546C67 ('glTF'), Chunk 0 type: 0x4E4F534A ('JSON')
    if (header.readUInt32LE(0) !== 0x46546C67) throw new Error('not a GLB: glTF magic header missing');
    if (header.readUInt32LE(16) !== 0x4E4F534A) throw new Error('first GLB chunk is not a JSON chunk');
    const chunkLength = header.readUInt32LE(12);
    if (20 + chunkLength > fileSize) {
      throw new Error(`truncated GLB: JSON chunk declares ${chunkLength} bytes, only ${fileSize - 20} present`);
    }
    const { buffer: jsonBuffer, bytesRead: jsonRead } = readFully(fd, chunkLength, 20);
    if (jsonRead !== chunkLength) {
      throw new Error(`read ${jsonRead} of the ${chunkLength}-byte JSON chunk`);
    }
    let gltf;
    try {
      gltf = JSON.parse(jsonBuffer.toString('utf-8'));
    } catch (err) {
      throw new Error(`invalid GLB JSON chunk: ${err.message}`);
    }
    if (gltf === null || typeof gltf !== 'object' || Array.isArray(gltf)) {
      throw new Error('invalid GLB JSON chunk: not a JSON object');
    }
    return gltf;
  } finally {
    fs.closeSync(fd);
  }
}

// True if any material of the GLB is doubleSided; throws when the file is not a readable GLB with a
// valid JSON chunk (the upload is then rejected with a 400).
function detectGlbDoubleSided(filePath) {
  const gltf = readGlbJson(filePath);
  if (gltf.materials === undefined) return false;
  if (!Array.isArray(gltf.materials)) throw new Error('invalid GLB JSON chunk: "materials" is not an array');
  return gltf.materials.some(mat => mat && mat.doubleSided === true);
}

// A client error: the upload is rejected with HTTP 400 and this message
class BadRequestError extends Error {}

// Allowed Optimization Parameters (the Python CLI accepts exactly these, after the aliases)
const FORMAT_ALIASES = { passthrough: 'original' };
const UV_MODE_ALIASES = { rechart: 'xatlas' };
const ALLOWED_FORMATS = ['ktx2', 'webp', 'original', ...Object.keys(FORMAT_ALIASES)];
const ALLOWED_UV_MODES = ['xatlas', 'uvatlas', ...Object.keys(UV_MODE_ALIASES)];
const ALLOWED_DOWNSCALE = ['on', 'off'];
const ALLOWED_MERGE_UV_ISLANDS = ['on', 'off'];
const ALLOWED_SIZE_MODES = ['exact', 'pot-up', 'pot-down'];
const ALLOWED_SMOOTH_NORMALS = ['on', 'off'];
const ALLOWED_REDUCE_ENGINES = ['cgal', 'meshlab'];
const ALLOWED_REDUCE_OPS = ['repair', 'self_intersection', 'isolated', 'hidden', 'merge'];
const DEFAULT_REDUCE_OPS = ['repair', 'isolated', 'hidden', 'merge'];
// Step 3 reads the shading budget off the model unless a number is sent instead
const AUTO_NORMAL_BUDGET = 'auto';
// Only these steps are optional; Step 0 (raw) and Step 7 (final) always run
const SKIPPABLE_STEPS = [1, 2, 3, 4, 5, 6];
// The options each optional step owns: sending one of them for a skipped step is a 400
const STEP_OPTION_FIELDS = {
  3: ['reduceEngine', 'reduceOps', 'reduceQualityBudget', 'reduceNormalBudget', 'reduceNormalFactor', 'reduceIsolatedMinFaces'],
  4: ['uvMode', 'downscale', 'sizeMode', 'mergeUvIslands'],
  6: ['smoothNormals']
};
const OPTION_FIELDS = [
  'format', 'uvMode', 'downscale', 'sizeMode', 'mergeUvIslands',
  'smoothNormals',
  'skipSteps', 'reduceEngine', 'reduceOps', 'reduceQualityBudget', 'reduceNormalBudget', 'reduceNormalFactor',
  'reduceIsolatedMinFaces'
];
const MULTIPART_FIELDS = ['file', 'samplePath', ...OPTION_FIELDS];
const JSON_FIELDS = ['samplePath', ...OPTION_FIELDS];

// Absent field -> default; any other value (including null) must match exactly (never coerced)
function validateChoice(name, value, allowed, defaultValue) {
  if (value === undefined) return defaultValue;
  if (!allowed.includes(value)) {
    throw new BadRequestError(`Unsupported ${name} '${value}' (allowed: ${allowed.join(', ')})`);
  }
  return value;
}

// Absent field -> default; otherwise a JSON array or a comma-separated string (multipart carries
// text fields only). Every item is validated by `parseItem` and may not be repeated.
function validateList(name, value, parseItem, defaultValue) {
  if (value === undefined) return defaultValue;
  let items;
  if (Array.isArray(value)) {
    items = value;
  } else if (typeof value === 'string') {
    items = value.split(',').map(item => item.trim()).filter(item => item !== '');
  } else {
    throw new BadRequestError(`Unsupported ${name} '${value}' (expected an array or a comma-separated string)`);
  }
  const parsed = items.map(parseItem);
  const duplicates = [...new Set(parsed.filter((item, i) => parsed.indexOf(item) !== i))];
  if (duplicates.length > 0) {
    throw new BadRequestError(`Duplicate ${name} value(s): ${duplicates.join(', ')}`);
  }
  return parsed;
}

// Absent field -> default; otherwise a number, or the text of one (multipart), that passes `check`
function validateNumber(name, value, defaultValue, check, expected) {
  if (value === undefined) return defaultValue;
  let num;
  if (typeof value === 'number') {
    num = value;
  } else if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
    num = Number(value);
  } else {
    throw new BadRequestError(`Unsupported ${name} '${value}' (expected ${expected})`);
  }
  if (!check(num)) {
    throw new BadRequestError(`Unsupported ${name} '${value}' (expected ${expected})`);
  }
  return num;
}

function parseSkipStep(item) {
  const step = typeof item === 'number' ? item : (/^\d+$/.test(item) ? Number(item) : NaN);
  if (!Number.isInteger(step) || !SKIPPABLE_STEPS.includes(step)) {
    throw new BadRequestError(`Unsupported skipSteps value '${item}' (allowed: ${SKIPPABLE_STEPS.join(', ')})`);
  }
  return step;
}

const parseReduceOp = (item) => validateChoice('reduceOps value', item, ALLOWED_REDUCE_OPS, undefined);

// An option belongs to the step that reads it: it must not be sent for a step this run skips.
// `fields` are the raw request fields, so a defaulted option is not counted as sent.
function rejectSkippedStepOptions(options, fields) {
  const skipped = new Set(options.skipSteps);
  for (const step of Object.keys(STEP_OPTION_FIELDS).map(Number)) {
    if (!skipped.has(step)) continue;
    const sent = STEP_OPTION_FIELDS[step].filter(name => fields[name] !== undefined);
    if (sent.length > 0) {
      throw new BadRequestError(`Step ${step} is skipped, so its option(s) must not be sent: ${sent.join(', ')}`);
    }
  }
  // Step 3's edge collapse throws the model's UVs away and only Step 4 re-charts them
  if (skipped.has(4) && !skipped.has(3) && options.reduceOps.includes('merge')) {
    throw new BadRequestError(
      "Step 3's reduceOps 'merge' invalidates the model's UVs, which only Step 4 can re-chart: " +
      'either keep Step 4 or drop merge from reduceOps'
    );
  }
}

// The shading budget is either an angle or 'auto', which lets Step 3 read one off the model.
// An empty field means the same as an absent one: the viewer leaves it blank to ask for 'auto'.
function validateNormalBudget(value) {
  if (value === undefined || (typeof value === 'string' && value.trim() === '')) return AUTO_NORMAL_BUDGET;
  if (value === AUTO_NORMAL_BUDGET) return AUTO_NORMAL_BUDGET;
  return validateNumber('reduceNormalBudget', value, AUTO_NORMAL_BUDGET, n => n > 0 && n <= 90,
    `'${AUTO_NORMAL_BUDGET}' or an angle above 0 and at most 90`);
}

function sanitizeOptimizationOptions(fields = {}) {
  const { format, uvMode, downscale, sizeMode, mergeUvIslands, smoothNormals,
    skipSteps, reduceEngine, reduceOps,
    reduceQualityBudget, reduceNormalBudget, reduceNormalFactor, reduceIsolatedMinFaces } = fields;
  const fmt = validateChoice('format', format, ALLOWED_FORMATS, 'ktx2');
  const uv = validateChoice('uvMode', uvMode, ALLOWED_UV_MODES, 'xatlas');
  const options = {
    format: FORMAT_ALIASES[fmt] || fmt,
    uvMode: UV_MODE_ALIASES[uv] || uv,
    downscale: validateChoice('downscale', downscale, ALLOWED_DOWNSCALE, 'on'),
    sizeMode: validateChoice('sizeMode', sizeMode, ALLOWED_SIZE_MODES, 'exact'),
    mergeUvIslands: validateChoice('mergeUvIslands', mergeUvIslands, ALLOWED_MERGE_UV_ISLANDS, 'on'),
    smoothNormals: validateChoice('smoothNormals', smoothNormals, ALLOWED_SMOOTH_NORMALS, 'on'),
    skipSteps: validateList('skipSteps', skipSteps, parseSkipStep, []),
    reduceEngine: validateChoice('reduceEngine', reduceEngine, ALLOWED_REDUCE_ENGINES, 'cgal'),
    reduceOps: validateList('reduceOps', reduceOps, parseReduceOp, DEFAULT_REDUCE_OPS),
    reduceQualityBudget: validateNumber('reduceQualityBudget', reduceQualityBudget, 0.1, n => n > 0, 'a number greater than 0'),
    reduceNormalBudget: validateNormalBudget(reduceNormalBudget),
    reduceNormalFactor: validateNumber('reduceNormalFactor', reduceNormalFactor, 1, n => n > 0 && n <= 30,
      'a number above 0 and at most 30'),
    reduceIsolatedMinFaces: validateNumber('reduceIsolatedMinFaces', reduceIsolatedMinFaces, 25,
      n => Number.isInteger(n) && n >= 0, 'a non-negative integer')
  };
  rejectSkippedStepOptions(options, fields);
  return options;
}

function rejectUnexpectedFields(names, allowed) {
  const unexpected = names.filter(name => !allowed.includes(name));
  if (unexpected.length > 0) {
    throw new BadRequestError(`Unexpected field(s): ${unexpected.join(', ')} (allowed: ${allowed.join(', ')})`);
  }
}

// samplePath must be one of the models /api/models lists (compared by real path)
function resolveListedModel(samplePath) {
  const realPathOf = (urlPath) => {
    try {
      return fs.realpathSync(path.resolve(__dirname, String(urlPath).replace(/^\/+/, '')));
    } catch (_) {
      return null; // does not exist -> cannot be a listed model
    }
  };
  const requested = realPathOf(samplePath);
  const listed = scanModels().map(m => realPathOf(m.url));
  if (requested === null || !listed.includes(requested)) {
    throw new BadRequestError(`samplePath '${samplePath}' is not one of the models listed by /api/models`);
  }
  return requested;
}

// In-Memory Job Management
const jobs = new Map();

function saveWorkspaceMetrics(job) {
  try {
    const metricsPath = path.join(job.workspaceDir, 'metrics.json');
    const payload = {
      jobId: job.id,
      status: job.status,
      config: job.config,
      textureClamped: Boolean(job.textureClamped),
      textureClampedMessage: job.textureClampedMessage || null,
      totalSteps: job.totalSteps,
      currentStep: job.currentStep,
      error: job.error,
      errorType: job.errorType,
      errorStep: job.errorStep,
      exitCode: job.exitCode,
      stderrTail: job.stderrTail,
      steps: job.metrics
    };
    fs.writeFileSync(metricsPath, JSON.stringify(payload, null, 2), 'utf-8');
  } catch (err) {
    console.error(`[Job ${job.id}] Failed to save metrics.json:`, err.message);
  }
}

function emitJobEvent(job, eventName, data) {
  const normalizedEvent = eventName;

  if (normalizedEvent === 'step_start') {
    job.currentStep = data.step;
  } else if (normalizedEvent === 'step_complete') {
    if (data.step !== undefined) {
      job.currentStep = data.step;
      if (!data.glbUrl) {
        data.glbUrl = `/workspaces/${job.id}/${data.file}`;
      }
      data.name = data.name || data.stepName || `Step ${data.step}`;
      data.stepName = data.name;

      if (job.textureClamped) {
        data.textureClamped = true;
        data.textureClampedMessage = job.textureClampedMessage;
      }

      const durSec = data.durationSeconds !== undefined ? Number(data.durationSeconds) : (data.metrics?.durationSeconds !== undefined ? Number(data.metrics.durationSeconds) : undefined);
      const durFmt = data.durationFormatted || data.metrics?.durationFormatted || (durSec !== undefined && !isNaN(durSec) ? (durSec < 0.1 ? `${Math.round(durSec * 1000)}ms` : `${durSec.toFixed(2)}s`) : undefined);
      const totDurSec = data.totalDurationSeconds !== undefined ? Number(data.totalDurationSeconds) : (data.metrics?.totalDurationSeconds !== undefined ? Number(data.metrics.totalDurationSeconds) : undefined);

      if (durSec !== undefined) data.durationSeconds = durSec;
      if (durFmt !== undefined) data.durationFormatted = durFmt;
      if (totDurSec !== undefined) data.totalDurationSeconds = totDurSec;

      if (data.metrics) {
        if (durSec !== undefined) data.metrics.durationSeconds = durSec;
        if (durFmt !== undefined) data.metrics.durationFormatted = durFmt;
        if (totDurSec !== undefined) data.metrics.totalDurationSeconds = totDurSec;
      }

      job.metrics[data.step] = {
        step: data.step,
        stepName: data.stepName,
        file: data.file,
        glbUrl: data.glbUrl,
        ...(data.metrics ? data.metrics : data),
        ...(durSec !== undefined ? { durationSeconds: durSec, durationFormatted: durFmt } : {}),
        ...(totDurSec !== undefined ? { totalDurationSeconds: totDurSec } : {}),
        ...(data.step === 4 && job.textureClamped ? { clamped: true, clampedMessage: job.textureClampedMessage } : {})
      };
    }
    saveWorkspaceMetrics(job);
  } else if (normalizedEvent === 'step_skipped') {
    job.currentStep = data.step;
    job.metrics[data.step] = { step: data.step, stepName: data.stepName, file: null, skipped: true };
    saveWorkspaceMetrics(job);
  } else if (normalizedEvent === 'texture_clamped') {
    job.textureClamped = true;
    job.textureClampedMessage = data.message || 'Original texture clamped (NO-UPSCALE policy)';
    saveWorkspaceMetrics(job);
  } else if (normalizedEvent === 'job_complete' || normalizedEvent === 'error') {
    // job.status / job.error are set by the caller (the close handler / failJob)
    saveWorkspaceMetrics(job);
  }

  const evtRecord = { event: normalizedEvent, data };
  job.events.push(evtRecord);

  // Broadcast to active SSE connections
  const sseMsg = `event: ${normalizedEvent}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const clientRes of job.clients) {
    try {
      clientRes.write(sseMsg);
    } catch (_) {
      job.clients.delete(clientRes);
    }
  }
}

// The job failed: record the concise reason and tell SSE clients. The first failure wins.
function failJob(job, { error, errorType = null, step = null }) {
  if (job.status === 'error') return;
  job.status = 'error';
  job.error = error;
  job.errorType = errorType;
  job.errorStep = step;
  console.error(`[Job ${job.id}] ❌ Failed${step !== null ? ` at step ${step}` : ''}: ${error}`);
  emitJobEvent(job, 'error', { jobId: job.id, message: error, error, errorType, step });
}

// The pipeline broke its stdout event protocol: fail the job and stop the process
function failJobProtocol(job, detail) {
  failJob(job, { error: `Pipeline protocol error: ${detail}`, errorType: 'PipelineProtocolError' });
  const proc = job.childProcess;
  if (proc && proc.exitCode === null && proc.signalCode === null) {
    proc.kill('SIGTERM');
  }
}

// A stdout line starting with '{' must be one pipeline event (see optimizer/step_pipeline.py)
function handlePipelineEvent(job, line) {
  if (job.status === 'error') return; // already failed: later output changes nothing
  const excerpt = line.length > 300 ? `${line.slice(0, 300)}…` : line;

  let payload;
  try {
    payload = JSON.parse(line);
  } catch (err) {
    failJobProtocol(job, `unparseable event line on stdout (${err.message}): ${excerpt}`);
    return;
  }
  if (payload === null || typeof payload !== 'object' || typeof payload.event !== 'string') {
    failJobProtocol(job, `stdout JSON line without an "event" field: ${excerpt}`);
    return;
  }

  // Inspect payload for texture clamping
  if (payload.textureClamped || payload.clamped || payload.metrics?.textureClamped || payload.metrics?.clamped) {
    const clampMsg = payload.clampedMessage || payload.metrics?.clampedMessage || payload.message || 'Original texture clamped (NO-UPSCALE policy)';
    job.textureClamped = true;
    job.textureClampedMessage = clampMsg;
    console.warn(`[Job ${job.id}][NO-UPSCALE] ${clampMsg}`);
    emitJobEvent(job, 'texture_clamped', {
      jobId: job.id,
      message: clampMsg,
      step: payload.step || 4,
      details: payload.metrics || payload
    });
  }

  switch (payload.event) {
    case 'step_complete':
      if (!Number.isInteger(payload.step) || payload.step < 0 || payload.step >= job.totalSteps ||
          typeof payload.file !== 'string' || payload.file === '') {
        failJobProtocol(job, `malformed step_complete event (needs an integer step 0-${job.totalSteps - 1} and a file): ${excerpt}`);
        return;
      }
      emitJobEvent(job, 'step_complete', payload);
      return;
    case 'step_skipped':
      if (!Number.isInteger(payload.step) || payload.step < 0 || payload.step >= job.totalSteps ||
          payload.file !== null) {
        failJobProtocol(job, `malformed step_skipped event (needs an integer step 0-${job.totalSteps - 1} and a null file): ${excerpt}`);
        return;
      }
      emitJobEvent(job, 'step_skipped', { ...payload, jobId: job.id });
      return;
    case 'pipeline_complete':
      // Completed only once the process exits 0 with all step files on disk (see the 'close' handler)
      job.pipelineComplete = payload;
      return;
    case 'pipeline_error':
      if (typeof payload.error !== 'string' || payload.error.trim() === '' ||
          !(payload.step === null || Number.isInteger(payload.step))) {
        failJobProtocol(job, `malformed pipeline_error event (needs an error reason and a step or null): ${excerpt}`);
        return;
      }
      failJob(job, { error: payload.error, errorType: payload.errorType ?? null, step: payload.step });
      return;
    default:
      failJobProtocol(job, `unknown event '${payload.event}': ${excerpt}`);
  }
}

// A plain-text stdout line: forwarded to SSE clients as a log event
function handlePipelineLogLine(job, trimmed) {
  // Check text logs for NO-UPSCALE policy (e.g. "[Step 4] Original texture... clamped to... (NO-UPSCALE policy)")
  const isClampedLog = trimmed.includes('NO-UPSCALE') ||
                       (trimmed.includes('[Step 4]') && trimmed.toLowerCase().includes('clamped')) ||
                       trimmed.toLowerCase().includes('clamped to');

  if (isClampedLog) {
    console.warn(`[Job ${job.id}][POLICY] ⚠️ ${trimmed}`);
    job.textureClamped = true;
    job.textureClampedMessage = trimmed;
    emitJobEvent(job, 'texture_clamped', {
      jobId: job.id,
      message: trimmed,
      step: 4
    });
  } else {
    console.log(`[Job ${job.id}] ${trimmed}`);
  }

  // Forward stdout line to SSE clients as a log event
  emitJobEvent(job, 'log', {
    jobId: job.id,
    message: trimmed,
    isClamped: isClampedLog,
    timestamp: Date.now()
  });
}

// Options must already be validated (sanitizeOptimizationOptions) and doubleSided detected
function startPipelineJob({ jobId, rawGlbPath, workspaceDir, format, uvMode, downscale, sizeMode,
  mergeUvIslands, smoothNormals, skipSteps, reduceEngine, reduceOps,
  reduceQualityBudget, reduceNormalBudget, reduceNormalFactor, reduceIsolatedMinFaces,
  doubleSided }) {
  const job = {
    id: jobId,
    workspaceDir,
    status: 'started',
    config: {
      format, uvMode, downscale, sizeMode, mergeUvIslands, smoothNormals,
      skipSteps, reduceEngine, reduceOps,
      reduceQualityBudget, reduceNormalBudget, reduceNormalFactor, reduceIsolatedMinFaces, doubleSided
    },
    startTime: Date.now(),
    totalSteps: TOTAL_STEPS,
    currentStep: 0,
    events: [],
    metrics: {},
    textureClamped: false,
    textureClampedMessage: null,
    clients: new Set(),
    childProcess: null,
    pipelineComplete: null, // the pipeline_complete event, once received
    error: null,
    errorType: null,
    errorStep: null,
    exitCode: null,
    stderrTail: null
  };
  jobs.set(jobId, job);

  console.log(
    `[Job ${jobId}] Initialized with downscale=${downscale}, sizeMode=${sizeMode}, format=${format}, ` +
    `uvMode=${uvMode}, mergeUvIslands=${mergeUvIslands}, ` +
    `smoothNormals=${smoothNormals}, ` +
    `skipSteps=${skipSteps.join(',') || 'none'}, ` +
    `reduceEngine=${reduceEngine}, reduceOps=${reduceOps.join(',') || 'none'}, ` +
    `reduceQualityBudget=${reduceQualityBudget}, reduceNormalBudget=${reduceNormalBudget}, ` +
    `reduceNormalFactor=${reduceNormalFactor}, ` +
    `reduceIsolatedMinFaces=${reduceIsolatedMinFaces}`
  );
  if (doubleSided) {
    console.log(`[Job ${jobId}] Auto-detected doubleSided: true in input model: enabling --double-sided flag`);
  }

  emitJobEvent(job, 'job_start', {
    jobId,
    status: 'started',
    totalSteps: TOTAL_STEPS,
    config: job.config
  });

  if (!fs.existsSync(PYTHON_BIN)) {
    failJob(job, { error: `Python virtualenv not found: ${PYTHON_BIN} is missing (run ./setup.sh)`, errorType: 'EnvironmentError' });
    return job;
  }

  const args = [
    '-m', 'optimizer.step_pipeline',
    rawGlbPath,
    '--output-dir', workspaceDir,
    '--format', format,
    '--downscale', downscale,
    '--size-mode', sizeMode,
    '--uv-mode', uvMode,
    '--merge-uv-islands', mergeUvIslands,
    smoothNormals === 'on' ? '--smooth-normals' : '--no-smooth-normals',
    '--reduce-engine', reduceEngine,
    '--reduce-ops', reduceOps.join(','),
    '--reduce-quality-budget', String(reduceQualityBudget),
    '--reduce-normal-budget', String(reduceNormalBudget),
    '--reduce-normal-factor', String(reduceNormalFactor),
    '--reduce-isolated-min-faces', String(reduceIsolatedMinFaces)
  ];
  if (skipSteps.length > 0) {
    args.push('--skip-steps', skipSteps.join(','));
  }
  if (doubleSided) {
    args.push('--double-sided');
  }

  console.log(`[Job ${jobId}] Spawning pipeline: ${PYTHON_BIN} ${args.join(' ')}`);

  const proc = spawn(PYTHON_BIN, args, {
    cwd: __dirname,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });
  job.childProcess = proc;
  job.status = 'running';
  saveWorkspaceMetrics(job); // a restart before the first step still finds this job (as interrupted)

  let stdoutBuffer = '';
  let stderrBuffer = '';
  const handleStdoutLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    if (trimmed.startsWith('{')) {
      handlePipelineEvent(job, trimmed);
    } else {
      handlePipelineLogLine(job, trimmed);
    }
  };

  proc.stdout.setEncoding('utf8');
  proc.stdout.on('data', (chunk) => {
    stdoutBuffer += chunk;
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop(); // keep partial line in buffer
    lines.forEach(handleStdoutLine);
  });

  proc.stderr.setEncoding('utf8');
  proc.stderr.on('data', (text) => {
    stderrBuffer += text;
    job.stderrTail = stderrBuffer.slice(-STDERR_TAIL_CHARS);
    console.error(`[Job ${jobId} ERR] ${text.trim()}`);

    // Also check stderr for clamping policy warnings
    if (text.includes('NO-UPSCALE') || text.includes('clamped to')) {
      const trimmed = text.trim();
      job.textureClamped = true;
      job.textureClampedMessage = trimmed;
      emitJobEvent(job, 'texture_clamped', {
        jobId,
        message: trimmed,
        step: 4
      });
    }
  });

  proc.on('close', (code, signal) => {
    handleStdoutLine(stdoutBuffer); // last line, if it had no trailing newline
    stdoutBuffer = '';
    job.exitCode = code;
    job.stderrTail = stderrBuffer.slice(-STDERR_TAIL_CHARS);

    if (job.status !== 'error') {
      const exit = signal ? `was killed by ${signal}` : `exited with code ${code}`;
      if (code !== 0) {
        const lastStderrLine = stderrBuffer.trim().split('\n').pop();
        const detail = job.pipelineComplete
          ? ' after pipeline_complete'
          : ` without a pipeline_error event${lastStderrLine ? `; last stderr line: ${lastStderrLine}` : ''}`;
        failJob(job, { error: `Step pipeline ${exit}${detail}` });
      } else if (!job.pipelineComplete) {
        failJob(job, { error: `Step pipeline ${exit} without a pipeline_complete event` });
      } else {
        const missing = missingStepFiles(job.workspaceDir, skipSteps);
        if (missing.length > 0) {
          failJob(job, { error: `Pipeline reported completion but step files are missing: ${missing.join(', ')}` });
        } else {
          job.status = 'completed';
          emitJobEvent(job, 'job_complete', {
            jobId,
            status: 'completed',
            totalSteps: TOTAL_STEPS,
            summary: job.pipelineComplete.summary,
            metrics: job.metrics,
            textureClamped: Boolean(job.textureClamped),
            textureClampedMessage: job.textureClampedMessage
          });
        }
      }
    }
    saveWorkspaceMetrics(job); // exit code & stderr tail, whatever the outcome
  });

  proc.on('error', (err) => {
    failJob(job, { error: `Failed to run the step pipeline (${PYTHON_BIN}): ${err.message}`, errorType: 'SpawnError' });
  });

  return job;
}

// A job this server process did not run, read back from its workspace metrics.json, with the status
// to report: the stored one, except that a job still started/running on disk was cut off by a server
// restart. Returns null when there is no metrics.json; throws when it cannot be read or parsed.
function loadStoredJob(jobId) {
  const wsDir = path.join(WORKSPACES_DIR, jobId);
  const metricsFile = path.join(wsDir, 'metrics.json');
  if (!fs.existsSync(metricsFile)) return null;

  let data;
  try {
    data = JSON.parse(fs.readFileSync(metricsFile, 'utf-8'));
  } catch (err) {
    throw new Error(`Cannot replay job ${jobId}: metrics.json is invalid (${err.message})`);
  }
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error(`Cannot replay job ${jobId}: metrics.json is not a JSON object`);
  }

  const stored = { ...data };
  if (data.status === undefined) {
    // Written by the Python pipeline itself (a CLI run, or a server that died while the pipeline
    // kept running): only its final payload with all step files on disk is a completed run
    const missing = missingStepFiles(wsDir, storedSkippedSteps(data));
    if (data.success === true && data.summary && missing.length === 0) {
      stored.status = 'completed';
    } else if (data.success === true && data.summary) {
      stored.status = 'error';
      stored.error = `step files are missing: ${missing.join(', ')}`;
    } else {
      stored.status = 'error';
      stored.error = INTERRUPTED_REASON;
    }
  } else if (data.status === 'started' || data.status === 'running') {
    stored.status = 'error';
    stored.error = INTERRUPTED_REASON;
    stored.errorStep = null;
  } else if (data.status !== 'completed' && data.status !== 'error') {
    throw new Error(`Cannot replay job ${jobId}: unknown status '${data.status}' in metrics.json`);
  }
  return stored;
}

const server = http.createServer(async (req, res) => {
  // CORS & Security Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
  res.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const host = req.headers.host || 'localhost';
  const parsedUrl = new URL(req.url, `http://${host}`);
  let pathname = decodeURIComponent(parsedUrl.pathname);

  // 1. API: List available models
  if (pathname === '/api/models' && req.method === 'GET') {
    let models;
    try {
      models = scanModels();
    } catch (err) {
      console.error(`[API /models] ${err.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `Cannot list the models: ${err.message}` }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(models, null, 2));
    return;
  }

  // 2. API: Upload GLB & Start Job
  if (pathname === '/api/upload' && req.method === 'POST') {
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    const wsDir = path.join(WORKSPACES_DIR, jobId);
    let jobStarted = false;
    try {
      const contentType = req.headers['content-type'] || '';
      let fields;
      let fileBuffer = null;

      if (contentType.includes('multipart/form-data')) {
        // (A) Multipart Form Data
        const webReq = new Request(`http://${host}${req.url}`, {
          method: req.method,
          headers: req.headers,
          body: Readable.toWeb(req),
          duplex: 'half'
        });
        let formData;
        try {
          formData = await webReq.formData();
        } catch (err) {
          throw new BadRequestError(`Invalid multipart form data: ${err.message}`);
        }
        rejectUnexpectedFields([...new Set(formData.keys())], MULTIPART_FIELDS);
        fields = {};
        const seen = new Set();
        for (const [name, value] of formData.entries()) {
          if (seen.has(name)) throw new BadRequestError(`Field '${name}' given more than once`);
          seen.add(name);
          if (name === 'file') {
            if (typeof value === 'string') throw new BadRequestError("Field 'file' must be a file upload");
            fileBuffer = Buffer.from(await value.arrayBuffer());
          } else {
            if (typeof value !== 'string') throw new BadRequestError(`Field '${name}' must be a text value`);
            fields[name] = value;
          }
        }
      } else if (contentType.includes('application/json')) {
        // (B) JSON Request (e.g. quick sample model run)
        let body = '';
        req.setEncoding('utf8');
        for await (const chunk of req) body += chunk;
        try {
          fields = JSON.parse(body);
        } catch (err) {
          throw new BadRequestError(`Invalid JSON body: ${err.message}`);
        }
        if (fields === null || typeof fields !== 'object' || Array.isArray(fields)) {
          throw new BadRequestError('JSON body must be an object');
        }
        rejectUnexpectedFields(Object.keys(fields), JSON_FIELDS);
      } else {
        res.writeHead(415, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Unsupported Content-Type. Use multipart/form-data or application/json.' }));
        return;
      }

      const options = sanitizeOptimizationOptions(fields);
      const sources = [
        fileBuffer !== null ? 'file' : null,
        fields.samplePath !== undefined ? 'samplePath' : null
      ].filter(Boolean);
      if (sources.length === 0) throw new BadRequestError('No file or samplePath provided');
      if (sources.length > 1) throw new BadRequestError('Provide only one of file / samplePath (got both)');

      let sampleModelPath = null;
      if (fileBuffer !== null) {
        // Validate GLB magic header "glTF" (0x46546C67)
        if (fileBuffer.length < 12 || fileBuffer.readUInt32LE(0) !== 0x46546C67) {
          throw new BadRequestError('Uploaded file is not a valid binary GLB model (glTF magic header missing).');
        }
      } else {
        sampleModelPath = resolveListedModel(fields.samplePath);
      }

      console.log(`[API /api/upload] ${sources[0]} request: downscale=${options.downscale}, sizeMode=${options.sizeMode}, format=${options.format}, uvMode=${options.uvMode}`);

      fs.mkdirSync(wsDir, { recursive: true });
      const rawGlbPath = path.join(wsDir, 'step_00_raw.glb');
      if (fileBuffer !== null) {
        fs.writeFileSync(rawGlbPath, fileBuffer);
      } else {
        fs.copyFileSync(sampleModelPath, rawGlbPath);
      }

      let doubleSided;
      try {
        doubleSided = detectGlbDoubleSided(rawGlbPath);
      } catch (err) {
        throw new BadRequestError(`Invalid GLB input: ${err.message}`);
      }

      const job = startPipelineJob({ jobId, rawGlbPath, workspaceDir: wsDir, ...options, doubleSided });
      jobStarted = true;

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ jobId, status: job.status, totalSteps: TOTAL_STEPS, config: job.config }));
    } catch (err) {
      if (!jobStarted) fs.rmSync(wsDir, { recursive: true, force: true });
      const status = err instanceof BadRequestError ? 400 : 500;
      if (status === 400) {
        console.warn(`[API /api/upload] Rejected (400): ${err.message}`);
      } else {
        console.error('Upload error:', err);
      }
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 3. API: SSE Event Stream for Job
  if (pathname.startsWith('/api/jobs/') && pathname.endsWith('/stream') && req.method === 'GET') {
    const match = pathname.match(/^\/api\/jobs\/([^\/]+)\/stream$/);
    if (!match) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid job stream path' }));
      return;
    }
    const jobId = match[1];

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    });

    // Send initial keep-alive comment
    res.write(': connected\n\n');

    let job = jobs.get(jobId);

    if (!job) {
      // Job run by an earlier server process: replay it from its workspace metrics.json
      let stored;
      try {
        stored = loadStoredJob(jobId);
      } catch (err) {
        console.error(`[SSE] ${err.message}`);
        res.write(`event: error\ndata: ${JSON.stringify({ jobId, message: err.message, error: err.message, step: null })}\n\n`);
        res.end();
        return;
      }
      if (!stored) {
        res.write(`event: error\ndata: ${JSON.stringify({ message: `Job ${jobId} not found`, jobId })}\n\n`);
        res.end();
        return;
      }

      res.write(`event: job_start\ndata: ${JSON.stringify({ jobId, totalSteps: TOTAL_STEPS, status: stored.status })}\n\n`);
      const steps = Array.isArray(stored.steps)
        ? stored.steps
        : (stored.steps ? Object.values(stored.steps) : []);
      for (const s of steps) {
        if (s.skipped === true) {
          res.write(`event: step_skipped\ndata: ${JSON.stringify({
            jobId,
            step: s.step,
            stepName: s.stepName,
            file: null
          })}\n\n`);
          continue;
        }
        const durSec = s.durationSeconds !== undefined ? s.durationSeconds : s.metrics?.durationSeconds;
        const durFmt = s.durationFormatted || s.metrics?.durationFormatted;
        const totDurSec = s.totalDurationSeconds !== undefined ? s.totalDurationSeconds : s.metrics?.totalDurationSeconds;
        res.write(`event: step_complete\ndata: ${JSON.stringify({
          step: s.step,
          stepName: s.stepName,
          file: s.file,
          glbUrl: s.glbUrl || `/workspaces/${jobId}/${s.file}`,
          durationSeconds: durSec,
          durationFormatted: durFmt,
          totalDurationSeconds: totDurSec,
          metrics: s.metrics || s
        })}\n\n`);
      }
      if (stored.status === 'completed') {
        res.write(`event: job_complete\ndata: ${JSON.stringify({ jobId, status: 'completed' })}\n\n`);
      } else {
        res.write(`event: error\ndata: ${JSON.stringify({
          jobId,
          message: stored.error,
          error: stored.error,
          errorType: stored.errorType ?? null,
          step: stored.errorStep ?? null
        })}\n\n`);
      }
      res.end();
      return;
    }

    // Replay historical events
    for (const evt of job.events) {
      res.write(`event: ${evt.event}\ndata: ${JSON.stringify(evt.data)}\n\n`);
    }

    if (job.status === 'completed' || job.status === 'error') {
      return;
    }

    // Register active SSE listener
    job.clients.add(res);
    req.on('close', () => {
      job.clients.delete(res);
    });
    return;
  }

  // 4. API: Query Job Metrics
  if (pathname.startsWith('/api/jobs/') && pathname.endsWith('/metrics') && req.method === 'GET') {
    const match = pathname.match(/^\/api\/jobs\/([^\/]+)\/metrics$/);
    if (!match) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid job metrics path' }));
      return;
    }
    const jobId = match[1];
    const job = jobs.get(jobId);

    if (job) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        jobId,
        status: job.status,
        currentStep: job.currentStep,
        totalSteps: job.totalSteps,
        config: job.config,
        textureClamped: Boolean(job.textureClamped),
        textureClampedMessage: job.textureClampedMessage || null,
        metrics: job.metrics,
        steps: Object.values(job.metrics),
        error: job.error,
        errorType: job.errorType,
        errorStep: job.errorStep,
        exitCode: job.exitCode,
        stderrTail: job.stderrTail
      }, null, 2));
      return;
    }

    // Job run by an earlier server process: its workspace metrics.json
    let stored;
    try {
      stored = loadStoredJob(jobId);
    } catch (err) {
      console.error(`[API /metrics] ${err.message}`);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
      return;
    }
    if (stored) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(stored, null, 2));
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: `Job ${jobId} not found` }));
    return;
  }

  // 5. Serve Workspaces Static Files (/workspaces/:id/:file)
  if (pathname.startsWith('/workspaces/') && req.method === 'GET') {
    const match = pathname.match(/^\/workspaces\/([^\/]+)\/(.+)$/);
    if (!match) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Workspace file path invalid');
      return;
    }
    const [, id, file] = match;
    const safeFilename = path.normalize(file).replace(/^(\.\.[\/\\])+/, '');
    const filePath = path.resolve(WORKSPACES_DIR, id, safeFilename);

    if (!filePath.startsWith(WORKSPACES_DIR)) {
      res.writeHead(403, { 'Content-Type': 'text/plain' });
      res.end('Forbidden');
      return;
    }

    fs.stat(filePath, (err, stats) => {
      if (err || !stats.isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end(`Workspace file not found: ${pathname}`);
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      let contentType = MIME_TYPES[ext] || 'application/octet-stream';
      if (ext === '.glb') {
        contentType = 'model/gltf-binary';
      }

      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': stats.size,
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache'
      });

      const stream = fs.createReadStream(filePath);
      stream.pipe(res);
    });
    return;
  }

  // 6. Static File Serving (Viewer & Examples)
  if (pathname === '/' || pathname === '/index.html') {
    pathname = '/viewer/index.html';
  }

  const filePath = path.resolve(__dirname, '.' + pathname);

  // Security: restrict static file serving strictly to allowed public directories
  const allowedDirs = [
    path.join(__dirname, 'viewer'),
    path.join(__dirname, 'examples'),
    path.join(__dirname, 'workspaces')
  ];
  const isAllowed = allowedDirs.some(dir => filePath.startsWith(dir));
  if (!isAllowed) {
    res.writeHead(403, { 'Content-Type': 'text/plain' });
    res.end('Forbidden');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`File not found: ${pathname}`);
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    res.writeHead(200, {
      'Content-Type': contentType,
      'Content-Length': stats.size,
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'no-cache'
    });

    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
  });
});

function startServer(port) {
  // Binding the requested port is required: never fall back to another port
  const onError = (err) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`❌ Port ${port} is already in use; not starting (stop the other server or set PORT).`);
    } else {
      console.error(`❌ Cannot listen on port ${port}:`, err);
    }
    process.exit(1);
  };

  server.once('error', onError);
  server.listen(port, () => {
    server.removeListener('error', onError);
    const actualPort = server.address().port;
    console.log(`\n=============================================================`);
    console.log(`🚀 3D Model Optimization Web Server & API is running!`);
    console.log(`📡 Local URL: http://localhost:${actualPort}`);
    console.log(`📁 Workspaces: ${WORKSPACES_DIR}`);
    console.log(`✨ Features:`);
    console.log(`   - POST /api/upload          (Multipart GLB upload & Workspace Job initialization)`);
    console.log(`   - GET  /api/jobs/:id/stream  (Real-time Server-Sent Events step streaming)`);
    console.log(`   - GET  /api/jobs/:id/metrics (Comprehensive multi-step metrics comparison)`);
    console.log(`   - GET  /workspaces/:id/:file (High-speed binary GLB delivery)`);
    console.log(`   - GET  /api/models          (Preset sample models catalog)`);
    console.log(`=============================================================\n`);
  });
}

startServer(PORT);
