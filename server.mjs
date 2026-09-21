import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exec, spawn } from 'node:child_process';
import { Readable } from 'node:stream';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const WORKSPACES_DIR = path.join(__dirname, 'workspaces');

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

function scanModels() {
  const models = [];
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
  return models;
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
      totalSteps: job.totalSteps,
      currentStep: job.currentStep,
      error: job.error,
      steps: job.metrics
    };
    fs.writeFileSync(metricsPath, JSON.stringify(payload, null, 2), 'utf-8');
  } catch (err) {
    console.error(`[Job ${job.id}] Failed to save metrics.json:`, err.message);
  }
}

function emitJobEvent(job, eventName, data) {
  // Normalize event names so SSE clients receive standard event names
  let normalizedEvent = eventName;
  if (eventName === 'pipeline_complete') normalizedEvent = 'job_complete';
  if (eventName === 'pipeline_error') normalizedEvent = 'error';

  if (normalizedEvent === 'step_start') {
    job.currentStep = data.step;
  } else if (normalizedEvent === 'step_complete') {
    if (data.step !== undefined) {
      job.currentStep = data.step;
      const file = data.file || data.modelFile;
      if (!data.file && file) {
        data.file = file;
      }
      if (!data.glbUrl && file) {
        data.glbUrl = `/workspaces/${job.id}/${file}`;
      }
      data.name = data.name || data.stepName || `Step ${data.step}`;
      data.stepName = data.name;

      job.metrics[data.step] = {
        step: data.step,
        stepName: data.stepName,
        file: data.file,
        glbUrl: data.glbUrl,
        ...(data.metrics ? data.metrics : data)
      };
    }
    saveWorkspaceMetrics(job);
  } else if (normalizedEvent === 'job_complete') {
    job.status = 'completed';
    saveWorkspaceMetrics(job);
  } else if (normalizedEvent === 'error') {
    job.status = 'error';
    job.error = data.message || data.error || 'Job execution error';
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

function startPipelineJob({ jobId, rawGlbPath, workspaceDir, resolution = 1024, format = 'ktx2', uvMode = 'direct' }) {
  const job = {
    id: jobId,
    workspaceDir,
    status: 'started',
    config: { resolution, format, uvMode },
    startTime: Date.now(),
    totalSteps: 7,
    currentStep: 0,
    events: [],
    metrics: {},
    clients: new Set(),
    childProcess: null,
    error: null
  };
  jobs.set(jobId, job);

  emitJobEvent(job, 'job_start', {
    jobId,
    status: 'started',
    totalSteps: 7,
    config: job.config
  });

  const venvPython = path.join(__dirname, '.venv', 'bin', 'python');
  const pythonBin = fs.existsSync(venvPython) ? venvPython : 'python3';
  const hasStepPipeline = fs.existsSync(path.join(__dirname, 'optimizer', 'step_pipeline.py'));

  let args;
  if (hasStepPipeline) {
    args = [
      '-m', 'optimizer.step_pipeline',
      rawGlbPath,
      '--output-dir', workspaceDir,
      '--resolution', String(resolution),
      '--format', String(format).toLowerCase()
    ];
    if (uvMode === 'rechart' || uvMode === 'xatlas') {
      args.push('--rechart-uv');
    }
  } else {
    args = [
      '-m', 'optimizer.cli',
      rawGlbPath,
      path.join(workspaceDir, 'step_06_final.glb'),
      '-r', String(resolution),
      '-f', String(format).toLowerCase(),
      '--export-steps', workspaceDir,
      '--step-events'
    ];
    if (uvMode === 'rechart' || uvMode === 'xatlas') {
      args.push('--rechart');
    }
  }

  console.log(`[Job ${jobId}] Spawning pipeline: ${pythonBin} ${args.join(' ')}`);

  const proc = spawn(pythonBin, args, {
    cwd: __dirname,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });
  job.childProcess = proc;
  job.status = 'running';

  let stdoutBuffer = '';
  let stderrBuffer = '';

  proc.stdout.on('data', (chunk) => {
    stdoutBuffer += chunk.toString('utf-8');
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop(); // keep partial line in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let payload = null;
      if (trimmed.startsWith('__STEP_EVENT__:')) {
        try {
          payload = JSON.parse(trimmed.slice('__STEP_EVENT__:'.length));
        } catch (_) {}
      } else if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        try {
          payload = JSON.parse(trimmed);
        } catch (_) {}
      }

      if (payload) {
        const eventName = payload.event || (payload.step !== undefined ? 'step_complete' : null);
        if (eventName) {
          emitJobEvent(job, eventName, payload);
          continue;
        }
      }
      console.log(`[Job ${jobId}] ${trimmed}`);
    }
  });

  proc.stderr.on('data', (chunk) => {
    const text = chunk.toString('utf-8');
    stderrBuffer += text;
    console.error(`[Job ${jobId} ERR] ${text.trim()}`);
  });

  proc.on('close', (code) => {
    if (stdoutBuffer.trim().startsWith('{') && stdoutBuffer.trim().endsWith('}')) {
      try {
        const payload = JSON.parse(stdoutBuffer.trim());
        if (payload.event) emitJobEvent(job, payload.event, payload);
      } catch (_) {}
    }

    if (code !== 0 && job.status !== 'completed') {
      job.status = 'error';
      job.error = stderrBuffer.trim() || `Step pipeline exited with code ${code}`;
      emitJobEvent(job, 'error', {
        jobId,
        message: job.error,
        code
      });
    } else {
      if (job.status !== 'completed') {
        job.status = 'completed';
        emitJobEvent(job, 'job_complete', {
          jobId,
          status: 'completed',
          totalSteps: 7,
          metrics: job.metrics
        });
      }
    }
  });

  proc.on('error', (err) => {
    job.status = 'error';
    job.error = err.message;
    emitJobEvent(job, 'error', {
      jobId,
      message: err.message
    });
  });

  return job;
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
    const models = scanModels();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(models, null, 2));
    return;
  }

  // 2. API: Upload GLB & Start Job
  if (pathname === '/api/upload' && req.method === 'POST') {
    try {
      const contentType = req.headers['content-type'] || '';
      const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
      const wsDir = path.join(WORKSPACES_DIR, jobId);
      fs.mkdirSync(wsDir, { recursive: true });
      const rawGlbPath = path.join(wsDir, 'step_00_raw.glb');

      // (A) Multipart Form Data
      if (contentType.includes('multipart/form-data')) {
        const webReq = new Request(`http://${host}${req.url}`, {
          method: req.method,
          headers: req.headers,
          body: Readable.toWeb(req),
          duplex: 'half'
        });

        const formData = await webReq.formData();
        const file = formData.get('file');
        const samplePath = formData.get('samplePath') || formData.get('sampleUrl');
        const resolution = parseInt(formData.get('resolution'), 10) || 1024;
        const format = (formData.get('format') || 'ktx2').toString();
        const uvMode = (formData.get('uvMode') || 'direct').toString();

        if (file && typeof file === 'object' && typeof file.arrayBuffer === 'function') {
          const ab = await file.arrayBuffer();
          const buffer = Buffer.from(ab);

          // Validate GLB magic header "glTF" (0x46546C67)
          if (buffer.length < 12 || buffer.readUInt32LE(0) !== 0x46546C67) {
            fs.rmSync(wsDir, { recursive: true, force: true });
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Uploaded file is not a valid binary GLB model (glTF magic header missing).' }));
            return;
          }

          fs.writeFileSync(rawGlbPath, buffer);
        } else if (samplePath) {
          const cleanPath = samplePath.toString().replace(/^\//, '');
          const srcAbsPath = path.resolve(__dirname, cleanPath);
          if (!fs.existsSync(srcAbsPath)) {
            fs.rmSync(wsDir, { recursive: true, force: true });
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: `Sample model not found: ${samplePath}` }));
            return;
          }
          fs.copyFileSync(srcAbsPath, rawGlbPath);
        } else {
          fs.rmSync(wsDir, { recursive: true, force: true });
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No file or samplePath provided in form data' }));
          return;
        }

        startPipelineJob({
          jobId,
          rawGlbPath,
          workspaceDir: wsDir,
          resolution,
          format,
          uvMode
        });

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ jobId, status: 'started', totalSteps: 7 }));
        return;
      }

      // (B) JSON Request (e.g. quick sample model run)
      if (contentType.includes('application/json')) {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
          try {
            const data = JSON.parse(body || '{}');
            const samplePath = data.samplePath || data.sampleUrl;
            const resolution = parseInt(data.resolution, 10) || 1024;
            const format = String(data.format || 'ktx2');
            const uvMode = String(data.uvMode || 'direct');

            if (!samplePath) {
              fs.rmSync(wsDir, { recursive: true, force: true });
              res.writeHead(400, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: 'Missing samplePath in JSON request' }));
              return;
            }

            const cleanPath = samplePath.toString().replace(/^\//, '');
            const srcAbsPath = path.resolve(__dirname, cleanPath);
            if (!fs.existsSync(srcAbsPath)) {
              fs.rmSync(wsDir, { recursive: true, force: true });
              res.writeHead(404, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: `Sample model not found: ${samplePath}` }));
              return;
            }

            fs.copyFileSync(srcAbsPath, rawGlbPath);

            startPipelineJob({
              jobId,
              rawGlbPath,
              workspaceDir: wsDir,
              resolution,
              format,
              uvMode
            });

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ jobId, status: 'started', totalSteps: 7 }));
          } catch (jsonErr) {
            fs.rmSync(wsDir, { recursive: true, force: true });
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: jsonErr.message }));
          }
        });
        return;
      }

      res.writeHead(415, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Unsupported Content-Type. Use multipart/form-data or application/json.' }));
    } catch (uploadErr) {
      console.error('Upload error:', uploadErr);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: uploadErr.message }));
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
      // Disk fallback: check if workspace exists with metrics.json
      const wsDir = path.join(WORKSPACES_DIR, jobId);
      const metricsFile = path.join(wsDir, 'metrics.json');
      if (fs.existsSync(metricsFile)) {
        try {
          const metricsData = JSON.parse(fs.readFileSync(metricsFile, 'utf-8'));
          res.write(`event: job_start\ndata: ${JSON.stringify({ jobId, totalSteps: 7 })}\n\n`);
          const steps = Array.isArray(metricsData.steps)
            ? metricsData.steps
            : (metricsData.steps ? Object.values(metricsData.steps) : []);
          for (const s of steps) {
            res.write(`event: step_complete\ndata: ${JSON.stringify({
              step: s.step,
              stepName: s.stepName,
              file: s.file,
              glbUrl: s.glbUrl || `/workspaces/${jobId}/${s.file}`,
              metrics: s
            })}\n\n`);
          }
          res.write(`event: job_complete\ndata: ${JSON.stringify({ jobId, status: 'completed' })}\n\n`);
          res.end();
          return;
        } catch (_) {}
      }

      res.write(`event: error\ndata: ${JSON.stringify({ message: `Job ${jobId} not found`, jobId })}\n\n`);
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
        metrics: job.metrics,
        steps: Object.values(job.metrics),
        error: job.error
      }, null, 2));
      return;
    }

    // Check disk
    const wsDir = path.join(WORKSPACES_DIR, jobId);
    const metricsFile = path.join(wsDir, 'metrics.json');
    if (fs.existsSync(metricsFile)) {
      try {
        const fileContent = fs.readFileSync(metricsFile, 'utf-8');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(fileContent);
        return;
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
        return;
      }
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

  // 6. Legacy API: Run single-shot optimization
  if (pathname === '/api/optimize' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const { input, resolution = 1024, format = 'ktx2' } = JSON.parse(body || '{}');
        if (!input) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Missing input model path' }));
          return;
        }

        const inputPath = path.resolve(__dirname, input.replace(/^\//, ''));
        const outputFilename = `opt_${Date.now()}_${path.basename(inputPath)}`;
        const outputPath = path.join(__dirname, 'examples', outputFilename);

        const cmd = `"${path.join(__dirname, 'bin', 'optimize-3d')}" "${inputPath}" "${outputPath}" -r ${resolution} -f ${format} --json`;

        exec(cmd, { cwd: __dirname }, (error, stdout, stderr) => {
          if (error) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: stderr || error.message }));
            return;
          }
          const outStat = fs.statSync(outputPath);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: true,
            outputUrl: `/examples/${outputFilename}`,
            sizeBytes: outStat.size,
            sizeFormatted: formatBytes(outStat.size),
            details: stdout
          }));
        });
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 7. Static File Serving (Viewer & Examples)
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
  const onError = (err) => {
    if (err.code === 'EADDRINUSE') {
      console.warn(`⚠️ Port ${port} is currently in use, trying port ${port + 1}...`);
      server.removeListener('error', onError);
      startServer(port + 1);
    } else {
      console.error('Server error:', err);
    }
  };

  server.once('error', onError);
  server.listen(port, () => {
    server.removeListener('error', onError);
    const actualPort = server.address()?.port || port;
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

const INITIAL_PORT = parseInt(process.env.PORT, 10) || 3000;
startServer(INITIAL_PORT);
