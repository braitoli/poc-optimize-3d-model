/**
 * tests/server.test.mjs  (run with `npm test`, i.e. `node --test`)
 *
 * server.mjs: upload parameter validation, how pipeline failures surface as job errors, and how jobs
 * are replayed from disk after a restart.
 *
 * Every server runs from a throw-away sandbox: a copy of server.mjs next to symlinks of examples/ and
 * viewer/, plus either the real optimizer/ + .venv, a fake step pipeline (a tiny Python module that
 * plays the scenario named in the uploaded GLB's extras), or no .venv at all. Job workspaces are
 * created inside the sandbox, never in the repo's workspaces/ directory.
 */

import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SAMPLE = '/examples/sample_dinoki.glb';
const INTERRUPTED = 'interrupted (server restarted)';
const SANDBOX_LINKS = ['examples', 'viewer', 'optimizer', '.venv'];

// Fake optimizer.step_pipeline: plays the scenario stored in the input GLB's extras.scenario
const FAKE_PIPELINE = `
import json, os, struct, sys, time
from pathlib import Path

FILES = ["step_00_raw.glb", "step_01_cleaned_grounded.glb", "step_02_oriented.glb",
         "step_03_texture_baked.glb", "step_04_palette_tagged.glb", "step_05_meshopt.glb", "step_06_final.glb"]


def emit(obj):
    print(json.dumps(obj), flush=True)


def wait_for_parent_exit(limit):
    parent, t0 = os.getppid(), time.time()
    while os.getppid() == parent and time.time() - t0 < limit:
        time.sleep(0.05)


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    data = src.read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    scenario = json.loads(data[20:20 + json_len]).get("extras", {}).get("scenario", "ok")

    def step(i, write=True):
        if write:
            (out / FILES[i]).write_bytes(b"fake")
        emit({"event": "step_complete", "step": i, "stepName": "s%d" % i, "file": FILES[i], "metrics": {"faces": 1}})

    if scenario in ("ok", "missing_final", "nonzero_after_complete"):
        for i in range(7):
            step(i, write=not (scenario == "missing_final" and i == 6))
        emit({"event": "pipeline_complete", "summary": {"elapsedSeconds": 0.1}})
        return 3 if scenario == "nonzero_after_complete" else 0
    if scenario == "pipeline_error":
        step(0)
        print("Traceback (most recent call last):\\n  noisy stderr line", file=sys.stderr, flush=True)
        emit({"event": "pipeline_error", "error": "fake concise reason", "errorType": "PipelineAbort", "step": 1})
        return 1
    if scenario == "exit0_no_complete":
        step(0)
        return 0
    if scenario == "crash_without_event":
        print("ModuleNotFoundError: No module named 'fake_dependency'", file=sys.stderr, flush=True)
        return 1
    if scenario == "malformed_event":
        step(0)
        print('{"event": "step_complete", "step": 1', flush=True)
    elif scenario == "step_only":
        print(json.dumps({"step": 0, "file": FILES[0]}), flush=True)
    elif scenario == "model_file":
        print(json.dumps({"event": "step_complete", "step": 0, "modelFile": FILES[0]}), flush=True)
    elif scenario == "hang":
        step(0)
    else:
        raise SystemExit("unknown scenario " + scenario)
    # The server must fail (and stop) the job on its own; exit once it is gone
    wait_for_parent_exit(60)
    return 0


sys.exit(main())
`;

function makeSandbox(kind) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `server-test-${kind}-`));
  fs.copyFileSync(path.join(REPO_ROOT, 'server.mjs'), path.join(dir, 'server.mjs'));
  fs.symlinkSync(path.join(REPO_ROOT, 'examples'), path.join(dir, 'examples'));
  fs.symlinkSync(path.join(REPO_ROOT, 'viewer'), path.join(dir, 'viewer'));
  if (kind === 'real') {
    fs.symlinkSync(path.join(REPO_ROOT, 'optimizer'), path.join(dir, 'optimizer'));
    fs.symlinkSync(path.join(REPO_ROOT, '.venv'), path.join(dir, '.venv'));
  } else if (kind === 'fake') {
    fs.mkdirSync(path.join(dir, 'optimizer'));
    fs.writeFileSync(path.join(dir, 'optimizer', '__init__.py'), '');
    fs.writeFileSync(path.join(dir, 'optimizer', 'step_pipeline.py'), FAKE_PIPELINE);
    fs.mkdirSync(path.join(dir, '.venv', 'bin'), { recursive: true });
    fs.symlinkSync(fs.realpathSync(path.join(REPO_ROOT, '.venv', 'bin', 'python')), path.join(dir, '.venv', 'bin', 'python'));
  }
  return dir;
}

function removeSandbox(dir) {
  // Unlink the symlinks first so nothing behind them (examples/, .venv, ...) can ever be touched
  for (const name of SANDBOX_LINKS) {
    const p = path.join(dir, name);
    if (fs.lstatSync(p, { throwIfNoEntry: false })?.isSymbolicLink()) fs.unlinkSync(p);
  }
  fs.rmSync(dir, { recursive: true, force: true });
}

function spawnServer(dir, port = '0') {
  const proc = spawn(process.execPath, ['server.mjs'], {
    cwd: dir,
    env: { ...process.env, PORT: port },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';
  proc.stdout.on('data', (d) => { output += d; });
  proc.stderr.on('data', (d) => { output += d; });
  return { proc, output: () => output };
}

async function startServer(dir) {
  const server = spawnServer(dir);
  server.url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`server did not start:\n${server.output()}`)), 15000);
    server.proc.stdout.on('data', () => {
      const m = server.output().match(/Local URL: (http:\/\/localhost:\d+)/);
      if (m) { clearTimeout(timer); resolve(m[1]); }
    });
    server.proc.on('exit', (code) => { clearTimeout(timer); reject(new Error(`server exited (${code}):\n${server.output()}`)); });
  });
  return server;
}

async function stopServer(server) {
  if (server && server.proc.exitCode === null && server.proc.signalCode === null) {
    server.proc.kill('SIGKILL');
    await once(server.proc, 'exit');
  }
}

// GLB from a glTF object (or raw JSON chunk bytes) and an optional BIN chunk
function makeGlb(json, bin = null) {
  let js = Buffer.isBuffer(json) ? json : Buffer.from(JSON.stringify(json), 'utf-8');
  js = Buffer.concat([js, Buffer.alloc((4 - (js.length % 4)) % 4, 0x20)]);
  const parts = [Buffer.alloc(12), Buffer.alloc(8), js];
  parts[1].writeUInt32LE(js.length, 0);
  parts[1].write('JSON', 4, 'ascii');
  if (bin) {
    const hdr = Buffer.alloc(8);
    hdr.writeUInt32LE(bin.length, 0);
    hdr.write('BIN\0', 4, 'ascii');
    parts.push(hdr, bin);
  }
  const total = parts.reduce((n, b) => n + b.length, 0);
  parts[0].write('glTF', 0, 'ascii');
  parts[0].writeUInt32LE(2, 4);
  parts[0].writeUInt32LE(total, 8);
  return Buffer.concat(parts);
}

const scenarioGlb = (scenario) => makeGlb({ asset: { version: '2.0' }, extras: { scenario } });

async function upload(url, fields) {
  const fd = new FormData();
  for (const [name, value] of Object.entries(fields)) {
    if (Buffer.isBuffer(value)) fd.append(name, new Blob([value]), 'model.glb');
    else fd.append(name, value);
  }
  const res = await fetch(`${url}/api/upload`, { method: 'POST', body: fd });
  return { status: res.status, body: await res.json() };
}

async function postJson(url, body) {
  const res = await fetch(`${url}/api/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return { status: res.status, body: await res.json() };
}

async function getMetrics(url, jobId) {
  const res = await fetch(`${url}/api/jobs/${jobId}/metrics`);
  return { status: res.status, body: await res.json() };
}

async function waitForJobEnd(url, jobId, timeoutMs = 20000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const { body } = await getMetrics(url, jobId);
    if (body.status === 'completed' || body.status === 'error') return body;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`job ${jobId} did not finish within ${timeoutMs} ms`);
}

// Terminal status can arrive (pipeline_error on stdout) before the process has exited
async function waitForExit(url, jobId) {
  await waitFor(async () => (await getMetrics(url, jobId)).body.exitCode !== null);
  return (await getMetrics(url, jobId)).body;
}

async function waitFor(predicate, timeoutMs = 10000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('condition not met in time');
}

// Reads the SSE stream until a terminal event (job_complete / error) or the end of the stream
async function readStream(url, jobId, timeoutMs = 10000) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  const events = [];
  try {
    const res = await fetch(`${url}/api/jobs/${jobId}/stream`, { signal: ac.signal });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return events;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const evt = {};
        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) evt.event = line.slice(7);
          else if (line.startsWith('data: ')) evt.data = JSON.parse(line.slice(6));
        }
        if (!evt.event) continue;
        events.push(evt);
        if (evt.event === 'job_complete' || evt.event === 'error') return events;
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') throw new Error(`stream for ${jobId} did not end within ${timeoutMs} ms: ${JSON.stringify(events)}`);
    throw err;
  } finally {
    clearTimeout(timer);
    ac.abort();
  }
}

const workspaceEntries = (dir) => fs.readdirSync(path.join(dir, 'workspaces')).filter((n) => !n.startsWith('.')).sort();

describe('upload parameter validation', () => {
  let dir;
  let server;
  before(async () => {
    dir = makeSandbox('fake');
    server = await startServer(dir);
  });
  after(async () => {
    await stopServer(server);
    removeSandbox(dir);
  });

  test('bogus format / uvMode, png, extra fields and foreign samplePath are 400 and create no job', async () => {
    const cases = [
      [{ samplePath: SAMPLE, format: 'png' }, /Unsupported format 'png'.*allowed: ktx2, webp, original/],
      [{ samplePath: SAMPLE, format: 'jpeg' }, /Unsupported format 'jpeg'/],
      [{ samplePath: SAMPLE, format: 'bogus' }, /Unsupported format 'bogus'/],
      [{ samplePath: SAMPLE, format: 'KTX2' }, /Unsupported format 'KTX2'/],
      [{ samplePath: SAMPLE, format: '' }, /Unsupported format ''/],
      [{ samplePath: SAMPLE, uvMode: 'bogus' }, /Unsupported uvMode 'bogus'.*allowed: xatlas, uvatlas, rechart/],
      [{ samplePath: SAMPLE, downscale: 'maybe' }, /Unsupported downscale 'maybe'/],
      [{ samplePath: SAMPLE, resolution: '512' }, /Unexpected field\(s\): resolution/],
      [{ samplePath: '/etc/passwd' }, /samplePath '\/etc\/passwd' is not one of the models listed by \/api\/models/],
      [{ samplePath: '/package.json' }, /is not one of the models listed/],
      [{ samplePath: '/examples/../server.mjs' }, /is not one of the models listed/],
      [{ samplePath: '/examples/missing.glb' }, /is not one of the models listed/],
      [{}, /No file or samplePath provided/],
      [{ file: Buffer.from('plain text, not a GLB at all') }, /not a valid binary GLB/],
      [{ file: makeGlb({ asset: { version: '2.0' } }).subarray(0, 24) }, /Invalid GLB input: .*truncated/],
      [{ file: makeGlb(Buffer.from('{not json')) }, /Invalid GLB input: invalid GLB JSON chunk/],
      [{ file: scenarioGlb('ok'), samplePath: SAMPLE }, /only one of/]
    ];
    const before = workspaceEntries(dir);
    for (const [fields, message] of cases) {
      const { status, body } = await upload(server.url, fields);
      assert.equal(status, 400, `${JSON.stringify(Object.keys(fields))}: ${JSON.stringify(body)}`);
      assert.match(body.error, message);
    }
    const jsonCases = [
      [{ samplePath: '/etc/passwd' }, /is not one of the models listed/],
      [{ samplePath: SAMPLE, resolution: 512 }, /Unexpected field\(s\): resolution/],
      [{ samplePath: SAMPLE, format: 'jpg' }, /Unsupported format 'jpg'/],
      [{ samplePath: SAMPLE, uvMode: 'rechart2' }, /Unsupported uvMode/],
      [{ samplePath: SAMPLE, format: null }, /Unsupported format 'null'/],
      [{ sampleUrl: SAMPLE }, /Unexpected field\(s\): sampleUrl/]
    ];
    for (const [body, message] of jsonCases) {
      const res = await postJson(server.url, body);
      assert.equal(res.status, 400, JSON.stringify(res.body));
      assert.match(res.body.error, message);
    }
    assert.deepEqual(workspaceEntries(dir), before, 'rejected uploads must not leave a workspace');
  });

  test('documented aliases passthrough -> original and rechart -> xatlas are accepted', async () => {
    const { status, body } = await upload(server.url, { samplePath: SAMPLE, format: 'passthrough', uvMode: 'rechart' });
    assert.equal(status, 200, JSON.stringify(body));
    assert.equal(body.config.format, 'original');
    assert.equal(body.config.uvMode, 'xatlas');
    const job = await waitForJobEnd(server.url, body.jobId);
    assert.equal(job.status, 'completed', JSON.stringify(job));
  });
});

describe('pipeline outcome handling (fake pipeline)', () => {
  let dir;
  let server;
  before(async () => {
    dir = makeSandbox('fake');
    server = await startServer(dir);
  });
  after(async () => {
    await stopServer(server);
    removeSandbox(dir);
  });

  async function runScenario(scenario) {
    const { status, body } = await upload(server.url, { file: scenarioGlb(scenario) });
    assert.equal(status, 200, JSON.stringify(body));
    return { jobId: body.jobId, job: await waitForJobEnd(server.url, body.jobId) };
  }

  test('a complete run with all 7 step files is completed', async () => {
    const { jobId, job } = await runScenario('ok');
    assert.equal(job.status, 'completed', JSON.stringify(job));
    assert.equal(job.error, null);
    const events = await readStream(server.url, jobId);
    assert.equal(events.at(-1).event, 'job_complete');
    assert.equal(events.filter((e) => e.event === 'step_complete').length, 7);
    const stored = JSON.parse(fs.readFileSync(path.join(dir, 'workspaces', jobId, 'metrics.json'), 'utf-8'));
    assert.equal(stored.status, 'completed');
  });

  test('pipeline_error sets the concise reason, keeps stderr separately and never completes', async () => {
    const { jobId } = await runScenario('pipeline_error');
    const job = await waitForExit(server.url, jobId);
    assert.equal(job.status, 'error');
    assert.equal(job.error, 'fake concise reason');
    assert.equal(job.errorType, 'PipelineAbort');
    assert.equal(job.errorStep, 1);
    assert.match(job.stderrTail, /noisy stderr line/);
    const events = await readStream(server.url, jobId);
    assert.equal(events.at(-1).event, 'error');
    assert.equal(events.at(-1).data.message, 'fake concise reason');
    assert.equal(events.at(-1).data.step, 1);
    assert.ok(!events.some((e) => e.event === 'job_complete'));
    assert.deepEqual(events.filter((e) => e.event === 'step_complete').map((e) => e.data.step), [0]);
    const stored = JSON.parse(fs.readFileSync(path.join(dir, 'workspaces', jobId, 'metrics.json'), 'utf-8'));
    assert.equal(stored.status, 'error');
    assert.equal(stored.error, 'fake concise reason');
  });

  const failures = [
    ['exit0_no_complete', /exited with code 0 without a pipeline_complete event/],
    ['missing_final', /step files are missing: step_06_final\.glb/],
    ['nonzero_after_complete', /exited with code 3 after pipeline_complete/],
    ['crash_without_event', /exited with code 1 without a pipeline_error event.*No module named 'fake_dependency'/],
    ['malformed_event', /unparseable event line/],
    ['step_only', /without an "event" field/],
    ['model_file', /malformed step_complete event/]
  ];
  for (const [scenario, message] of failures) {
    test(`${scenario} ends the job in error`, async () => {
      const { job } = await runScenario(scenario);
      assert.equal(job.status, 'error', JSON.stringify(job));
      assert.match(job.error, message);
    });
  }
});

describe('replay from disk after a restart', () => {
  let dir;
  let server;
  before(() => { dir = makeSandbox('fake'); });
  after(async () => {
    await stopServer(server);
    removeSandbox(dir);
  });

  function writeStoredJob(jobId, content) {
    fs.mkdirSync(path.join(dir, 'workspaces', jobId), { recursive: true });
    fs.writeFileSync(path.join(dir, 'workspaces', jobId, 'metrics.json'),
      typeof content === 'string' ? content : JSON.stringify(content));
  }

  test('an errored job still reports its error, a job cut off mid-run reports interrupted', async () => {
    server = await startServer(dir);
    const failed = await upload(server.url, { file: scenarioGlb('pipeline_error') });
    await waitForJobEnd(server.url, failed.body.jobId);
    const hung = await upload(server.url, { file: scenarioGlb('hang') });
    await waitFor(async () => (await getMetrics(server.url, hung.body.jobId)).body.steps.length === 1);
    await stopServer(server); // killed mid-job

    server = await startServer(dir);
    const failedMetrics = await getMetrics(server.url, failed.body.jobId);
    assert.equal(failedMetrics.status, 200);
    assert.equal(failedMetrics.body.status, 'error');
    assert.equal(failedMetrics.body.error, 'fake concise reason');
    const failedEvents = await readStream(server.url, failed.body.jobId);
    assert.equal(failedEvents.at(-1).event, 'error');
    assert.equal(failedEvents.at(-1).data.message, 'fake concise reason');
    assert.ok(!failedEvents.some((e) => e.event === 'job_complete'));

    const hungMetrics = await getMetrics(server.url, hung.body.jobId);
    assert.equal(hungMetrics.body.status, 'error');
    assert.equal(hungMetrics.body.error, INTERRUPTED);
    const hungEvents = await readStream(server.url, hung.body.jobId);
    assert.equal(hungEvents.at(-1).event, 'error');
    assert.equal(hungEvents.at(-1).data.message, INTERRUPTED);
    assert.deepEqual(hungEvents.filter((e) => e.event === 'step_complete').map((e) => e.data.step), [0]);
  });

  test('stored completed / pipeline-written / corrupt metrics.json', async () => {
    server ??= await startServer(dir);
    writeStoredJob('job_stored_completed', { jobId: 'job_stored_completed', status: 'completed', error: null, steps: {} });
    writeStoredJob('job_stored_started', { jobId: 'job_stored_started', status: 'started', error: null, steps: {} });
    // Written by the Python pipeline itself (no server status) and cut off after step 2
    writeStoredJob('job_stored_partial', { success: true, steps: [], lastCompletedStep: 2 });
    writeStoredJob('job_stored_corrupt', '{"status": "completed", ');

    assert.equal((await readStream(server.url, 'job_stored_completed')).at(-1).event, 'job_complete');
    for (const jobId of ['job_stored_started', 'job_stored_partial']) {
      const { body } = await getMetrics(server.url, jobId);
      assert.equal(body.status, 'error', jobId);
      assert.equal(body.error, INTERRUPTED, jobId);
      assert.equal((await readStream(server.url, jobId)).at(-1).data.message, INTERRUPTED, jobId);
    }

    const corrupt = await getMetrics(server.url, 'job_stored_corrupt');
    assert.equal(corrupt.status, 500);
    assert.match(corrupt.body.error, /metrics\.json is invalid/);
    const corruptEvents = await readStream(server.url, 'job_stored_corrupt');
    assert.equal(corruptEvents.at(-1).event, 'error');
    assert.match(corruptEvents.at(-1).data.message, /metrics\.json is invalid/);

    const missing = await getMetrics(server.url, 'job_does_not_exist');
    assert.equal(missing.status, 404);
  });
});

describe('real pipeline', () => {
  let dir;
  let server;
  before(async () => {
    dir = makeSandbox('real');
    server = await startServer(dir);
  });
  after(async () => {
    await stopServer(server);
    removeSandbox(dir);
  });

  test('a GLB the pipeline cannot process ends in error with the pipeline_error reason', async () => {
    // Valid GLB container (passes upload validation) without any mesh
    const { status, body } = await upload(server.url, { file: makeGlb({ asset: { version: '2.0' } }) });
    assert.equal(status, 200, JSON.stringify(body));
    await waitForJobEnd(server.url, body.jobId, 60000);
    const job = await waitForExit(server.url, body.jobId);
    assert.equal(job.status, 'error', JSON.stringify(job));
    assert.ok(job.error && !job.error.includes('Traceback'), job.error);
    assert.ok(job.errorType);
    assert.ok(Number.isInteger(job.errorStep));
    assert.match(job.stderrTail, /Traceback \(most recent call last\)/);
    const stored = JSON.parse(fs.readFileSync(path.join(dir, 'workspaces', body.jobId, 'metrics.json'), 'utf-8'));
    assert.equal(stored.status, 'error');
    assert.equal(stored.error, job.error);
  });
});

describe('startup and environment', () => {
  const dirs = [];
  const servers = [];
  after(async () => {
    for (const s of servers) await stopServer(s);
    for (const d of dirs) removeSandbox(d);
  });

  test('a missing .venv/bin/python fails the job with a clear error (no system python3)', async () => {
    const dir = makeSandbox('no-venv');
    dirs.push(dir);
    const server = await startServer(dir);
    servers.push(server);
    const { status, body } = await upload(server.url, { samplePath: SAMPLE });
    assert.equal(status, 200, JSON.stringify(body));
    const job = await waitForJobEnd(server.url, body.jobId);
    assert.equal(job.status, 'error');
    assert.match(job.error, /\.venv\/bin\/python/);
  });

  test('a port in use makes the server exit non-zero instead of binding another port', async () => {
    const dir = makeSandbox('no-venv');
    dirs.push(dir);
    const first = await startServer(dir);
    servers.push(first);
    const port = new URL(first.url).port;
    const second = spawnServer(dir, port);
    servers.push(second);
    const [code] = await once(second.proc, 'exit');
    assert.equal(code, 1);
    assert.match(second.output(), new RegExp(`Port ${port} is already in use`));
    assert.doesNotMatch(second.output(), /Local URL/);
  });

  test('an invalid PORT makes the server exit non-zero', async () => {
    const dir = makeSandbox('no-venv');
    dirs.push(dir);
    const server = spawnServer(dir, 'abc');
    servers.push(server);
    const [code] = await once(server.proc, 'exit');
    assert.equal(code, 1);
    assert.match(server.output(), /Invalid PORT 'abc'/);
  });
});
