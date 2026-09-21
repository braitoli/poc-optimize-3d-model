#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { NodeIO, Primitive } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression, KHRTextureBasisu, EXTTextureWebP } from '@gltf-transform/extensions';
import {
  weld,
  compactPrimitive,
  reorder,
  quantize,
  prune,
  dedup,
  listTextureSlots,
  getTextureColorSpace
} from '@gltf-transform/functions';
import {
  MeshoptEncoder,
  MeshoptDecoder
} from 'meshoptimizer';
import { computeStandardSmoothNormals } from './smooth_normals.mjs';

const execFileAsync = promisify(execFile);

export const KTX2_MODES = ['uastc', 'etc1s'];

/** sharp decodes, measures, converts and resizes every texture; there is no other image backend. */
async function requireSharp() {
  try {
    const mod = await import('sharp');
    return mod.default || mod;
  } catch (err) {
    throw new Error(`sharp is required for texture compression but could not be loaded: ${err.message}`);
  }
}

/** Encoded bytes of texture i; an image without data is an error, never skipped. */
function textureBytes(tex, i) {
  const image = tex.getImage();
  if (!image || image.byteLength === 0) {
    throw new Error(`texture ${i} (${tex.getName() || 'unnamed'}) has an empty image`);
  }
  return Buffer.from(image);
}

/** Real format and pixel size of an encoded PNG / JPEG / WebP texture. */
async function readImageInfo(sharp, buffer, i) {
  let meta;
  try {
    meta = await sharp(buffer).metadata();
  } catch (err) {
    throw new Error(`texture ${i}: cannot decode the image (${err.message})`);
  }
  if (!['png', 'jpeg', 'webp'].includes(meta.format)) {
    throw new Error(`texture ${i}: unsupported image format '${meta.format}' (expected PNG, JPEG or WebP)`);
  }
  if (!(meta.width > 0 && meta.height > 0)) {
    throw new Error(`texture ${i}: cannot read the image size`);
  }
  return { format: meta.format, width: meta.width, height: meta.height };
}

function checkMaxDim(maxDim) {
  if (maxDim !== null && !(Number.isInteger(maxDim) && maxDim > 0)) {
    throw new Error(`maxDim must be a positive integer or null, got ${maxDim}`);
  }
  return maxDim;
}

/** Size with the longer side reduced to maxDim, aspect ratio kept; null when the image already fits (never upscales). */
function fitWithin(width, height, maxDim) {
  const longer = Math.max(width, height);
  if (maxDim === null || longer <= maxDim) return null;
  const scale = maxDim / longer;
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/** First non-empty stderr line of a failed child process (the full error stays on the stack). */
function firstErrorLine(err) {
  const lines = `${err.stderr || ''}\n${err.message || ''}`.split('\n').map(l => l.trim()).filter(Boolean);
  return lines.find(l => /error/i.test(l)) || lines[0] || String(err);
}

export async function compressTexturesKtx2(doc, options = {}) {
  const mode = options.mode ?? 'uastc';
  if (!KTX2_MODES.includes(mode)) {
    throw new Error(`Unsupported KTX2 mode '${mode}' (expected ${KTX2_MODES.join(' or ')})`);
  }
  const uastcLevel = options.uastcLevel ?? 2;
  const uastcRdo = options.uastcRdo ?? 1.0;
  const uastcRdoD = options.uastcRdoD ?? 2048;
  const etc1sQuality = options.etc1sQuality ?? 128;
  const compLevel = options.compLevel ?? 1;
  const maxDim = checkMaxDim(options.maxDim ?? null);
  const generateMipmaps = options.mipmaps !== false;
  const totalCores = options.threads ?? (os.cpus()?.length || 4);

  const textures = doc.getRoot().listTextures();
  const pendingTextures = [];
  let totalBefore = 0;

  for (let i = 0; i < textures.length; i++) {
    const tex = textures[i];
    // Already KTX2: skipped, so re-running the compression is a no-op
    if (tex.getMimeType() === 'image/ktx2') continue;

    const imgBuffer = textureBytes(tex, i);

    // glTF: only colour slots (baseColor, emissive, ...) are sRGB; normal/metallicRoughness/occlusion are linear data.
    // Textures with no material slot keep the previous (sRGB) behaviour.
    const slots = listTextureSlots(tex);
    let colorSpace = 'srgb';
    if (getTextureColorSpace(tex) === 'srgb') {
      const dataSlots = slots.filter(s => !/color|emissive|diffuse/i.test(s)); // gltf-transform's colour-slot name rule
      if (dataSlots.length > 0) {
        console.warn(`   ⚠️ Warning: texture ${i} is shared by colour and data slots (${slots.join(', ')}); encoding as sRGB.`);
      }
    } else if (slots.length > 0) {
      colorSpace = slots.includes('normalTexture') ? 'normal' : 'linear';
    }

    totalBefore += imgBuffer.byteLength;
    pendingTextures.push({ tex, i, imgBuffer, colorSpace });
  }

  if (pendingTextures.length === 0) {
    return { count: 0, beforeBytes: 0, afterBytes: 0 };
  }

  try {
    await execFileAsync('basisu', ['-version']);
  } catch (err) {
    const why = err.code === 'ENOENT' ? 'not found on PATH' : firstErrorLine(err);
    throw new Error(`basisu CLI is required for KTX2 compression but cannot be run (${why})`);
  }

  const sharp = await requireSharp();
  for (const item of pendingTextures) {
    item.info = await readImageInfo(sharp, item.imgBuffer, item.i);
  }

  // Determine concurrency and thread allocation per texture
  const concurrency = Math.min(pendingTextures.length, Math.max(1, Math.floor(totalCores / 2)));
  const threadsPerTex = Math.max(1, Math.floor(totalCores / concurrency));

  let totalAfter = 0;
  let processedCount = 0;
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'poc-ktx2-'));

  try {
    async function processTextureItem({ tex, i, imgBuffer, colorSpace, info }) {
      let inPath;
      if (info.format === 'webp') {
        // basisu reads PNG / JPEG only
        inPath = path.join(tmpDir, `tex_${i}.png`);
        await fs.writeFile(inPath, await sharp(imgBuffer).png().toBuffer());
      } else {
        inPath = path.join(tmpDir, `tex_${i}${info.format === 'jpeg' ? '.jpg' : '.png'}`);
        await fs.writeFile(inPath, imgBuffer);
      }

      const outPath = path.join(tmpDir, `tex_${i}.ktx2`);
      const basisuArgs = ['-ktx2', '-max_threads', String(threadsPerTex)];

      if (mode === 'uastc') {
        basisuArgs.push('-uastc');
        basisuArgs.push('-uastc_level', String(uastcLevel));
        if (uastcRdo > 0) {
          basisuArgs.push('-uastc_rdo_l', String(uastcRdo));
          if (uastcRdoD && uastcRdoD > 0) {
            basisuArgs.push('-uastc_rdo_d', String(uastcRdoD));
          }
        }
      } else {
        basisuArgs.push('-q', String(etc1sQuality));
        basisuArgs.push('-comp_level', String(compLevel));
      }

      // Without these basisu assumes sRGB and tags the KTX2 DFD as sRGB, so viewers decode
      // normals through the sRGB curve. Both flags imply linear metrics, linear mip filtering
      // and a linear transfer function (verified with basisu v2.50).
      if (colorSpace === 'normal') {
        basisuArgs.push('-normal_map');
      } else if (colorSpace === 'linear') {
        basisuArgs.push('-linear');
      }

      if (generateMipmaps) {
        basisuArgs.push('-mipmap', '-mip_fast');
      }

      // Only a texture larger than maxDim is resampled: longer side to maxDim, aspect ratio kept
      const resized = fitWithin(info.width, info.height, maxDim);
      if (resized) {
        basisuArgs.push('-resample', String(resized.width), String(resized.height));
      }

      basisuArgs.push('-output_path', tmpDir);
      basisuArgs.push(inPath);

      try {
        await execFileAsync('basisu', basisuArgs);
      } catch (err) {
        throw new Error(`texture ${i}: basisu KTX2 encoding failed: ${firstErrorLine(err)}`, { cause: err });
      }

      const ktx2Data = await fs.readFile(outPath);
      tex.setImage(new Uint8Array(ktx2Data));
      tex.setMimeType('image/ktx2');

      return ktx2Data.byteLength;
    }

    for (let idx = 0; idx < pendingTextures.length; idx += concurrency) {
      const batch = pendingTextures.slice(idx, idx + concurrency);
      const results = await Promise.all(batch.map(item => processTextureItem(item)));
      for (const bytes of results) {
        totalAfter += bytes;
        processedCount++;
      }
    }

    if (processedCount > 0) {
      doc.createExtension(KHRTextureBasisu).setRequired(true);
      const webpExt = doc.getRoot().listExtensionsUsed().find(e => e.extensionName === 'EXT_texture_webp');
      if (webpExt && !doc.getRoot().listTextures().some(t => t.getMimeType() === 'image/webp')) {
        webpExt.dispose();
      }
    }
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }

  return {
    count: processedCount,
    beforeBytes: totalBefore,
    afterBytes: totalAfter,
    mode
  };
}

export async function compressTexturesWebp(doc, options = {}) {
  const quality = options.quality ?? 85;
  const maxDim = checkMaxDim(options.maxDim ?? null);

  const textures = doc.getRoot().listTextures();
  if (textures.length === 0) {
    return { count: 0, beforeBytes: 0, afterBytes: 0 };
  }

  const sharp = await requireSharp();
  let totalBefore = 0;
  let totalAfter = 0;
  let processedCount = 0;

  for (let i = 0; i < textures.length; i++) {
    const tex = textures[i];
    const imgBuffer = textureBytes(tex, i);
    totalBefore += imgBuffer.byteLength;

    let pipeline = sharp(imgBuffer);
    if (maxDim !== null) {
      // Longer side to maxDim, aspect ratio kept, never upscaled
      pipeline = pipeline.resize(maxDim, maxDim, { fit: 'inside', withoutEnlargement: true });
    }
    let webpData;
    try {
      webpData = await pipeline.webp({ quality }).toBuffer();
    } catch (err) {
      throw new Error(`texture ${i}: WebP encoding failed (${err.message})`, { cause: err });
    }

    totalAfter += webpData.byteLength;
    processedCount++;

    tex.setImage(new Uint8Array(webpData));
    tex.setMimeType('image/webp');
  }

  if (processedCount > 0) {
    doc.createExtension(EXTTextureWebP).setRequired(true);
  }

  return {
    count: processedCount,
    beforeBytes: totalBefore,
    afterBytes: totalAfter,
    quality
  };
}

function getStats(doc) {
  let triangles = 0;
  let vertices = 0;
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute('POSITION');
      if (pos) vertices += pos.getCount();
      const ind = prim.getIndices();
      if (ind) triangles += ind.getCount() / 3;
      else if (pos) triangles += pos.getCount() / 3;
    }
  }
  return { triangles: Math.round(triangles), vertices: Math.round(vertices) };
}

/** Integer CLI value in [min, max]; anything else throws. */
function intArg(flag, raw, min, max) {
  if (!/^[+-]?\d+$/.test(raw)) throw new Error(`${flag} expects an integer, got '${raw}'`);
  return inRange(flag, Number(raw), min, max);
}

/** Finite numeric CLI value in [min, max]; anything else throws. */
function numberArg(flag, raw, min, max) {
  const n = raw.trim() === '' ? NaN : Number(raw);
  if (!Number.isFinite(n)) throw new Error(`${flag} expects a number, got '${raw}'`);
  return inRange(flag, n, min, max);
}

function inRange(flag, n, min, max) {
  if (n < min || n > max) {
    throw new Error(`${flag} must be ${max === Infinity ? `>= ${min}` : `between ${min} and ${max}`}, got ${n}`);
  }
  return n;
}

async function runCli() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`Usage: node optimize_meshopt.mjs <input.glb> <output.glb> [options]
Options:
  --textures-only          Only compress textures (no normals / weld / quantize / reorder)
  --smooth-normals         Compute angle-weighted smooth normals across seams (default: ON)
  --no-smooth-normals      Disable smooth normals
  --keep-uv-float32        Keep UV as Float32 (default: OFF, UV quantized to 16-bit; UVs outside [0,1] stay Float32)
  --pos-bits <1-16>        Quantize position bits (default: 14)
  --weld <tol>             Weld tolerance >= 0 (default: 0.0001)
  --reorder                Reorder for GPU cache (default: ON)
  --meshopt                Enable EXT_meshopt_compression (default: ON)
  --no-meshopt             Disable EXT_meshopt_compression
  --ktx2                   Enable KTX2 UASTC texture compression (default: ON)
  --no-ktx2                Disable KTX2
  --ktx2-mode <mode>       uastc (default) or etc1s
  --ktx2-level <0-4>       UASTC level (default: 2)
  --ktx2-rdo <float>       UASTC RDO lambda >= 0, 0 disables RDO (default: 1.0)
  --ktx2-rdo-d <bytes>     UASTC RDO dictionary size 1-65536 (default: 2048)
  --ktx2-threads <n>       basisu threads in total (default: CPU count)
  --texture-max-dim <px>   Downscale textures whose longer side exceeds px (aspect kept, never upscales)
  --webp                   Use WebP instead of KTX2
  --webp-quality <1-100>   WebP quality (default: 85)
  --single-sided           Force single-sided material
  --keep-double-sided      Keep double-sided material
  --json                   Output machine-readable JSON
  --ratio, --target-faces  Accepted and ignored: zero-decimation, every triangle is kept
`);
    process.exit(0);
  }

  let inputFile = null;
  let outputFile = null;
  let enableSmoothNormals = true;
  let keepUvFloat32 = false;
  let posBits = 14;
  let weldTol = 0.0001;
  let enableReorder = true;
  let enableMeshopt = true;
  let enableKtx2 = true;
  let enableWebp = false;
  let webpQuality = 85;
  let ktx2Mode = 'uastc';
  let ktx2Level = 2;
  let ktx2Rdo = 1.0;
  let ktx2RdoD = 2048;
  let ktx2Threads = os.cpus()?.length || 4;
  let textureMaxDim = null;
  let forceSingleSided = false;
  let keepDoubleSided = false;
  let enableJson = false;
  let texturesOnly = false;

  // Every argument is either a known option (with its value) or one of the two paths; anything else throws.
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    const value = () => {
      const v = args[i + 1];
      if (v === undefined || v.startsWith('--')) throw new Error(`${a} requires a value`);
      i++;
      return v;
    };
    if (a === '--textures-only') texturesOnly = true;
    else if (a === '--smooth-normals') enableSmoothNormals = true;
    else if (a === '--no-smooth-normals') enableSmoothNormals = false;
    else if (a === '--keep-uv-float32') keepUvFloat32 = true;
    else if (a === '--pos-bits') posBits = intArg(a, value(), 1, 16);
    else if (a === '--weld') weldTol = numberArg(a, value(), 0, Infinity);
    else if (a === '--reorder') enableReorder = true;
    else if (a === '--meshopt') enableMeshopt = true;
    else if (a === '--no-meshopt') enableMeshopt = false;
    else if (a === '--ktx2') { enableKtx2 = true; enableWebp = false; }
    else if (a === '--no-ktx2') enableKtx2 = false;
    else if (a === '--ktx2-mode') {
      ktx2Mode = value();
      if (!KTX2_MODES.includes(ktx2Mode)) throw new Error(`${a} must be ${KTX2_MODES.join(' or ')}, got '${ktx2Mode}'`);
    }
    else if (a === '--ktx2-level') ktx2Level = intArg(a, value(), 0, 4);
    else if (a === '--ktx2-rdo') ktx2Rdo = numberArg(a, value(), 0, Infinity);
    else if (a === '--ktx2-rdo-d') ktx2RdoD = intArg(a, value(), 1, 65536);
    else if (a === '--ktx2-threads') ktx2Threads = intArg(a, value(), 1, Infinity);
    else if (a === '--texture-max-dim' || a === '--ktx2-max-dim') textureMaxDim = intArg(a, value(), 1, Infinity);
    else if (a === '--webp') { enableWebp = true; enableKtx2 = false; }
    else if (a === '--webp-quality') webpQuality = intArg(a, value(), 1, 100);
    else if (a === '--single-sided') forceSingleSided = true;
    else if (a === '--keep-double-sided' || a === '--double-sided') keepDoubleSided = true;
    else if (a === '--json') enableJson = true;
    else if (a === '--ratio' || a === '--target-faces' || a.startsWith('--ratio=') || a.startsWith('--target-faces=')) {
      if (!a.includes('=')) value();
      // RULE 11 GUARD: Ignore decimation requests
      if (!enableJson) console.warn('   ⚠️ [RULE 11 GUARD] Mesh decimation request bypassed. 100% triangles preserved.');
    }
    else if (a.startsWith('-')) throw new Error(`Unknown option '${a}' (see --help)`);
    else if (!inputFile) inputFile = a;
    else if (!outputFile) outputFile = a;
    else throw new Error(`Unexpected argument '${a}': only <input.glb> <output.glb> are positional`);
  }

  // When --keep-double-sided is active, forceSingleSided must NEVER override it
  if (keepDoubleSided) {
    forceSingleSided = false;
  }

  if (!inputFile || !outputFile) {
    throw new Error('Both input.glb and output.glb paths are required.');
  }

  const startTime = Date.now();
  const inputBytes = await fs.readFile(inputFile);
  const inputSize = inputBytes.byteLength;

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'meshopt.decoder': MeshoptDecoder,
      'meshopt.encoder': MeshoptEncoder
    });

  const doc = await io.readBinary(new Uint8Array(inputBytes));
  const beforeStats = getStats(doc);

  if (!enableJson) {
    console.log(`📦 Input: ${inputFile} (${(inputSize / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`   Geometry: ${beforeStats.triangles.toLocaleString()} triangles, ${beforeStats.vertices.toLocaleString()} vertices`);
  }

  // 1. Compute angle-weighted smooth normals across seams
  if (!texturesOnly && enableSmoothNormals) {
    if (!enableJson) console.log('   * Computing Angle-Weighted Smooth Normals (spatial seam welding)...');
    computeStandardSmoothNormals(doc, { smoothAcrossUvSeams: true, spatialTolerance: 1e-5 });
  }

  // 2. Material double-sided adjustments
  if (keepDoubleSided) {
    for (const mat of doc.getRoot().listMaterials()) {
      mat.setDoubleSided(true);
    }
    if (!enableJson) console.log('   * Preserved/Set double-sided materials (--keep-double-sided active)');
  } else if (forceSingleSided) {
    for (const mat of doc.getRoot().listMaterials()) {
      mat.setDoubleSided(false);
    }
  }

  // 3. glTF Transform pipeline
  if (!texturesOnly) {
    const transforms = [
      dedup(),
      prune(),
      weld({ tolerance: weldTol })
    ];

    if (enableReorder) {
      transforms.push(reorder({ encoder: MeshoptEncoder }));
    }

    // Quantize: 16-bit UV (TEXCOORD excluded with --keep-uv-float32). NORMAL is left
    // Float32 here; the EXT_meshopt_compression OCTAHEDRAL filter encodes it below.
    const quantizePattern = keepUvFloat32
      ? /^(POSITION|COLOR.*|JOINTS.*|WEIGHTS.*)$/
      : /^(POSITION|TEXCOORD.*|COLOR.*|JOINTS.*|WEIGHTS.*)$/;

    transforms.push(
      quantize({
        pattern: quantizePattern,
        quantizePosition: posBits,
        quantizeTexcoord: 16
      })
    );

    await doc.transform(...transforms);
  }

  // 4. EXT_meshopt_compression
  if (enableMeshopt) {
    doc.createExtension(EXTMeshoptCompression)
      .setRequired(true)
      .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.FILTER });
  }

  // 5. Texture compression (KTX2 UASTC or WebP)
  let ktx2Result = null;
  let webpResult = null;

  if (enableWebp) {
    if (!enableJson) console.log(`   * Compressing textures to WebP (Q=${webpQuality}, maxDim=${textureMaxDim || 'native'})...`);
    webpResult = await compressTexturesWebp(doc, { quality: webpQuality, maxDim: textureMaxDim });
  } else if (enableKtx2) {
    if (!enableJson) console.log(`   * Compressing textures to KTX2 UASTC (Level ${ktx2Level}, maxDim=${textureMaxDim || 'native'})...`);
    ktx2Result = await compressTexturesKtx2(doc, {
      mode: ktx2Mode,
      uastcLevel: ktx2Level,
      uastcRdo: ktx2Rdo,
      uastcRdoD: ktx2RdoD,
      threads: ktx2Threads,
      maxDim: textureMaxDim,
      mipmaps: true
    });
  }

  // 6. Write output
  const outputGlb = await io.writeBinary(doc);
  await fs.mkdir(path.dirname(outputFile), { recursive: true });
  await fs.writeFile(outputFile, Buffer.from(outputGlb));

  const outputSize = outputGlb.byteLength;
  const afterStats = getStats(doc);
  const elapsed = (Date.now() - startTime) / 1000;

  const resultSummary = {
    success: true,
    inputFile,
    outputFile,
    inputBytes: inputSize,
    outputBytes: outputSize,
    savedBytes: inputSize - outputSize,
    savedPercent: Number(((1 - outputSize / inputSize) * 100).toFixed(2)),
    trianglesBefore: beforeStats.triangles,
    trianglesAfter: afterStats.triangles,
    verticesBefore: beforeStats.vertices,
    verticesAfter: afterStats.vertices,
    elapsedSeconds: Number(elapsed.toFixed(2)),
    ktx2: ktx2Result,
    webp: webpResult
  };

  if (enableJson) {
    console.log(JSON.stringify(resultSummary));
  } else {
    console.log(`✅ Completed in ${elapsed.toFixed(2)}s:`);
    console.log(`   Output: ${outputFile} (${(outputSize / 1024 / 1024).toFixed(2)} MB, saved ${resultSummary.savedPercent}%)`);
    console.log(`   Triangles: ${afterStats.triangles.toLocaleString()} (100% Zero-Decimation preserved)`);
    console.log(`   Vertices: ${beforeStats.vertices.toLocaleString()} -> ${afterStats.vertices.toLocaleString()}`);
  }
}

const isDirectRun = process.argv[1] && (
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1]) ||
  process.argv[1].endsWith('optimize_meshopt.mjs')
);

if (isDirectRun) {
  runCli().catch(err => {
    // First stderr line: the one-line reason (step_pipeline reports it); then the full error
    console.error(`Fatal optimization error: ${String(err?.message ?? err).split('\n')[0]}`);
    console.error(err);
    process.exit(1);
  });
}
