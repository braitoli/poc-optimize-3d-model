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
  dedup
} from '@gltf-transform/functions';
import {
  MeshoptEncoder,
  MeshoptDecoder
} from 'meshoptimizer';
import { computeStandardSmoothNormals } from './smooth_normals.mjs';

const execFileAsync = promisify(execFile);

let _cachedSharp = undefined;
async function getSharpModule() {
  if (_cachedSharp !== undefined) return _cachedSharp;
  try {
    const mod = await import('sharp');
    _cachedSharp = mod.default || mod;
  } catch {
    _cachedSharp = null;
  }
  return _cachedSharp;
}

function getImageDimensions(buffer) {
  if (!buffer || buffer.length < 24) return null;
  // PNG: IHDR chunk width/height at byte 16 and 20
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) {
    return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
  }
  // JPEG: scan for SOF marker
  if (buffer[0] === 0xff && buffer[1] === 0xd8) {
    let offset = 2;
    while (offset < buffer.length - 8) {
      if (buffer[offset] !== 0xff) { offset++; continue; }
      const marker = buffer[offset + 1];
      if (marker === 0xd9 || marker === 0xda) break;
      const len = buffer.readUInt16BE(offset + 2);
      if ((marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) || (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)) {
        return { height: buffer.readUInt16BE(offset + 5), width: buffer.readUInt16BE(offset + 7) };
      }
      offset += 2 + len;
    }
  }
  return null;
}

export async function compressTexturesKtx2(doc, options = {}) {
  const mode = options.mode || 'uastc';
  const uastcLevel = options.uastcLevel ?? 2;
  const uastcRdo = options.uastcRdo ?? 1.0;
  const uastcRdoD = options.uastcRdoD ?? 2048;
  const etc1sQuality = options.etc1sQuality ?? 128;
  const compLevel = options.compLevel ?? 1;
  const maxDim = options.maxDim ?? null;
  const generateMipmaps = options.mipmaps !== false;
  const totalCores = options.threads ?? (os.cpus()?.length || 4);

  const textures = doc.getRoot().listTextures();
  if (textures.length === 0) {
    return { count: 0, beforeBytes: 0, afterBytes: 0 };
  }

  // Verify basisu availability
  try {
    await execFileAsync('basisu', ['-version']);
  } catch {
    console.warn('   ⚠️ Warning: `basisu` CLI not found in system PATH. Skipping KTX2 compression.');
    return { count: 0, beforeBytes: 0, afterBytes: 0, skipped: true };
  }

  const sharpMod = await getSharpModule();
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'poc-ktx2-'));

  const pendingTextures = [];
  let totalBefore = 0;

  for (let i = 0; i < textures.length; i++) {
    const tex = textures[i];
    const mime = tex.getMimeType();
    if (mime === 'image/ktx2') continue;

    const imgBuffer = Buffer.from(tex.getImage());
    if (!imgBuffer || imgBuffer.byteLength === 0) continue;

    totalBefore += imgBuffer.byteLength;
    pendingTextures.push({ tex, i, imgBuffer, mime });
  }

  if (pendingTextures.length === 0) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    return { count: 0, beforeBytes: 0, afterBytes: 0 };
  }

  // Determine concurrency and thread allocation per texture
  const concurrency = Math.min(pendingTextures.length, Math.max(1, Math.floor(totalCores / 2)));
  const threadsPerTex = Math.max(1, Math.floor(totalCores / concurrency));

  let totalAfter = 0;
  let processedCount = 0;

  try {
    async function processTextureItem({ tex, i, imgBuffer, mime }) {
      const isJpeg = mime === 'image/jpeg' || mime === 'image/jpg';
      const isWebp = mime === 'image/webp' || (imgBuffer.length >= 12 && imgBuffer.subarray(0, 4).toString() === 'RIFF' && imgBuffer.subarray(8, 12).toString() === 'WEBP');

      let inPath;
      if (isWebp) {
        const pngPath = path.join(tmpDir, `tex_${i}.png`);
        if (sharpMod) {
          const pngBuf = await sharpMod(imgBuffer).png().toBuffer();
          await fs.writeFile(pngPath, pngBuf);
        } else {
          const webpPath = path.join(tmpDir, `tex_${i}.webp`);
          await fs.writeFile(webpPath, imgBuffer);
          try {
            await execFileAsync('sips', ['-s', 'format', 'png', webpPath, '--out', pngPath]);
          } catch {
            await execFileAsync('python3', ['-c', `from PIL import Image; Image.open('${webpPath}').save('${pngPath}')`]);
          }
        }
        inPath = pngPath;
      } else {
        const ext = isJpeg ? '.jpg' : '.png';
        inPath = path.join(tmpDir, `tex_${i}${ext}`);
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

      if (generateMipmaps) {
        basisuArgs.push('-mipmap', '-mip_fast');
      }

      // Check if image actually exceeds maxDim before adding -resample
      const dims = getImageDimensions(imgBuffer);
      const needsResample = maxDim && maxDim > 0 && (!dims || dims.width > maxDim || dims.height > maxDim);
      if (needsResample) {
        basisuArgs.push('-resample', String(maxDim), String(maxDim));
      }

      basisuArgs.push('-output_path', tmpDir);
      basisuArgs.push(inPath);

      await execFileAsync('basisu', basisuArgs);

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
  const maxDim = options.maxDim ?? null;

  const textures = doc.getRoot().listTextures();
  if (textures.length === 0) {
    return { count: 0, beforeBytes: 0, afterBytes: 0 };
  }

  const sharpMod = await getSharpModule();
  const tmpDir = sharpMod ? null : await fs.mkdtemp(path.join(os.tmpdir(), 'poc-webp-'));
  let totalBefore = 0;
  let totalAfter = 0;
  let processedCount = 0;

  try {
    for (let i = 0; i < textures.length; i++) {
      const tex = textures[i];
      const mime = tex.getMimeType();
      const imgBuffer = Buffer.from(tex.getImage());
      if (!imgBuffer || imgBuffer.byteLength === 0) continue;

      totalBefore += imgBuffer.byteLength;
      let webpData;

      if (sharpMod) {
        let pipeline = sharpMod(imgBuffer);
        if (maxDim && maxDim > 0) {
          pipeline = pipeline.resize(maxDim, maxDim, { fit: 'inside', withoutEnlargement: true });
        }
        webpData = await pipeline.webp({ quality }).toBuffer();
      } else {
        const isWebp = mime === 'image/webp' || (imgBuffer.length >= 12 && imgBuffer.subarray(0, 4).toString() === 'RIFF' && imgBuffer.subarray(8, 12).toString() === 'WEBP');
        const isJpeg = mime === 'image/jpeg' || mime === 'image/jpg';
        const ext = isJpeg ? '.jpg' : (isWebp ? '.webp' : '.png');
        const inPath = path.join(tmpDir, `tex_${i}${ext}`);
        const outPath = path.join(tmpDir, `tex_${i}_out.webp`);
        await fs.writeFile(inPath, imgBuffer);

        const pyScript = `
from PIL import Image
im = Image.open('${inPath}')
if ${maxDim ? maxDim : 0} > 0 and (im.width > ${maxDim || 0} or im.height > ${maxDim || 0}):
    im.thumbnail((${maxDim || 0}, ${maxDim || 0}), Image.Resampling.LANCZOS)
im.save('${outPath}', 'WEBP', quality=${quality})
`;
        await execFileAsync('python3', ['-c', pyScript]);
        webpData = await fs.readFile(outPath);
      }

      totalAfter += webpData.byteLength;
      processedCount++;

      tex.setImage(new Uint8Array(webpData));
      tex.setMimeType('image/webp');
    }

    if (processedCount > 0) {
      doc.createExtension(EXTTextureWebP).setRequired(true);
    }
  } finally {
    if (tmpDir) {
      await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    }
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

async function runCli() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`Usage: node optimize_meshopt.mjs <input.glb> <output.glb> [options]
Options:
  --smooth-normals         Compute angle-weighted smooth normals across seams (default: ON)
  --no-smooth-normals      Disable smooth normals
  --keep-uv-float32        Keep UV as Float32 (default: ON)
  --pos-bits <bits>        Quantize position bits (default: 14)
  --normal-bits <bits>     Quantize normal bits (default: 12)
  --weld <tol>             Weld tolerance (default: 0.0001)
  --reorder                Reorder for GPU cache (default: ON)
  --meshopt                Enable EXT_meshopt_compression (default: ON)
  --ktx2                   Enable KTX2 UASTC texture compression (default: ON)
  --no-ktx2                Disable KTX2
  --ktx2-mode <mode>       uastc (default) or etc1s
  --ktx2-level <level>     UASTC level 0-4 (default: 2)
  --ktx2-rdo <float>       UASTC RDO lambda (default: 1.0)
  --texture-max-dim <px>   Texture max dimension (e.g. 1024, 2048)
  --webp                   Use WebP instead of KTX2
  --webp-quality <1-100>   WebP quality (default: 85)
  --single-sided           Force single-sided material
  --keep-double-sided      Keep double-sided material
  --json                   Output machine-readable JSON
`);
    process.exit(0);
  }

  let inputFile = null;
  let outputFile = null;
  let enableSmoothNormals = true;
  let keepUvFloat32 = true;
  let posBits = 14;
  let normalBits = 12;
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
  let enableJson = false;
  let exportIntermediateMeshopt = null;
  let texturesOnly = false;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--textures-only') texturesOnly = true;
    else if (a === '--smooth-normals') enableSmoothNormals = true;
    else if (a === '--no-smooth-normals') enableSmoothNormals = false;
    else if (a === '--keep-uv-float32') keepUvFloat32 = true;
    else if (a === '--export-intermediate-meshopt' && args[i + 1]) exportIntermediateMeshopt = args[++i];
    else if (a === '--pos-bits' && args[i + 1]) posBits = parseInt(args[++i], 10);
    else if (a === '--normal-bits' && args[i + 1]) normalBits = parseInt(args[++i], 10);
    else if (a === '--weld' && args[i + 1]) weldTol = parseFloat(args[++i]);
    else if (a === '--reorder') enableReorder = true;
    else if (a === '--meshopt') enableMeshopt = true;
    else if (a === '--no-meshopt') enableMeshopt = false;
    else if (a === '--ktx2') { enableKtx2 = true; enableWebp = false; }
    else if (a === '--no-ktx2') enableKtx2 = false;
    else if (a === '--ktx2-mode' && args[i + 1]) ktx2Mode = args[++i];
    else if (a === '--ktx2-level' && args[i + 1]) ktx2Level = parseInt(args[++i], 10);
    else if (a === '--ktx2-rdo' && args[i + 1]) ktx2Rdo = parseFloat(args[++i]);
    else if (a === '--ktx2-rdo-d' && args[i + 1]) ktx2RdoD = parseInt(args[++i], 10);
    else if (a === '--ktx2-threads' && args[i + 1]) ktx2Threads = parseInt(args[++i], 10);
    else if ((a === '--texture-max-dim' || a === '--ktx2-max-dim') && args[i + 1]) textureMaxDim = parseInt(args[++i], 10);
    else if (a === '--webp') { enableWebp = true; enableKtx2 = false; }
    else if (a === '--webp-quality' && args[i + 1]) webpQuality = parseInt(args[++i], 10);
    else if (a === '--single-sided') forceSingleSided = true;
    else if (a === '--keep-double-sided' || a === '--double-sided') forceSingleSided = false;
    else if (a === '--json') enableJson = true;
    else if (a.startsWith('--ratio') || a.startsWith('--target-faces')) {
      // RULE 11 GUARD: Ignore decimation requests
      if (!enableJson) console.warn('   ⚠️ [RULE 11 GUARD] Mesh decimation request bypassed. 100% triangles preserved.');
    } else if (!inputFile) inputFile = a;
    else if (!outputFile) outputFile = a;
  }

  if (!inputFile || !outputFile) {
    console.error('Error: Both input.glb and output.glb paths are required.');
    process.exit(1);
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
  if (forceSingleSided) {
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

    // Quantize: exclude TEXCOORD from pattern to keep Float32 UV
    const quantizePattern = keepUvFloat32
      ? /^(POSITION|NORMAL|COLOR.*|JOINTS.*|WEIGHTS.*)$/
      : /.*/;

    transforms.push(
      quantize({
        pattern: quantizePattern,
        quantizePosition: posBits,
        quantizeNormal: normalBits,
        quantizeTexcoord: 12
      })
    );

    await doc.transform(...transforms);
  }

  // 4. EXT_meshopt_compression
  if (enableMeshopt) {
    doc.createExtension(EXTMeshoptCompression).setRequired(true);
  }

  // Intermediate Step 5 Export (Geometry + Meshopt compressed, before texture transcode)
  if (exportIntermediateMeshopt) {
    const intermediateMeshoptGlb = await io.writeBinary(doc);
    await fs.mkdir(path.dirname(exportIntermediateMeshopt), { recursive: true });
    await fs.writeFile(exportIntermediateMeshopt, Buffer.from(intermediateMeshoptGlb));
    if (!enableJson) console.log(`   * Exported Step 5 Meshopt intermediate: ${exportIntermediateMeshopt}`);
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
    console.error('Fatal optimization error:', err);
    process.exit(1);
  });
}
