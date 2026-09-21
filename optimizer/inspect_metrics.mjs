#!/usr/bin/env node
/**
 * inspect_metrics.mjs
 *
 * Comprehensive 3D Model GLB Metrics Extraction Engine.
 * Utilizes @gltf-transform/core, @gltf-transform/functions (inspect),
 * and meshoptimizer to inspect and output standard metrics:
 * - fileSizeBytes & fileSizeFormatted
 * - faces (triangle count)
 * - vertices count
 * - meshes count & primitives count
 * - drawCalls estimate
 * - uvChannels (TEXCOORD_0, ...) with bounds & stats
 * - textures (mime, resolution, fileSizeBytes, uncompressed vs KTX2 GPU VRAM)
 * - totalGpuVramBytes & totalGpuVramFormatted
 * - materials (name, alphaMode, doubleSided, etc.)
 * - boundingBox (min, max, dimensions, center)
 * - palette & primaryColor (from extras)
 * - extensions (used & required)
 *
 * Usage via CLI:
 *   node optimizer/inspect_metrics.mjs <model.glb> [--json]
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { inspect } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';

export function formatBytes(bytes, decimals = 2) {
  if (!+bytes || bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Extracts comprehensive 3D metrics from a GLB buffer or file path.
 * @param {string|Uint8Array|Buffer} input - Path to .glb file or GLB byte buffer
 * @returns {Promise<Object>} Comprehensive 3D metrics
 */
export async function inspectGlbMetrics(input) {
  let fileSizeBytes = 0;
  let inputPath = null;
  let glbBuffer = null;

  if (typeof input === 'string') {
    inputPath = path.resolve(input);
    const stat = await fs.stat(inputPath);
    fileSizeBytes = stat.size;
    glbBuffer = await fs.readFile(inputPath);
  } else if (Buffer.isBuffer(input) || input instanceof Uint8Array) {
    fileSizeBytes = input.byteLength;
    glbBuffer = input;
  } else {
    throw new TypeError('Input must be a file path string or a Buffer/Uint8Array.');
  }

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'meshopt.decoder': MeshoptDecoder,
      'meshopt.encoder': MeshoptEncoder
    });

  const doc = await io.readBinary(new Uint8Array(glbBuffer));
  const report = inspect(doc);
  const root = doc.getRoot();

  // 1. Faces, Vertices, Meshes, Primitives
  let totalFaces = 0;
  let totalVertices = 0;
  let primitiveCount = 0;
  const uvChannelMap = new Map();

  // Fallback direct accessor bounding box
  let fallbackBboxMin = [Infinity, Infinity, Infinity];
  let fallbackBboxMax = [-Infinity, -Infinity, -Infinity];

  for (const mesh of root.listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      primitiveCount++;

      // POSITION
      const pos = prim.getAttribute('POSITION');
      if (pos) {
        totalVertices += pos.getCount();
        const pMin = pos.getMin([0, 0, 0]);
        const pMax = pos.getMax([0, 0, 0]);
        if (pMin && pMax) {
          for (let i = 0; i < 3; i++) {
            fallbackBboxMin[i] = Math.min(fallbackBboxMin[i], pMin[i]);
            fallbackBboxMax[i] = Math.max(fallbackBboxMax[i], pMax[i]);
          }
        }
      }

      // TRIANGLES
      const indices = prim.getIndices();
      if (indices) {
        totalFaces += Math.floor(indices.getCount() / 3);
      } else if (pos) {
        totalFaces += Math.floor(pos.getCount() / 3);
      }

      // UV CHANNELS
      for (const sem of prim.listSemantics()) {
        if (sem.startsWith('TEXCOORD_')) {
          const uvAcc = prim.getAttribute(sem);
          if (!uvAcc) continue;
          const uMin = uvAcc.getMin([0, 0]) || [0, 0];
          const uMax = uvAcc.getMax([0, 0]) || [0, 0];
          const count = uvAcc.getCount();

          if (!uvChannelMap.has(sem)) {
            uvChannelMap.set(sem, {
              channel: sem,
              count,
              min: [Number(uMin[0].toFixed(5)), Number(uMin[1].toFixed(5))],
              max: [Number(uMax[0].toFixed(5)), Number(uMax[1].toFixed(5))],
              bounds: {
                uMin: Number(uMin[0].toFixed(5)),
                uMax: Number(uMax[0].toFixed(5)),
                vMin: Number(uMin[1].toFixed(5)),
                vMax: Number(uMax[1].toFixed(5)),
                uRange: Number((uMax[0] - uMin[0]).toFixed(5)),
                vRange: Number((uMax[1] - uMin[1]).toFixed(5))
              },
              normalized: uMin[0] >= -0.01 && uMax[0] <= 1.01 && uMin[1] >= -0.01 && uMax[1] <= 1.01
            });
          } else {
            const existing = uvChannelMap.get(sem);
            existing.count += count;
            existing.min[0] = Math.min(existing.min[0], Number(uMin[0].toFixed(5)));
            existing.min[1] = Math.min(existing.min[1], Number(uMin[1].toFixed(5)));
            existing.max[0] = Math.max(existing.max[0], Number(uMax[0].toFixed(5)));
            existing.max[1] = Math.max(existing.max[1], Number(uMax[1].toFixed(5)));
            existing.bounds.uMin = existing.min[0];
            existing.bounds.uMax = existing.max[0];
            existing.bounds.vMin = existing.min[1];
            existing.bounds.vMax = existing.max[1];
            existing.bounds.uRange = Number((existing.max[0] - existing.min[0]).toFixed(5));
            existing.bounds.vRange = Number((existing.max[1] - existing.min[1]).toFixed(5));
            existing.normalized = existing.bounds.uMin >= -0.01 && existing.bounds.uMax <= 1.01 && existing.bounds.vMin >= -0.01 && existing.bounds.vMax <= 1.01;
          }
        }
      }
    }
  }

  // 2. Draw calls estimate
  let drawCalls = 0;
  const scenes = root.listScenes();
  const activeScene = root.getDefaultScene() || (scenes.length > 0 ? scenes[0] : null);
  if (activeScene) {
    activeScene.traverse(node => {
      const mesh = node.getMesh();
      if (mesh) {
        drawCalls += mesh.listPrimitives().length;
      }
    });
  }
  if (drawCalls === 0) {
    drawCalls = primitiveCount;
  }

  // 3. Bounding box (prefer inspect scenes bbox which dequantizes & applies transforms)
  let bboxMin = [0, 0, 0];
  let bboxMax = [0, 0, 0];
  let hasBbox = false;

  if (report.scenes && report.scenes.properties && report.scenes.properties.length > 0) {
    const sceneBbox = report.scenes.properties[0];
    if (sceneBbox && sceneBbox.bboxMin && sceneBbox.bboxMax) {
      bboxMin = sceneBbox.bboxMin.map(v => Number(v.toFixed(5)));
      bboxMax = sceneBbox.bboxMax.map(v => Number(v.toFixed(5)));
      hasBbox = true;
    }
  }

  if (!hasBbox && isFinite(fallbackBboxMin[0])) {
    bboxMin = fallbackBboxMin.map(v => Number(v.toFixed(5)));
    bboxMax = fallbackBboxMax.map(v => Number(v.toFixed(5)));
    hasBbox = true;
  }

  const dimensions = [
    Number((bboxMax[0] - bboxMin[0]).toFixed(5)),
    Number((bboxMax[1] - bboxMin[1]).toFixed(5)),
    Number((bboxMax[2] - bboxMin[2]).toFixed(5))
  ];
  const center = [
    Number(((bboxMin[0] + bboxMax[0]) / 2).toFixed(5)),
    Number(((bboxMin[1] + bboxMax[1]) / 2).toFixed(5)),
    Number(((bboxMin[2] + bboxMax[2]) / 2).toFixed(5))
  ];

  // 4. Textures & GPU VRAM
  const textures = [];
  let totalGpuVramBytes = 0;
  let totalGpuVramUncompressedBytes = 0;
  const reportTextures = report.textures?.properties || [];

  const docTextures = root.listTextures();
  for (let i = 0; i < docTextures.length; i++) {
    const tex = docTextures[i];
    const repTex = reportTextures[i] || {};
    const mimeType = tex.getMimeType() || repTex.mimeType || 'unknown';
    const isKtx2 = mimeType === 'image/ktx2';
    const isWebp = mimeType === 'image/webp';
    const isPng = mimeType === 'image/png';
    const isJpeg = mimeType === 'image/jpeg' || mimeType === 'image/jpg';

    let formatName = 'UNKNOWN';
    if (isKtx2) formatName = 'KTX2';
    else if (isWebp) formatName = 'WebP';
    else if (isPng) formatName = 'PNG';
    else if (isJpeg) formatName = 'JPEG';

    // Resolution, read from the encoded image; without it the texture and VRAM metrics would be made up
    const size = tex.getSize();
    if (!size || !(size[0] > 0) || !(size[1] > 0)) {
      throw new Error(
        `Cannot read the size of texture ${i} (${tex.getName() || 'unnamed'}, ${mimeType}, ` +
        `${tex.getImage()?.byteLength ?? 0} bytes): the image is missing, corrupt or in an unsupported format`
      );
    }
    const [width, height] = size;

    const imgBuffer = tex.getImage();
    const texFileBytes = imgBuffer ? imgBuffer.byteLength : (repTex.size || 0);

    // GPU VRAM calculation:
    // Uncompressed RGBA8888 with mipmaps: W * H * 4 bytes * 1.3333333333
    // KTX2 UASTC/BC7/ETC2 with mipmaps: W * H * 1 byte * 1.3333333333
    const vramUncompressed = Math.round(width * height * 4 * (4 / 3));
    const vramKtx2 = Math.round(width * height * 1 * (4 / 3));

    // Estimated GPU VRAM based on format
    let estimatedVram = repTex.gpuSize || (isKtx2 ? vramKtx2 : vramUncompressed);
    if (!estimatedVram && width > 0 && height > 0) {
      estimatedVram = isKtx2 ? vramKtx2 : vramUncompressed;
    }

    totalGpuVramBytes += estimatedVram;
    totalGpuVramUncompressedBytes += vramUncompressed;

    textures.push({
      name: tex.getName() || repTex.name || `texture_${i}`,
      mimeType,
      format: formatName,
      compression: repTex.compression || (isKtx2 ? 'UASTC' : 'None'),
      resolution: [width, height],
      resolutionFormatted: `${width}x${height}`,
      fileSizeBytes: texFileBytes,
      fileSizeFormatted: formatBytes(texFileBytes),
      slots: repTex.slots || [],
      gpuVramUncompressedBytes: vramUncompressed,
      gpuVramUncompressedFormatted: formatBytes(vramUncompressed),
      gpuVramKtx2Bytes: vramKtx2,
      gpuVramKtx2Formatted: formatBytes(vramKtx2),
      estimatedGpuVramBytes: estimatedVram,
      estimatedGpuVramFormatted: formatBytes(estimatedVram)
    });
  }

  // 5. Materials
  const materials = [];
  const reportMaterials = report.materials?.properties || [];
  const docMaterials = root.listMaterials();
  for (let i = 0; i < docMaterials.length; i++) {
    const mat = docMaterials[i];
    const repMat = reportMaterials[i] || {};
    materials.push({
      name: mat.getName() || repMat.name || `material_${i}`,
      alphaMode: mat.getAlphaMode() || repMat.alphaMode || 'OPAQUE',
      doubleSided: mat.getDoubleSided() ?? repMat.doubleSided ?? false,
      roughnessFactor: mat.getRoughnessFactor(),
      metallicFactor: mat.getMetallicFactor(),
      hasBaseColorTexture: !!mat.getBaseColorTexture()
    });
  }

  // 6. Extras (Palette, Primary Color)
  const extras = root.getExtras() || {};
  const palette = Array.isArray(extras.palette) ? extras.palette : [];
  const primaryColor = typeof extras.primaryColor === 'string' ? extras.primaryColor : (palette[0] || null);
  const paletteDetails = Array.isArray(extras.paletteDetails) ? extras.paletteDetails : [];

  // 7. Extensions
  const extensionsUsed = root.listExtensionsUsed().map(e => e.extensionName);
  const extensionsRequired = root.listExtensionsRequired().map(e => e.extensionName);

  return {
    fileSizeBytes,
    fileSizeFormatted: formatBytes(fileSizeBytes),
    faces: totalFaces,
    vertices: totalVertices,
    meshes: root.listMeshes().length,
    primitives: primitiveCount,
    drawCalls,
    uvChannels: Array.from(uvChannelMap.values()),
    textures,
    totalGpuVramBytes,
    totalGpuVramFormatted: formatBytes(totalGpuVramBytes),
    totalGpuVramUncompressedBytes,
    totalGpuVramUncompressedFormatted: formatBytes(totalGpuVramUncompressedBytes),
    materials,
    boundingBox: {
      min: bboxMin,
      max: bboxMax,
      dimensions,
      center
    },
    palette,
    primaryColor,
    paletteDetails,
    extensions: {
      used: extensionsUsed,
      required: extensionsRequired
    },
    extras
  };
}

async function runCli() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: node optimizer/inspect_metrics.mjs <file.glb> [--compact]`);
    process.exit(0);
  }

  const inputFile = args.find(a => !a.startsWith('-'));
  const isCompact = args.includes('--compact');

  if (!inputFile) {
    console.error('Error: Please provide a valid .glb file path.');
    process.exit(1);
  }

  try {
    const metrics = await inspectGlbMetrics(inputFile);
    if (isCompact) {
      console.log(JSON.stringify(metrics));
    } else {
      console.log(JSON.stringify(metrics, null, 2));
    }
  } catch (err) {
    // First stderr line: the one-line reason (step_pipeline reports it); then the full error
    console.error(`Failed to inspect GLB metrics: ${String(err?.message ?? err).split('\n')[0]}`);
    console.error(err);
    process.exit(1);
  }
}

const isDirectRun = process.argv[1] && (
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1]) ||
  process.argv[1].endsWith('inspect_metrics.mjs')
);

if (isDirectRun) {
  runCli();
}
