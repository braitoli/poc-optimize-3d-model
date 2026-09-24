import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dequantize } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';
import sharp from 'sharp';

const execFileAsync = promisify(execFile);

export async function unpackModel(inputPath, outputPath) {
  await MeshoptDecoder.ready;
  await MeshoptEncoder.ready;

  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
    'meshopt.decoder': MeshoptDecoder,
    'meshopt.encoder': MeshoptEncoder,
  });

  const doc = await io.read(inputPath);

  // Dequantize attributes to standard float32 so external renderers / trimesh can read exact coordinates
  await doc.transform(dequantize());

  const textures = doc.getRoot().listTextures();

  for (let i = 0; i < textures.length; i++) {
    const tex = textures[i];
    const mime = tex.getMimeType();
    const rawBytes = Buffer.from(tex.getImage());

    if (mime === 'image/ktx2' || (rawBytes.length >= 12 && rawBytes.subarray(0, 7).toString('ascii') === '\xabKTX 20')) {
      // Unpack KTX2 to PNG using basisu
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ktx-unpack-'));
      const ktxPath = path.join(tmpDir, 'tex.ktx2');
      await fs.writeFile(ktxPath, rawBytes);
      try {
        await execFileAsync('basisu', ['-unpack', ktxPath], { cwd: tmpDir });
        const files = await fs.readdir(tmpDir);
        const rgbPngFile = files.find(f => f.includes('_rgb_') && (f.includes('level_0') || f.includes('level0')) && f.endsWith('.png'))
          || files.find(f => f.includes('_rgb_') && f.endsWith('.png'))
          || files.find(f => !f.includes('_a_') && f.endsWith('.png'));

        if (rgbPngFile) {
          const pngBuf = await fs.readFile(path.join(tmpDir, rgbPngFile));
          tex.setImage(pngBuf);
          tex.setMimeType('image/png');
        } else {
          console.warn('Could not find unpacked RGB PNG in', files);
        }
      } finally {
        await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
      }
    } else if (mime === 'image/webp') {
      const pngBuf = await sharp(rawBytes).png().toBuffer();
      tex.setImage(pngBuf);
      tex.setMimeType('image/png');
    }
  }

  // Remove compression extensions so standard Assimp / Open3D / Trimesh can load it
  const extensionsUsed = doc.getRoot().listExtensionsUsed();
  for (const ext of extensionsUsed) {
    if (
      ext.extensionName === 'EXT_meshopt_compression' ||
      ext.extensionName === 'KHR_texture_basisu' ||
      ext.extensionName === 'EXT_texture_webp' ||
      ext.extensionName === 'KHR_mesh_quantization'
    ) {
      ext.dispose();
    }
  }

  const glbBytes = await io.writeBinary(doc);
  await fs.writeFile(outputPath, Buffer.from(glbBytes));
}

// CLI execution
if (process.argv[1] && (process.argv[1].endsWith('unpack_meshopt.mjs') || process.argv[1].endsWith('unpack_for_render.mjs'))) {
  const [inputGlb, outputGlb] = process.argv.slice(2);
  if (!inputGlb || !outputGlb) {
    console.error('Usage: node unpack_meshopt.mjs <input.glb> <output.glb>');
    process.exit(1);
  }
  unpackModel(inputGlb, outputGlb).catch(err => {
    console.error('Unpack error:', err);
    process.exit(1);
  });
}
