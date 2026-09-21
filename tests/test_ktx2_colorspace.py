"""
test_ktx2_colorspace.py

Step 6 KTX2 colour space (optimizer/node/optimize_meshopt.mjs, compressTexturesKtx2):
- normal / metallicRoughness / occlusion textures are linear data -> KTX2 DFD transfer function LINEAR (1).
- baseColor / emissive stay sRGB colour -> transfer function SRGB (2).
- A texture shared by a colour slot and a data slot is encoded as sRGB with a warning.

Run via:
    pytest tests/test_ktx2_colorspace.py
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO_ROOT / "optimizer" / "node" / "optimize_meshopt.mjs"

KHR_DF_TRANSFER_LINEAR = 1
KHR_DF_TRANSFER_SRGB = 2

# Builds a quad GLB: material "all" uses one texture per slot, material "shared" uses one
# texture as both baseColor and occlusion. argv: out.glb, then PNG paths in SLOTS order + shared.
BUILD_JS = r"""
import fs from 'node:fs';
import { Document, NodeIO } from '@gltf-transform/core';
const [out, base, normal, mr, occ, emissive, shared] = process.argv.slice(1);
const doc = new Document();
const buf = doc.createBuffer();
const tex = (name, p) => doc.createTexture(name).setImage(new Uint8Array(fs.readFileSync(p))).setMimeType('image/png');
const quad = (mat) => {
  const acc = (type, arr) => doc.createAccessor().setType(type).setBuffer(buf).setArray(arr);
  const prim = doc.createPrimitive()
    .setAttribute('POSITION', acc('VEC3', new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0])))
    .setAttribute('NORMAL', acc('VEC3', new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])))
    .setAttribute('TEXCOORD_0', acc('VEC2', new Float32Array([0, 1, 1, 1, 1, 0, 0, 0])))
    .setIndices(acc('SCALAR', new Uint16Array([0, 1, 2, 0, 2, 3])))
    .setMaterial(mat);
  return doc.createMesh().addPrimitive(prim);
};
const all = doc.createMaterial('all')
  .setBaseColorTexture(tex('baseColor', base))
  .setNormalTexture(tex('normal', normal))
  .setMetallicRoughnessTexture(tex('metallicRoughness', mr))
  .setOcclusionTexture(tex('occlusion', occ))
  .setEmissiveTexture(tex('emissive', emissive))
  .setEmissiveFactor([1, 1, 1]);
const sharedTex = tex('shared', shared);
const sharedMat = doc.createMaterial('shared').setBaseColorTexture(sharedTex).setOcclusionTexture(sharedTex);
const scene = doc.createScene();
scene.addChild(doc.createNode('a').setMesh(quad(all)));
scene.addChild(doc.createNode('b').setMesh(quad(sharedMat)));
await new NodeIO().write(out, doc);
"""

# Prints {textureName: {mime, transferFunction, levels}} for every texture of a GLB.
DUMP_JS = r"""
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
import { read } from 'ktx-parse';
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(process.argv[1]);
const out = {};
for (const t of doc.getRoot().listTextures()) {
  const row = { mime: t.getMimeType() };
  if (row.mime === 'image/ktx2') {
    const k = read(t.getImage());
    row.transferFunction = k.dataFormatDescriptor[0].transferFunction;
    row.levels = k.levels.length;
  }
  out[t.getName()] = row;
}
console.log(JSON.stringify(out));
"""


def run_node(script: str, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script, *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"node script failed: {proc.stderr or proc.stdout}")
    return proc


@unittest.skipIf(shutil.which("basisu") is None, "basisu CLI not on PATH (Step 6 skips KTX2 without it)")
class TestKtx2ColorSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        size = 32
        images = {
            "baseColor": lambda x, y: (x * 8, y * 8, 60),
            "normal": lambda x, y: (128 + (x - 16) * 3, 128 + (y - 16) * 3, 240),
            "mr": lambda x, y: (255, x * 8, y * 8),
            "occ": lambda x, y: ((x + y) * 4,) * 3,
            "emissive": lambda x, y: (255 * ((x >> 3) & 1), 0, 128),
            "shared": lambda x, y: (200 - x * 2, 200 - y * 2, 200),
        }
        paths = []
        for name, fn in images.items():
            img = Image.new("RGB", (size, size))
            img.putdata([fn(x, y) for y in range(size) for x in range(size)])
            p = tmp / f"{name}.png"
            img.save(p)
            paths.append(str(p))

        src = tmp / "src.glb"
        dst = tmp / "step6.glb"
        run_node(BUILD_JS, str(src), *paths)

        # Same flags as the Step 6 call in optimizer/step_pipeline.py.
        cmd = [
            "node", str(NODE_SCRIPT), str(src), str(dst),
            "--textures-only", "--meshopt", "--texture-max-dim", "1024", "--json",
            "--ktx2", "--ktx2-mode", "uastc", "--ktx2-level", "2", "--ktx2-rdo", "1.0",
            "--ktx2-rdo-d", "2048", "--ktx2-threads", "2", "--single-sided",
        ]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"optimize_meshopt.mjs failed: {proc.stderr or proc.stdout}")
        cls.stderr = proc.stderr
        cls.textures = json.loads(run_node(DUMP_JS, str(dst)).stdout.strip().splitlines()[-1])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_all_textures_are_mipmapped_ktx2(self):
        self.assertEqual(len(self.textures), 6)
        for name, row in self.textures.items():
            self.assertEqual(row["mime"], "image/ktx2", name)
            self.assertGreater(row["levels"], 1, name)

    def test_data_textures_are_linear(self):
        for name in ("normal", "metallicRoughness", "occlusion"):
            self.assertEqual(self.textures[name]["transferFunction"], KHR_DF_TRANSFER_LINEAR, name)

    def test_color_textures_stay_srgb(self):
        for name in ("baseColor", "emissive"):
            self.assertEqual(self.textures[name]["transferFunction"], KHR_DF_TRANSFER_SRGB, name)

    def test_shared_color_and_data_texture_is_srgb_with_warning(self):
        self.assertEqual(self.textures["shared"]["transferFunction"], KHR_DF_TRANSFER_SRGB)
        self.assertIn("shared by colour and data slots", self.stderr)


if __name__ == "__main__":
    unittest.main()
