"""
test_meshopt_encoding.py

Acceptance tests for Step 5 geometry encoding (optimizer/node/optimize_meshopt.mjs):
- TEXCOORD_0 quantized to 16-bit normalized (opt-out via --keep-uv-float32).
- NORMAL encoded with the EXT_meshopt_compression OCTAHEDRAL filter.
- Rule 11: triangle count and index buffer untouched by the encoding change.

And for seam-welded smooth normals (optimizer/node/smooth_normals.mjs):
- Vertices split by a UV seam (same position, different UV) share one normal.

Run via:
    pytest tests/test_meshopt_encoding.py
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO_ROOT / "optimizer" / "node" / "optimize_meshopt.mjs"

# Decodes a GLB (including meshopt) and prints accessor layout + decoded arrays as JSON.
# Positions are emitted in world space so node-level quantization transforms are applied.
DUMP_JS = r"""
import fs from 'node:fs';
import { NodeIO, Logger } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';
await MeshoptDecoder.ready;

const bytes = fs.readFileSync(process.argv[1]);
const jsonLen = bytes.readUInt32LE(12);
const gltf = JSON.parse(bytes.subarray(20, 20 + jsonLen).toString('utf8'));
const primDef = gltf.meshes[0].primitives[0];
const layout = {};
for (const [semantic, index] of Object.entries(primDef.attributes)) {
  const acc = gltf.accessors[index];
  const view = gltf.bufferViews[acc.bufferView] || {};
  const meshopt = (view.extensions || {}).EXT_meshopt_compression || {};
  layout[semantic] = { componentType: acc.componentType, normalized: !!acc.normalized, filter: meshopt.filter || 'NONE' };
}

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const doc = await io.readBinary(new Uint8Array(bytes));
doc.setLogger(new Logger(Logger.Verbosity.ERROR));
const node = doc.getRoot().listNodes().find(n => n.getMesh());
const m = node.getWorldMatrix();
const prim = node.getMesh().listPrimitives()[0];
const read = (semantic) => {
  const acc = prim.getAttribute(semantic);
  if (!acc) return null;
  const out = [], el = [];
  for (let i = 0; i < acc.getCount(); i++) { acc.getElement(i, el); out.push([...el]); }
  return out;
};
const position = read('POSITION').map(([x, y, z]) => [
  m[0] * x + m[4] * y + m[8] * z + m[12],
  m[1] * x + m[5] * y + m[9] * z + m[13],
  m[2] * x + m[6] * y + m[10] * z + m[14],
]);
console.log(JSON.stringify({
  layout,
  position,
  normal: read('NORMAL'),
  uv: read('TEXCOORD_0'),
  indices: Array.from(prim.getIndices().getArray()),
}));
"""

GL_BYTE = 5120
GL_UNSIGNED_SHORT = 5123
GL_FLOAT = 5126


def make_uv_band_glb(path: Path, u_offset: float = 0.0, seam_jitter: float = 0.0, lat: int = 24, lon: int = 48) -> int:
    """
    Writes a textured spherical band (no poles, so no degenerate triangles) with an explicit
    UV seam: column j == lon duplicates column j == 0 at the same position with u shifted by 1.
    seam_jitter offsets the duplicate column along z (the seam lies at z == 0), mimicking the
    float noise real exporters leave between seam copies. Faces are wound CCW seen from outside.
    Returns the triangle count.
    """
    thetas = np.linspace(0.2 * np.pi, 0.8 * np.pi, lat + 1)
    verts, uvs = [], []
    for i, theta in enumerate(thetas):
        for j in range(lon + 1):
            phi = 2.0 * np.pi * (j % lon) / lon  # j == lon reuses the exact j == 0 position
            z = np.sin(theta) * np.sin(phi) + (seam_jitter if j == lon else 0.0)
            verts.append([np.sin(theta) * np.cos(phi), np.cos(theta), z])
            uvs.append([j / lon + u_offset, i / lat])
    verts = np.asarray(verts, dtype=np.float64)

    faces = []
    for i in range(lat):
        for j in range(lon):
            a = i * (lon + 1) + j
            b = a + lon + 1
            faces += [[a, a + 1, b], [a + 1, b + 1, b]]
    faces = np.asarray(faces, dtype=np.int64)

    # Orient every face outward (face normal agrees with the radial direction of its centroid).
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", fn, tri.mean(axis=1)) < 0
    faces[inward] = faces[inward][:, ::-1]

    image = Image.new("RGB", (64, 64), (180, 90, 40))
    visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs), image=image)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
    mesh.export(str(path), file_type="glb")
    return len(faces)


def run_optimizer(src: Path, dst: Path, *flags: str) -> dict:
    cmd = ["node", str(NODE_SCRIPT), str(src), str(dst), "--no-ktx2", "--json", *flags]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"optimize_meshopt.mjs failed: {proc.stderr or proc.stdout}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON summary in output: {proc.stdout}")


def dump_glb(path: Path) -> dict:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", DUMP_JS, str(path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GLB dump failed: {proc.stderr}")
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    for key in ("position", "normal", "uv", "indices"):
        if data[key] is not None:
            data[key] = np.asarray(data[key])
    return data


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)))


class TestStep5GeometryEncoding(unittest.TestCase):
    """Default Step 5 output: 16-bit UV + octahedral normals, triangles untouched."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.src = tmp / "band.glb"
        cls.src_oor = tmp / "band_out_of_range.glb"
        cls.face_count = make_uv_band_glb(cls.src)
        make_uv_band_glb(cls.src_oor, u_offset=0.5)

        cls.out_default = tmp / "default.glb"
        cls.out_float_uv = tmp / "float_uv.glb"
        cls.out_no_meshopt = tmp / "no_meshopt.glb"
        cls.out_oor = tmp / "out_of_range.glb"

        cls.summary_default = run_optimizer(cls.src, cls.out_default)
        cls.summary_float_uv = run_optimizer(cls.src, cls.out_float_uv, "--keep-uv-float32")
        cls.summary_no_meshopt = run_optimizer(cls.src, cls.out_no_meshopt, "--no-meshopt")
        cls.summary_oor = run_optimizer(cls.src_oor, cls.out_oor)

        cls.default = dump_glb(cls.out_default)
        cls.float_uv = dump_glb(cls.out_float_uv)
        cls.no_meshopt = dump_glb(cls.out_no_meshopt)
        cls.oor = dump_glb(cls.out_oor)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_texcoord_quantized_to_16bit_normalized(self):
        uv = self.default["layout"]["TEXCOORD_0"]
        self.assertEqual(uv["componentType"], GL_UNSIGNED_SHORT)
        self.assertTrue(uv["normalized"])

    def test_normals_use_octahedral_filter(self):
        self.assertEqual(self.default["layout"]["NORMAL"]["filter"], "OCTAHEDRAL")
        self.assertEqual(self.default["layout"]["NORMAL"]["componentType"], GL_BYTE)

    def test_uv_error_below_tenth_of_texel_at_4k(self):
        # Same weld/reorder in both runs, so vertex order matches 1:1.
        self.assertEqual(self.default["uv"].shape, self.float_uv["uv"].shape)
        max_texel_error = np.abs(self.default["uv"] - self.float_uv["uv"]).max() * 4096
        self.assertLess(max_texel_error, 0.1)

    def test_octahedral_normal_error_below_2_degrees(self):
        self.assertEqual(self.default["normal"].shape, self.no_meshopt["normal"].shape)
        self.assertLess(angle_deg(self.default["normal"], self.no_meshopt["normal"]).max(), 2.0)

    def test_rule11_triangles_and_indices_preserved(self):
        for summary in (self.summary_default, self.summary_float_uv, self.summary_oor):
            self.assertEqual(summary["trianglesBefore"], self.face_count)
            self.assertEqual(summary["trianglesAfter"], self.face_count)
        np.testing.assert_array_equal(self.default["indices"], self.float_uv["indices"])

    def test_keep_uv_float32_opt_out(self):
        self.assertEqual(self.float_uv["layout"]["TEXCOORD_0"]["componentType"], GL_FLOAT)

    def test_out_of_range_uv_stays_float(self):
        # gltf-transform quantize() skips TEXCOORD outside [0,1]; output must stay valid float UV.
        self.assertEqual(self.oor["layout"]["TEXCOORD_0"]["componentType"], GL_FLOAT)
        self.assertGreater(self.oor["uv"][:, 0].max(), 1.0)

    def test_default_output_smaller_than_float_uv(self):
        self.assertLess(self.summary_default["outputBytes"], self.summary_float_uv["outputBytes"])


class TestSmoothNormalsSeamWelding(unittest.TestCase):
    """Smooth normals must be continuous across UV seams (same position => same normal)."""

    SEAM_JITTER = 0.0

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        src = tmp / "band.glb"
        out = tmp / "smoothed.glb"
        make_uv_band_glb(src, seam_jitter=cls.SEAM_JITTER)
        # --no-meshopt keeps NORMAL as float so the check is exact.
        run_optimizer(src, out, "--smooth-normals", "--no-meshopt")
        cls.data = dump_glb(out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_seam_vertices_share_identical_normals(self):
        positions, normals = self.data["position"], self.data["normal"]
        _, group, counts = np.unique(positions, axis=0, return_inverse=True, return_counts=True)
        group = group.reshape(-1)
        seam_groups = np.flatnonzero(counts > 1)
        self.assertGreater(len(seam_groups), 0, "fixture lost its UV seam; test would be vacuous")
        for g in seam_groups:
            members = normals[group == g]
            np.testing.assert_allclose(members, np.broadcast_to(members[0], members.shape), atol=1e-6)

    def test_normals_point_outward(self):
        radial = self.data["position"] / np.linalg.norm(self.data["position"], axis=1, keepdims=True)
        normals = self.data["normal"] / np.linalg.norm(self.data["normal"], axis=1, keepdims=True)
        self.assertGreater(np.einsum("ij,ij->i", normals, radial).min(), 0.99)


class TestSmoothNormalsNearCoincidentSeam(TestSmoothNormalsSeamWelding):
    """Seam copies 1e-6 apart (below the 1e-5 welding tolerance) must still share one normal."""

    SEAM_JITTER = 1e-6


if __name__ == "__main__":
    unittest.main()
