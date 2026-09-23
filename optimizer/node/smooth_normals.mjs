/**
 * smooth_normals.mjs
 *
 * Computes Angle-Weighted Smooth Vertex Normals (Thürmer & Wüthrich / Bærentzen & Aanaes).
 * Smooths normals across adjacent faces with weight proportional to the face vertex incident angle.
 * Snaps positions to a spatial tolerance grid and remaps them (meshoptimizer's generatePositionRemap)
 * so that vertices split by UV seams share continuous, seamless smooth normals without seam creases,
 * dimples, or inverted normals.
 *
 * A hard edge is encoded the same way a UV seam is - one position carrying several vertices - and the
 * only thing that tells them apart is the normals those vertices arrive with: a seam's agree, a
 * crease's do not. So the grid cell is subdivided by incoming normal direction, and only vertices
 * that already agree to within `hardEdgeDegrees` are welded. Welding a cell whole flattens every
 * crease in the model (measured on dinoki: 5,195 creased positions in, 0 out).
 */

import { Primitive } from '@gltf-transform/core';
import { MeshoptSimplifier } from 'meshoptimizer';

await MeshoptSimplifier.ready;

export function computeStandardSmoothNormals(doc, options = {}) {
  const smoothAcrossUvSeams = options.smoothAcrossUvSeams !== false;
  const spatialTol = options.spatialTolerance || 1e-5;
  const invTol = 1.0 / spatialTol;
  // Matches HARD_EDGE_DEGREES in optimizer/core/face_reduce.py, which is where Step 3 rebuilds the
  // creases this must not undo.
  const cosHardEdge = Math.cos(((options.hardEdgeDegrees ?? 15) * Math.PI) / 180);

  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      if (prim.getMode() !== Primitive.Mode.TRIANGLES) continue;

      const posAcc = prim.getAttribute('POSITION');
      if (!posAcc) continue;

      const posArr = posAcc.getArray();
      const vCount = posAcc.getCount();
      if (vCount === 0) continue;

      const indAcc = prim.getIndices();
      const indArr = indAcc ? indAcc.getArray() : null;
      const triCount = indArr ? Math.floor(indArr.length / 3) : Math.floor(vCount / 3);
      if (triCount === 0) continue;

      // Map each vertex to a spatial group if smoothing across UV seams: positions are snapped to the
      // tolerance grid (round(coord / spatialTol)) and then remapped, so remap[i] is the index of the
      // first vertex in the same grid cell and groups are keyed by vertex index (buffers sized vCount;
      // slots of non-representative vertices stay unused and are never read back).
      let vertToSpatial;
      const spatialCount = vCount;
      if (smoothAcrossUvSeams) {
        // Cell indices are exact in float32 only up to 2^24 (|coord| < ~167 units at 1e-5);
        // beyond that neighbouring cells merge, i.e. the tolerance effectively grows.
        const snapped = new Float32Array(vCount * 3);
        for (let i = 0; i < vCount * 3; i++) snapped[i] = Math.round(posArr[i] * invTol);
        vertToSpatial = MeshoptSimplifier.generatePositionRemap(snapped, 3);

        // Split each cell again by the direction its vertices already point in, so the two sides
        // of a crease stay apart while the two sides of a UV seam are welded. Without an incoming
        // NORMAL there is nothing to tell the two cases apart and the cell is welded whole.
        const inNormAcc = prim.getAttribute('NORMAL');
        if (inNormAcc) {
          const subgroups = new Map();   // cell representative -> vertex indices representing it
          const n = [0, 0, 0];
          const normals = new Float64Array(vCount * 3);
          for (let i = 0; i < vCount; i++) {
            inNormAcc.getElement(i, n);
            const len = Math.hypot(n[0], n[1], n[2]) || 1;
            normals[i * 3] = n[0] / len;
            normals[i * 3 + 1] = n[1] / len;
            normals[i * 3 + 2] = n[2] / len;
          }
          const refined = new Int32Array(vCount);
          for (let i = 0; i < vCount; i++) {
            const cell = vertToSpatial[i];
            let reps = subgroups.get(cell);
            if (reps === undefined) { reps = []; subgroups.set(cell, reps); }
            let found = -1;
            for (const r of reps) {
              const d = normals[i * 3] * normals[r * 3]
                      + normals[i * 3 + 1] * normals[r * 3 + 1]
                      + normals[i * 3 + 2] * normals[r * 3 + 2];
              if (d >= cosHardEdge) { found = r; break; }
            }
            if (found < 0) { reps.push(i); found = i; }
            refined[i] = found;
          }
          vertToSpatial = refined;
        }
      } else {
        vertToSpatial = new Int32Array(vCount);
        for (let i = 0; i < vCount; i++) vertToSpatial[i] = i;
      }

      // Normal accumulation buffers
      const normX = new Float64Array(spatialCount);
      const normY = new Float64Array(spatialCount);
      const normZ = new Float64Array(spatialCount);

      // Iterate through all triangles
      for (let t = 0; t < triCount; t++) {
        let i0, i1, i2;
        if (indArr) {
          i0 = indArr[t * 3];
          i1 = indArr[t * 3 + 1];
          i2 = indArr[t * 3 + 2];
        } else {
          i0 = t * 3;
          i1 = t * 3 + 1;
          i2 = t * 3 + 2;
        }

        if (i0 >= vCount || i1 >= vCount || i2 >= vCount) continue;
        if (i0 === i1 || i1 === i2 || i2 === i0) continue; // Degenerate triangle index

        const p0x = posArr[i0 * 3], p0y = posArr[i0 * 3 + 1], p0z = posArr[i0 * 3 + 2];
        const p1x = posArr[i1 * 3], p1y = posArr[i1 * 3 + 1], p1z = posArr[i1 * 3 + 2];
        const p2x = posArr[i2 * 3], p2y = posArr[i2 * 3 + 1], p2z = posArr[i2 * 3 + 2];

        // Edge vectors: e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2
        const e0x = p1x - p0x, e0y = p1y - p0y, e0z = p1z - p0z;
        const e1x = p2x - p1x, e1y = p2y - p1y, e1z = p2z - p1z;
        const e2x = p0x - p2x, e2y = p0y - p2y, e2z = p0z - p2z;

        // Face normal: e0 x (-e2) = (p1 - p0) x (p2 - p0)
        const fnx = e0y * (-e2z) - e0z * (-e2y);
        const fny = e0z * (-e2x) - e0x * (-e2z);
        const fnz = e0x * (-e2y) - e0y * (-e2x);
        const fnLen = Math.hypot(fnx, fny, fnz);
        if (fnLen < 1e-12) continue; // Degenerate area

        const invFnLen = 1.0 / fnLen;
        const ufnX = fnx * invFnLen;
        const ufnY = fny * invFnLen;
        const ufnZ = fnz * invFnLen;

        // Edge lengths
        const len0 = Math.hypot(e0x, e0y, e0z);
        const len1 = Math.hypot(e1x, e1y, e1z);
        const len2 = Math.hypot(e2x, e2y, e2z);
        if (len0 < 1e-9 || len1 < 1e-9 || len2 < 1e-9) continue;

        // Angle at vertex 0: between e0 and -e2
        const dot0 = e0x * (-e2x) + e0y * (-e2y) + e0z * (-e2z);
        const angle0 = Math.atan2(fnLen, dot0);

        // Angle at vertex 1: between e1 and -e0
        const dot1 = e1x * (-e0x) + e1y * (-e0y) + e1z * (-e0z);
        const angle1 = Math.atan2(fnLen, dot1);

        // Angle at vertex 2: between e2 and -e1
        const dot2 = e2x * (-e1x) + e2y * (-e1y) + e2z * (-e1z);
        const angle2 = Math.atan2(fnLen, dot2);

        const s0 = vertToSpatial[i0];
        const s1 = vertToSpatial[i1];
        const s2 = vertToSpatial[i2];

        normX[s0] += ufnX * angle0;
        normY[s0] += ufnY * angle0;
        normZ[s0] += ufnZ * angle0;

        normX[s1] += ufnX * angle1;
        normY[s1] += ufnY * angle1;
        normZ[s1] += ufnZ * angle1;

        normX[s2] += ufnX * angle2;
        normY[s2] += ufnY * angle2;
        normZ[s2] += ufnZ * angle2;
      }

      // Normalize spatial normals
      for (let s = 0; s < spatialCount; s++) {
        const nx = normX[s];
        const ny = normY[s];
        const nz = normZ[s];
        const len = Math.hypot(nx, ny, nz);
        if (len > 1e-9) {
          const invLen = 1.0 / len;
          normX[s] = nx * invLen;
          normY[s] = ny * invLen;
          normZ[s] = nz * invLen;
        } else {
          normY[s] = 1.0; // Fallback up-vector
        }
      }

      // Write back to vertex normals
      const outNormArr = new Float32Array(vCount * 3);
      for (let i = 0; i < vCount; i++) {
        const s = vertToSpatial[i];
        outNormArr[i * 3] = normX[s];
        outNormArr[i * 3 + 1] = normY[s];
        outNormArr[i * 3 + 2] = normZ[s];
      }

      let normAcc = prim.getAttribute('NORMAL');
      if (!normAcc) {
        normAcc = doc.createAccessor('NORMAL')
          .setType('VEC3')
          .setArray(outNormArr);
        prim.setAttribute('NORMAL', normAcc);
      } else {
        normAcc.setArray(outNormArr);
      }
    }
  }
}
