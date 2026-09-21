import { NodeIO } from '@gltf-transform/core';
import { KHRONOS_EXTENSIONS } from '@gltf-transform/extensions';

async function inspect(filePath) {
    const io = new NodeIO().registerExtensions(KHRONOS_EXTENSIONS);

    const doc = await io.read(filePath);
    const root = doc.getRoot();
    
    let totalFaces = 0;
    let degenerateUVs = 0;
    let flippedNormals = 0;
    let zeroLengthNormals = 0;
    
    root.listMeshes().forEach(mesh => {
        mesh.listPrimitives().forEach(prim => {
            const position = prim.getAttribute('POSITION');
            const normal = prim.getAttribute('NORMAL');
            const texcoord = prim.getAttribute('TEXCOORD_0');
            const indices = prim.getIndices();
            
            if (indices) {
                totalFaces += indices.getCount() / 3;
                
                if (normal) {
                    for (let i = 0; i < normal.getCount(); i++) {
                        const n = normal.getElement(i, []);
                        const len = Math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                        if (len < 0.001) zeroLengthNormals++;
                    }
                }
            }
        });
    });

    console.log(`--- File: ${filePath} ---`);
    console.log(`Total Faces: ${totalFaces}`);
    console.log(`Zero Length Normals: ${zeroLengthNormals}`);
    
    root.listMaterials().forEach((mat, i) => {
        console.log(`Material ${i}:`);
        console.log(`  AlphaMode: ${mat.getAlphaMode()}`);
        console.log(`  AlphaCutoff: ${mat.getAlphaCutoff()}`);
        console.log(`  DoubleSided: ${mat.getDoubleSided()}`);
        const baseColorTex = mat.getBaseColorTexture();
        if (baseColorTex) {
            console.log(`  BaseColorTexture: ${baseColorTex.getMimeType()}, Size: ${baseColorTex.getImage().byteLength}`);
        }
    });
}

async function main() {
    await inspect(process.argv[2]);
    await inspect(process.argv[3]);
}

main().catch(console.error);
