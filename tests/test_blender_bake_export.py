"""
tests/test_blender_bake_export.py

Unit tests for doubleSided logic, material state consolidation, and glTF binary verification.
"""

import unittest
import tempfile
from pathlib import Path
from optimizer.blender.blender_bake_export import (
    is_material_double_sided,
    resolve_consolidated_double_sided,
    verify_gltf_double_sided
)


class DummyMaterial:
    def __init__(self, use_backface_culling: bool):
        self.use_backface_culling = use_backface_culling


class TestBlenderBakeExport(unittest.TestCase):
    def test_is_material_double_sided_rule(self):
        """
        Rule:
          material.use_backface_culling = False  ==> doubleSided = True  (Both sides rendered)
          material.use_backface_culling = True   ==> doubleSided = False (FrontSide only)
        """
        mat_double = DummyMaterial(use_backface_culling=False)
        mat_single = DummyMaterial(use_backface_culling=True)

        self.assertTrue(is_material_double_sided(mat_double), "use_backface_culling=False must be doubleSided=True")
        self.assertFalse(is_material_double_sided(mat_single), "use_backface_culling=True must be doubleSided=False")

        # None / empty slot fallback
        self.assertTrue(is_material_double_sided(None, empty_slot_fallback=True))
        self.assertFalse(is_material_double_sided(None, empty_slot_fallback=False))

    def test_multi_material_consolidation_strategies(self):
        """
        Verify multi-material consolidation strategies:
        - auto / any: True if at least one material is double-sided
        - all: True only if all materials are double-sided
        - force_double: Always True
        - force_single: Always False
        """
        # Mixed materials (e.g. Tree: Bark=Single-sided, Leaves=Double-sided)
        mixed_state = {
            "materials": {
                "M_Bark": {"double_sided": False, "use_backface_culling": True},
                "M_Leaves": {"double_sided": True, "use_backface_culling": False}
            }
        }

        # 'auto' / 'any' must prevent leaves from culling/disappearing
        self.assertTrue(resolve_consolidated_double_sided(mixed_state, strategy="auto"))
        self.assertTrue(resolve_consolidated_double_sided(mixed_state, strategy="any"))

        # 'all' fails because Bark is single-sided
        self.assertFalse(resolve_consolidated_double_sided(mixed_state, strategy="all"))

        # Forced overrides
        self.assertTrue(resolve_consolidated_double_sided(mixed_state, strategy="force_double"))
        self.assertFalse(resolve_consolidated_double_sided(mixed_state, strategy="force_single"))

        # Pure solid model (all single-sided, e.g. Dinoki)
        solid_state = {
            "materials": {
                "M_Body": {"double_sided": False, "use_backface_culling": True},
                "M_Eyes": {"double_sided": False, "use_backface_culling": True}
            }
        }
        self.assertFalse(resolve_consolidated_double_sided(solid_state, strategy="auto"))
        self.assertFalse(resolve_consolidated_double_sided(solid_state, strategy="any"))

    def test_verify_sample_glbs(self):
        """Verify reading actual binary GLBs from repository."""
        koidrax = Path("examples/models/koidrax_raw.glb")
        if koidrax.exists():
            res = verify_gltf_double_sided(str(koidrax), expected_double_sided=True)
            self.assertTrue(res["valid"])
            self.assertTrue(res["matches_expected"])
            self.assertTrue(res["materials"][0]["doubleSided"])

        flamibo = Path("examples/models/flamibo_raw.glb")
        if flamibo.exists():
            res = verify_gltf_double_sided(str(flamibo), expected_double_sided=False)
            self.assertTrue(res["valid"])
            self.assertTrue(res["matches_expected"])
            self.assertFalse(res["materials"][0]["doubleSided"])


if __name__ == "__main__":
    unittest.main()
