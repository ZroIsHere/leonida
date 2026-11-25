"""image_get_lod on a cube: the query reads the 6-layer 2D-array view a cube binds as.

Standard library only (spirv-as and spirv-val are used when they are on PATH); no GPU, no LLVM, no game files.

The translator maps a cube sample onto the 2D-array view of the six faces (`cube_coords`): the hardware cube s and t
are the projected sc, tc / |cubema| + 1.5, and the third address word is 8 * slice + face. `image_get_lod` refused
every dimension but 2D and 2D array; on a cube it now subtracts one from s and t (the array view's u and v) and asks
OpImageQueryLod on the cube's %simg2da view. The face word takes no part in the LOD, so it is not read, and `a16`
cube stays refused because a16 packs the three coordinates differently.

No corpus shader uses cube `image_get_lod` (705 in the 26 503-shader set: 702 2D and 3 2D array), so this closes a
form rather than fixing a title frame. The rule is the RDNA2 ISA MIMG GET_LOD dimension form and the existing cube
sample mapping (docs/design.md); the emitted module validates for Vulkan 1.2.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import support

DESC = "s[0:7], s[8:11]"
CUBE = "dim:SQ_RSRC_IMG_CUBE"
EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]


def translate(instruction):
    return support.translate_lines("ps_get_lod_cube", [instruction, *EPILOGUE])


def source_vgpr(spv, value):
    """The VGPR index the id `value` was loaded from (through bitcasts), or None."""
    consts = {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}
    for _ in range(8):
        m = re.search(rf"^\s*{re.escape(value)} = (\w+) (?:%\w+ )?(%\w+)", spv, re.M)
        if not m:
            return None
        op, src = m.groups()
        if op == "OpLoad":
            m2 = re.search(rf"^\s*{re.escape(src)} = OpAccessChain %\S+ %vgpr (%\w+)", spv, re.M)
            return consts.get(m2.group(1)) if m2 else None
        value = src
    return None


class GetLodCube(unittest.TestCase):
    def one(self):
        return translate(f"image_get_lod v[0:1], v[4:6], {DESC} dmask:0x3 {CUBE}")

    def test_cube_query_uses_the_six_layer_2d_array_view(self):
        spv = self.one()
        self.assertIn("%img2da = OpTypeImage %float 2D 0 1 0 1 Unknown", spv)
        self.assertIn("%simg2da", spv)
        self.assertIn("OpImageQueryLod %v2float", spv)

    def test_s_and_t_shift_by_one_and_the_face_word_is_not_read(self):
        spv = self.one()
        subs = re.findall(r"(%\w+) = OpFSub %float (%\w+) %f1$", spv, re.M)
        self.assertEqual(len(subs), 2)
        self.assertEqual([source_vgpr(spv, src) for _, src in subs], [4, 5])   # the projected s and t, not the face
        uv = re.search(r"(%\w+) = OpCompositeConstruct %v2float (%\w+) (%\w+)$", spv, re.M)
        self.assertIsNotNone(uv)
        self.assertEqual(list(uv.groups()[1:]), [sid for sid, _ in subs])
        lod = re.search(r"OpImageQueryLod %v2float %\w+ (%\w+)$", spv, re.M)
        self.assertEqual(lod.group(1), uv.group(1))
        # v6 holds 8 * slice + face; the query does not need it
        self.assertNotIn("%reg_v6", spv)

    def test_a16_cube_stays_refused(self):
        with self.assertRaises(support.rdna2spirv.Unsupported):
            translate(f"image_get_lod v[0:1], v[4:5], {DESC} dmask:0x3 {CUBE} a16")

    def test_a_cube_with_too_few_address_registers_is_refused(self):
        with self.assertRaises(support.rdna2spirv.Unsupported):
            translate(f"image_get_lod v[0:1], v[4:5], {DESC} dmask:0x3 {CUBE}")

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_module_validates(self):
        spv = self.one()
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
            with open(a, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.2", b], check=True)


if __name__ == "__main__":
    unittest.main()
