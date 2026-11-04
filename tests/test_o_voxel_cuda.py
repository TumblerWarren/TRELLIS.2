"""Static and subprocess-free regression tests for O-Voxel GPU diagnosis."""
from __future__ import annotations
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "check_o_voxel_kernels.py"
REPAIR = ROOT / "scripts" / "rebuild_o_voxel.sh"


class OVoxelTests(unittest.TestCase):
    def test_python_smoke_syntax_and_help_without_torch(self):
        ast.parse(SMOKE.read_text(encoding="utf-8"))
        run = subprocess.run([sys.executable, str(SMOKE), "--help"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("O-Voxel", run.stdout)

    def test_shell_syntax(self):
        run = subprocess.run(["bash", "-n", str(REPAIR)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_exact_mesh_hashmap_kernels_are_exercised(self):
        src = SMOKE.read_text(encoding="utf-8")
        self.assertIn("hashmap_insert_3d_idx_as_val_cuda", src)
        self.assertIn("hashmap_lookup_3d_cuda", src)
        self.assertIn("torch.cuda.synchronize", src)

    def test_rebuild_pins_official_source_and_preserves_dependencies(self):
        src = REPAIR.read_text(encoding="utf-8")
        self.assertIn("https://github.com/microsoft/TRELLIS.2.git", src)
        self.assertIn("75fbf0183001ed9876c8dbb35de6b68552ee08bd", src)
        self.assertIn("sparse-checkout set o-voxel", src)
        self.assertIn("o-voxel/third_party/eigen", src)
        self.assertIn('export TORCH_CUDA_ARCH_LIST="$gpu_arch"', src)
        self.assertIn("--no-deps", src)
        self.assertIn("--no-build-isolation", src)
        self.assertIn("scripts/check_o_voxel_kernels.py", src)

    def test_preflight_covers_both_cuda_extensions(self):
        preflight = (ROOT / "scripts" / "check_runpod.py").read_text(encoding="utf-8")
        self.assertIn('with_name("check_cuda_kernels.py")', preflight)
        self.assertIn('with_name("check_o_voxel_kernels.py")', preflight)
        self.assertIn("TRELLIS_SKIP_KERNEL_PREFLIGHT", preflight)
        self.assertLess(
            preflight.index('with_name("check_cuda_kernels.py")'),
            preflight.index('with_name("check_o_voxel_kernels.py")'),
        )


if __name__ == "__main__":
    unittest.main()
