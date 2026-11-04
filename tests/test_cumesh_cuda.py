"""CPU-only regression checks for CuMesh native-arch rebuild workflow."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "check_cumesh_kernels.py"
REBUILD = ROOT / "scripts" / "rebuild_cumesh.sh"


class CuMeshKernelTests(unittest.TestCase):
    def test_python_valid_and_help_without_gpu(self):
        ast.parse(SMOKE.read_text(encoding="utf-8"))
        p = subprocess.run([sys.executable, str(SMOKE), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("CuMesh", p.stdout)

    def test_shell_syntax(self):
        p = subprocess.run(["bash", "-n", str(REBUILD)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_smoke_reproduces_exact_failure(self):
        src = SMOKE.read_text(encoding="utf-8")
        self.assertIn("mesh.get_edges()", src)
        self.assertIn("torch.cuda.synchronize(device)", src)
        self.assertIn("dtype=torch.int32", src)
        self.assertIn("dtype=torch.float32", src)

    def test_rebuild_keeps_other_extensions(self):
        src = REBUILD.read_text(encoding="utf-8")
        self.assertIn("12289e1062f0603f2f0d0771b02e1395d247f26f", src)
        self.assertIn("JeffreyXiang/CuMesh.git", src)
        self.assertIn("submodule update --init --recursive", src)
        self.assertIn('export TORCH_CUDA_ARCH_LIST="$arch"', src)
        self.assertIn("--no-deps", src)
        self.assertIn("--no-build-isolation", src)
        self.assertIn("python scripts/check_cumesh_kernels.py", src)
        self.assertNotIn("pip uninstall torch", src)

    def test_preflight_runs_cumesh_after_prior_checks(self):
        src = (ROOT / "scripts/check_runpod.py").read_text(encoding="utf-8")
        self.assertIn('with_name("check_cumesh_kernels.py")', src)
        self.assertLess(src.index('with_name("check_o_voxel_kernels.py")'),
                        src.index('with_name("check_cumesh_kernels.py")'))


if __name__ == "__main__":
    unittest.main()
