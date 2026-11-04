"""Static tests for the GPU-specific FlexGEMM diagnostic and repair workflow.

CI has no NVIDIA GPU or CUDA toolkit; these tests do not claim a working binary.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts/check_cuda_kernels.py"
REBUILD = ROOT / "scripts/rebuild_flexgemm.sh"


class FlexgemmKernelChecks(unittest.TestCase):
    def test_smoke_test_syntax_and_help_without_torch(self):
        ast.parse(SMOKE.read_text(encoding="utf-8"))
        cmd = subprocess.run(
            [sys.executable, str(SMOKE), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(cmd.returncode, 0, cmd.stderr)
        self.assertIn("FlexGEMM", cmd.stdout)

    def test_rebuild_shell_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(REBUILD)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_gpu_arch_is_selected(self):
        script = REBUILD.read_text(encoding="utf-8")
        self.assertIn("torch.cuda.get_device_capability(0)", script)
        self.assertIn('export TORCH_CUDA_ARCH_LIST="$gpu_arch"', script)
        self.assertIn("--no-build-isolation", script)
        self.assertIn("--no-deps", script)
        self.assertIn("scripts/check_cuda_kernels.py", script)
        self.assertIn("v1.1.0", script)

    def test_preflight_runs_gpu_smoke_before_gradio(self):
        preflight = (ROOT / "scripts/check_runpod.py").read_text(encoding="utf-8")
        runpod = (ROOT / "scripts/run_runpod.sh").read_text(encoding="utf-8")
        self.assertIn('with_name("check_cuda_kernels.py")', preflight)
        self.assertIn("subprocess.run", preflight)
        self.assertIn("TRELLIS_SKIP_KERNEL_PREFLIGHT", preflight)
        self.assertLess(runpod.index("python scripts/check_runpod.py"), runpod.index("exec python app.py"))


if __name__ == "__main__":
    unittest.main()
