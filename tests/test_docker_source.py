"""Docker structure and missing-weight safeguards; no Docker/GPU required."""
from __future__ import annotations
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

class DockerSourceTests(unittest.TestCase):
    def test_dockerfile_uses_matched_cuda_and_lazy_runtime(self):
        s = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("nvidia/cuda:13.0.2-devel-ubuntu24.04", s)
        self.assertIn("TRELLIS_LAZY_LOAD=true", s)
        self.assertIn("TRELLIS_SKIP_MODEL_PREFLIGHT=true", s)
        self.assertIn("HF_HOME=/workspace/hf-cache", s)
        self.assertIn("ARG CUDA_ARCHS=8.9", s)
        self.assertIn("scripts/run_runpod.sh", s)
        self.assertNotIn("hf download ", s)
        self.assertNotIn("snapshot_download", s)

    def test_dockerignore_blocks_weights_and_secrets(self):
        s = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for pattern in ("env/", ".git", "*.safetensors", "*.gguf",
                        ".env", "assets/hdri/*.exr"):
            self.assertIn(pattern, s)

    def test_native_extensions_are_compiled_for_configurable_gpu(self):
        s = (ROOT / "scripts/docker_build_extensions.sh").read_text(encoding="utf-8")
        self.assertIn("TORCH_CUDA_ARCH_LIST", s)
        self.assertIn("--no-deps", s)
        for package in ("FlexGEMM", "CuMesh", "o-voxel", "nvdiffrast", "nvdiffrec"):
            self.assertIn(package, s)
        r = subprocess.run(["bash", "-n", str(ROOT / "scripts/docker_build_extensions.sh")],
                           capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)

if __name__ == "__main__":
    unittest.main()
