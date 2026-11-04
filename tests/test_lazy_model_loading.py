"""CPU-only source checks for lazy-loading model cache and Docker behavior."""
from __future__ import annotations
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LazyModelLoadTests(unittest.TestCase):
    def test_app_source_compiles_without_torch(self):
        ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))

    def test_singleton_and_callback_paths(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("_model_init_lock = threading.Lock()", source)
        self.assertIn("def ensure_models_loaded()", source)
        self.assertIn("with _model_init_lock:", source)
        self.assertIn('os.getenv("TRELLIS_LAZY_LOAD", "false")', source)
        self.assertIn('os.getenv("TRELLIS_TMP_DIR"', source)
        self.assertIn("new_pipeline = Trellis2ImageTo3DPipeline.from_pretrained", source)
        self.assertIn("ensure_models_loaded()\n    outputs, latents = pipeline.run(", source)
        self.assertIn("ensure_models_loaded()\n    return local_rmbg(", source)
        self.assertIn("ensure_models_loaded()\n    mesh = pipeline.decode_latent(", source)

    def test_preflight_allows_lazy_skip_but_still_tests_gpu(self):
        code = (ROOT / "scripts/check_runpod.py").read_text(encoding="utf-8")
        ast.parse(code)
        self.assertIn("TRELLIS_SKIP_MODEL_PREFLIGHT", code)
        self.assertLess(code.index("TRELLIS_SKIP_MODEL_PREFLIGHT"), code.index("check_cuda_kernels.py"))
        self.assertIn("check_o_voxel_kernels.py", code)
        self.assertIn("check_cumesh_kernels.py", code)


if __name__ == "__main__":
    unittest.main()
