"""Dependency-free static tests; actual inference requires a GPU pod."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RunPodSourceTests(unittest.TestCase):
    def test_python_sources_compile(self):
        for name in (
            "app.py",
            "trellis2/modules/image_feature_extractor.py",
            "trellis2/pipelines/rembg/BiRefNet.py",
            "trellis2/pipelines/trellis2_image_to_3d.py",
            "scripts/check_runpod.py",
        ):
            with self.subTest(name=name):
                ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_no_remote_zerogpu_dependency(self):
        app = (ROOT / "app.py").read_text()
        self.assertNotIn("rmbg_client", app)
        self.assertNotIn("spaces.GPU", app)
        self.assertIn("local_rmbg", app)
        self.assertIn("ensure_hdri_assets()", app)
        self.assertIn("resolve_gradio_access()", app)

    def test_feature_extractor_accepts_nested_encoder(self):
        source = (ROOT / "trellis2/modules/image_feature_extractor.py").read_text()
        self.assertIn('getattr(self.model, "model", self.model)', source)

    def test_rembg_uses_model_device(self):
        source = (ROOT / "trellis2/pipelines/rembg/BiRefNet.py").read_text()
        self.assertIn("next(self.model.parameters()).device", source)
        self.assertNotIn('.to("cuda")', source)

    def test_pipeline_allows_public_rembg_override(self):
        source = (ROOT / "trellis2/pipelines/trellis2_image_to_3d.py").read_text()
        self.assertIn("rembg_model_name: str | None", source)
        self.assertIn("rembg_args['model_name'] = rembg_model_name", source)

if __name__ == "__main__":
    unittest.main()
