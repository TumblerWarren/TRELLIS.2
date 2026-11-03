"""RunPod downloader compatibility: no network or GPU required."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hf_download_compat import configure_hf_download_environment, explain_hub_error


class DownloadEnvironmentTests(unittest.TestCase):
    def test_absent_setting_is_noop(self):
        env = {"HF_TOKEN": "secret", "HF_XET_HIGH_PERFORMANCE": "1"}
        self.assertFalse(configure_hf_download_environment(env, notify=lambda _: self.fail("spurious warning")))
        self.assertEqual(env["HF_TOKEN"], "secret")
        self.assertEqual(env["HF_XET_HIGH_PERFORMANCE"], "1")

    def test_all_legacy_values_are_removed(self):
        for value in ("1", "true", "ON", "yes", "0", "false", "", "invalid"):
            with self.subTest(value=value):
                messages = []
                env = {
                    "HF_HUB_ENABLE_HF_TRANSFER": value,
                    "HF_TOKEN": "secret",
                    "HF_HOME": "/cache",
                    "HF_HUB_DISABLE_XET": "1",
                    "HF_HUB_OFFLINE": "1",
                    "HTTPS_PROXY": "https://proxy.example",
                }
                self.assertTrue(configure_hf_download_environment(env, notify=messages.append))
                self.assertNotIn("HF_HUB_ENABLE_HF_TRANSFER", env)
                self.assertEqual(env["HF_TOKEN"], "secret")
                self.assertEqual(env["HF_HOME"], "/cache")
                self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
                self.assertEqual(env["HF_HUB_OFFLINE"], "1")
                self.assertEqual(env["HTTPS_PROXY"], "https://proxy.example")
                self.assertEqual(len(messages), int(value.strip().lower() in {"1", "true", "on", "yes"}))

    def test_app_configures_before_gradio_import(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertLess(source.index("configure_hf_download_environment()"), source.index("import gradio as gr"))
        self.assertLess(source.index("configure_hf_download_environment()"), source.index("from huggingface_hub import"))

    def test_preflight_configures_before_hub_import(self):
        source = (ROOT / "scripts/check_runpod.py").read_text(encoding="utf-8")
        self.assertLess(source.index("configure_hf_download_environment()"), source.index("from huggingface_hub import"))
        self.assertIn("explain_hub_error(exc)", source)

    def test_runner_cleans_inherited_flag(self):
        script = (ROOT / "scripts/run_runpod.sh").read_text(encoding="utf-8")
        self.assertIn("unset HF_HUB_ENABLE_HF_TRANSFER", script)
        self.assertLess(script.index("unset HF_HUB_ENABLE_HF_TRANSFER"), script.index("python scripts/check_runpod.py"))

    def test_integration_subprocess_with_inherited_flag(self):
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        snippet = (
            "from hf_download_compat import configure_hf_download_environment; "
            "import os; configure_hf_download_environment(); "
            "assert 'HF_HUB_ENABLE_HF_TRANSFER' not in os.environ"
        )
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            env=env, cwd=str(ROOT / "scripts"),
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_error_guidance_gated_and_disk(self):
        self.assertIn("gated", explain_hub_error(type("GatedRepoError", (Exception,), {})("restricted")).lower())
        self.assertIn("disk space", explain_hub_error(OSError("No space left on device")).lower())


if __name__ == "__main__":
    unittest.main()
