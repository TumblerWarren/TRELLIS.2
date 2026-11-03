"""Unit tests for default public-link and secure access configuration."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from public_access import resolve_gradio_access, explain_gradio_access, parse_flag


class PublicAccessTests(unittest.TestCase):
    def test_defaults_request_public_link_and_secure_login(self):
        config = resolve_gradio_access({})
        self.assertTrue(config.share)
        self.assertEqual(config.server_name, "0.0.0.0")
        self.assertEqual(config.server_port, 7860)
        self.assertTrue(config.generated_credentials)
        self.assertEqual(config.auth[0], "admin")
        self.assertGreaterEqual(len(config.auth[1]), 24)

    def test_share_false_still_secures_externally_bound_proxy(self):
        config = resolve_gradio_access({"GRADIO_SHARE": "false"})
        self.assertFalse(config.share)
        self.assertTrue(config.auth)
        self.assertTrue(config.generated_credentials)

    def test_explicit_credentials_are_preserved(self):
        config = resolve_gradio_access({
            "GRADIO_SHARE":"true",
            "GRADIO_AUTH_USER":"owner",
            "GRADIO_AUTH_PASSWORD":"strong-supplied-pass",
            "GRADIO_SERVER_PORT":"8080",
        })
        self.assertEqual(config.auth, ("owner","strong-supplied-pass"))
        self.assertFalse(config.generated_credentials)
        self.assertEqual(config.server_port,8080)

    def test_private_loopback_without_share_does_not_force_login(self):
        config = resolve_gradio_access({
            "GRADIO_SHARE":"false",
            "GRADIO_SERVER_NAME":"127.0.0.1",
        })
        self.assertIsNone(config.auth)

    def test_explicit_insecure_opt_out(self):
        config = resolve_gradio_access({
            "TRELLIS_ALLOW_UNAUTHENTICATED":"true",
            "GRADIO_SHARE":"true",
        })
        self.assertIsNone(config.auth)

    def test_partial_credentials_refused(self):
        for values in (
            {"GRADIO_AUTH_USER":"owner"},
            {"GRADIO_AUTH_PASSWORD":"password"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    resolve_gradio_access(values)

    def test_invalid_inputs_refused(self):
        for values in (
            {"GRADIO_SHARE":"aut0"},
            {"GRADIO_SERVER_PORT":"70000"},
            {"GRADIO_SERVER_PORT":"foo"},
            {"TRELLIS_ALLOW_UNAUTHENTICATED":"sometimes"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    resolve_gradio_access(values)

    def test_no_fake_success_message_for_tunnel(self):
        from io import StringIO
        from contextlib import redirect_stdout
        sink = StringIO()
        with redirect_stdout(sink):
            explain_gradio_access(resolve_gradio_access({"GRADIO_AUTH_USER":"owner","GRADIO_AUTH_PASSWORD":"secret"}))
        self.assertIn("only if Gradio's outbound share tunnel succeeds", sink.getvalue())
        self.assertNotIn("Running on public URL:", sink.getvalue())

    def test_app_launch_uses_access_config(self):
        app = (ROOT / "app.py").read_text()
        ast.parse(app)
        self.assertIn("access = resolve_gradio_access()", app)
        self.assertIn("share=access.share", app)
        self.assertIn("auth=access.auth", app)

    def test_runner_sets_share_default_but_allows_override(self):
        runner = (ROOT / "scripts/run_runpod.sh").read_text()
        self.assertIn('export GRADIO_SHARE="${GRADIO_SHARE:-true}"', runner)


if __name__ == "__main__":
    unittest.main()
