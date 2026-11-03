"""Test browser-login integration without GPU, Hugging Face access, or downloads."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "login_huggingface.sh"
RUNNER = ROOT / "scripts" / "run_runpod.sh"


def invoke(extra, env):
    return subprocess.run(
        ["bash", str(SCRIPT), *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )


class BrowserLoginTests(unittest.TestCase):
    def env(self, tmp):
        env = os.environ.copy()
        env.pop("HF_TOKEN", None)
        env.pop("HF_HUB_OFFLINE", None)
        env.update({
            "HOME": str(tmp / "home"),
            "HF_HOME": str(tmp / "hf-cache"),
            "HF_BROWSER_CLI_VENV": str(tmp / "cli"),
        })
        return env

    def test_shell_syntax(self):
        p = subprocess.run(["bash", "-n", str(SCRIPT), str(RUNNER)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_env_token_override_refused_without_leaking_secret(self):
        with tempfile.TemporaryDirectory() as d:
            env = self.env(Path(d))
            env["HF_TOKEN"] = "secret-test-value"
            p = invoke(["login"], env)
            self.assertEqual(p.returncode, 2)
            self.assertIn("unset HF_TOKEN", p.stderr)
            self.assertNotIn("secret-test-value", p.stdout + p.stderr)

    def test_missing_cli_check_fails_with_actionable_help(self):
        with tempfile.TemporaryDirectory() as d:
            p = invoke(["--check"], self.env(Path(d)))
            self.assertEqual(p.returncode, 3)
            self.assertIn("login_huggingface.sh", p.stderr)

    def test_check_uses_isolated_cli_and_same_hf_home(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            env = self.env(tmp)
            bindir = Path(env["HF_BROWSER_CLI_VENV"]) / "bin"
            bindir.mkdir(parents=True)
            hf = bindir / "hf"
            hf.write_text("#!/bin/sh\nprintf 'account-check=%s\\n' \"$HF_HOME\"\n", encoding="utf-8")
            hf.chmod(0o700)
            p = invoke(["--check"], env)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn(env["HF_HOME"], p.stdout)

    def test_login_stays_in_separate_cli_environment(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            env = self.env(tmp)
            bindir = Path(env["HF_BROWSER_CLI_VENV"]) / "bin"
            bindir.mkdir(parents=True)
            hf = bindir / "hf"
            hf.write_text(
                "#!/bin/sh\n"
                "echo cli-auth-$1-$2-$3\n", encoding="utf-8"
            )
            hf.chmod(0o700)
            py = bindir / "python"
            py.write_text("#!/bin/sh\necho pip-in-cli\n", encoding="utf-8")
            py.chmod(0o700)
            p = invoke(["login"], env)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("pip-in-cli", p.stdout)
            self.assertIn("cli-auth-auth-login---force", p.stdout)
            self.assertIn(env["HF_HOME"], p.stdout)

    def test_runpod_launcher_refreshes_before_preflight(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("bash scripts/login_huggingface.sh --check", source)
        self.assertLess(
            source.index("bash scripts/login_huggingface.sh --check"),
            source.index("python scripts/check_runpod.py"),
        )
        self.assertIn("HF_HUB_OFFLINE", source)


if __name__ == "__main__":
    unittest.main()
