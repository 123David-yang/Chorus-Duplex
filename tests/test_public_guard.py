import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.production.publishing.public_guard import check_revision


class PublicGuardTests(unittest.TestCase):
    def test_detects_a_removed_credential_in_git_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def git(*args):
                subprocess.run(["git", "-C", str(root), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            git("init")
            git("config", "user.name", "Guard Test")
            git("config", "user.email", "guard@example.invalid")
            path = root / "example.txt"
            path.write_text("".join(("gh", "p_", "A" * 36)), encoding="utf-8")
            git("add", path.name)
            git("commit", "-m", "Add fixture")
            path.write_text("redacted", encoding="utf-8")
            git("add", path.name)
            git("commit", "-m", "Remove fixture")
            with patch("scripts.production.publishing.public_guard.ROOT", root):
                findings, skipped = check_revision("HEAD")
            self.assertEqual(findings, ["GitHub token pattern in reachable history: example.txt"])
            self.assertFalse(skipped)

    def test_detects_a_removed_internal_report_in_git_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def git(*args):
                subprocess.run(["git", "-C", str(root), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            git("init")
            git("config", "user.name", "Guard Test")
            git("config", "user.email", "guard@example.invalid")
            report = root / "docs" / "paper" / "internal.md"
            report.parent.mkdir(parents=True)
            report.write_text("internal", encoding="utf-8")
            git("add", "docs/paper/internal.md")
            git("commit", "-m", "Add fixture")
            git("rm", "docs/paper/internal.md")
            git("commit", "-m", "Remove fixture")
            with patch("scripts.production.publishing.public_guard.ROOT", root):
                findings, skipped = check_revision("HEAD")
            self.assertEqual(findings, ["Local-only path appears in reachable history: docs/paper/internal.md"])
            self.assertFalse(skipped)


if __name__ == "__main__":
    unittest.main()
