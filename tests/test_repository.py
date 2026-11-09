import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in {".git", ".astra-home", "research", "hermes-agent", ".venv", "live-output", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_file():
            yield path


def test_synced_copies():
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/sync_shared.py"), "--check"], check=True)
    source = (ROOT / "shared/gmi_media.py").read_bytes()
    for kind in ("image", "video", "speech"):
        assert (ROOT / f"gmi-{kind}/gmi_media.py").read_bytes() == source


def test_repository_hygiene():
    patterns = [r"\bgmi_" + r"[A-Za-z0-9_-]{8,}", r"\bsk-" + r"[A-Za-z0-9_-]{16,}", r"Bearer " + r"[A-Za-z0-9_-]{16,}"]
    for path in files():
        assert path.suffix != ".pyc", str(path)
        if "fixtures" not in path.parts:
            assert path.stat().st_size <= 200_000, str(path)
        text = path.read_text()
        assert not any(char in text for char in (chr(0x2013), chr(0x2014))), str(path)
        for pattern in patterns:
            assert not re.search(pattern, text.replace("test-key-not-real", "FAKE")), str(path)


def test_public_hermes_imports_and_secret_access():
    allowed = {"agent.image_gen_provider", "agent.video_gen_provider", "agent.tts_provider", "agent.secret_scope", "hermes_cli.config"}
    for kind in ("image", "video", "speech"):
        for path in (ROOT / f"gmi-{kind}").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and not node.level:
                    module = node.module or ""
                    if module.startswith(("agent", "hermes_cli", "tools", "plugins", "run_agent")):
                        assert module in allowed
                if isinstance(node, ast.Import):
                    assert not any(n.name.startswith(("agent", "hermes_cli", "tools", "plugins", "run_agent")) for n in node.names)
                if isinstance(node, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "GMI_API_KEY" for a in node.args):
                    assert isinstance(node.func, ast.Name) and node.func.id == "get_secret_str"
