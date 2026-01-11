from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pdf2tree import __version__

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_dir = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_dir}:{existing}" if existing else src_dir

    cmd = [
        sys.executable,
        "-c",
        "from pdf2tree.core import main; raise SystemExit(main())",
        *args,
    ]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)


def test_cli_version_flag_prints_only_version() -> None:
    result = _run_cli(["--version"])
    assert result.returncode == 0
    assert result.stdout == f"{__version__}\n"
    assert result.stderr == ""


def test_cli_ver_flag_prints_only_version() -> None:
    result = _run_cli(["--ver"])
    assert result.returncode == 0
    assert result.stdout == f"{__version__}\n"
    assert result.stderr == ""
