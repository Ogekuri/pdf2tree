import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _venv_python(root: Path) -> Path:
    for name in ("python3", "python"):
        candidate = root / ".venv" / "bin" / name
        if candidate.exists():
            return candidate
    return None


def test_run_with_project_venv(tmp_path):
    root = Path(__file__).resolve().parents[1]
    venv_python = _venv_python(root)
    if venv_python is None:
        pytest.skip("Project venv not found at .venv/")

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not available")

    sample_dir = root / "pdf_sample"
    tex_file = sample_dir / "pdf_sample.tex"
    pdf_file = sample_dir / "pdf_sample.pdf"

    if not tex_file.exists():
        pytest.skip("Sample .tex file not found")

    # Compile the sample PDF.
    compile_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(sample_dir),
        str(tex_file),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"pdflatex failed: {result.stderr.strip()}")

    if not pdf_file.exists():
        pytest.skip("Sample PDF was not generated")

    output_dir = root / "temp" / "pdf_sample"
    legacy_typo_dir = root / "temp" / "pdf_sampe"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if legacy_typo_dir.exists():
        shutil.rmtree(legacy_typo_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    run_cmd = [
        str(venv_python),
        "-c",
        "from pdf2tree.core import main; raise SystemExit(main())",
        "--from-file",
        str(pdf_file),
        "--to-dir",
        str(output_dir),
    ]

    run = subprocess.run(run_cmd, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (output_dir / "project_manifest.json").exists()
