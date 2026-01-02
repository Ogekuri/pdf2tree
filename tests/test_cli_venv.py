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
    math_imgs = list(output_dir.glob("**/math_formula_*.png"))
    assert math_imgs, "Expected math formula images to be generated"
    sample_content = "\n".join((output_dir.glob("*/content.md").__iter__()) and [p.read_text(encoding="utf-8") for p in output_dir.glob("*/content.md")])
    assert "$$" in sample_content or "![Formula" in sample_content


def test_run_dry_run(tmp_path):
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

    output_dir = root / "temp" / "pdf_sample_dryrun"
    if output_dir.exists():
        shutil.rmtree(output_dir)

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
        "--dry-run",
    ]

    run = subprocess.run(run_cmd, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert not output_dir.exists(), "Dry-run should not create output directory or files"


def test_run_with_annotation(tmp_path):
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

    output_dir = root / "temp" / "pdf_sample_annotation"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'tests' / 'stubs'}{os.pathsep}{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PDF2TREE_GEMINI_STUB_PATH"] = str(root / "tests" / "stubs")
    env["PDF2TREE_GEMINI_MODULE"] = "pdf2tree_gemini_stub"
    env["GEMINI_API_KEY"] = "dummy-key-for-tests"

    run_cmd = [
        str(venv_python),
        "-c",
        "from pdf2tree.core import main; raise SystemExit(main())",
        "--from-file",
        str(pdf_file),
        "--to-dir",
        str(output_dir),
        "--annotate-images",
    ]

    run = subprocess.run(run_cmd, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    content_files = sorted(output_dir.glob("*/content.md"))
    assert content_files, "No content files generated with annotation"
    annotated_found = False
    math_block_found = False
    math_replaced = False
    for path in content_files:
        text = path.read_text(encoding="utf-8")
        if "Descrizione immagine stub" in text:
            annotated_found = True
        if "math_formula" in text or "math_page" in text or "$$" in text:
            math_block_found = True
        if r"\[" not in text and "![Formula" in text:
            math_replaced = True
    assert annotated_found, "Missing annotated image descriptions in generated content"
    assert math_block_found, "Missing math formula rendering in annotated content"
    assert math_replaced, "Math formulas not replaced in annotated content"
