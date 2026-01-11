from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Iterator

import fitz
import pytest
from pdf2tree import __version__
from pdf2tree.core import (
    PostProcessingConfig,
    extract_image_basenames_in_order,
    load_prompts_file,
    generate_markdown_toc_file,
    normalize_markdown_headings,
    normalize_markdown_format,
    remove_markdown_index,
    add_pdf_toc_to_markdown,
    select_annotation_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "pdf_sample"
TEMP_DIR = ROOT / "temp" / "pdf_sample"
TEMP_DIR_CUSTOM = ROOT / "temp" / "pdf_sample_custom"
TEMP_DIR_FORM_ENABLED = ROOT / "temp" / "pdf_sample_form_enabled"
TEMP_DIR_VECTOR_ENABLED = ROOT / "temp" / "pdf_sample_vector_on"
TEMP_DIR_NONEMPTY = ROOT / "temp" / "pdf_sample_nonempty"
TEMP_DIR_OPENCV = ROOT / "temp" / "pdf_sample_opencv_missing"
TEMP_DIR_CV2_STUB = ROOT / "temp" / "cv2_stub"
TEMP_DIR_FORM_BASE = ROOT / "temp" / "pdf_sample_form_base"
TEMP_DIR_FORM_CROP = ROOT / "temp" / "pdf_sample_form_crop"
TEMP_DIR_POST = ROOT / "temp" / "pdf_sample_postproc"
TEMP_DIR_POST_ONLY = ROOT / "temp" / "pdf_sample_post_only"
TEMP_DIR_POST_ONLY_RESUME = ROOT / "temp" / "pdf_sample_post_only_resume"
TEMP_DIR_POST_ONLY_MISSING_PDF = ROOT / "temp" / "pdf_sample_post_only_missing_pdf"
TEMP_DIR_MANIFEST_REBUILD = ROOT / "temp" / "pdf_sample_manifest_rebuild"
TEMP_DIR_ANNOTATE_IMG = ROOT / "temp" / "pdf_sample_annotate_img"
TEMP_DIR_ANNOTATE_EQ = ROOT / "temp" / "pdf_sample_annotate_eq"
TEMP_DIR_N_PAGES = ROOT / "temp" / "pdf_sample_n_pages"
TEMP_DIR_REMOVE_SMALL = ROOT / "temp" / "pdf_sample_remove_small"
TEMP_DIR_REMOVE_SMALL_DISABLED = ROOT / "temp" / "pdf_sample_remove_small_disabled"
TEMP_DIR_PROMPTS = ROOT / "temp" / "pdf_sample_prompts"
TEMP_DIR_TOC_VALIDATION = ROOT / "temp" / "pdf_sample_toc_validation"
TEMP_DIR_TOC_VALIDATION_FULL = ROOT / "temp" / "pdf_sample_toc_validation_full"
TEMP_DIR_TOC_NORMALIZATION = ROOT / "temp" / "pdf_sample_toc_normalization"
TEMP_DIR_CONTEXT = ROOT / "temp" / "pdf_sample_context"
TEMP_DIR_CLEANUP_DISABLED = ROOT / "temp" / "pdf_sample_cleanup_disabled"
TEMP_DIR_TOC_DISABLED = ROOT / "temp" / "pdf_sample_toc_disabled"
TEMP_DIR_PAGES_REF = ROOT / "temp" / "pdf_sample_pages_ref"

EXIT_OUTPUT_DIR_NOT_EMPTY = 7
EXIT_OPENCV_MISSING = 8
EXIT_POSTPROC_ARTIFACT = 9
EXIT_INVALID_ARGS = 6
EXIT_POSTPROC_DEP = 10

pytestmark = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex non disponibile nel PATH")

ALLOWED_SOURCES = {"pymupdf", "form-xobject", "vector-image"}

os.environ.setdefault("PDF2TREE_TEST_MODE", "1")

TEST_START_PAGE = 2
TEST_PAGE_COUNT = 1
TEST_PAGE_RANGE_ARGS = ["--start-page", str(TEST_START_PAGE), "--n-pages", str(TEST_PAGE_COUNT)]


def _with_test_page_range(args: list[str], *, start_page: int = TEST_START_PAGE, page_count: int = TEST_PAGE_COUNT) -> list[str]:
    updated = list(args)
    if "--start-page" not in updated:
        updated += ["--start-page", str(start_page)]
    if "--n-pages" not in updated:
        updated += ["--n-pages", str(page_count)]
    return updated


def _test_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(overrides)
    return env


@pytest.fixture(scope="session")
def pdf_artifacts() -> dict[str, Path]:
    _cleanup_sample_dir()

    _run_pdflatex()

    pdf_path = SAMPLE_DIR / "pdf_sample.pdf"
    toc_path = SAMPLE_DIR / "pdf_sample.toc"
    pdf_no_toc = SAMPLE_DIR / "pdf_sample_no-toc.pdf"
    assert pdf_path.exists(), "La compilazione iniziale deve produrre pdf_sample.pdf"
    assert toc_path.exists(), "La compilazione iniziale deve produrre pdf_sample.toc"

    if pdf_no_toc.exists():
        pdf_no_toc.unlink()
    pdf_path.rename(pdf_no_toc)

    _run_pdflatex()
    pdf_with_toc = SAMPLE_DIR / "pdf_sample.pdf"
    assert pdf_with_toc.exists(), "La seconda compilazione deve produrre il PDF con TOC"

    return {
        "with_toc": pdf_with_toc,
        "without_toc": pdf_no_toc,
        "toc": toc_path,
    }


def _run_pdflatex() -> None:
    """Compila il sorgente LaTeX e fallisce con dettaglio se pdflatex esce con errore."""
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(SAMPLE_DIR),
        "pdf_sample.tex",
    ]
    result = subprocess.run(cmd, cwd=SAMPLE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"pdflatex failed (code {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def _cleanup_sample_dir() -> None:
    for path in SAMPLE_DIR.iterdir():
        if path.name == "pdf_sample.tex":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _image_dimensions(path: Path) -> tuple[int, int]:
    pix = fitz.Pixmap(str(path))
    dims = (pix.width, pix.height)
    return dims


def _list_png_without_forms(images_dir: Path) -> list[str]:
    return sorted(path.name for path in images_dir.glob("*.png") if "-form-" not in path.name)


def _page_markdown(md_text: str, page_no: int) -> str:
    marker = f"--- start of page.page_number={page_no} ---"
    parts = md_text.split(marker)
    if len(parts) < 2:
        return ""
    after = parts[1]
    return after.split("--- start of page.page_number=", 1)[0]


def _hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _iter_manifest_toc_nodes(nodes: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Restituisce tutti i nodi della TOC serializzata esplorandoli in profondità."""

    for node in nodes:
        yield node
        yield from _iter_manifest_toc_nodes(node.get("children") or [])


def _normalize_title(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text or "")
    norm = norm.replace("’", "'").replace("‘", "'")
    norm = norm.replace("“", '"').replace("”", '"')
    norm = norm.replace("\u00a0", " ")
    norm = re.sub(r"[*_`]+", "", norm)
    norm = re.sub(r"^\d+(?:\.\d+)*\s+", "", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip()


def test_extract_image_order_preserves_sequence() -> None:
    md = "![](images/one.png) text ![](images/two.png) again ![](images/one.png)"
    assert extract_image_basenames_in_order(md) == ["one.png", "two.png", "one.png"]


def test_pdf_sample_cli_workflow(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    pdf_no_toc = pdf_artifacts["without_toc"]
    toc_path = pdf_artifacts["toc"]

    assert pdf_no_toc.exists(), "Il PDF senza TOC deve essere disponibile per i test"
    assert toc_path.exists(), "Il file TOC deve essere stato generato"
    assert pdf_path.exists(), "Il PDF con TOC deve essere disponibile"

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    cmd_fail = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_no_toc.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR.relative_to(ROOT)),
        "--verbose",
        "--debug",
    ])
    result_fail = subprocess.run(cmd_fail, cwd=ROOT, capture_output=True, text=True)
    assert result_fail.returncode == 4, f"Atteso codice 4 su PDF senza TOC, ottenuto {result_fail.returncode}\nSTDOUT:\n{result_fail.stdout}\nSTDERR:\n{result_fail.stderr}"
    assert "TOC not found" in (result_fail.stdout + result_fail.stderr), "I messaggi di errore devono essere in inglese"

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    cmd_ok = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR.relative_to(ROOT)),
        "--header",
        "8",
        "--footer",
        "20",
        "--verbose",
        "--debug",
    ])
    result_ok = subprocess.run(cmd_ok, cwd=ROOT, capture_output=True, text=True)
    assert result_ok.returncode == 0, f"Esecuzione attesa con successo, ottenuto {result_ok.returncode}\nSTDOUT:\n{result_ok.stdout}\nSTDERR:\n{result_ok.stderr}"

    verbose_output = result_ok.stdout + result_ok.stderr
    banner = f"*** pdf2tree ({__version__}) ***"
    assert banner in verbose_output, "Il banner con nome e versione deve essere stampato prima del processing"
    assert "Parameter summary:" in verbose_output, "Il riepilogo dei parametri deve seguire il banner"
    assert "Form XObject: OFF" in verbose_output, "Il riepilogo deve indicare lo stato di Form XObject"
    assert "Vector extraction: OFF" in verbose_output, "Il riepilogo deve indicare lo stato dell'estrazione vettoriale"
    phase_markers = re.findall(r"--- [^-][^-]* ---", verbose_output)
    assert phase_markers, "Devono essere stampate le intestazioni di fase"
    assert verbose_output.count("done.") >= len(phase_markers), "Ogni fase deve terminare con 'done.'"
    assert "Processing page" in verbose_output, "La modalità verbose deve mostrare l'avanzamento per pagina"
    assert "vector extraction disabled (use --enable-vector-images to activate)" in verbose_output, "Il verbose deve indicare lo stato dell'estrazione vettoriale"
    assert "Post-processing flag not provided; execution stops after writing Markdown." in verbose_output, "Il verbose deve chiarire quando il post-processing non viene richiesto"

    assert TEMP_DIR.exists(), "La cartella di output deve essere creata dopo l'esecuzione riuscita"
    md_files = [path for path in TEMP_DIR.glob("*.md") if not path.name.endswith(".processing.md")]
    assert md_files, "Deve essere generato almeno un file Markdown nell'output"
    backup_path = md_files[0].with_suffix(md_files[0].suffix + ".processing.md")
    assert backup_path.exists(), "Il backup .processing.md del Markdown deve essere generato"
    assert backup_path.read_text(encoding="utf-8") == md_files[0].read_text(encoding="utf-8"), "Il backup deve corrispondere al Markdown generato"

    if TEMP_DIR_CUSTOM.exists():
        shutil.rmtree(TEMP_DIR_CUSTOM)

    cmd_custom = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_CUSTOM.relative_to(ROOT)),
        "--header",
        "5",
        "--footer",
        "15",
        "--verbose",
        "--debug",
    ])
    result_custom = subprocess.run(cmd_custom, cwd=ROOT, capture_output=True, text=True)
    assert result_custom.returncode == 0, f"Esecuzione con header/footer personalizzati deve riuscire, ottenuto {result_custom.returncode}\nSTDOUT:\n{result_custom.stdout}\nSTDERR:\n{result_custom.stderr}"
    assert TEMP_DIR_CUSTOM.exists(), "La cartella di output custom deve essere creata"

    if TEMP_DIR_FORM_ENABLED.exists():
        shutil.rmtree(TEMP_DIR_FORM_ENABLED)

    cmd_form_enabled = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_FORM_ENABLED.relative_to(ROOT)),
        "--enable-form-xobject",
        "--verbose",
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--debug",
    ], start_page=5)
    result_form_enabled = subprocess.run(cmd_form_enabled, cwd=ROOT, capture_output=True, text=True)
    assert result_form_enabled.returncode == 0, f"Esecuzione con --enable-form-xobject deve riuscire, ottenuto {result_form_enabled.returncode}\nSTDOUT:\n{result_form_enabled.stdout}\nSTDERR:\n{result_form_enabled.stderr}"
    form_images = sorted((TEMP_DIR_FORM_ENABLED / "images").glob("*-form-*.png"))
    assert form_images, "Le immagini dei Form XObject devono essere generate quando abilitate"
    manifest_form = json.loads((TEMP_DIR_FORM_ENABLED / f"{pdf_path.stem}.json").read_text(encoding="utf-8"))
    assert any(entry.get("source") == "form-xobject" for entry in manifest_form.get("images", [])), "Il manifest deve elencare i Form XObject quando abilitati"

    if TEMP_DIR_VECTOR_ENABLED.exists():
        shutil.rmtree(TEMP_DIR_VECTOR_ENABLED)

    cmd_vector_enabled = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_VECTOR_ENABLED.relative_to(ROOT)),
        "--enable-vector-images",
        "--verbose",
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
    ], start_page=5)
    result_vector_enabled = subprocess.run(cmd_vector_enabled, cwd=ROOT, capture_output=True, text=True)
    assert result_vector_enabled.returncode == 0, (
        f"Esecuzione con --enable-vector-images deve riuscire, ottenuto {result_vector_enabled.returncode}\n"
        f"STDOUT:\n{result_vector_enabled.stdout}\nSTDERR:\n{result_vector_enabled.stderr}"
    )
    vector_on = list((TEMP_DIR_VECTOR_ENABLED / "images").glob("*-vector-*.png"))
    assert vector_on, "Le immagini vettoriali devono essere generate quando abilitate"
    manifest_vector_on = json.loads((TEMP_DIR_VECTOR_ENABLED / f"{pdf_path.stem}.json").read_text(encoding="utf-8"))
    assert any(entry.get("source") == "vector-image" for entry in manifest_vector_on.get("images", [])), "Il manifest deve indicare source='vector-image' quando l'opzione è attiva"
    verbose_vector = result_vector_enabled.stdout + result_vector_enabled.stderr
    assert "vector extraction enabled" in verbose_vector, "Il verbose deve mostrare l'attivazione esplicita dell'estrazione vettoriale"


def test_page_range_limits_processing(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_N_PAGES.exists():
        shutil.rmtree(TEMP_DIR_N_PAGES)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_N_PAGES.relative_to(ROOT)),
        "--verbose",
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Esecuzione con --n-pages 1 deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_N_PAGES / f"{pdf_path.stem}.json"
    assert manifest_path.exists(), "Il manifest deve essere generato quando --post-processing è richiesto"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    toc_nodes = manifest.get("markdown", {}).get("toc_tree", [])
    assert toc_nodes, "Il manifest deve includere la TOC serializzata"
    for node in _iter_manifest_toc_nodes(toc_nodes):
        assert "pdf_source_page" not in node, "I riferimenti pagina devono essere rimossi di default"
        assert "page" not in node, "Il vecchio campo page non deve più comparire nella TOC"
        for field in ("start_line", "end_line", "start_char", "end_char"):
            assert field in node, f"Il nodo TOC deve avere il campo {field}"
        assert int(node["start_line"]) <= int(node["end_line"]), "L'intervallo di righe deve essere valido"
        assert int(node["start_char"]) <= int(node["end_char"]), "L'intervallo di caratteri deve essere valido"

    def _assert_nested_ranges(node: dict[str, Any]) -> None:
        for child in node.get("children", []) or []:
            assert int(node["start_line"]) <= int(child["start_line"]), "Il padre deve iniziare prima del figlio"
            assert int(node["end_line"]) < int(child["start_line"]), "Il padre deve terminare prima dell'inizio del figlio"
            assert int(node["start_char"]) <= int(child["start_char"]), "Il padre deve iniziare prima del figlio (caratteri)"
            assert int(node["end_char"]) < int(child["start_char"]), "Il padre deve terminare prima del figlio (caratteri)"
            _assert_nested_ranges(child)

    for root in toc_nodes:
        _assert_nested_ranges(root)

    md_files = [path for path in TEMP_DIR_N_PAGES.glob("*.md") if not path.name.endswith(".processing.md")]
    assert md_files, "Deve essere generato il Markdown anche con --n-pages"
    md_text = md_files[0].read_text(encoding="utf-8")
    assert "--- start of page.page_number=" not in md_text, "I marker di pagina devono essere rimossi dalla fase di cleanup di default"
    assert "** PDF TOC **" in md_text, "La TOC derivata dal PDF deve restare disponibile dopo il cleanup"
    match = re.search(r"page_count:\s*(\d+)", md_text)
    assert match and int(match.group(1)) == 1, "Il front matter deve riflettere il nuovo numero di pagine"

    for entry in manifest.get("tables", []):
        assert "pdf_source_page" not in entry, "Le tabelle non devono riportare il riferimento pagina di default"

    for entry in manifest.get("images", []):
        assert "pdf_source_page" not in entry, "Le immagini non devono riportare il riferimento pagina di default"

    verbose_output = result.stdout + result.stderr
    assert "Processing page" in verbose_output, "La modalità verbose deve restare attiva"
    assert "[1/1]" in verbose_output, "L'avanzamento deve riflettere il limite di una pagina"
    assert f"PDF page {TEST_START_PAGE}" in verbose_output, "Il log verbose deve riportare il numero pagina originale"


def test_pdf_pages_ref_preserved_when_enabled(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_PAGES_REF.exists():
        shutil.rmtree(TEMP_DIR_PAGES_REF)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_PAGES_REF.relative_to(ROOT)),
        "--verbose",
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--enable-pdf-pages-ref",
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Esecuzione con --enable-pdf-pages-ref deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_PAGES_REF / f"{pdf_path.stem}.json"
    assert manifest_path.exists(), "Il manifest deve essere generato con il flag attivo"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    toc_nodes = manifest.get("markdown", {}).get("toc_tree", [])
    assert toc_nodes, "La TOC deve essere presente nel manifest"
    for node in _iter_manifest_toc_nodes(toc_nodes):
        assert "pdf_source_page" in node, "Il flag deve conservare il riferimento pagina nei nodi TOC"
        page_no = node.get("pdf_source_page")
        assert isinstance(page_no, int) and page_no >= TEST_START_PAGE, "Il riferimento pagina deve riflettere l'intervallo selezionato"
        assert "page" not in node, "Il vecchio campo page non deve ricomparire"

    for entry in manifest.get("tables", []):
        page_no = entry.get("pdf_source_page")
        assert isinstance(page_no, int) and page_no >= TEST_START_PAGE, "Le tabelle devono riportare il riferimento pagina quando abilitato"

    for entry in manifest.get("images", []):
        page_no = entry.get("pdf_source_page")
        assert isinstance(page_no, int) and page_no >= TEST_START_PAGE, "Le immagini devono riportare il riferimento pagina quando abilitato"


def test_output_dir_must_be_empty(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]

    if TEMP_DIR_NONEMPTY.exists():
        shutil.rmtree(TEMP_DIR_NONEMPTY)
    TEMP_DIR_NONEMPTY.mkdir(parents=True, exist_ok=True)
    (TEMP_DIR_NONEMPTY / "keep.txt").write_text("existing", encoding="utf-8")

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_NONEMPTY.relative_to(ROOT)),
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == EXIT_OUTPUT_DIR_NOT_EMPTY, (
        f"Atteso codice {EXIT_OUTPUT_DIR_NOT_EMPTY} su directory di output non vuota, "
        f"ottenuto {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_missing_opencv_fails_fast(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]

    if TEMP_DIR_OPENCV.exists():
        shutil.rmtree(TEMP_DIR_OPENCV)
    if TEMP_DIR_CV2_STUB.exists():
        shutil.rmtree(TEMP_DIR_CV2_STUB)

    TEMP_DIR_CV2_STUB.mkdir(parents=True, exist_ok=True)
    (TEMP_DIR_CV2_STUB / "cv2.py").write_text("raise ImportError('cv2 stub')\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TEMP_DIR_CV2_STUB}:{env.get('PYTHONPATH', '')}"

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_OPENCV.relative_to(ROOT)),
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == EXIT_OPENCV_MISSING, (
        f"Atteso codice {EXIT_OPENCV_MISSING} quando OpenCV manca, "
        f"ottenuto {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_form_xobject_images_not_affected_by_header_footer_crop(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]

    for path in (TEMP_DIR_FORM_BASE, TEMP_DIR_FORM_CROP):
        if path.exists():
            shutil.rmtree(path)

    cmd_base = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_FORM_BASE.relative_to(ROOT)),
        "--enable-form-xobject",
        "--enable-vector-images",
        "--verbose",
    ], start_page=5)
    result_base = subprocess.run(cmd_base, cwd=ROOT, capture_output=True, text=True)
    assert result_base.returncode == 0, (
        f"Esecuzione senza margini deve riuscire, ottenuto {result_base.returncode}\n"
        f"STDOUT:\n{result_base.stdout}\nSTDERR:\n{result_base.stderr}"
    )
    form_images_base = sorted((TEMP_DIR_FORM_BASE / "images").glob("*-form-*.png"))
    assert form_images_base, "Devono essere generate immagini dei Form XObject"
    dims_base = {path.name: _image_dimensions(path) for path in form_images_base}
    hashes_base = {path.name: _hash_file(path) for path in form_images_base}
    vector_images_base = sorted((TEMP_DIR_FORM_BASE / "images").glob("*-vector-*.png"))
    assert vector_images_base, "Devono essere generate immagini vettoriali"
    images_base_no_form = _list_png_without_forms(TEMP_DIR_FORM_BASE / "images")

    cmd_crop = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_FORM_CROP.relative_to(ROOT)),
        "--header",
        "10",
        "--footer",
        "10",
        "--enable-form-xobject",
        "--enable-vector-images",
        "--verbose",
    ], start_page=5)
    result_crop = subprocess.run(cmd_crop, cwd=ROOT, capture_output=True, text=True)
    assert result_crop.returncode == 0, (
        f"Esecuzione con header/footer deve riuscire, ottenuto {result_crop.returncode}\n"
        f"STDOUT:\n{result_crop.stdout}\nSTDERR:\n{result_crop.stderr}"
    )
    form_images_crop = sorted((TEMP_DIR_FORM_CROP / "images").glob("*-form-*.png"))
    assert form_images_crop, "Devono essere generate immagini dei Form XObject con margini impostati"
    dims_crop = {path.name: _image_dimensions(path) for path in form_images_crop}
    hashes_crop = {path.name: _hash_file(path) for path in form_images_crop}
    vector_images_crop = sorted((TEMP_DIR_FORM_CROP / "images").glob("*-vector-*.png"))
    assert vector_images_crop, "Devono essere generate immagini vettoriali anche con margini"
    images_crop_no_form = _list_png_without_forms(TEMP_DIR_FORM_CROP / "images")

    assert set(dims_crop) == set(dims_base), "I Form XObject generati devono coincidere con e senza margini verticali"
    for name, dims in dims_base.items():
        assert dims[0] > 0 and dims[1] > 0, f"L'immagine {name} non deve essere vuota"
        assert dims_crop[name] == dims, f"L'immagine {name} deve restare invariata applicando header/footer"
        assert hashes_crop[name] == hashes_base[name], f"L'immagine {name} deve avere contenuto invariato con header/footer"
    assert {path.name for path in vector_images_crop} == {path.name for path in vector_images_base}, "Le immagini vettoriali devono coincidere con e senza margini"
    assert images_crop_no_form == images_base_no_form, "Le immagini raster/vettoriali standard devono coincidere con e senza margini"


def test_post_processing_with_pix2tex_pipeline_marks_equations(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_POST.exists():
        shutil.rmtree(TEMP_DIR_POST)

    env = _test_env(PDF2TREE_TEST_PIX2TEX_FORMULA="E=mc^2")

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST.relative_to(ROOT)),
        "--post-processing",
        "--enable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
        "--debug",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"Esecuzione con post-processing deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_POST / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    equations = [entry for entry in manifest.get("images", []) if entry.get("type") == "equation"]
    assert equations, "Il post-processing Pix2Tex deve marcare almeno un'immagine come equation"
    assert all(entry.get("equation") for entry in equations), "Le equation devono includere il campo 'equation' nel manifest"
    md_file = TEMP_DIR_POST / manifest["markdown"]["file"]
    md_text = md_file.read_text(encoding="utf-8")
    assert "$$E=mc^2$$" in md_text.replace(" ", ""), "La formula Pix2Tex deve essere inserita nel Markdown"
    equation_path = equations[0].get("file")
    assert equation_path, "Le immagini equation devono indicare il file nel manifest"
    equation_base = Path(str(equation_path)).name
    start_marker = f"**----- Start of equation: {equation_base} -----**"
    end_marker = f"**----- End of equation: {equation_base} -----**"
    assert start_marker in md_text and end_marker in md_text, "I marcatori Start/End devono essere presenti"
    assert str(equation_path) in md_text, "Il Markdown deve contenere il link all'immagine dell'equazione"
    assert md_text.index(start_marker) < md_text.index(end_marker) < md_text.index(str(equation_path)), (
        "L'ordine deve essere Start marker -> End marker -> immagine"
    )

    toc_path = md_file.with_suffix(".toc")
    assert toc_path.exists(), "Il file .toc deve essere generato durante il post-processing"
    toc_content = toc_path.read_text(encoding="utf-8")
    assert "#" in toc_content, "La TOC Markdown deve contenere le intestazioni estratte"
    verbose_output = result.stdout + result.stderr
    assert "Pix2Tex images[" in verbose_output, "La modalità verbose deve riportare il riferimento posizionale delle immagini Pix2Tex"
    assert "validation result: PASSED" in verbose_output, "La modalità verbose deve stampare l'esito della validazione LaTeX"
    assert "Pix2Tex test mode" in verbose_output, "La modalità debug deve evidenziare l'uso della risposta proforma"


def test_post_processing_normalizes_headings_against_pdf_toc(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_TOC_NORMALIZATION.exists():
        shutil.rmtree(TEMP_DIR_TOC_NORMALIZATION)

    start_page = 3
    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_TOC_NORMALIZATION.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--verbose",
    ], start_page=start_page)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Esecuzione con post-processing deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_TOC_NORMALIZATION / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    md_file = TEMP_DIR_TOC_NORMALIZATION / manifest["markdown"]["file"]
    toc_path = md_file.with_suffix(".toc")
    toc_content = toc_path.read_text(encoding="utf-8")

    toc_heading_titles: list[str] = []
    for line in toc_content.splitlines():
        match = re.match(r"- \[(.+?)\]\(#", line.strip())
        if match:
            toc_heading_titles.append(match.group(1).strip())

    assert toc_heading_titles, "La TOC Markdown deve contenere intestazioni estratte"

    with fitz.open(str(pdf_path)) as doc:
        toc_pdf = doc.get_toc() or []

    page_start = start_page
    page_end = start_page + TEST_PAGE_COUNT - 1
    pdf_titles_range: list[str] = []
    for entry in toc_pdf:
        if len(entry) < 3:
            continue
        title = str(entry[1]).strip()
        try:
            page_no = int(entry[2])
        except Exception:
            continue
        if title.lower() in {"indice", "toc"}:
            continue
        if page_start <= page_no <= page_end:
            pdf_titles_range.append(title)

    assert pdf_titles_range, "La TOC del PDF deve contenere voci nel range di pagine testato"
    assert len(toc_heading_titles) == len(pdf_titles_range), (
        "Il numero di intestazioni Markdown deve allinearsi alla TOC del PDF per il range selezionato"
    )

    for md_title, pdf_title in zip(toc_heading_titles, pdf_titles_range):
        assert _normalize_title(md_title) == _normalize_title(pdf_title), (
            "Le intestazioni normalizzate del Markdown devono corrispondere a quelle del PDF"
        )


def test_manifest_context_matches_toc_entries(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_CONTEXT.exists():
        shutil.rmtree(TEMP_DIR_CONTEXT)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_CONTEXT.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--verbose",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Esecuzione con post-processing deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_CONTEXT / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    md_file = TEMP_DIR_CONTEXT / manifest["markdown"]["file"]
    toc_titles: list[str] = []
    for line in md_file.with_suffix(".toc").read_text(encoding="utf-8").splitlines():
        match = re.match(r"- \[(.+?)\]\(#", line.strip())
        if match:
            toc_titles.append(match.group(1).strip())

    assert toc_titles, "La TOC Markdown deve contenere intestazioni estratte"

    def _assert_context(entry: dict[str, object]) -> None:
        ctx_path = entry.get("context_path") or []
        assert isinstance(ctx_path, list), "context_path deve essere una lista"
        assert entry.get("context") == " > ".join(ctx_path), "context deve essere join di context_path"
        if ctx_path:
            assert ctx_path[-1] in toc_titles, "L'ultimo elemento di context_path deve provenire dalla TOC"

    images = manifest.get("images", [])
    tables = manifest.get("tables", [])
    assets = images + tables
    assert assets, "Il manifest deve elencare almeno un asset"
    for entry in assets:
        _assert_context(entry)


def test_post_processing_can_be_disabled(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_POST.exists():
        shutil.rmtree(TEMP_DIR_POST)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST.relative_to(ROOT)),
        "--post-processing",
        "--enable-pic2tex",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Esecuzione con post-processing disabilitato deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    manifest_path = TEMP_DIR_POST / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all(entry.get("type") == "image" for entry in manifest.get("images", [])), (
        "Con --disable-pic2tex le immagini devono restare type='image'"
    )


def test_post_processing_skips_invalid_latex_equations(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_POST.exists():
        shutil.rmtree(TEMP_DIR_POST)

    env = _test_env(PDF2TREE_TEST_PIX2TEX_FORMULA=r"\left(")

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST.relative_to(ROOT)),
        "--post-processing",
        "--enable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"Esecuzione con post-processing deve riuscire anche con formula non valida, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_POST / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images", [])
    assert images, "Il manifest deve elencare le immagini"
    assert all(entry.get("type") == "image" for entry in images), "Le immagini devono restare type='image' se la formula è invalida"
    assert all("equation" not in entry for entry in images), "Le immagini non devono contenere il campo equation quando la validazione fallisce"
    md_file = TEMP_DIR_POST / manifest["markdown"]["file"]
    md_text = md_file.read_text(encoding="utf-8")
    assert r"\left(" not in md_text, "Formule non valide non devono essere inserite nel Markdown"


def test_post_processing_only_requires_markdown_and_backup(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_POST_ONLY.exists():
        shutil.rmtree(TEMP_DIR_POST_ONLY)
    TEMP_DIR_POST_ONLY.mkdir(parents=True, exist_ok=True)
    (TEMP_DIR_POST_ONLY / "dummy.md").write_text("placeholder", encoding="utf-8")

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST_ONLY.relative_to(ROOT)),
        "--post-processing-only",
        "--disable-annotate-images",
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == EXIT_POSTPROC_ARTIFACT, (
        f"Atteso codice {EXIT_POSTPROC_ARTIFACT} quando mancano gli artefatti di post-processing, "
        f"ottenuto {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    if TEMP_DIR_POST_ONLY.exists():
        shutil.rmtree(TEMP_DIR_POST_ONLY)
    TEMP_DIR_POST_ONLY.mkdir(parents=True, exist_ok=True)
    md_path = TEMP_DIR_POST_ONLY / f"{pdf_path.stem}.md"
    backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
    md_content = "Placeholder content for post-processing-only\n"
    md_path.write_text(md_content, encoding="utf-8")
    backup_path.write_text(md_content, encoding="utf-8")

    manifest_path = TEMP_DIR_POST_ONLY / f"{pdf_path.stem}.json"
    if manifest_path.exists():
        manifest_path.unlink()

    cmd_success = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST_ONLY.relative_to(ROOT)),
        "--post-processing-only",
        "--disable-annotate-images",
    ])
    result_success = subprocess.run(cmd_success, cwd=ROOT, capture_output=True, text=True)
    assert result_success.returncode == 0, (
        f"Attesa riuscita del post-processing-only con Markdown e backup pronti, "
        f"ottenuto {result_success.returncode}\nSTDOUT:\n{result_success.stdout}\nSTDERR:\n{result_success.stderr}"
    )
    assert manifest_path.exists(), "Il manifest deve essere ricreato a partire da Markdown e PDF"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("markdown", {}).get("file") == md_path.name, (
        "Il manifest creato dal post-processing-only deve riferire il file Markdown ripristinato"
    )

    restored_md = md_path.read_text(encoding="utf-8")
    assert "Placeholder content for post-processing-only" in restored_md, (
        "Il Markdown deve essere ripristinato dal backup prima di eventuali modifiche della pipeline"
    )

    if TEMP_DIR_POST_ONLY.exists():
        shutil.rmtree(TEMP_DIR_POST_ONLY)
    TEMP_DIR_POST_ONLY.mkdir(parents=True, exist_ok=True)
    md_path = TEMP_DIR_POST_ONLY / f"{pdf_path.stem}.md"
    md_path.write_text(md_content, encoding="utf-8")

    cmd_missing_backup = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST_ONLY.relative_to(ROOT)),
        "--post-processing-only",
        "--disable-annotate-images",
    ])
    result_missing_backup = subprocess.run(cmd_missing_backup, cwd=ROOT, capture_output=True, text=True)
    assert result_missing_backup.returncode == EXIT_POSTPROC_ARTIFACT, (
        f"Atteso codice {EXIT_POSTPROC_ARTIFACT} quando manca il backup .processing.md, "
        f"ottenuto {result_missing_backup.returncode}\nSTDOUT:\n{result_missing_backup.stdout}\nSTDERR:\n{result_missing_backup.stderr}"
    )


def test_post_processing_only_requires_existing_pdf(pdf_artifacts: dict[str, Path]) -> None:
    pdf_missing = SAMPLE_DIR / "pdf_missing_does_not_exist.pdf"

    if TEMP_DIR_POST_ONLY_MISSING_PDF.exists():
        shutil.rmtree(TEMP_DIR_POST_ONLY_MISSING_PDF)
    TEMP_DIR_POST_ONLY_MISSING_PDF.mkdir(parents=True, exist_ok=True)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_missing.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST_ONLY_MISSING_PDF.relative_to(ROOT)),
        "--post-processing-only",
        "--disable-annotate-images",
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == EXIT_POSTPROC_ARTIFACT, (
        f"Atteso codice {EXIT_POSTPROC_ARTIFACT} quando il PDF sorgente manca, "
        f"ottenuto {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "Source file not found for post-processing-only" in combined, "Il log deve indicare l'assenza del PDF sorgente"


def test_post_processing_only_restores_from_backup(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_POST_ONLY_RESUME.exists():
        shutil.rmtree(TEMP_DIR_POST_ONLY_RESUME)

    cmd_initial = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST_ONLY_RESUME.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
    ])
    result_initial = subprocess.run(cmd_initial, cwd=ROOT, capture_output=True, text=True)
    assert result_initial.returncode == 0, (
        f"L'esecuzione iniziale con post-processing deve riuscire, ottenuto {result_initial.returncode}\n"
        f"STDOUT:\n{result_initial.stdout}\nSTDERR:\n{result_initial.stderr}"
    )

    md_files = [path for path in TEMP_DIR_POST_ONLY_RESUME.glob("*.md") if not path.name.endswith(".processing.md")]
    assert md_files, "Deve essere presente il file Markdown dopo la prima esecuzione"
    md_file = md_files[0]
    backup_path = md_file.with_suffix(md_file.suffix + ".processing.md")
    assert backup_path.exists(), "Il backup .processing.md deve esistere dopo la prima esecuzione"
    expected_md = md_file.read_text(encoding="utf-8")
    manifest_path = TEMP_DIR_POST_ONLY_RESUME / f"{pdf_path.stem}.json"
    expected_manifest = manifest_path.read_text(encoding="utf-8")

    md_file.write_text("corrupted", encoding="utf-8")

    cmd_resume = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_POST_ONLY_RESUME.relative_to(ROOT)),
        "--post-processing-only",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
    ])
    result_resume = subprocess.run(cmd_resume, cwd=ROOT, capture_output=True, text=True)
    assert result_resume.returncode == 0, (
        f"La ripresa del post-processing deve riuscire, ottenuto {result_resume.returncode}\n"
        f"STDOUT:\n{result_resume.stdout}\nSTDERR:\n{result_resume.stderr}"
    )

    restored_md = md_file.read_text(encoding="utf-8")
    restored_manifest = manifest_path.read_text(encoding="utf-8")
    assert restored_md == expected_md, "Il Markdown deve essere ripristinato dal backup prima del post-processing"
    assert restored_manifest == expected_manifest, "Il manifest deve essere riscritto senza variazioni inattese"


def test_post_processing_only_rebuilds_manifest_from_pdf(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]

    if TEMP_DIR_MANIFEST_REBUILD.exists():
        shutil.rmtree(TEMP_DIR_MANIFEST_REBUILD)

    cmd_initial = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_MANIFEST_REBUILD.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--verbose",
    ], start_page=3)
    result_initial = subprocess.run(cmd_initial, cwd=ROOT, capture_output=True, text=True)
    assert result_initial.returncode == 0, (
        f"L'esecuzione iniziale deve riuscire, ottenuto {result_initial.returncode}\n"
        f"STDOUT:\n{result_initial.stdout}\nSTDERR:\n{result_initial.stderr}"
    )

    manifest_path = TEMP_DIR_MANIFEST_REBUILD / f"{pdf_path.stem}.json"
    manifest_baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_baseline.get("images"), "Il manifest iniziale deve contenere immagini"

    manifest_tampered = dict(manifest_baseline)
    manifest_tampered["images"] = []
    safe_content = json.dumps(manifest_tampered, ensure_ascii=False, indent=2)
    manifest_path.write_text(safe_content + "\n", encoding="utf-8")

    cmd_resume = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_MANIFEST_REBUILD.relative_to(ROOT)),
        "--post-processing-only",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--verbose",
    ])
    result_resume = subprocess.run(cmd_resume, cwd=ROOT, capture_output=True, text=True)
    assert result_resume.returncode == 0, (
        f"La ripresa del post-processing deve riuscire, ottenuto {result_resume.returncode}\n"
        f"STDOUT:\n{result_resume.stdout}\nSTDERR:\n{result_resume.stderr}"
    )

    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after.get("images"), "Il manifest ricostruito deve popolare la sezione images"
    assert len(manifest_after.get("images", [])) >= len(manifest_baseline.get("images", [])), (
        "La ricostruzione deve ripristinare almeno il numero di immagini del manifest iniziale"
    )
    assert manifest_after.get("markdown") == manifest_baseline.get("markdown"), "Il blocco markdown deve restare coerente dopo la ricostruzione"


def test_annotation_requires_api_key(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_ANNOTATE_IMG.exists():
        shutil.rmtree(TEMP_DIR_ANNOTATE_IMG)

    cmd = _with_test_page_range([
        str(ROOT / ".venv" / "bin" / "python"),
        "-c",
        "from pdf2tree.core import main; raise SystemExit(main())",
        "--from-file",
        str(pdf_path),
        "--to-dir",
        str(TEMP_DIR_ANNOTATE_IMG),
        "--post-processing",
    ])
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == EXIT_INVALID_ARGS, (
        f"Atteso codice {EXIT_INVALID_ARGS} quando manca la chiave Gemini, "
        f"ottenuto {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_annotation_is_added_to_manifest_and_markdown(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_ANNOTATE_IMG.exists():
        shutil.rmtree(TEMP_DIR_ANNOTATE_IMG)

    env = _test_env(
        GEMINI_API_KEY="dummy-key",
        PDF2TREE_TEST_GEMINI_IMAGE_ANNOTATION="Annotated description for test image",
    )

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_ANNOTATE_IMG.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--verbose",
    ], start_page=8)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"Esecuzione con annotazione immagini deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_ANNOTATE_IMG / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images", [])
    assert images, "Il manifest deve elencare le immagini"
    first_image = images[0]
    assert first_image.get("annotation"), "Le immagini devono includere il campo annotation dopo l'annotazione"
    md_file = TEMP_DIR_ANNOTATE_IMG / manifest["markdown"]["file"]
    md_text = md_file.read_text(encoding="utf-8")
    base_name = Path(first_image["file"]).name
    start_marker = f"**----- Start of annotation: {base_name} -----**"
    end_marker = f"**----- End of annotation: {base_name} -----**"
    assert start_marker in md_text and end_marker in md_text, "I marcatori Start/End devono essere presenti per l'annotazione"
    assert "Annotated description for test image" in md_text, "Il testo di annotazione deve essere presente nel Markdown"
    assert md_text.index(start_marker) < md_text.index("Annotated description for test image") < md_text.index(end_marker), (
        "L'annotazione deve apparire tra i marcatori Start/End"
    )
    verbose_output = result.stdout + result.stderr
    assert "Annotating image" in verbose_output, "La modalità verbose deve riportare le immagini annotate"
    assert "Gemini annotation test mode" in verbose_output, "La modalità verbose deve indicare la risposta proforma"


def test_annotation_for_equations_uses_pix2tex_output(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_ANNOTATE_EQ.exists():
        shutil.rmtree(TEMP_DIR_ANNOTATE_EQ)

    env = _test_env(
        GEMINI_API_KEY="dummy-key",
        PDF2TREE_TEST_PIX2TEX_FORMULA="E=mc^2",
    )

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_ANNOTATE_EQ.relative_to(ROOT)),
        "--post-processing",
        "--enable-pic2tex",
        "--enable-annotate-equations",
        "--verbose",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"Esecuzione con annotazione equazioni deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_ANNOTATE_EQ / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eq_entries = [entry for entry in manifest.get("images", []) if entry.get("type") == "equation"]
    assert eq_entries, "Devono essere presenti immagini marcate come equation"
    assert any(entry.get("annotation") for entry in eq_entries), "Le equazioni devono includere l'annotazione nel manifest"
    assert any("E=mc^2" in entry.get("annotation", "") for entry in eq_entries), "Le annotazioni devono incorporare la formula Pix2Tex"
    md_file = TEMP_DIR_ANNOTATE_EQ / manifest["markdown"]["file"]
    md_text = md_file.read_text(encoding="utf-8")
    annotated_eq = [entry for entry in eq_entries if entry.get("annotation")]
    eq_base = Path(annotated_eq[0]["file"]).name
    assert f"**----- Start of annotation: {eq_base} -----**" in md_text, "L'annotazione deve comparire nel Markdown con i marcatori Start/End"
    assert "Representative LaTeX: $$ E=mc^2 $$" in md_text, "Il Markdown deve riportare la formula canned"
    verbose_output = result.stdout + result.stderr
    assert "Annotating equation" in verbose_output, "La modalità verbose deve riportare le equazioni annotate"
    assert "Gemini annotation test mode" in verbose_output, "La modalità verbose deve indicare la risposta proforma"


def test_toc_validation_fails_on_partial_pdf_when_forced(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]

    if TEMP_DIR_TOC_VALIDATION.exists():
        shutil.rmtree(TEMP_DIR_TOC_VALIDATION)

    env = _test_env(PDF2TREE_FORCE_TOC_VALIDATION="1")

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_TOC_VALIDATION.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == EXIT_POSTPROC_DEP, (
        f"Con validazione TOC forzata il run parziale deve fallire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "TOC mismatch" in combined, "Il log deve indicare la mancata corrispondenza della TOC"
    assert TEMP_DIR_TOC_VALIDATION.exists(), "La cartella di output deve essere creata anche in caso di mismatch TOC"
    md_files = [path for path in TEMP_DIR_TOC_VALIDATION.glob("*.md") if not path.name.endswith(".processing.md")]
    assert md_files, "Il Markdown deve essere comunque generato prima del codice di errore finale"
    manifest_path = TEMP_DIR_TOC_VALIDATION / f"{pdf_path.stem}.json"
    assert manifest_path.exists(), "Il manifest deve essere scritto anche se la validazione TOC fallisce"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("markdown", {}).get("file") == md_files[0].name, "Il manifest deve riferire il Markdown generato"
    toc_file = md_files[0].with_suffix(".toc")
    assert toc_file.exists(), "Il file .toc deve essere generato prima del rilevamento del mismatch"


def test_toc_validation_succeeds_on_full_pdf_when_forced(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]

    if TEMP_DIR_TOC_VALIDATION_FULL.exists():
        shutil.rmtree(TEMP_DIR_TOC_VALIDATION_FULL)

    env = _test_env(PDF2TREE_FORCE_TOC_VALIDATION="1")

    cmd = [
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_TOC_VALIDATION_FULL.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--verbose",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        f"La validazione TOC sul PDF completo deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "TOC mismatch" not in combined, "La validazione TOC sul PDF completo non deve riportare errori"


def test_remove_small_images_stage_prunes_manifest_and_markdown(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_REMOVE_SMALL.exists():
        shutil.rmtree(TEMP_DIR_REMOVE_SMALL)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_REMOVE_SMALL.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--min-size-x",
        "4096",
        "--min-size-y",
        "4096",
        "--verbose",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"La fase remove-small-images deve completarsi con successo, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_REMOVE_SMALL / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest.get("images"), "Tutte le immagini devono essere rimosse dal manifest quando superano la soglia impostata"

    md_file = TEMP_DIR_REMOVE_SMALL / manifest["markdown"]["file"]
    md_text = md_file.read_text(encoding="utf-8")
    assert "![](images/" not in md_text, "Il Markdown non deve contenere riferimenti ad immagini rimosse"

    images_dir = TEMP_DIR_REMOVE_SMALL / "images"
    assert any(images_dir.glob("*.png")), "I file immagine non devono essere cancellati dal disco, solo manifest e Markdown devono essere aggiornati"

    verbose_output = result.stdout + result.stderr
    assert "remove-small-images" in verbose_output, "La modalità verbose deve riportare l'esito della fase remove-small-images"
    assert "REMOVE" in verbose_output, "Il log verbose deve indicare l'eliminazione delle immagini"


def test_remove_small_images_stage_can_be_disabled(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_REMOVE_SMALL_DISABLED.exists():
        shutil.rmtree(TEMP_DIR_REMOVE_SMALL_DISABLED)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_REMOVE_SMALL_DISABLED.relative_to(ROOT)),
        "--post-processing",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
        "--min-size-x",
        "4096",
        "--min-size-y",
        "4096",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"L'esecuzione con remove-small-images disabilitato deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    manifest_path = TEMP_DIR_REMOVE_SMALL_DISABLED / f"{pdf_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images", [])
    assert images, "Con remove-small-images disabilitato le immagini devono restare nel manifest"

    images_dir = TEMP_DIR_REMOVE_SMALL_DISABLED / "images"
    assert any(images_dir.glob("*.png")), "I file immagine non devono essere rimossi quando la fase è disattivata"


def test_disable_cleanup_preserves_markers(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_CLEANUP_DISABLED.exists():
        shutil.rmtree(TEMP_DIR_CLEANUP_DISABLED)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_CLEANUP_DISABLED.relative_to(ROOT)),
        "--post-processing",
        "--disable-cleanup",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
    ])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"L'esecuzione con cleanup disabilitato deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    md_files = [path for path in TEMP_DIR_CLEANUP_DISABLED.glob("*.md") if not path.name.endswith(".processing.md")]
    assert md_files, "Il Markdown deve essere presente nell'output"
    md_text = md_files[0].read_text(encoding="utf-8")
    assert "--- start of page.page_number=" in md_text, "I marker di pagina devono restare quando il cleanup è disabilitato"
    assert re.search(r"toc", md_text, flags=re.IGNORECASE), "Il cleanup disabilitato deve preservare la TOC duplicata (anche se normalizzata)"


def test_disable_toc_flag_skips_toc_insertion(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    if TEMP_DIR_TOC_DISABLED.exists():
        shutil.rmtree(TEMP_DIR_TOC_DISABLED)

    cmd = _with_test_page_range([
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(TEMP_DIR_TOC_DISABLED.relative_to(ROOT)),
        "--post-processing",
        "--disable-toc",
        "--disable-pic2tex",
        "--disable-annotate-images",
        "--disable-remove-small-images",
    ], start_page=3)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Esecuzione con TOC disabilitata deve riuscire, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    md_files = [path for path in TEMP_DIR_TOC_DISABLED.glob("*.md") if not path.name.endswith(".processing.md")]
    assert md_files, "Il Markdown deve essere generato anche con TOC disabilitata"
    md_text = md_files[0].read_text(encoding="utf-8")
    assert "--- start of page.page_number=" not in md_text, "Il cleanup deve comunque rimuovere i marker di pagina"
    assert "** PDF TOC **" not in md_text, "Con --disable-toc la TOC Markdown non deve essere inserita"


def test_write_prompts_creates_file_and_exits() -> None:
    if TEMP_DIR_PROMPTS.exists():
        shutil.rmtree(TEMP_DIR_PROMPTS)
    TEMP_DIR_PROMPTS.mkdir(parents=True, exist_ok=True)
    target = TEMP_DIR_PROMPTS / "prompts.json"

    cmd = ["./pdf2tree.sh", "--write-prompts", str(target.relative_to(ROOT))]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"La scrittura dei prompt di default deve uscire con successo, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert target.exists(), "Il file dei prompt deve essere creato"
    prompts_data = json.loads(target.read_text(encoding="utf-8"))
    for key in ("prompt_equation", "prompt_non_equation", "prompt_uncertain"):
        assert key in prompts_data and prompts_data[key], f"La chiave {key} deve essere presente e non vuota"


def test_prompts_file_missing_key_causes_error(pdf_artifacts: dict[str, Path]) -> None:
    pdf_path = pdf_artifacts["with_toc"]
    bad_prompts_dir = TEMP_DIR_PROMPTS / "invalid"
    if bad_prompts_dir.exists():
        shutil.rmtree(bad_prompts_dir)
    bad_prompts_dir.mkdir(parents=True, exist_ok=True)
    bad_file = bad_prompts_dir / "prompts.json"
    bad_file.write_text(json.dumps({"prompt_equation": "x"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        load_prompts_file(bad_file)

    out_dir = TEMP_DIR_PROMPTS / "run"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        "./pdf2tree.sh",
        "--from-file",
        str(pdf_path.relative_to(ROOT)),
        "--to-dir",
        str(out_dir.relative_to(ROOT)),
        "--prompts",
        str(bad_file.relative_to(ROOT)),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == EXIT_INVALID_ARGS, (
        f"Il caricamento di un file di prompt invalido deve fallire con codice {EXIT_INVALID_ARGS}, ottenuto {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_select_annotation_prompt_rules() -> None:
    cfg = PostProcessingConfig(
        enable_pix2tex=True,
        disable_pix2tex=False,
        equation_min_len=5,
        verbose=False,
        debug=False,
        annotate_images=True,
        annotate_equations=True,
        gemini_api_key="k",
        gemini_model="m",
        gemini_module="mod",
        test_mode=False,
        disable_remove_small_images=False,
        disable_cleanup=False,
        disable_toc=False,
        enable_pdf_pages_ref=False,
        min_size_x=1,
        min_size_y=1,
        prompt_equation="EQ",
        prompt_non_equation="NE",
        prompt_uncertain="UN",
    )

    assert select_annotation_prompt(True, True, cfg) == "EQ", "Con Pix2Tex e is_equation deve usare prompt_equation"
    assert select_annotation_prompt(False, True, cfg) == "NE", "Con Pix2Tex e immagine deve usare prompt_non_equation"
    assert select_annotation_prompt(True, False, cfg) == "UN", "Senza Pix2Tex deve usare prompt_uncertain per equazioni"
    assert select_annotation_prompt(False, False, cfg) == "UN", "Senza Pix2Tex deve usare prompt_uncertain per immagini"


def test_remove_markdown_index() -> None:
    duplicate = """Prefazione iniziale
--- start of page.page_number=1 ---
## Sommario provvisorio
Test di sommario
--- end of page.page_number=1 ---
## Introduzione Stravagante
Contenuto Primario
"""
    pdf_toc = [(1, "Introduzione Stravagante", 1)]
    cleaned = remove_markdown_index(duplicate, pdf_toc)
    assert "Prefazione iniziale" not in cleaned
    assert "## Sommario provvisorio" not in cleaned
    assert "## Introduzione Stravagante" in cleaned
    assert "--- start of page.page_number=1 ---" in cleaned
    assert "--- end of page.page_number=1 ---" in cleaned
    section = cleaned.split("--- end of page.page_number=1 ---", 1)[-1]
    assert "## Introduzione Stravagante" in section


def test_normalize_markdown_format_br_to_newline() -> None:
    source = "Line1<br>Line2<BR/>Line3<br />Line4"
    normalized = normalize_markdown_format(source)
    assert normalized == "Line1\nLine2\nLine3\nLine4"


def test_normalize_markdown_headings_inserts_hashes() -> None:
    source = """** PDF TOC **
## **Introduzione Stravagante**
## **1.1 Motivazioni improbabili**
"""
    headings = [(2, "Introduzione Stravagante"), (3, "1.1 Motivazioni improbabili")]
    normalized = normalize_markdown_headings(source, headings)
    assert "## Introduzione Stravagante" in normalized
    assert "### 1.1 Motivazioni improbabili" in normalized
    assert "- [Introduzione Stravagante](#introduzione-stravagante)" not in normalized


def test_generate_markdown_toc_file_writes_anchor(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_content = """** PDF TOC **
## Introduzione Stravagante (pag. 3)
### 1.1 Motivazioni improbabili (pag. 3)
"""
    md_path.write_text(md_content, encoding="utf-8")
    toc_path, headings = generate_markdown_toc_file(md_content, md_path, tmp_path)
    toc_lines = toc_path.read_text(encoding="utf-8").splitlines()
    assert any(line.strip() == "- [Introduzione Stravagante](#introduzione-stravagante) (pag. 3)" for line in toc_lines)
    assert any(line.strip() == "- [1.1 Motivazioni improbabili](#11-motivazioni-improbabili) (pag. 3)" for line in toc_lines)
    assert headings == [(2, "Introduzione Stravagante"), (3, "1.1 Motivazioni improbabili")]


def test_add_pdf_toc_to_markdown_inserts_hierarchical_toc() -> None:
    md = """--- start of page.page_number=1 ---
## Introduzione Stravagante
Contenuto
"""
    pdf_toc = [(1, "Introduzione Stravagante", 1), (2, "1.1 Motivazioni improbabili", 1)]
    updated = add_pdf_toc_to_markdown(md, pdf_toc)
    lines = updated.splitlines()
    assert lines[0].strip() == "--- start of page.page_number=1 ---"
    assert "** PDF TOC **" in updated
    assert "- [Introduzione Stravagante](#introduzione-stravagante)" in updated
    assert "  - [1.1 Motivazioni improbabili](#11-motivazioni-improbabili)" in updated


def test_add_pdf_toc_to_markdown_normalizes_pdf_toc_heading() -> None:
    md = """--- start of page.page_number=1 ---
## TOC
Contenuto fittizio
"""
    pdf_toc = [(1, "Introduzione Stravagante", 1)]
    updated = add_pdf_toc_to_markdown(md, pdf_toc)

    lines = updated.splitlines()
    assert lines[1].strip() == "** PDF TOC **", "L'intestazione TOC deve essere normalizzata al formato in grassetto"
    assert "## TOC" not in updated, "Varianti legacy di TOC devono essere normalizzate al formato in grassetto"
