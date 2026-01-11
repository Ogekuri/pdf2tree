"""
CLI per convertire PDF in Markdown con estrazione di immagini/tabelle e gestione di fallback sui Form XObject.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings:.*",
    category=UserWarning,
)

import argparse
import csv
import importlib
import json
import logging
import mimetypes
import re
import subprocess
import sys
import unicodedata
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pymupdf  # Libreria PyMuPDF

from pdf2tree.latex import validate_latex_formula

LOG = logging.getLogger("pdf2tree")
MM_TO_PT = 72.0 / 25.4
VECTOR_SKIP_Y_RATIO = 0.15
MIN_VECTOR_SIZE_PT = 100.0
MAX_SEPARATOR_WIDTH_RATIO = 0.8
MIN_VECTOR_PATHS = 3
CLUSTER_X_TOLERANCE = 10.0
CLUSTER_Y_TOLERANCE = 10.0
VECTOR_PADDING = 8.0
EXIT_INVALID_ARGS = 6
EXIT_OUTPUT_DIR = 7
EXIT_OPENCV_MISSING = 8
EXIT_POSTPROC_ARTIFACT = 9
EXIT_POSTPROC_DEP = 10
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
TEST_MODE_ENV = "PDF2TREE_TEST_MODE"
TEST_PIX2TEX_FORMULA_ENV = "PDF2TREE_TEST_PIX2TEX_FORMULA"
TEST_GEMINI_IMAGE_ENV = "PDF2TREE_TEST_GEMINI_IMAGE_ANNOTATION"
TEST_GEMINI_EQUATION_ENV = "PDF2TREE_TEST_GEMINI_EQUATION_ANNOTATION"
TEST_PIX2TEX_DEFAULT_FORMULA = r"\int_{0}^{1} x^2 \, dx = \frac{1}{3}"
TEST_GEMINI_IMAGE_DEFAULT = "Test image annotation generated during automated test execution."
TEST_GEMINI_EQUATION_DEFAULT = (
    "Test equation annotation generated during automated test execution.\n\n"
    "Representative LaTeX: $$ {formula} $$"
)
GITHUB_LATEST_RELEASE_URL = "https://api.github.com/repos/Ogekuri/pdf2tree/releases/latest"
UPDATE_CHECK_TIMEOUT_SECONDS = 1.0
PROMPT_EQUATION_DEFAULT = (
    """
You are annotating an image for Retrieval-Augmented Generation (RAG).
Goal: detailed, faithful description optimized for retrieval, clearly grounded explanation.

Return ENGLISH Markdown optimized for RAG with the following sections (keep headings exactly):

## Overview
One sentence describing what the image contains (e.g., "A single equation", "A system of equations with a diagram", etc.).

## Mathematical transcription
- Transcribe ALL mathematical expressions visible (even if multiple).
- Preserve symbols, subscripts/superscripts, Greek letters, limits, summations, matrices, piecewise definitions.
- If something is unreadable, write: [UNREADABLE] and do NOT guess.

## LaTeX (MathJax)
Provide the equation(s) as MathJax-ready LaTeX in Markdown:
- Use one or more display blocks:
  $$ ... $$
- If multiple equations, use separate $$ blocks, in reading order.
- Do NOT add extra equations not present in the image.

## Definitions / Variables (only if explicitly present)
List variable meanings ONLY if they are written in the image. Otherwise write "Not specified."

## Notes on layout (only if useful)
Mention important spatial structure: aligned equations, braces, arrows, numbered steps, boxed results, etc.

## Ambiguities / Unclear parts
Bullet list of any uncertain characters/symbols and where they appear.
"""
)
PROMPT_NON_EQUATION_DEFAULT = (
    """
You are annotating an image for Retrieval-Augmented Generation (RAG).
Goal: detailed, faithful description optimized for retrieval, clearly grounded explanation.

Return ENGLISH Markdown optimized for RAG with the following sections (keep headings exactly):

## Overview
One sentence describing the image type (photo, diagram, chart, UI screenshot, table, flowchart, etc.).

## Visible text (verbatim)
Transcribe all readable text exactly as shown (line breaks if meaningful).
If text is unreadable, write: [UNREADABLE] and do NOT guess.

## Entities and layout
Describe the layout and key elements (objects, labels, axes, boxes, arrows, regions, legend, callouts).
Including all details useful to understand mathematical graphs, processes flows, flow charts, temporal diagrams, mind maps and graphs behaviors.
If there are flows/process steps, describe them in order.

## Tables (if any)
Recreate any table as Markdown table, preserving headers and cell values.
If a cell is unreadable, use [UNREADABLE].

## Quantities / Data (if any)
List numeric values, units, ranges, axis ticks, categories exactly as visible.

## Ambiguities / Unclear parts
Bullet list of any uncertain text/labels and where they appear.
"""
)
PROMPT_UNCERTAIN_DEFAULT = (
    """
You are annotating an image for Retrieval-Augmented Generation (RAG).
Goal: detailed, faithful description optimized for retrieval, clearly grounded explanation.

First decide whether the image contains a mathematical equation/expression that should be transcribed as LaTeX.
Then produce ENGLISH Markdown optimized for RAG in ONE of the two formats below.

### If the image contains mathematical equation(s):
Use EXACTLY these sections:

## Overview
One sentence describing the content.

## Classification
Equation: YES (confidence 0-100)

## Mathematical transcription
Transcribe ALL mathematical expressions visible. Do NOT guess unreadable parts; use [UNREADABLE].

## LaTeX (MathJax)
Provide ONLY the equation(s) that appear in the image as display blocks:
$$ ... $$
Use multiple blocks if needed, in reading order.

## Non-math context (if present)
Briefly describe any accompanying diagram/text/table that changes how the equation is read (e.g., variable definitions, constraints, figure references).

## Ambiguities / Unclear parts
List uncertain symbols/text and location.

### If the image does NOT contain equations:
Use EXACTLY these sections:

## Overview
One sentence describing the content.

## Classification
Equation: NO (confidence 0-100)

## Visible text (verbatim)
Transcribe all readable text exactly; use [UNREADABLE] when needed.

## Entities and layout
Describe the layout, objects, labels, arrows/flows, charts, and tables.
Including all details useful to understand mathematical graphs, processes flows, flow charts, temporal diagrams, mind maps and graphs behaviors.
If there are flows/process steps, describe them in order.

## Tables (if any)
Recreate as Markdown table.

## Ambiguities / Unclear parts
List uncertain parts and location.

Important rules:
- Do NOT invent equations or text.
- If unsure, choose the most likely classification but reflect uncertainty via confidence and Ambiguities.
"""
)
DEFAULT_PROMPTS = {
    "prompt_equation": PROMPT_EQUATION_DEFAULT.strip(),
    "prompt_non_equation": PROMPT_NON_EQUATION_DEFAULT.strip(),
    "prompt_uncertain": PROMPT_UNCERTAIN_DEFAULT.strip(),
}


def _env_flag_enabled(value: Optional[str]) -> bool:
    """Interpreta una variabile di ambiente come flag booleano abilitato."""

    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def is_test_mode() -> bool:
    """Rileva la modalità di test tramite variabili d'ambiente o esecuzione pytest."""

    env_flag = os.environ.get(TEST_MODE_ENV)
    if env_flag is not None:
        return _env_flag_enabled(env_flag)
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _get_test_pix2tex_formula() -> str:
    """Restituisce la formula LaTeX fittizia per la modalità di test Pix2Tex."""

    override = os.environ.get(TEST_PIX2TEX_FORMULA_ENV)
    if override is not None and override.strip():
        return override.strip()
    return TEST_PIX2TEX_DEFAULT_FORMULA


def _get_test_annotation_text(is_equation: bool, equation_text: Optional[str]) -> str:
    """Genera testo di annotazione deterministico in modalità test per immagini o equazioni."""

    if is_equation:
        override = os.environ.get(TEST_GEMINI_EQUATION_ENV)
        if override is not None and override.strip():
            return override.strip()
        formula = equation_text or _get_test_pix2tex_formula() or TEST_PIX2TEX_DEFAULT_FORMULA
        return TEST_GEMINI_EQUATION_DEFAULT.format(formula=formula)
    override = os.environ.get(TEST_GEMINI_IMAGE_ENV)
    if override is not None and override.strip():
        return override.strip()
    return TEST_GEMINI_IMAGE_DEFAULT


def _write_prompts_file(path: Path) -> None:
    """Scrive su disco i prompt di default creando le cartelle genitore."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_PROMPTS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_prompts_file(path: Path) -> Dict[str, str]:
    """Carica e valida un file JSON di prompt che deve contenere le tre chiavi richieste."""
    try:
        data_raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to read prompts file {path}: {exc}") from exc

    prompts: Dict[str, str] = {}
    for key in ("prompt_equation", "prompt_non_equation", "prompt_uncertain"):
        value = data_raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Prompts file {path} missing non-empty key: {key}")
        prompts[key] = value.strip()
    return prompts


def select_annotation_prompt(is_equation: bool, pix2tex_executed: bool, config: "PostProcessingConfig") -> str:
    """Seleziona il prompt in base al tipo di immagine e all'esito dell'esecuzione Pix2Tex."""
    if pix2tex_executed:
        return config.prompt_equation if is_equation else config.prompt_non_equation
    return config.prompt_uncertain


def _resolve_log_level(verbose: bool, debug: bool) -> int:
    """Determina il livello di log partendo dai flag verbose/debug."""

    return logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)


def _configure_pdf2tree_logger(level: int) -> None:
    """Imposta il logger pdf2tree conservando handler e formato anche se cambia il root logger."""
    LOG.setLevel(level)
    LOG.propagate = False
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        handler.setLevel(level)
        LOG.addHandler(handler)
    else:
        for handler in LOG.handlers:
            handler.setLevel(level)
            if handler.formatter is None:
                handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))


@dataclass
class PostProcessingConfig:
    enable_pix2tex: bool
    disable_pix2tex: bool
    equation_min_len: int
    verbose: bool
    debug: bool
    annotate_images: bool
    annotate_equations: bool
    gemini_api_key: Optional[str]
    gemini_model: str
    gemini_module: str
    test_mode: bool
    disable_remove_small_images: bool
    disable_cleanup: bool
    disable_toc: bool
    enable_pdf_pages_ref: bool
    min_size_x: int
    min_size_y: int
    prompt_equation: str
    prompt_non_equation: str
    prompt_uncertain: str
    skip_toc_validation: bool = False


def progress_label(prefix: str, current: int, total: int) -> str:
    """Restituisce un'etichetta di avanzamento leggibile (contatore e percentuale se nota)."""
    if total > 0:
        percent = (current / total) * 100.0
        return f"{prefix} [{current}/{total}] ({percent:.1f}%)"
    return f"{prefix} [{current}]"


def _progress_bar_line(current: int, total: int, width: int = 24) -> str:
    """Costruisce una barra di avanzamento ASCII per aggiornamenti in modalità verbose."""

    if total <= 0:
        return "[?]"
    clamped = max(0, min(current, total))
    filled = int((clamped / total) * width)
    filled = min(filled, width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _log_verbose_progress(prefix: str, current: int, total: int, detail: Optional[str] = None) -> None:
    """Emette una riga di avanzamento con barra visiva solo in modalità verbose."""

    bar = _progress_bar_line(current, total)
    counter = f"[{current}/{total}]" if total > 0 else f"[{current}]"
    if total > 0:
        msg = f"{prefix} {bar} {counter} ({(current / total) * 100.0:.1f}%)"
    else:
        msg = f"{prefix} {bar} {counter}"
    if detail:
        msg = f"{msg} | {detail}"
    LOG.info(msg)


def setup_logging(verbose: bool, debug: bool) -> None:
    """Configura livello e formato del logging in base ai flag della CLI."""
    level = _resolve_log_level(verbose, debug)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    _configure_pdf2tree_logger(level)


def _format_flag(value: bool) -> str:
    """Converte un booleano in etichetta ON/OFF."""

    return "ON" if value else "OFF"


def print_parameter_summary(
    *,
    args: argparse.Namespace,
    post_config: PostProcessingConfig,
    pdf_path: Path,
    out_dir: Path,
    post_processing_active: bool,
    post_processing_only: bool,
    form_xobject_enabled: bool,
    vector_images_enabled: bool,
) -> None:
    """Stampa un riepilogo deterministico dei parametri CLI validati."""

    if post_processing_only:
        mode_desc = "post-processing-only"
    elif post_processing_active:
        mode_desc = "conversion + post-processing"
    else:
        mode_desc = "conversion"

    header_mm = 0.0 if args.header is None else float(args.header)
    footer_mm = 0.0 if args.footer is None else float(args.footer)
    start_page = args.start_page if args.start_page is not None else 1
    page_count = args.n_pages if args.n_pages is not None else "ALL"

    pix2tex_active = bool(post_config.enable_pix2tex and not post_config.disable_pix2tex)
    remove_small_active = not post_config.disable_remove_small_images
    cleanup_active = not post_config.disable_cleanup
    toc_active = not post_config.disable_toc
    annotate_images_active = bool(post_config.annotate_images)
    annotate_equations_active = bool(post_config.annotate_equations)
    gemini_key_present = bool(post_config.gemini_api_key)
    pdf_pages_ref_active = bool(post_config.enable_pdf_pages_ref)

    lines = [
        "Parameter summary:",
        f"  - Mode: {mode_desc}",
        f"  - Source PDF: {pdf_path}",
        f"  - Output directory: {out_dir}",
        f"  - Header/Footer crop (mm): {header_mm} / {footer_mm}",
        f"  - Page range: start={start_page}, count={page_count}",
        f"  - Verbose: {_format_flag(bool(args.verbose))}",
        f"  - Debug: {_format_flag(bool(args.debug))}",
        f"  - Form XObject: {_format_flag(form_xobject_enabled)}",
        f"  - Vector extraction: {_format_flag(vector_images_enabled)}",
        f"  - Post-processing: {_format_flag(post_processing_active)}",
        f"  - Remove small images: {_format_flag(remove_small_active)} (min {post_config.min_size_x}x{post_config.min_size_y}px)",
        f"  - Cleanup: {_format_flag(cleanup_active)}",
        f"  - Add TOC: {_format_flag(toc_active)}",
        f"  - PDF page refs: {_format_flag(pdf_pages_ref_active)}",
        f"  - Pix2Tex: {_format_flag(pix2tex_active)} (threshold {post_config.equation_min_len})",
        f"  - Annotate images: {_format_flag(annotate_images_active)}",
        f"  - Annotate equations: {_format_flag(annotate_equations_active)}",
        f"  - Gemini API key: {_format_flag(gemini_key_present)}",
        f"  - Gemini module/model: {post_config.gemini_module} / {post_config.gemini_model}",
    ]

    print("\n".join(lines))


def program_version() -> str:
    """Restituisce la versione del pacchetto, oppure 'unknown' se non disponibile."""
    try:
        from pdf2tree import __version__ as pkg_version  # type: ignore
    except Exception:
        pkg_version = "unknown"
    return str(pkg_version)


def _extract_numeric_version(text: str) -> Optional[str]:
    """Estrae una versione numerica tipo X.Y.Z da una stringa (es. tag GitHub `v0.0.7`)."""

    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)+", text.strip())
    if not match:
        return None
    return match.group(0)


def _version_tuple(version: str) -> Optional[Tuple[int, ...]]:
    """Converte una stringa versione numerica in tupla di interi per confronto."""

    normalized = _extract_numeric_version(version)
    if not normalized:
        return None
    try:
        parts = tuple(int(p) for p in normalized.split("."))
    except Exception:
        return None
    return parts


def _is_version_greater(candidate: str, current: str) -> bool:
    """Confronta due versioni numeriche; ritorna True se candidate > current."""

    cand_t = _version_tuple(candidate)
    curr_t = _version_tuple(current)
    if not cand_t or not curr_t:
        return False

    max_len = max(len(cand_t), len(curr_t))
    cand_padded = cand_t + (0,) * (max_len - len(cand_t))
    curr_padded = curr_t + (0,) * (max_len - len(curr_t))
    return cand_padded > curr_padded


def _fetch_latest_release_version(*, timeout_seconds: float) -> Optional[str]:
    """Interroga GitHub Releases per ottenere la versione latest; ritorna None su errore."""

    req = urllib.request.Request(
        GITHUB_LATEST_RELEASE_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "pdf2tree",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read()
    except Exception:
        return None

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None

    tag = payload.get("tag_name") if isinstance(payload, dict) else None
    if not isinstance(tag, str) or not tag.strip():
        return None

    return _extract_numeric_version(tag.strip())


def maybe_print_new_version_notice(*, program_name: str = "pdf2tree") -> None:
    """Stampa un avviso se è disponibile una nuova versione; ignora errori e procede."""

    # CORE-DES-086: in modalità test non eseguire richieste di rete.
    if is_test_mode():
        return

    current = program_version()
    if not current or current == "unknown":
        return

    latest = _fetch_latest_release_version(timeout_seconds=UPDATE_CHECK_TIMEOUT_SECONDS)
    if not latest:
        return

    if not _is_version_greater(latest, current):
        return

    print(
        f"A new version of {program_name} is available: current {current}, latest {latest}. "
        f"To upgrade, run: {program_name} --upgrade"
    )


def run_self_upgrade(*, package_name: str = "pdf2tree") -> int:
    """Esegue l'upgrade del pacchetto tramite pip e ritorna un exit code."""

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
    try:
        result = subprocess.run(cmd)
    except Exception as exc:
        print(f"Upgrade failed: {exc}")
        return 1
    return int(result.returncode)


def print_program_banner(name: str = "pdf2tree") -> None:
    """Stampa il banner del programma con nome e versione."""
    print(f"*** {name} ({program_version()}) ***")


def start_phase(description: str) -> None:
    """Stampa il separatore che precede una fase di elaborazione."""
    print(f"\n--- {description} ---")


def end_phase() -> None:
    """Stampa il marcatore di completamento di una fase di elaborazione."""
    print("done.")


def slugify_filename(name: str) -> str:
    """Normalizza un nome file in uno slug sicuro per il filesystem."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "document"


def safe_write_text(path: Path, text: str) -> None:
    """Scrive testo su disco creando le cartelle necessarie e normalizzando le newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: List[List[str]]) -> None:
    """Scrive righe CSV creando la gerarchia di cartelle se mancante."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def unique_target(path: Path) -> Path:
    """Restituisce un percorso libero aggiungendo suffissi incrementali se il file esiste."""

    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def normalize_path_for_md(p: str) -> str:
    """Normalizza un percorso per l'uso nei link Markdown."""

    p = p.replace("\\", "/")
    if p.startswith("file://"):
        p = p[len("file://") :]
    return p


def relative_to_output(path: Path, out_dir: Path) -> str:
    """Calcola un percorso relativo alla cartella di output, con fallback al nome file."""

    try:
        return str(path.resolve().relative_to(out_dir.resolve()))
    except Exception:
        return path.name


def has_opencv() -> bool:
    """Verifica se OpenCV è importabile nell'ambiente corrente."""

    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class TocNode:
    title: str
    page: int
    level: int
    children: List["TocNode"]


@dataclass
class TocValidationResult:
    """Risultato strutturato della validazione TOC."""

    ok: bool
    pdf_titles: List[str]
    md_titles: List[str]
    mismatches: List[Tuple[int, str, str]]
    pdf_count: int
    md_count: int
    reason: str


def build_toc_tree(toc_list: List[List[Any]]) -> TocNode:
    """Costruisce l'albero nidificato della TOC PyMuPDF preservando la gerarchia."""
    root = TocNode(title="root", page=0, level=0, children=[])
    stack: List[TocNode] = [root]

    for entry in toc_list or []:
        if len(entry) < 3:
            continue
        level_raw, title_raw, page_raw = entry[0], entry[1], entry[2]
        try:
            level = int(level_raw)
            page_no = int(page_raw)
        except Exception:
            continue
        title = str(title_raw).strip()
        node = TocNode(title=title, page=page_no, level=level, children=[])

        while stack and level <= stack[-1].level:
            stack.pop()
        parent = stack[-1] if stack else root
        parent.children.append(node)
        stack.append(node)

    return root


def serialize_toc_tree(node: TocNode) -> List[Dict[str, Any]]:
    """Serializza l'albero TOC in una lista di dizionari ricorsiva."""

    return [
        {
            "title": child.title,
            "pdf_source_page": child.page,
            "children": serialize_toc_tree(child),
        }
        for child in node.children
    ]


def find_context_for_page(root: TocNode, page_no: Optional[int]) -> List[str]:
    """Individua il percorso di titoli che meglio corrisponde al numero di pagina."""
    if page_no is None:
        return []

    best_path: List[TocNode] = []
    best_page = -1
    fallback_path: List[TocNode] = []
    fallback_page: Optional[int] = None

    def dfs(node: TocNode, ancestors: List[TocNode]) -> None:
        nonlocal best_path, best_page, fallback_path, fallback_page
        current_path = ancestors + ([node] if node.level > 0 else [])
        if node.level > 0:
            if fallback_page is None or node.page < fallback_page:
                fallback_page = node.page
                fallback_path = current_path
            if node.page <= page_no:
                if node.page > best_page:
                    best_page = node.page
                    best_path = current_path
        for child in node.children:
            dfs(child, current_path)

    dfs(root, [])
    if best_path:
        return [n.title for n in best_path]
    return [n.title for n in fallback_path]


def find_context(
    toc_path: Optional[Path],
    toc_root: Optional[TocNode],
    asset_names: Iterable[str],
    fallback_page: Optional[int],
) -> Tuple[List[str], str]:
    """Trova il contesto di un asset usando prima il file .toc e in fallback la TOC del PDF."""

    names = {n for n in asset_names if n}
    if toc_path and toc_path.exists() and names:
        try:
            lines = toc_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []

        heading_re = re.compile(r"^(?P<indent>\s*)-\s*\[(?P<title>[^\]]+)\]\([^)]*\)")
        stack: List[str] = []
        found: Optional[List[str]] = None

        for raw in lines:
            if not raw.strip():
                continue
            h_match = heading_re.match(raw)
            if h_match:
                indent = h_match.group("indent")
                level = int(len(indent) / 2) + 1
                title = h_match.group("title").strip()

                while len(stack) >= level:
                    stack.pop()
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(title)
                continue

            if any(name in raw for name in names):
                found = [item for item in stack if item]
                break

        if found is not None:
            context_path = list(found)
            context_str = " > ".join(context_path)
            return context_path, context_str

    context_titles = (
        find_context_for_page(toc_root or build_toc_tree([]), fallback_page)
        if fallback_page is not None
        else []
    )
    context_str = " > ".join(context_titles)
    return context_titles, context_str


def build_context_metadata(context_titles: List[str]) -> Tuple[str, List[str]]:
    """Costruisce `context` e `context_path` usando solo i titoli TOC trovati."""

    context_path = list(context_titles)
    context_str = " > ".join(context_path)
    return context_str, context_path


PAGE_IN_NAME_RE = re.compile(r"-([0-9]{3,4})-")


def guess_page_from_filename(name: str) -> Optional[int]:
    """Prova a dedurre il numero di pagina da un nome file."""

    match = PAGE_IN_NAME_RE.search(name)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    alt = re.search(r"page[_-]?([0-9]{3,4})", name, re.IGNORECASE)
    if alt:
        try:
            return int(alt.group(1))
        except Exception:
            return None
    return None


IMG_LINK_RE = re.compile(r"(!\[[^\]]*\]\()([^\)]+)(\))")


def extract_image_basenames_from_markdown(md: str) -> Set[str]:
    """Estrae i basename dei file immagine presenti nei link Markdown."""

    out: Set[str] = set()
    for match in IMG_LINK_RE.finditer(md or ""):
        url = match.group(2).strip().strip('"').strip("'")
        url = normalize_path_for_md(url)
        url = url.split("?", 1)[0].split("#", 1)[0]
        base = url.split("/")[-1].strip()
        if base:
            out.add(base)
    return out


def extract_image_basenames_in_order(md: str) -> List[str]:
    """Restituisce i basename delle immagini nell'ordine in cui compaiono nel Markdown."""

    ordered: List[str] = []
    for match in IMG_LINK_RE.finditer(md or ""):
        url = match.group(2).strip().strip('"').strip("'")
        url = normalize_path_for_md(url)
        url = url.split("?", 1)[0].split("#", 1)[0]
        base = url.split("/")[-1].strip()
        if base:
            ordered.append(base)
    return ordered


def rewrite_image_links_to_images_subdir(md: str, subdir: str = "images") -> str:
    """Forza tutti i link immagine a puntare a images/<basename>."""

    def repl(match: re.Match) -> str:
        before, url, after = match.group(1), match.group(2), match.group(3)
        url_clean = normalize_path_for_md(url.strip().strip('"').strip("'"))
        url_clean = url_clean.split("?", 1)[0].split("#", 1)[0]
        base = url_clean.split("/")[-1]
        return f"{before}{subdir}/{base}{after}"

    return IMG_LINK_RE.sub(repl, md)


def yaml_front_matter(metadata: dict, source_path: Path, page_count: int) -> str:
    """Crea il front matter YAML con i metadati del PDF."""

    def clean(value: Any) -> str:
        text = str(value).replace("\n", " ").strip()
        text = text.replace('"', "'")
        return text

    title = clean(metadata.get("title", "")) if metadata else ""
    author = clean(metadata.get("author", "")) if metadata else ""
    subject = clean(metadata.get("subject", "")) if metadata else ""
    creator = clean(metadata.get("creator", "")) if metadata else ""

    lines = [
        "---",
        f'source_file: "{clean(source_path.name)}"',
        f"page_count: {page_count}",
    ]
    if title:
        lines.append(f'title: "{title}"')
    if author:
        lines.append(f'author: "{author}"')
    if subject:
        lines.append(f'subject: "{subject}"')
    if creator:
        lines.append(f'creator: "{creator}"')
    lines.append("---\n")
    return "\n".join(lines)


def build_toc_markdown(toc: List[List[Any]]) -> str:
    """Genera la sezione Markdown della TOC con rientri per livello."""

    if not toc:
        return ""
    lines: List[str] = ["** PDF TOC **\n"]
    for level, title, page_no in toc:
        indent = "  " * max(0, int(level) - 1)
        safe_title = str(title).strip()
        lines.append(f"{indent}- {safe_title} (pag. {page_no})")
    return "\n".join(lines).strip() + "\n"


def looks_like_markdown_table(text: str) -> bool:
    """Verifica se un testo assomiglia a una tabella Markdown."""

    if "|" not in (text or ""):
        return False
    if re.search(r"^\s*\|?.+\|.+\|?\s*$", text or "", flags=re.MULTILINE) and re.search(
        r"^\s*\|?\s*:?-{2,}:?\s*\|", text or "", flags=re.MULTILINE
    ):
        return True
    return False


def extract_tables_fallback(doc: Any, page_index: int) -> List[Tuple[str, List[List[str]]]]:
    """Esegue l'estrazione di tabelle via fallback PyMuPDF se disponibile."""

    page = doc[page_index]
    try:
        tables = page.find_tables()
    except Exception as exc:  # pragma: no cover - dipende dal supporto PyMuPDF
        LOG.debug("find_tables failed on page %d: %s", page_index + 1, exc)
        return []

    results: List[Tuple[str, List[List[str]]]] = []
    for table in getattr(tables, "tables", []) or []:
        md = ""
        rows: List[List[str]] = []

        try:
            if hasattr(table, "to_markdown"):
                md = table.to_markdown() or ""
        except Exception as exc:
            LOG.debug("Table.to_markdown failed page %d: %s", page_index + 1, exc)

        try:
            raw = table.extract()
            if isinstance(raw, list):
                rows = [[("" if cell is None else str(cell)) for cell in row] for row in raw]
        except Exception as exc:
            LOG.debug("Table.extract failed page %d: %s", page_index + 1, exc)

        if md.strip() or rows:
            results.append((md, rows))
    return results


def export_tables_files(tables_dir: Path, page_no: int, tables: List[Tuple[str, List[List[str]]]]) -> List[List[Path]]:
    """Esporta le tabelle estratte in Markdown e CSV restituendo i percorsi creati."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    exported: List[List[Path]] = []
    for t_idx, (table_md, table_rows) in enumerate(tables, start=1):
        stem = f"page-{page_no:03d}-table-{t_idx:02d}"
        files: List[Path] = []
        if table_md.strip():
            path_md = tables_dir / f"{stem}.md"
            safe_write_text(path_md, table_md.strip() + "\n")
            files.append(path_md)
        if table_rows:
            path_csv = tables_dir / f"{stem}.csv"
            write_csv(path_csv, table_rows)
            files.append(path_csv)
        if files:
            exported.append(files)
    return exported


def format_table_references(exported: List[List[Path]], out_dir: Path) -> List[str]:
    """Crea i riferimenti Markdown per ogni tabella esportata, restituendo blocchi separati."""

    blocks: List[str] = []
    for files in exported:
        rel_md: Optional[str] = None
        rel_csv: Optional[str] = None
        for path in files:
            rel = relative_to_output(path, out_dir)
            if path.suffix.lower() == ".md":
                rel_md = rel
            elif path.suffix.lower() == ".csv":
                rel_csv = rel

        lines: List[str] = []
        if rel_md:
            lines.append(f"[Markdown]({rel_md})")
        if rel_csv:
            if lines:
                lines.append("")
            lines.append(f"[CSV]({rel_csv})")

        blocks.append("\n".join(lines) + ("\n" if lines else ""))

    return blocks


IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mat_mul(A: Tuple[float, float, float, float, float, float], B: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float, float, float, float]:
    """Esegue la moltiplicazione di due matrici di trasformazione 2D affine."""

    a, b, c, d, e, f = A
    a2, b2, c2, d2, e2, f2 = B
    return (
        a * a2 + b * c2,
        a * b2 + b * d2,
        c * a2 + d * c2,
        c * b2 + d * d2,
        e * a2 + f * c2 + e2,
        e * b2 + f * d2 + f2,
    )


def apply_mat(M: Tuple[float, float, float, float, float, float], x: float, y: float) -> Tuple[float, float]:
    """Applica una trasformazione affine 2D a un punto."""

    a, b, c, d, e, f = M
    return (a * x + c * y + e, b * x + d * y + f)


def parse_content_tokens(stream_bytes: bytes) -> List[bytes]:
    """Tokenizza lo stream di contenuto PDF in nomi e numeri utili all'analisi."""

    return re.findall(rb"/[A-Za-z0-9_.]+|[-+]?\d*\.?\d+|[A-Za-z]+", stream_bytes)


@dataclass
class FormPlacement:
    name: str
    xref: int
    bbox: Tuple[float, float, float, float]
    ctm: Tuple[float, float, float, float, float, float]
    rect_mupdf: pymupdf.Rect
    order: int


def get_xobject_subtype_and_bbox(doc: pymupdf.Document, xref: int) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]:
    """Recupera subtype e bounding box di uno XObject a partire dal suo xref."""

    source = doc.xref_object(xref) or ""
    subtype_match = re.search(r"/Subtype\s*/([A-Za-z0-9]+)", source)
    subtype = subtype_match.group(1) if subtype_match else None
    bbox_match = re.search(r"/BBox\s*\[\s*([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s*\]", source)
    bbox = None
    if bbox_match:
        bbox = tuple(float(bbox_match.group(i)) for i in range(1, 5))  # type: ignore
    return subtype, bbox


def find_form_placements_on_page(doc: pymupdf.Document, page: pymupdf.Page, page_no: int, debug: bool = False) -> List[FormPlacement]:
    """Parse page content stream to locate Form XObject placements with approximate bounding boxes."""
    xobjs = page.get_xobjects() or []
    by_name: Dict[str, int] = {}
    for entry in xobjs:
        xref = int(entry[0])
        name = str(entry[1])
        by_name[name] = xref
    if not by_name:
        return []

    try:
        mediabox = getattr(page, "mediabox", None)
        base_height = float(mediabox.height) if mediabox else float(page.rect.height)
    except Exception:
        base_height = float(page.rect.height)

    cropbox = getattr(page, "cropbox", None)
    crop_x0 = float(getattr(cropbox, "x0", 0.0) if cropbox else 0.0)
    crop_y0 = float(getattr(cropbox, "y0", 0.0) if cropbox else 0.0)
    top_ref = float(base_height - crop_y0)
    crop_height = float(getattr(cropbox, "height", 0.0) if cropbox else 0.0)
    if crop_height <= 0.0:
        crop_height = float(base_height)

    stream_all = b""
    for content_xref in page.get_contents() or []:
        try:
            stream_all += doc.xref_stream(content_xref) + b"\n"
        except Exception:
            pass

    tokens = parse_content_tokens(stream_all)

    placements: List[FormPlacement] = []
    ctm = IDENT
    stack: List[Tuple[float, float, float, float, float, float]] = []
    last_nums: List[float] = []
    last_name: Optional[str] = None
    order = 0

    meta_cache: Dict[int, Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]] = {}

    for tok in tokens:
        if re.fullmatch(rb"[-+]?\d*\.?\d+", tok):
            last_nums.append(float(tok))
            continue

        if tok == b"q":
            stack.append(ctm)
            last_nums.clear()
            last_name = None
            continue

        if tok == b"Q":
            ctm = stack.pop() if stack else IDENT
            last_nums.clear()
            last_name = None
            continue

        if tok == b"cm":
            if len(last_nums) >= 6:
                a, b, c, d, e, f = last_nums[-6:]
                M = (a, b, c, d, e, f)
                ctm = mat_mul(M, ctm)
            last_nums.clear()
            last_name = None
            continue

        if tok.startswith(b"/"):
            last_name = tok[1:].decode("utf-8", errors="ignore")
            last_nums.clear()
            continue

        if tok == b"Do":
            if last_name and last_name in by_name:
                xref = by_name[last_name]
                if xref not in meta_cache:
                    meta_cache[xref] = get_xobject_subtype_and_bbox(doc, xref)
                subtype, bbox = meta_cache[xref]
                if subtype == "Form" and bbox is not None:
                    x0, y0, x1, y1 = bbox
                    corners = [
                        apply_mat(ctm, x0, y0),
                        apply_mat(ctm, x1, y0),
                        apply_mat(ctm, x0, y1),
                        apply_mat(ctm, x1, y1),
                    ]
                    xs = [pt[0] - crop_x0 for pt in corners]
                    ys = [pt[1] - crop_y0 for pt in corners]

                    top = top_ref if top_ref > 0 else crop_height if crop_height > 0 else page.rect.height
                    rect_mupdf = pymupdf.Rect(min(xs), top - max(ys), max(xs), top - min(ys))

                    placements.append(
                        FormPlacement(
                            name=last_name,
                            xref=xref,
                            bbox=bbox,
                            ctm=ctm,
                            rect_mupdf=rect_mupdf,
                            order=order,
                        )
                    )
                    order += 1

            last_name = None
            last_nums.clear()
            continue

        last_nums.clear()

    filtered: List[FormPlacement] = []
    for placement in placements:
        rect = placement.rect_mupdf
        if rect.is_empty:
            continue
        if rect.width < 20 or rect.height < 20:
            continue
        rect_clipped = rect & page.rect
        if rect_clipped.is_empty:
            continue
        placement.rect_mupdf = rect_clipped
        filtered.append(placement)

    if debug and filtered:
        LOG.debug("Page %d: found %d Form placements: %s", page_no, len(filtered), [p.name for p in filtered])

    return filtered


def render_and_save_form_images(
    page: pymupdf.Page,
    placements: List[FormPlacement],
    images_dir: Path,
    pdf_filename_prefix: str,
    page_no: int,
    dpi: int = 150,
) -> List[Tuple[FormPlacement, str]]:
    """Rasterizza e salva i Form XObject individuati restituendo i nomi file creati."""

    images_dir.mkdir(parents=True, exist_ok=True)
    out: List[Tuple[FormPlacement, str]] = []
    for idx, placement in enumerate(sorted(placements, key=lambda p: p.order), start=1):
        fname = f"{pdf_filename_prefix}-{page_no:04d}-form-{idx:02d}.png"
        target = unique_target(images_dir / fname)
        pix = page.get_pixmap(clip=placement.rect_mupdf, dpi=dpi, alpha=False)
        pix.save(str(target))
        out.append((placement, target.name))
    return out


def find_vector_regions(page: pymupdf.Page, debug: bool = False) -> List[pymupdf.Rect]:
    """Individua regioni candidate di disegni vettoriali da rasterizzare come immagini."""
    try:
        drawings = page.get_drawings()
    except Exception as exc:  # pragma: no cover - dipende dalla versione di PyMuPDF
        LOG.debug("get_drawings failed on page %d: %s", page.number + 1, exc)
        return []

    if not drawings:
        return []

    page_rect = page.rect
    width = float(page_rect.width)
    height = float(page_rect.height)
    skip_top = float(page_rect.y0) + height * VECTOR_SKIP_Y_RATIO
    skip_bottom = float(page_rect.y1) - height * VECTOR_SKIP_Y_RATIO

    filtered = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None or rect.is_empty:
            continue
        if rect.width < MIN_VECTOR_SIZE_PT or rect.height < MIN_VECTOR_SIZE_PT:
            continue
        if rect.y1 < skip_top or rect.y0 > skip_bottom:
            continue
        if rect.width > width * 0.9 and rect.height > height * 0.9:
            continue
        if rect.width > width * MAX_SEPARATOR_WIDTH_RATIO and rect.height < 20:
            continue
        filtered.append(d)

    if not filtered:
        return []

    try:
        clusters = page.cluster_drawings(
            drawings=filtered, x_tolerance=CLUSTER_X_TOLERANCE, y_tolerance=CLUSTER_Y_TOLERANCE
        )
    except AttributeError:
        LOG.debug("cluster_drawings not available on page %d", page.number + 1)
        clusters = []
    except Exception as exc:
        LOG.debug("cluster_drawings failed on page %d: %s", page.number + 1, exc)
        clusters = []

    if not clusters and len(filtered) >= MIN_VECTOR_PATHS:
        rects = [d.get("rect") for d in filtered if d.get("rect")]
        if rects:
            try:
                union = rects[0]
                for r in rects[1:]:
                    union = union | r
                clusters = [union]
            except Exception:
                clusters = []

    regions: List[pymupdf.Rect] = []
    for rect in clusters:
        if rect.width < MIN_VECTOR_SIZE_PT or rect.height < MIN_VECTOR_SIZE_PT:
            continue
        count = 0
        for d in filtered:
            d_rect = d.get("rect")
            if d_rect and d_rect.intersects(rect):
                count += 1
        if count < MIN_VECTOR_PATHS:
            continue
        padded = pymupdf.Rect(rect)
        padded.x0 = max(page_rect.x0, padded.x0 - VECTOR_PADDING)
        padded.y0 = max(page_rect.y0, padded.y0 - VECTOR_PADDING)
        padded.x1 = min(page_rect.x1, padded.x1 + VECTOR_PADDING)
        padded.y1 = min(page_rect.y1, padded.y1 + VECTOR_PADDING)
        clipped = padded & page_rect
        if clipped.is_empty:
            continue
        regions.append(clipped)

    regions.sort(key=lambda r: r.width * r.height, reverse=True)
    if debug and regions:
        LOG.debug("Page %d: vector regions %s", page.number + 1, regions)
    return regions


def render_vector_regions(
    doc: pymupdf.Document,
    doc_page_index: int,
    regions: List[pymupdf.Rect],
    images_dir: Path,
    pdf_filename_prefix: str,
    pdf_page_no: int,
    dpi: int = 200,
) -> List[str]:
    """Rasterizza le regioni vettoriali trovate e salva le immagini risultanti."""

    if not regions:
        return []
    images_dir.mkdir(parents=True, exist_ok=True)
    out: List[str] = []
    try:
        page = doc[doc_page_index]
    except Exception as exc:  # pragma: no cover - dipende dall'integrità del PDF
        LOG.debug("Page %d access failed: %s", pdf_page_no, exc)
        return []

    for idx, rect in enumerate(regions, start=1):
        fname = f"{pdf_filename_prefix}-{pdf_page_no:04d}-vector-{idx:02d}.png"
        target = unique_target(images_dir / fname)
        try:
            pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
            pix.save(str(target))
            out.append(target.name)
        except Exception as exc:
            LOG.debug("Vector render failed page %d idx %d: %s", pdf_page_no, idx, exc)
            continue
    return out


CAPTION_RE = re.compile(r"(?m)^(Figura\s+\d+(\.\d+)?\s*:.*)$")
PJM_PLACEHOLDER_RE = re.compile(r"\*\*==>\s*picture\s*\[[^\]]+\]\s*intentionally\s*omitted\s*<==\*\*", re.IGNORECASE)
ANNOTATION_LINE_RE = re.compile(r"^>\s*\[annotation:([^\]]+)\]:", re.IGNORECASE)
_BR_TAG_RE = r"(?:<br\s*/?>\s*)?"
EQUATION_BLOCK_START_RE = re.compile(
    rf"^(?:\*\*)?-----\s*Start of equation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
EQUATION_BLOCK_END_RE = re.compile(
    rf"^(?:\*\*)?-----\s*End of equation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
ANNOTATION_BLOCK_START_RE = re.compile(
    rf"^(?:\*\*)?-----\s*Start of annotation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
ANNOTATION_BLOCK_END_RE = re.compile(
    rf"^(?:\*\*)?-----\s*End of annotation:\s*(.+?)\s*-----\s*(?:\*\*)?\s*{_BR_TAG_RE}$",
    re.IGNORECASE,
)
PAGE_END_MARKER_RE = re.compile(r"(?m)^\s*---\s*end of page\.page_number=\d+\s*---\s*$\n?")
PAGE_START_MARKER_CAPTURE_RE = re.compile(r"^\s*---\s*start of page\.page_number=(\d+)\s*---\s*$", re.IGNORECASE)
PAGE_END_MARKER_CAPTURE_RE = re.compile(r"^\s*---\s*end of page\.page_number=(\d+)\s*---\s*$", re.IGNORECASE)


def inject_images_into_page_markdown(page_text: str, rel_image_paths: List[str]) -> Tuple[str, str]:
    """
    Inserisce le immagini nel Markdown della pagina preservando eventuali marker alfabetici.
    Strategia:
      0) sostituisce in ordine i placeholder PyMuPDF "**==> picture [...] intentionally omitted <==**"
      1) se esistono linee con singole lettere (es. "A"/"B"), mantiene la lettera e inserisce l'immagine dopo
      2) altrimenti inserisce le immagini prima delle didascalie "Figura ..."
      3) in mancanza, appende le immagini al termine del contenuto
    """

    lines = (page_text or "").splitlines()
    out_lines: List[str] = []
    img_index = 0

    def consume_placeholder(md: str) -> Tuple[str, int]:
        """Sostituisce progressivamente i placeholder PyMuPDF con le immagini disponibili."""

        nonlocal img_index
        replaced = 0
        parts = []
        last = 0
        for match in PJM_PLACEHOLDER_RE.finditer(md):
            parts.append(md[last:match.start()])
            if img_index < len(rel_image_paths):
                parts.append(f"![]({rel_image_paths[img_index]})")
                img_index += 1
            replaced += 1  # removes the placeholder even when not replaced by an image
            last = match.end()
        parts.append(md[last:])
        return "".join(parts), replaced

    text, replaced = consume_placeholder("\n".join(lines))
    if replaced:
        lines = text.splitlines()
    else:
        text = "\n".join(lines)

    for line in lines:
        if img_index < len(rel_image_paths) and re.fullmatch(r"\s*[A-Z]\s*", line or ""):
            out_lines.append(line)
            out_lines.append("")
            out_lines.append(f"![]({rel_image_paths[img_index]})")
            out_lines.append("")
            img_index += 1
        else:
            out_lines.append(line)

    text = "\n".join(out_lines)

    if img_index == len(rel_image_paths):
        return text, "inserted_after_single_letter_lines" if replaced == 0 else "placeholders_and_letters"

    captions = list(CAPTION_RE.finditer(text))
    if captions:
        pieces: List[str] = []
        last = 0
        cap_idx = 0
        while img_index < len(rel_image_paths) and cap_idx < len(captions):
            match = captions[cap_idx]
            pieces.append(text[last:match.start()])
            pieces.append(f"![]({rel_image_paths[img_index]})\n")
            img_index += 1
            last = match.start()
            cap_idx += 1
        pieces.append(text[last:])
        txt2 = "".join(pieces)
        if img_index == len(rel_image_paths):
            return txt2, "inserted_before_captions"

        txt2 = txt2.rstrip() + "\n" + "\n".join(f"![]({p})" for p in rel_image_paths[img_index:]) + "\n"
        return txt2, "captions_then_append"

    txt3 = text.rstrip() + "\n" + "\n".join(f"![]({p})" for p in rel_image_paths[img_index:]) + "\n"
    return txt3, "appended_end"


def iter_search_dirs(pdf_path: Path, out_dir: Path) -> List[Path]:
    """Restituisce i percorsi da ispezionare per spostare file generati dal PDF."""

    dirs = [pdf_path.parent.resolve(), Path.cwd().resolve(), out_dir.resolve()]
    seen = set()
    uniq: List[Path] = []
    for directory in dirs:
        if directory not in seen:
            seen.add(directory)
            uniq.append(directory)
    return uniq


def move_files_by_name(names: Iterable[str], search_dirs: List[Path], dest_dir: Path, verbose: bool = False) -> int:
    """Sposta file noti da più cartelle verso la destinazione garantendo nomi univoci."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in names:
        found: Optional[Path] = None
        for directory in search_dirs:
            candidate = directory / name
            if candidate.exists() and candidate.is_file():
                found = candidate
                break
        if not found:
            continue
        if found.resolve().parent == dest_dir.resolve():
            continue
        target = unique_target(dest_dir / found.name)
        try:
            shutil.move(str(found), str(target))
            moved += 1
            if verbose:
                LOG.info("Moved image: %s -> %s", found, target)
        except Exception as exc:
            LOG.debug("Unable to move %s -> %s (%s)", found, target, exc)
    return moved


def collect_probable_generated_images(pdf_path: Path, fmt: str, search_dirs: List[Path]) -> Set[str]:
    """
    Raccoglie i PNG plausibili generati da pymupdf4llm anche se non linkati (es. pdf_sample.pdf-0003-10.png).
    I render dei Form XObject creati direttamente in images/ sono volutamente esclusi.
    """
    prefixes = [pdf_path.name, pdf_path.stem]
    pattern = re.compile(
        rf"^({'|'.join(re.escape(p) for p in prefixes)})-\d{{4}}-\d+\.{re.escape(fmt)}$",
        re.IGNORECASE,
    )

    names: Set[str] = set()
    for directory in search_dirs:
        for path in directory.glob(f"*.{fmt}"):
            if path.is_file() and pattern.match(path.name):
                names.add(path.name)
    return names


def mm_to_points(value_mm: float) -> float:
    """Converte millimetri in punti tipografici."""

    return float(value_mm) * MM_TO_PT


def apply_vertical_crop_margins(
    doc: pymupdf.Document, header_mm: float, footer_mm: float, debug: bool = False, verbose: bool = False
) -> None:
    """Applica il crop verticale in base ai margini header/footer mantenendo l'area valida."""
    if header_mm <= 0 and footer_mm <= 0:
        return

    header_pts = mm_to_points(max(0.0, header_mm))
    footer_pts = mm_to_points(max(0.0, footer_mm))

    for page in doc:
        rect = page.rect
        max_removable = max(rect.height - 1.0, 0.0)
        total_requested = header_pts + footer_pts
        if total_requested > max_removable and total_requested > 0:
            scale = max_removable / total_requested
            top_adj = header_pts * scale
            bottom_adj = footer_pts * scale
        else:
            top_adj = header_pts
            bottom_adj = footer_pts
        new_rect = pymupdf.Rect(rect.x0, rect.y0 + top_adj, rect.x1, rect.y1 - bottom_adj)
        page.set_cropbox(new_rect)
        if debug or verbose:
            LOG.log(
                logging.DEBUG if debug else logging.INFO,
                "Applied vertical crop on page %d: top %.2f pt (%.2f mm) bottom %.2f pt (%.2f mm) -> rect %s",
                page.number + 1,
                top_adj,
                top_adj / MM_TO_PT,
                bottom_adj,
                bottom_adj / MM_TO_PT,
                new_rect,
            )


def embed_equations_in_markdown(md_text: str, formulas_by_base: Dict[str, str]) -> str:
    """Inserisce le formule LaTeX immediatamente prima dei link immagine corrispondenti."""
    if not formulas_by_base:
        return md_text

    lines = (md_text or "").splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        start_match = EQUATION_BLOCK_START_RE.match(stripped)
        if start_match:
            block_base = start_match.group(1).strip()
            if block_base in formulas_by_base:
                i += 1
                while i < len(lines):
                    if EQUATION_BLOCK_END_RE.match(lines[i].strip()):
                        i += 1
                        break
                    i += 1
                continue
            output.append(line)
            i += 1
            while i < len(lines):
                output.append(lines[i])
                if EQUATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            continue

        matches = list(IMG_LINK_RE.finditer(line))
        if not matches:
            output.append(line)
            i += 1
            continue

        formula_match = None
        formula_base = ""
        for match in matches:
            url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
            base = url.split("/")[-1]
            if base in formulas_by_base:
                formula_match = match
                formula_base = base
                break

        if not formula_match:
            output.append(line)
            i += 1
            continue

        before = line[: formula_match.start()]
        after = line[formula_match.end() :]
        if before.strip():
            output.append(before.rstrip())

        if output and re.match(r"^\s*\$\$.*\$\$\s*$", output[-1]):
            output.pop()

        if output and output[-1].strip():
            output.append("")

        start_line = f"**----- Start of equation: {formula_base} -----**\n\n"
        output.append(start_line)
        output.append(f"$${formulas_by_base[formula_base]}$$")
        output.append(f"\n**----- End of equation: {formula_base} -----**\n")
        if output and output[-1].strip():
            output.append("")
        output.append(formula_match.group(0))
        if after.strip():
            output.append(after.lstrip())
        i += 1

    return "\n".join(output)


def embed_annotations_in_markdown(md_text: str, annotations: Dict[str, str]) -> str:
    """Inserisce i blocchi di annotazione subito dopo i link immagine corrispondenti."""
    if not annotations:
        return md_text

    lines = (md_text or "").splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        ann_old = ANNOTATION_LINE_RE.match(stripped)
        if ann_old:
            base_existing = ann_old.group(1)
            if base_existing in annotations:
                i += 1
                while i < len(lines) and lines[i].startswith(">"):
                    i += 1
                continue
            output.append(line)
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                output.append(lines[i])
                i += 1
            continue

        ann_start = ANNOTATION_BLOCK_START_RE.match(stripped)
        if ann_start:
            base_block = ann_start.group(1).strip()
            if base_block in annotations:
                i += 1
                while i < len(lines):
                    if ANNOTATION_BLOCK_END_RE.match(lines[i].strip()):
                        i += 1
                        break
                    i += 1
                continue
            output.append(line)
            i += 1
            while i < len(lines):
                output.append(lines[i])
                if ANNOTATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            continue

        output.append(line)
        matches = list(IMG_LINK_RE.finditer(line))
        for match in matches:
            url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
            base = url.split("/")[-1]
            if base not in annotations:
                continue
            text = annotations[base].strip()
            if not text:
                continue
            if output and output[-1].strip():
                output.append("")
            start_line = f"**----- Start of annotation: {base} -----**\n\n"
            output.append(start_line)
            ann_lines = text.splitlines() or [text]
            for extra in ann_lines:
                output.append(extra.strip())
            output.append(f"\n**----- End of annotation: {base} -----**\n")
            output.append("")
        i += 1

    return "\n".join(output)


def _strip_image_links_from_line(line: str, basenames: Set[str]) -> Tuple[str, bool]:
    """Rimuove i link a immagini specifiche da una singola riga Markdown."""

    if not basenames:
        return line, False
    result_parts: List[str] = []
    last_index = 0
    removed = False
    for match in IMG_LINK_RE.finditer(line):
        result_parts.append(line[last_index : match.start()])
        url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
        url = url.split("?", 1)[0].split("#", 1)[0]
        base = url.split("/")[-1]
        if base in basenames:
            removed = True
        else:
            result_parts.append(match.group(0))
        last_index = match.end()
    if not removed:
        return line, False
    result_parts.append(line[last_index:])
    new_line = "".join(result_parts)
    if not new_line.strip():
        return "", True
    return new_line, True


def strip_image_references_from_markdown(md_text: str, basenames: Set[str]) -> str:
    """Rimuove link immagine e blocchi Pix2Tex/annotazioni per i basename indicati."""
    if not basenames:
        return md_text

    lines = md_text.splitlines()
    output: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        eq_start = EQUATION_BLOCK_START_RE.match(stripped)
        if eq_start and eq_start.group(1).strip() in basenames:
            i += 1
            while i < len(lines):
                if EQUATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        ann_start = ANNOTATION_BLOCK_START_RE.match(stripped)
        if ann_start and ann_start.group(1).strip() in basenames:
            i += 1
            while i < len(lines):
                if ANNOTATION_BLOCK_END_RE.match(lines[i].strip()):
                    i += 1
                    break
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        new_line, removed = _strip_image_links_from_line(line, basenames)
        if removed and not new_line.strip():
            if output and not output[-1].strip():
                pass
            else:
                output.append("")
        else:
            output.append(new_line)
        i += 1

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def remove_small_images_phase(
    manifest: Dict[str, Any],
    md_text: str,
    out_dir: Path,
    config: PostProcessingConfig,
) -> Tuple[str, Dict[str, Any]]:
    """Filtra le immagini troppo piccole dal manifest e dal Markdown mantenendo i file su disco."""
    images = list(manifest.get("images") or [])
    if not images:
        return md_text, manifest

    kept: List[Dict[str, Any]] = []
    removed_basenames: Set[str] = set()
    min_width = max(1, int(config.min_size_x))
    min_height = max(1, int(config.min_size_y))
    total = len(images)

    for index, entry in enumerate(images, start=1):
        rel_file = entry.get("file")
        if not rel_file:
            kept.append(entry)
            continue
        rel_path = normalize_path_for_md(str(rel_file))
        path = out_dir / rel_path
        base_name = Path(rel_path).name
        width: Optional[int] = None
        height: Optional[int] = None
        if path.exists():
            pix_obj: Optional[pymupdf.Pixmap] = None
            try:
                pix_obj = pymupdf.Pixmap(str(path))
                width, height = int(pix_obj.width), int(pix_obj.height)
            except Exception as exc:
                if config.debug:
                    LOG.debug("remove-small-images unable to read %s: %s", path, exc)
            finally:
                pix_obj = None
        else:
            if config.debug:
                LOG.debug("remove-small-images missing file: %s", path)

        should_remove = bool(width is not None and height is not None and width < min_width and height < min_height)
        if config.verbose:
            size_label = f"{width}x{height}px" if width is not None and height is not None else "unknown size"
            status = "REMOVE" if should_remove else "KEEP"
            _log_verbose_progress(
                "remove-small-images",
                index,
                total,
                detail=f"{rel_file} -> {status} ({size_label})",
            )

        if should_remove:
            removed_basenames.add(base_name)
            if config.debug:
                LOG.debug("remove-small-images flagged %s for removal but retains disk file", path)
            continue

        kept.append(entry)

    manifest["images"] = kept
    if not removed_basenames:
        return md_text, manifest

    cleaned_md = strip_image_references_from_markdown(md_text, removed_basenames)
    return cleaned_md, manifest


def remove_markdown_index(md_text: str, pdf_toc: List[Tuple[int, str, Optional[int]]]) -> str:
    """Rimuove il contenuto iniziale fino alla prima voce del pdf_toc mantenendo i marker di pagina."""

    if not md_text or not pdf_toc:
        return md_text

    first_title = _normalize_title_for_toc(str(pdf_toc[0][1]).strip()) if len(pdf_toc[0]) >= 2 else ""
    if not first_title:
        return md_text

    heading_re = re.compile(r"^(?P<prefix>\s*)(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

    lines = md_text.splitlines()
    output: List[str] = []
    keep_content = False
    matched_heading = False

    # Preserva l'eventuale front matter YAML iniziale.
    idx = 0
    if lines and lines[0].strip() == "---":
        output.append(lines[0])
        idx = 1
        while idx < len(lines):
            output.append(lines[idx])
            if lines[idx].strip() == "---":
                idx += 1
                break
            idx += 1
    else:
        idx = 0

    for line in lines[idx:]:
        stripped = line.strip()

        if PAGE_START_MARKER_CAPTURE_RE.match(stripped) or PAGE_END_MARKER_CAPTURE_RE.match(stripped):
            output.append(line)
            continue

        if not keep_content:
            match = heading_re.match(line)
            if match:
                candidate = _normalize_title_for_toc(match.group("title"))
                if candidate == first_title:
                    keep_content = True
                    matched_heading = True
                    output.append(line)
                    continue
            continue

        output.append(line)

    if not matched_heading:
        return md_text

    result = "\n".join(output)
    if md_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def cleanup_markdown(md_text: str, pdf_headings: Optional[List[Tuple[int, str, Optional[int]]]] = None) -> str:
    """Pulisce il Markdown degradando le intestazioni fuori TOC e rimuovendo i marker di pagina."""

    if not md_text:
        return md_text

    base_text = clean_markdown_headings(md_text, pdf_headings or []) if pdf_headings else md_text

    cleaned_lines: List[str] = []
    for raw in base_text.splitlines():
        stripped = raw.strip()
        if PAGE_START_MARKER_CAPTURE_RE.match(stripped) or PAGE_END_MARKER_CAPTURE_RE.match(stripped):
            continue
        cleaned_lines.append(raw)

    result = "\n".join(cleaned_lines)
    if base_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _slugify_markdown_heading(title: str) -> str:
    """Genera uno slug Markdown da un titolo per collegamenti interni."""

    slug = unicodedata.normalize("NFKD", title or "").lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug or "section"


def _scan_markdown_for_toc_entries(md_text: str) -> List[Tuple[str, int, str, str]]:
    """Analizza il Markdown e restituisce la sequenza delle voci TOC e dei riferimenti."""

    lines = md_text.splitlines()
    toc_sequence: List[Tuple[str, int, str, str]] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    image_re = re.compile(r"!\[[^\]]*\]\(\s*images/[^)]+\)")
    table_re = re.compile(r"\(\s*tables/[^)]+\)")

    for raw in lines:
        line = raw.rstrip()
        heading = heading_re.match(line)
        if heading:
            level = len(heading.group(1))
            title_raw = heading.group(2).strip()
            page_note = ""
            match = re.search(r"\(pag\.\s*([^)]+)\)", title_raw)
            if match:
                page_note = match.group(1).strip()
                title_raw = title_raw.replace(match.group(0), "").strip()
            title_display = re.sub(r"[*_`]+", "", title_raw).strip()
            title_clean = re.sub(r"[*_`]+", "", title_raw).strip()
            title_clean = re.sub(r"^\d+(?:\.\d+)*\s+", "", title_clean).strip()
            if not title_display:
                continue
            if title_clean.lower() in {"indice", "toc"} or title_clean.lower().startswith("capitolo "):
                continue
            toc_sequence.append(("heading", level, title_display, page_note))
            continue
        if image_re.search(line):
            toc_sequence.append(("raw", 0, line.strip(), ""))
            continue
        if table_re.search(line):
            toc_sequence.append(("raw", 0, line.strip(), ""))
            continue

    return toc_sequence


def normalize_markdown_headings(md_text: str, toc_headings: List[Tuple[Any, ...]]) -> str:
    """Normalizza intestazioni e TOC nel Markdown in base ai titoli presenti nella TOC del PDF."""

    lines = md_text.splitlines()

    normalized_toc: List[Tuple[int, str, Optional[int]]] = []
    for entry in toc_headings:
        if len(entry) < 2:
            continue
        level = int(entry[0])
        title = str(entry[1])
        page: Optional[int] = None
        if len(entry) >= 3:
            try:
                page = int(entry[2])
            except Exception:
                page = None
        normalized_toc.append((level, title, page))

    search_pos = 0
    for level, title, _ in normalized_toc:
        target = _normalize_title_for_toc(title)
        for idx in range(search_pos, len(lines)):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("<!--"):
                continue
            candidate = re.sub(r"^#{1,6}\s*", "", stripped).strip()
            candidate = re.sub(r"[*_`]+", "", candidate).strip()
            candidate = re.sub(r"^\d+(?:\.\d+)*\s+", "", candidate).strip()
            if not candidate:
                continue
            if _normalize_title_for_toc(candidate) == target:
                heading_line = f"{'#' * max(1, min(level, 6))} {title}"
                lines[idx] = heading_line
                search_pos = idx + 1
                break

    result = "\n".join(lines)
    if md_text.endswith("\n"):
        result += "\n"
    return result


def clean_markdown_headings(md_text: str, pdf_headings: List[Tuple[int, str, Optional[int]]]) -> str:
    """Degrada le intestazioni Markdown non presenti nella TOC in testo maiuscolo in grassetto."""

    normalized_titles = {
        _normalize_title_for_toc(str(title))
        for _, title, *rest in pdf_headings
        if str(title).strip()
    }

    heading_re = re.compile(r"^(?P<prefix>\s*)(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

    def _strip_bold(text: str) -> str:
        """Rimuove il grassetto esterno se presente restituendo il testo pulito."""

        t = text.strip()
        if (t.startswith("**") and t.endswith("**")) or (t.startswith("__") and t.endswith("__")):
            return t[2:-2].strip()
        return t

    lines = md_text.splitlines()
    output: List[str] = []
    for line in lines:
        match = heading_re.match(line)
        if not match:
            output.append(line)
            continue

        title_raw = match.group("title").strip()
        normalized = _normalize_title_for_toc(title_raw)

        if normalized in normalized_titles:
            output.append(line)
            continue

        clean_title = _strip_bold(title_raw).upper()
        bold_title = f"**{clean_title}**"
        output.append(f"{match.group('prefix')}{bold_title}")

    result = "\n".join(output)
    if md_text.endswith("\n"):
        result += "\n"
    return result


def add_pdf_toc_to_markdown(md_text: str, pdf_headings: List[Tuple[int, str, Optional[int]]]) -> str:
    """Inserisce una TOC Markdown derivata dal pdf_toc all'inizio del documento."""

    if not md_text or not pdf_headings:
        return md_text

    def _normalize_toc_heading_variants(content: str) -> str:
        """Normalizza intestazioni TOC alternative sul formato canonico in grassetto."""

        lines = content.splitlines()
        pattern = re.compile(
            r"^\s*(?:#{1,6}\s*)?(?:\*{0,2}\s*)?(?:pdf\s+toc|toc)\s*(?:\*{0,2}\s*)?$",
            re.IGNORECASE,
        )
        normalized: List[str] = []
        for line in lines:
            if pattern.match(line.strip()):
                normalized.append("** PDF TOC **")
            else:
                normalized.append(line)
        result = "\n".join(normalized)
        if content.endswith("\n") and not result.endswith("\n"):
            result += "\n"
        return result

    md_text = _normalize_toc_heading_variants(md_text)

    toc_lines: List[str] = ["** PDF TOC **", ""]
    for level, title, _ in pdf_headings:
        indent = "  " * max(0, int(level) - 1)
        anchor = _slugify_markdown_heading(title)
        toc_lines.append(f"{indent}- [{title}](#{anchor})")
    toc_lines.append("")

    lines = md_text.splitlines()
    insert_at = 0
    for idx, raw in enumerate(lines):
        if PAGE_START_MARKER_CAPTURE_RE.match(raw.strip()):
            insert_at = idx + 1
            break

    new_lines = lines[:insert_at] + toc_lines + lines[insert_at:]
    result = "\n".join(new_lines)
    if md_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def normalize_markdown_format(md_text: str) -> str:
    """Normalizza il formato Markdown sostituendo elementi HTML come <br> con newline."""

    if not md_text:
        return md_text
    return re.sub(r"(?i)<br\s*/?>", "\n", md_text)


def generate_markdown_toc_file(md_text: str, md_path: Path, out_dir: Path) -> Tuple[Path, List[Tuple[int, str]]]:
    """Estrae voci di TOC dal Markdown e scrive un file .toc in formato Markdown."""

    toc_sequence = _scan_markdown_for_toc_entries(md_text)
    toc_lines: List[str] = []
    heading_entries: List[Tuple[int, str]] = []

    for item in toc_sequence:
        kind, level, text, page_note = item
        if kind == "heading":
            indent = "  " * max(0, level - 1)
            anchor = _slugify_markdown_heading(text)
            page_suffix = f" (pag. {page_note})" if page_note else ""
            toc_lines.append(f"{indent}- [{text}](#{anchor}){page_suffix}")
            heading_entries.append((level, text))
        else:
            toc_lines.append(text)

    toc_path = md_path.with_suffix(".toc")
    content = "\n".join([ln for ln in toc_lines if ln.strip()])
    safe_write_text(toc_path, (content + "\n") if content else "")
    return toc_path, heading_entries


def normalize_markdown_file(pdf_path: Path, md_path: Path, out_dir: Path, *, add_toc: bool = True) -> Tuple[str, List[Tuple[int, str]], List[List[Any]], Path, List[Tuple[int, str, Optional[int]]]]:
    """Normalizza il Markdown ripristinato usando la TOC del PDF e rigenera il file .toc."""

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Unable to open source PDF {pdf_path}: {exc}") from exc

    try:
        toc_raw = doc.get_toc() or []
    except Exception as exc:
        try:
            doc.close()
        except Exception:
            pass
        raise RuntimeError(f"Unable to read PDF TOC from {pdf_path}: {exc}") from exc
    finally:
        try:
            doc.close()
        except Exception:
            pass

    pdf_headings: List[Tuple[int, str, Optional[int]]] = []
    for entry in toc_raw:
        if len(entry) < 2:
            continue
        try:
            level = int(entry[0])
        except Exception:
            continue
        title = str(entry[1]).strip()
        page_no: Optional[int] = None
        if len(entry) >= 3:
            try:
                page_no = int(entry[2])
            except Exception:
                page_no = None
        if not title or title.lower() in {"indice", "toc"}:
            continue
        pdf_headings.append((level, title, page_no))

    md_text = md_path.read_text(encoding="utf-8")
    md_text = normalize_markdown_format(md_text)
    md_text = remove_markdown_index(md_text, pdf_headings)
    normalized_md = normalize_markdown_headings(md_text, pdf_headings)
    cleaned_md = clean_markdown_headings(normalized_md, pdf_headings)
    final_md = add_pdf_toc_to_markdown(cleaned_md, pdf_headings) if add_toc else cleaned_md
    safe_write_text(md_path, final_md)
    toc_path, toc_headings = generate_markdown_toc_file(final_md, md_path, out_dir)

    return final_md, toc_headings, toc_raw, toc_path, pdf_headings


def _normalize_title_for_toc(title: str) -> str:
    """Normalizza un titolo per confronti TOC eliminando differenze stilistiche."""

    norm = unicodedata.normalize("NFKD", title or "")
    norm = norm.replace("\u2019", "'").replace("\u2018", "'")
    norm = norm.replace("\u201c", '"').replace("\u201d", '"')
    norm = norm.replace("\u00a0", " ")
    norm = re.sub(r"[*_`]+", "", norm)
    norm = re.sub(r"^\d+(?:\.\d+)*\s+", "", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip()


def validate_markdown_toc_against_pdf(pdf_path: Path, headings_md: List[Tuple[int, str]]) -> TocValidationResult:
    """Confronta la TOC estratta dal Markdown con quella nativa del PDF restituendo l'esito senza interrompere il flusso."""

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Unable to open source PDF {pdf_path}: {exc}") from exc

    try:
        toc_pdf = doc.get_toc() or []
    except Exception as exc:
        doc.close()
        raise RuntimeError(f"Unable to read PDF TOC from {pdf_path}: {exc}") from exc

    pdf_headings: List[Tuple[int, str]] = []
    for entry in toc_pdf:
        if len(entry) < 2:
            continue
        try:
            level = int(entry[0])
        except Exception:
            continue
        title = str(entry[1]).strip()
        if title.lower() in {"indice", "toc"}:
            continue
        pdf_headings.append((level, title))

    try:
        doc.close()
    except Exception:
        pass

    titles_pdf = [_normalize_title_for_toc(title) for _, title in pdf_headings]
    titles_md = [_normalize_title_for_toc(title) for _, title in headings_md]

    mismatches: List[Tuple[int, str, str]] = []
    max_len = max(len(titles_pdf), len(titles_md))
    for idx in range(max_len):
        pdf_title = titles_pdf[idx] if idx < len(titles_pdf) else "<missing>"
        md_title = titles_md[idx] if idx < len(titles_md) else "<missing>"
        if pdf_title != md_title:
            mismatches.append((idx, pdf_title, md_title))

    ok = len(mismatches) == 0
    if ok:
        reason = ""
    elif len(titles_pdf) != len(titles_md):
        reason = f"TOC length differs (pdf={len(titles_pdf)}, md={len(titles_md)})"
    else:
        first = mismatches[0]
        reason = (
            f"TOC content differs at position {first[0] + 1}: pdf='{first[1]}' vs md='{first[2]}'"
        )

    return TocValidationResult(
        ok=ok,
        pdf_titles=titles_pdf,
        md_titles=titles_md,
        mismatches=mismatches,
        pdf_count=len(titles_pdf),
        md_count=len(titles_md),
        reason=reason,
    )


def log_toc_validation_result(result: TocValidationResult, *, verbose: bool, debug: bool) -> None:
    """Emette log di esito TOC includendo evidenza dettagliata in verbose/debug."""

    if result.ok:
        if verbose:
            LOG.info("TOC validation passed (%d entries)", result.pdf_count)
        return

    base_msg = f"TOC mismatch between PDF and Markdown .toc (pdf={result.pdf_count}, md={result.md_count})"
    if result.mismatches:
        first = result.mismatches[0]
        base_msg += (
            f"; first difference at position {first[0] + 1}: pdf='{first[1]}' vs md='{first[2]}'"
        )
    if result.reason:
        base_msg = f"{base_msg} | {result.reason}"
    LOG.error(base_msg)

    if verbose:
        LOG.info("TOC comparison (PDF vs Markdown):")
        mismatch_idx = {idx for idx, _, _ in result.mismatches}
        max_len = max(result.pdf_count, result.md_count)
        for idx in range(max_len):
            pdf_title = result.pdf_titles[idx] if idx < result.pdf_count else "<missing>"
            md_title = result.md_titles[idx] if idx < result.md_count else "<missing>"
            status = "OK" if idx not in mismatch_idx else "FAIL"
            LOG.info("[%d] %s | PDF=\"%s\" | MD=\"%s\"", idx + 1, status, pdf_title or "<empty>", md_title or "<empty>")

    if debug:
        LOG.debug("TOC normalized PDF titles (%d): %s", result.pdf_count, result.pdf_titles)
        LOG.debug("TOC normalized Markdown titles (%d): %s", result.md_count, result.md_titles)
        LOG.debug("TOC mismatches indexes: %s", [idx for idx, _, _ in result.mismatches])


def build_manifest_from_outputs(
    *,
    pdf_path: Path,
    md_path: Path,
    out_dir: Path,
    images_dir: Path,
    tables_dir: Path,
    toc_path: Optional[Path] = None,
    toc_root: Optional[TocNode] = None,
    toc_raw: Optional[List[List[Any]]] = None,
    manifest_tables: Optional[List[Dict[str, Any]]] = None,
    image_page_map: Optional[Dict[str, int]] = None,
    image_source: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Costruisce il manifest consolidando tabelle e immagini preservando i metadati disponibili."""

    current_manifest = {}
    tables: List[Dict[str, Any]] = list(manifest_tables or current_manifest.get("tables") or [])
    toc_root_local = toc_root
    toc_path_local = toc_path or md_path.with_suffix(".toc")
    doc: Optional[pymupdf.Document] = None

    try:
        md_text_for_tables = md_path.read_text(encoding="utf-8")
    except Exception:
        md_text_for_tables = ""
    table_ref_re = re.compile(r"tables/([^)\s]+?)(?:\\.|\)|\s)")
    referenced_table_stems: Set[str] = set()
    for match in table_ref_re.finditer(md_text_for_tables):
        stem = Path(match.group(1)).stem
        if stem:
            referenced_table_stems.add(stem)

    if toc_root_local is None:
        try:
            if toc_raw is not None:
                toc_root_local = build_toc_tree(toc_raw)
            else:
                doc = pymupdf.open(str(pdf_path))
                raw = doc.get_toc() or []
                toc_root_local = build_toc_tree(raw)
        except Exception as exc:
            LOG.debug("doc.get_toc failed during manifest build: %s", exc)
            toc_root_local = build_toc_tree([])

    manifest_markdown = {
        "file": relative_to_output(md_path, out_dir),
        "toc_tree": serialize_toc_tree(toc_root_local),
    }

    existing_table_stems: Set[str] = set()
    for entry in tables:
        for f in entry.get("files", []):
            existing_table_stems.add(Path(f).stem)

    if tables_dir.exists():
        grouped_tables: Dict[str, List[Path]] = {}
        for path in tables_dir.iterdir():
            if not path.is_file():
                continue
            grouped_tables.setdefault(path.stem, []).append(path)

        for stem, files in grouped_tables.items():
            if stem in existing_table_stems:
                continue
            if referenced_table_stems and stem not in referenced_table_stems:
                continue
            page_for_table = guess_page_from_filename(stem)
            table_names = [p.name for p in files]
            context_path, context_str = find_context(toc_path_local, toc_root_local, table_names, page_for_table)
            title = context_path[-1] if context_path else ""
            tables.append(
                {
                    "pdf_source_page": page_for_table,
                    "title": title,
                    "context": context_str,
                    "context_path": context_path,
                    "files": [relative_to_output(p, out_dir) for p in sorted(files)],
                }
            )

    manifest_images: List[Dict[str, Any]] = []
    old_images_by_base: Dict[str, Dict[str, Any]] = {}
    for entry in current_manifest.get("images", []) or []:
        base = Path(str(entry.get("file", ""))).name
        if base:
            old_images_by_base[base] = entry

    images_dir_exists = images_dir.exists()
    if images_dir_exists:
        for path in sorted(images_dir.glob("*")):
            if not path.is_file():
                continue
            base_name = path.name
            previous = old_images_by_base.get(base_name, {})
            page_for_img: Optional[int] = None
            if image_page_map:
                page_for_img = image_page_map.get(base_name)
            if page_for_img is None:
                page_for_img = previous.get("pdf_source_page")
            if page_for_img is None:
                guessed = guess_page_from_filename(base_name)
                page_for_img = guessed if guessed is not None else None

            context_path, context_str = find_context(toc_path_local, toc_root_local, [base_name], page_for_img)
            title = context_path[-1] if context_path else ""

            manifest_images.append(
                {
                    "file": relative_to_output(path, out_dir),
                    "pdf_source_page": page_for_img,
                    "title": title,
                    "context": context_str,
                    "context_path": context_path,
                    "source": previous.get("source")
                    or (image_source or {}).get(base_name)
                    or "pymupdf",
                    "type": previous.get("type", "image"),
                    **({"equation": previous["equation"]} if "equation" in previous else {}),
                    **({"annotation": previous["annotation"]} if "annotation" in previous else {}),
                }
            )

    manifest = {
        "source_pdf": pdf_path.name,
        "markdown": manifest_markdown,
        "tables": tables,
        "images": manifest_images,
    }

    if doc is not None:
        try:
            doc.close()
        except Exception:
            pass

    return manifest, tables, manifest_images


def generate_item_ids(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Assegna ID univoci e coerenti a nodi TOC, tabelle e immagini del manifest."""

    toc_counter = 1

    def _assign_toc_ids(nodes: List[Dict[str, Any]]) -> None:
        nonlocal toc_counter
        for node in nodes:
            node["id"] = node.get("id") or f"toc-{toc_counter}"
            toc_counter += 1
            children = node.get("children") or []
            node["children"] = children
            _assign_toc_ids(children)

    markdown_section = manifest.get("markdown") or {}
    toc_tree = markdown_section.get("toc_tree") or []
    _assign_toc_ids(toc_tree)

    table_counter = 1
    for table in manifest.get("tables") or []:
        table["id"] = table.get("id") or f"table-{table_counter}"
        table_counter += 1

    image_counter = 1
    for image in manifest.get("images") or []:
        image["id"] = image.get("id") or f"img-{image_counter}"
        image_counter += 1

    return manifest


def referring_toc(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Popola relazioni parent/prev/next per i nodi TOC seguendo la gerarchia."""

    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []

    ordered_nodes: List[Tuple[Dict[str, Any], Optional[str]]] = []

    def _collect(nodes: List[Dict[str, Any]], parent_id: Optional[str]) -> None:
        for node in nodes:
            node_id = node.get("id")
            children = node.get("children") or []
            if node_id:
                ordered_nodes.append((node, parent_id))
                _collect(children, node_id)
            else:
                _collect(children, parent_id)

    _collect(toc_tree, None)

    for idx, (node, parent_id) in enumerate(ordered_nodes):
        if parent_id:
            node["parent_id"] = parent_id
        else:
            node.pop("parent_id", None)

        if idx > 0:
            node["prev_id"] = ordered_nodes[idx - 1][0].get("id")
        else:
            node.pop("prev_id", None)

        if idx + 1 < len(ordered_nodes):
            node["next_id"] = ordered_nodes[idx + 1][0].get("id")
        else:
            node.pop("next_id", None)

    return manifest


def _find_toc_parent_id_from_context(
    toc_tree: List[Dict[str, Any]], context_path: List[str]
) -> Optional[str]:
    """Risoluzione best-effort del nodo TOC partendo dal context_path."""

    if not context_path:
        return None

    current_nodes = toc_tree
    parent_id: Optional[str] = None
    for title in context_path:
        target = _normalize_title_for_toc(title)
        match: Optional[Dict[str, Any]] = None
        for candidate in current_nodes:
            cand_title = _normalize_title_for_toc(str(candidate.get("title", "")))
            if cand_title == target:
                match = candidate
                break
        if match is None:
            parent_id = None
            break
        parent_id = match.get("id")
        current_nodes = match.get("children") or []

    if parent_id:
        return parent_id

    last_title = _normalize_title_for_toc(context_path[-1])
    stack = list(toc_tree)
    while stack:
        node = stack.pop(0)
        if _normalize_title_for_toc(str(node.get("title", ""))) == last_title:
            return node.get("id")
        stack.extend(node.get("children") or [])

    return None


def referring_tables(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Collega le tabelle al nodo TOC contenitore e ai fratelli adiacenti."""

    tables = manifest.get("tables") or []
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []

    parent_groups: Dict[str, List[Dict[str, Any]]] = {}

    for table in tables:
        context_path = table.get("context_path") or []
        parent_id = _find_toc_parent_id_from_context(toc_tree, context_path)
        if parent_id:
            table["parent_id"] = parent_id
            parent_groups.setdefault(parent_id, []).append(table)
        else:
            table.pop("parent_id", None)

    for siblings in parent_groups.values():
        if len(siblings) <= 1:
            if siblings:
                siblings[0].pop("prev_id", None)
                siblings[0].pop("next_id", None)
            continue
        for idx, table in enumerate(siblings):
            if idx > 0:
                table["prev_id"] = siblings[idx - 1].get("id")
            else:
                table.pop("prev_id", None)
            if idx + 1 < len(siblings):
                table["next_id"] = siblings[idx + 1].get("id")
            else:
                table.pop("next_id", None)

    return manifest


def referring_images(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Collega le immagini al nodo TOC contenitore e ai fratelli adiacenti."""

    images = manifest.get("images") or []
    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []

    parent_groups: Dict[str, List[Dict[str, Any]]] = {}

    for image in images:
        context_path = image.get("context_path") or []
        parent_id = _find_toc_parent_id_from_context(toc_tree, context_path)
        if parent_id:
            image["parent_id"] = parent_id
            parent_groups.setdefault(parent_id, []).append(image)
        else:
            image.pop("parent_id", None)

    for siblings in parent_groups.values():
        if len(siblings) <= 1:
            if siblings:
                siblings[0].pop("prev_id", None)
                siblings[0].pop("next_id", None)
            continue
        for idx, image in enumerate(siblings):
            if idx > 0:
                image["prev_id"] = siblings[idx - 1].get("id")
            else:
                image.pop("prev_id", None)
            if idx + 1 < len(siblings):
                image["next_id"] = siblings[idx + 1].get("id")
            else:
                image.pop("next_id", None)

    return manifest


def populate_tables(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Aggiunge gli ID delle tabelle ai nodi TOC referenziati come parent."""

    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []
    tables = manifest.get("tables") or []

    node_by_id: Dict[str, Dict[str, Any]] = {}

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                node_by_id[node_id] = node
                node.setdefault("tables", [])
                node.setdefault("images", [])
            _collect(node.get("children") or [])

    _collect(toc_tree)

    for table in tables:
        parent_id = table.get("parent_id")
        if not parent_id:
            continue
        node = node_by_id.get(parent_id)
        if node is None:
            continue
        node.setdefault("tables", [])
        if table.get("id") and table["id"] not in node["tables"]:
            node["tables"].append(table["id"])

    return manifest


def populate_images(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Aggiunge gli ID delle immagini ai nodi TOC referenziati come parent."""

    toc_tree = (manifest.get("markdown") or {}).get("toc_tree") or []
    images = manifest.get("images") or []

    node_by_id: Dict[str, Dict[str, Any]] = {}

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                node_by_id[node_id] = node
                node.setdefault("tables", [])
                node.setdefault("images", [])
            _collect(node.get("children") or [])

    _collect(toc_tree)

    for image in images:
        parent_id = image.get("parent_id")
        if not parent_id:
            continue
        node = node_by_id.get(parent_id)
        if node is None:
            continue
        node.setdefault("images", [])
        if image.get("id") and image["id"] not in node["images"]:
            node["images"].append(image["id"])

    return manifest


def cleanup_manifest(manifest: Dict[str, Any], keep_pdf_pages_ref: bool) -> Dict[str, Any]:
    """Rimuove i campi `pdf_source_page` dal manifest salvo abilitazione esplicita."""

    if keep_pdf_pages_ref:
        return manifest

    markdown = manifest.get("markdown") or {}
    toc_nodes = markdown.get("toc_tree") or []

    def _strip_pdf_page(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node.pop("pdf_source_page", None)
            _strip_pdf_page(node.get("children") or [])

    _strip_pdf_page(toc_nodes)

    for table in manifest.get("tables") or []:
        table.pop("pdf_source_page", None)

    for image in manifest.get("images") or []:
        image.pop("pdf_source_page", None)

    return manifest


def _build_line_index(md_text: str) -> Tuple[List[str], List[int], List[int]]:
    """Costruisce gli indici di riga e byte sulla base del Markdown attuale."""

    lines = md_text.splitlines()
    line_starts: List[int] = []
    newline_lens: List[int] = []
    offset = 0
    trailing_newline = md_text.endswith("\n")
    for idx, line in enumerate(lines):
        line_starts.append(offset)
        has_newline = idx < len(lines) - 1 or trailing_newline
        newline_len = 1 if has_newline else 0
        newline_lens.append(newline_len)
        offset += len(line) + newline_len
    return lines, line_starts, newline_lens


def _line_char_range(
    line_starts: List[int], newline_lens: List[int], lines: List[str], start_idx: int, end_idx: int
) -> Tuple[int, int]:
    """Calcola l'intervallo di caratteri (offset byte invariato) per un blocco di righe."""

    start_char = line_starts[start_idx]
    end_offset = line_starts[end_idx] + len(lines[end_idx]) + newline_lens[end_idx]
    end_char = end_offset - 1 if end_offset > start_char else start_char
    return start_char, end_char


def set_toc_lines(manifest: Dict[str, Any], md_text: str) -> Dict[str, Any]:
    """Annota i nodi TOC assegnando intervalli limitati alla singola voce, senza includere i figli.

    Se le intestazioni reali (#, ##, ...) non sono presenti (es. range pagine limitato),
    ricorre ai link della TOC Markdown inserita ("- [Titolo](#ancora)") per ancorare i
    nodi e mantenere comunque gli intervalli di righe/byte."""

    lines, line_starts, newline_lens = _build_line_index(md_text)
    if not lines:
        return manifest

    toc_nodes = (manifest.get("markdown") or {}).get("toc_tree") or []
    if not toc_nodes:
        return manifest

    heading_re = re.compile(r"^#{1,6}\s+(.+?)\s*$")
    toc_link_re = re.compile(r"^\s*-\s*\[(.+?)\]\([^)]*\)\s*$")

    start_map: Dict[int, int] = {}
    end_map: Dict[int, int] = {}
    total_lines = len(lines)

    flattened: List[Dict[str, Any]] = []

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            flattened.append(node)
            _collect(node.get("children") or [])

    _collect(toc_nodes)
    if not flattened:
        return manifest

    positions: List[Tuple[Dict[str, Any], int]] = []

    def _find_line_for_title(target_norm: str, start_from: int) -> Optional[int]:
        """Cerca prima un'intestazione reale, poi un link TOC se assente."""

        for idx in range(start_from, len(lines)):
            match = heading_re.match(lines[idx])
            if not match:
                continue
            candidate = _normalize_title_for_toc(match.group(1))
            if candidate == target_norm:
                return idx

        for idx in range(start_from, len(lines)):
            match = toc_link_re.match(lines[idx])
            if not match:
                continue
            candidate = _normalize_title_for_toc(match.group(1))
            if candidate == target_norm:
                return idx

        return None

    search_start = 0
    for node in flattened:
        title = str(node.get("title") or "").strip()
        if not title:
            continue
        target_norm = _normalize_title_for_toc(title)
        found_idx = _find_line_for_title(target_norm, search_start)
        if found_idx is not None:
            positions.append((node, found_idx))
            search_start = found_idx + 1

    if not positions:
        return manifest

    # Ordina per posizione trovata nel Markdown così l'end di ciascuna voce termina
    # immediatamente prima della voce successiva (o a fine file per l'ultima).
    positions.sort(key=lambda item: item[1])

    for index, (node, start_idx) in enumerate(positions):
        start_map[id(node)] = start_idx
        next_start_idx = positions[index + 1][1] if index + 1 < len(positions) else total_lines
        end_idx = next_start_idx - 1 if next_start_idx > 0 else total_lines - 1
        if end_idx < start_idx:
            end_idx = start_idx
        end_map[id(node)] = end_idx

    for node, start_idx in positions:
        end_idx = end_map.get(id(node), start_idx)
        start_char, end_char = _line_char_range(line_starts, newline_lens, lines, start_idx, end_idx)
        node["start_line"] = start_idx + 1
        node["end_line"] = end_idx + 1
        node["start_char"] = start_char
        node["end_char"] = end_char

    return manifest


def set_tables_lines(manifest: Dict[str, Any], md_text: str) -> Dict[str, Any]:
    """Annota ogni tabella con il blocco di righe e byte che contiene i riferimenti Markdown/CSV."""

    lines, line_starts, newline_lens = _build_line_index(md_text)
    if not lines:
        return manifest

    tables = manifest.get("tables") or []
    for table in tables:
        files = table.get("files") or []
        hit_lines: List[int] = []
        for idx, line in enumerate(lines):
            for file_ref in files:
                if file_ref and file_ref in line:
                    hit_lines.append(idx)
        if not hit_lines:
            continue
        start_idx = min(hit_lines)
        end_idx = max(hit_lines)

        table_heading_re = re.compile(r"^#{3,6}\s+tabella", re.IGNORECASE)
        table_block_marker_re = re.compile(r"^###\s+tabelle", re.IGNORECASE)

        # Estende l'intervallo verso l'alto per includere il blocco tabellare immediatamente precedente
        # ai link [Markdown]/[CSV], fermandosi a separatori pagina o a contenuto non correlato.
        expanded_start = start_idx
        seen_block = False
        for i in range(start_idx - 1, -1, -1):
            stripped = lines[i].strip()

            if PAGE_START_MARKER_CAPTURE_RE.match(stripped) or PAGE_END_MARKER_CAPTURE_RE.match(stripped):
                break

            is_heading = bool(table_heading_re.match(stripped) or table_block_marker_re.match(stripped))
            is_table_row = "|" in stripped and not stripped.startswith("---")
            is_fallback_marker = stripped.lower().startswith("<!-- extracted_tables_fallback")

            if is_heading or is_table_row or is_fallback_marker:
                expanded_start = i
                seen_block = True
                continue

            if stripped == "":
                if seen_block:
                    expanded_start = i
                    continue
                break

            if seen_block:
                break
            break

        start_idx = expanded_start
        start_char, end_char = _line_char_range(line_starts, newline_lens, lines, start_idx, end_idx)
        table["start_line"] = start_idx + 1
        table["end_line"] = end_idx + 1
        table["start_char"] = start_char
        table["end_char"] = end_char

    return manifest


def set_images_lines(manifest: Dict[str, Any], md_text: str) -> Dict[str, Any]:
    """Annota ogni immagine con il blocco che include i blocchi equation/annotation e il link."""

    lines, line_starts, newline_lens = _build_line_index(md_text)
    if not lines:
        return manifest

    for image in manifest.get("images") or []:
        file_rel = image.get("file")
        if not file_rel:
            continue
        base = Path(str(file_rel)).name
        if not base:
            continue
        candidates: List[int] = []
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            start_eq = EQUATION_BLOCK_START_RE.match(stripped)
            if start_eq and start_eq.group(1).strip() == base:
                candidates.append(idx)
                continue
            end_eq = EQUATION_BLOCK_END_RE.match(stripped)
            if end_eq and end_eq.group(1).strip() == base:
                candidates.append(idx)
                continue
            start_ann = ANNOTATION_BLOCK_START_RE.match(stripped)
            if start_ann and start_ann.group(1).strip() == base:
                candidates.append(idx)
                continue
            end_ann = ANNOTATION_BLOCK_END_RE.match(stripped)
            if end_ann and end_ann.group(1).strip() == base:
                candidates.append(idx)
                continue
            for match in IMG_LINK_RE.finditer(raw):
                url = normalize_path_for_md(match.group(2).strip().strip('"').strip("'"))
                if url.split("/")[-1] == base:
                    candidates.append(idx)
                    break
        if not candidates:
            continue
        start_idx = min(candidates)
        end_idx = max(candidates)
        start_char, end_char = _line_char_range(line_starts, newline_lens, lines, start_idx, end_idx)
        image["start_line"] = start_idx + 1
        image["end_line"] = end_idx + 1
        image["start_char"] = start_char
        image["end_char"] = end_char

    return manifest


def _guess_mime_type(path: Path) -> str:
    """Indovina il MIME type di un file immagine."""

    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def _build_gemini_parts(prompt: str, mime_type: str, image_bytes: bytes) -> List[Dict[str, Any]]:
    """Costruisce il payload di richiesta Gemini usando inline_data per le immagini."""
    return [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
            ],
        }
    ]


def _init_gemini_model(api_key: str, model_name: str, module_path: str = "google.genai") -> Any:
    """Inizializza il modello Gemini usando esclusivamente il modulo python-genai (google.genai)."""

    try:
        genai = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - dipendente dall'ambiente
        raise RuntimeError(f"Unable to import module {module_path}: {exc}") from exc

    if hasattr(genai, "configure"):
        genai.configure(api_key=api_key)

    if hasattr(genai, "GenerativeModel"):
        return genai.GenerativeModel(model_name)

    if hasattr(genai, "Client"):
        client = genai.Client(api_key=api_key)

        class _ClientWrapper:
            def __init__(self, client_obj: Any, name: str) -> None:
                self._client = client_obj
                self._model = name

            def generate_content(self, parts: Any) -> Any:
                return self._client.models.generate_content(model=self._model, contents=parts)

        return _ClientWrapper(client, model_name)

    raise RuntimeError(f"Module {module_path} does not provide GenerativeModel or Client APIs")


def _run_pix2tex_test_mode(manifest: Dict[str, Any], md_text: str, config: PostProcessingConfig) -> Tuple[str, Dict[str, Any]]:
    """Simula Pix2Tex con output deterministico durante gli unit test."""
    formula_raw = _get_test_pix2tex_formula()
    formulas: Dict[str, str] = {}
    if not formula_raw:
        if config.debug:
            LOG.debug("Pix2Tex test mode skipped because canned formula is empty")
        return md_text, manifest

    threshold = config.equation_min_len
    length = len(formula_raw)
    if length < threshold:
        if config.verbose:
            LOG.info(
                "Pix2Tex test mode skipped because canned formula length %d < threshold %d",
                length,
                threshold,
            )
        return md_text, manifest

    is_valid = validate_latex_formula(formula_raw)
    if not is_valid:
        if config.verbose:
            LOG.info("Pix2Tex test mode skipped because canned formula failed validation")
        return md_text, manifest

    images = manifest.get("images") or []
    total_images = len(images)
    if config.verbose:
        LOG.info("Pix2Tex test mode active: using canned formula for %d images", total_images)
    for index, entry in enumerate(images, start=1):
        file_rel = entry.get("file")
        if not file_rel:
            continue
        base_name = Path(str(file_rel)).name
        formulas[base_name] = formula_raw
        entry["type"] = "equation"
        entry["equation"] = formula_raw
        if config.verbose:
            LOG.info("Pix2Tex images[%d/%d] %s validation result: PASSED (test mode)", index, total_images, file_rel)

    if not formulas:
        return md_text, manifest

    if config.debug:
        LOG.debug("Pix2Tex test mode formula applied: %s", formula_raw)

    md_text = embed_equations_in_markdown(md_text, formulas)
    return md_text, manifest


def run_pix2tex_phase(manifest: Dict[str, Any], md_text: str, out_dir: Path, config: PostProcessingConfig) -> Tuple[str, Dict[str, Any]]:
    """Esegue Pix2Tex sulle immagini del manifest aggiornando tipo e inserendo formule LaTeX."""
    if config.test_mode:
        return _run_pix2tex_test_mode(manifest, md_text, config)

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow not available: {exc}") from exc

    try:
        from pix2tex.cli import LatexOCR  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pix2tex not available: {exc}") from exc

    model = LatexOCR()
    formulas: Dict[str, str] = {}
    images = manifest.get("images") or []
    total_images = len(images)
    for index, entry in enumerate(images, start=1):
        file_rel = entry.get("file")
        if not file_rel:
            continue
        path = out_dir / normalize_path_for_md(str(file_rel))
        pos_ref = f"images[{index}/{total_images}]" if total_images else "images[?]"
        if config.verbose:
            _log_verbose_progress("Pix2Tex", index, total_images, detail=str(file_rel))
        try:
            img = Image.open(path)
        except Exception as exc:
            LOG.debug("Pix2Tex skipped %s (cannot open image): %s", path, exc)
            continue

        try:
            formula_raw = str(model(img)).strip()
        except Exception as exc:
            LOG.debug("Pix2Tex failed on %s: %s", path, exc)
            try:
                img.close()
            except Exception:
                pass
            continue
        finally:
            try:
                img.close()
            except Exception:
                pass

        if config.debug:
            LOG.debug("Pix2Tex %s raw output: %s", pos_ref, formula_raw)

        length = len(formula_raw)
        threshold = config.equation_min_len
        if length >= threshold:
            is_valid = validate_latex_formula(formula_raw)
            if config.verbose:
                status = "PASSED" if is_valid else "FAILED"
                LOG.info("Pix2Tex %s %s validation result: %s", pos_ref, file_rel, status)
            if is_valid:
                base_name = Path(file_rel).name
                formulas[base_name] = formula_raw
                entry["type"] = "equation"
                entry["equation"] = formula_raw
        else:
            if config.verbose:
                LOG.info("Pix2Tex %s %s validation result: SKIPPED (len=%d < threshold=%d)", pos_ref, file_rel, length, threshold)

    if formulas:
        md_text = embed_equations_in_markdown(md_text, formulas)
    return md_text, manifest


def run_annotation_phase(
    manifest: Dict[str, Any], md_text: str, out_dir: Path, config: PostProcessingConfig, pix2tex_executed: bool
) -> Tuple[str, Dict[str, Any]]:
    """Annota immagini e/o equazioni con Gemini aggiornando Markdown e manifest."""
    if not (config.annotate_images or config.annotate_equations):
        return md_text, manifest

    if not config.gemini_api_key:
        raise RuntimeError("Gemini API key missing for annotation")

    model = None
    if not config.test_mode:
        module_path = config.gemini_module or "google.genai"
        try:
            model = _init_gemini_model(
                config.gemini_api_key,
                config.gemini_model or GEMINI_DEFAULT_MODEL,
                module_path=module_path,
            )
        except Exception as exc:  # pragma: no cover - dipende da dipendenze esterne
            raise RuntimeError(f"Unable to initialize Gemini model: {exc}") from exc
    elif config.verbose:
        LOG.info("Gemini annotation test mode active: using canned responses")

    annotations: Dict[str, str] = {}
    images = manifest.get("images") or []
    total_images = len(images)
    for index, entry in enumerate(images, start=1):
        file_rel = entry.get("file")
        entry_type = str(entry.get("type") or "").lower()
        if not file_rel:
            continue
        is_equation = entry_type == "equation"
        if is_equation and not config.annotate_equations:
            continue
        if not is_equation and not config.annotate_images:
            continue
        if config.verbose:
            _log_verbose_progress("Annotation", index, total_images, detail=str(file_rel))

        path = out_dir / normalize_path_for_md(str(file_rel))
        if config.verbose:
            LOG.info("Annotating %s: %s", "equation" if is_equation else "image", file_rel)
        try:
            if config.test_mode:
                annotation_final = _get_test_annotation_text(is_equation, entry.get("equation"))
            else:
                prompt = select_annotation_prompt(is_equation, pix2tex_executed, config)
                try:
                    image_bytes = path.read_bytes()
                except Exception as exc:
                    raise RuntimeError(f"Unable to read image {path}: {exc}") from exc

                mime_type = _guess_mime_type(path)
                parts = _build_gemini_parts(prompt, mime_type, image_bytes)
                try:
                    model_response = model.generate_content(parts)
                except Exception as exc:
                    if config.debug:
                        LOG.debug("Gemini request failed for %s", path.name, exc_info=exc)
                    raise RuntimeError(f"Gemini request failed for {path.name}: {exc}") from exc

                if config.debug:
                    LOG.debug("Gemini raw response for %s: %r", path.name, model_response)

                annotation_text = getattr(model_response, "text", "") if model_response is not None else ""
                if not annotation_text and hasattr(model_response, "candidates"):
                    candidates = getattr(model_response, "candidates", []) or []
                    if candidates:
                        annotation_text = getattr(candidates[0], "text", "") or getattr(candidates[0], "content", "") or ""

                if not annotation_text:
                    raise RuntimeError(f"Empty annotation from Gemini for {path.name}")

                annotation_final = str(annotation_text).strip()

            annotations[Path(file_rel).name] = annotation_final
            entry["annotation"] = annotation_final
            if config.verbose:
                LOG.info("Annotation completed: %s", file_rel)
            if config.debug:
                LOG.debug("Annotation content for %s: %s", file_rel, annotation_final)
        except Exception as exc:
            LOG.error("Annotation failed for %s: %s", file_rel, exc)
            if config.debug:
                LOG.debug("Annotation failure details for %s", file_rel, exc_info=exc)
            continue

    if annotations:
        md_text = embed_annotations_in_markdown(md_text, annotations)

    return md_text, manifest


def processing_prepare_output_dirs(out_dir: Path, debug_enabled: bool) -> Tuple[Path, Path, Optional[Path]]:
    """Prepara le cartelle di output per immagini, tabelle e debug."""

    start_phase("Preparing output directories")
    images_dir = out_dir / "images"
    tables_dir = out_dir / "tables"
    debug_dir = out_dir / "debug" if debug_enabled else None
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    end_phase()
    return images_dir, tables_dir, debug_dir


def processing_setup_layout(verbose: bool) -> bool:
    """Inizializza i moduli di layout opzionali restituendo lo stato abilitato."""

    layout_enabled = False
    if verbose:
        LOG.info("Loading PDF and preparing layout modules...")
    try:
        import pymupdf.layout  # noqa: F401

        layout_enabled = True
    except Exception as exc:
        LOG.warning("pymupdf-layout not active (legacy fallback). Detail: %s", exc)
    return layout_enabled


def processing_import_pymupdf4llm() -> Any:
    """Importa pymupdf4llm fallendo in modo esplicito in caso di errore."""

    try:
        import pymupdf4llm

        return pymupdf4llm
    except Exception as exc:
        LOG.error("Unable to import pymupdf4llm: %s", exc)
        raise


def processing_open_documents(pdf_path: Path) -> Tuple[pymupdf.Document, pymupdf.Document]:
    """Apre due handle PyMuPDF separati per immagini e testo."""

    doc_images = pymupdf.open(str(pdf_path))
    doc_text = pymupdf.open(str(pdf_path))
    return doc_images, doc_text


def processing_select_pages_and_toc(
    *,
    doc_images: pymupdf.Document,
    doc_text: pymupdf.Document,
    pdf_path: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[List[Any]], TocNode, int, int, int, int, int, int]:
    """Applica la selezione pagine, raccoglie metadata e TOC, restituendo i parametri di range."""

    def _close_documents() -> None:
        try:
            doc_images.close()
        except Exception:
            pass
        try:
            doc_text.close()
        except Exception:
            pass

    metadata: Dict[str, Any] = {}
    toc_raw: List[List[Any]] = []
    try:
        metadata = doc_text.metadata or {}
    except Exception as exc:
        LOG.debug("doc.metadata failed: %s", exc)
    try:
        toc_raw = doc_text.get_toc() or []
    except Exception as exc:
        LOG.debug("doc.get_toc failed: %s", exc)
    original_page_count = getattr(doc_text, "page_count", 0)
    start_page = int(args.start_page or 1)
    page_offset = max(start_page - 1, 0)
    requested_limit: Optional[int] = None
    page_range_end: Optional[int] = None

    if original_page_count > 0:
        if start_page > original_page_count:
            _close_documents()
            raise RuntimeError(
                f"Start page {start_page} exceeds document page count {original_page_count}"
            )
        available_from_start = original_page_count - page_offset
        if available_from_start <= 0:
            _close_documents()
            raise RuntimeError(
                f"Start page {start_page} exceeds document page count {original_page_count}"
            )
        if args.n_pages is not None:
            requested_limit = int(args.n_pages)
            if requested_limit > available_from_start:
                _close_documents()
                raise RuntimeError(
                    f"Requested page range {start_page}-{start_page + requested_limit - 1} exceeds document page count {original_page_count}"
                )
            if args.verbose and page_offset == 0 and requested_limit == original_page_count:
                LOG.info(
                    "Requested page limit %d covers all %d page(s); processing entire document",
                    args.n_pages,
                    original_page_count,
                )
        else:
            requested_limit = available_from_start

        page_range_end = start_page + requested_limit - 1
        if page_offset > 0 or requested_limit < original_page_count:
            page_indices = list(range(page_offset, page_offset + requested_limit))
            doc_images.select(page_indices)
            doc_text.select(page_indices)
            if args.verbose:
                LOG.info(
                    "Limiting processing to PDF pages %d-%d (%d page(s) selected out of %d)",
                    start_page,
                    page_range_end,
                    requested_limit,
                    original_page_count,
                )
    else:
        requested_limit = getattr(doc_text, "page_count", 0) or 0
        page_range_end = start_page + max(requested_limit - 1, 0)

    apply_vertical_crop_margins(
        doc_text, header_mm=args.header, footer_mm=args.footer, debug=bool(args.debug), verbose=bool(args.verbose)
    )

    processed_page_count = requested_limit or getattr(doc_text, "page_count", original_page_count)
    toc = toc_raw
    if toc_raw and page_range_end is not None:
        selection_start = start_page
        selection_end = page_range_end
        if selection_start > 1 or (original_page_count and selection_end < original_page_count):
            toc_limited = []
            for entry in toc_raw:
                if len(entry) < 3:
                    continue
                try:
                    page_no = int(entry[2])
                except Exception:
                    continue
                if selection_start <= page_no <= selection_end:
                    toc_limited.append(entry)
            if toc_limited:
                toc = toc_limited

    toc_root = build_toc_tree(toc)

    if not toc_raw:
        _close_documents()
        raise RuntimeError(f"TOC not found in PDF: {pdf_path}")

    return (
        metadata,
        toc,
        toc_root,
        processed_page_count,
        page_offset,
        page_range_end or processed_page_count,
        requested_limit or processed_page_count,
        original_page_count,
        start_page,
    )


def processing_extract_chunks(
    *,
    doc_images: pymupdf.Document,
    doc_text: pymupdf.Document,
    images_dir: Path,
    args: argparse.Namespace,
    layout_enabled: bool,
    pymupdf4llm: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Estrae i chunk Markdown con e senza immagini inline."""

    md_kwargs_common = dict(
        page_chunks=True,
        force_text=True,
        show_progress=bool(args.verbose),
        page_separators=bool(args.debug),
        header=True,
        footer=True,
        use_ocr=bool(layout_enabled),
        table_strategy="lines_strict",
    )

    md_kwargs_images = dict(
        md_kwargs_common,
        write_images=True,
        embed_images=False,
        image_path=str(images_dir),
        image_format="png",
        dpi=150,
    )
    md_kwargs_text = dict(
        md_kwargs_common,
        write_images=False,
        embed_images=False,
        image_path=str(images_dir),
        image_format="png",
        dpi=150,
    )

    start_phase("Extracting Markdown and assets")
    if args.verbose:
        LOG.info("Extracting Markdown with inline image generation...")
    chunks_images = pymupdf4llm.to_markdown(doc_images, **md_kwargs_images)
    if args.verbose:
        LOG.info("Extracting Markdown text without embedding images...")
    chunks_text = pymupdf4llm.to_markdown(doc_text, **md_kwargs_text)
    end_phase()

    if isinstance(chunks_images, str):
        chunks_images = [{"text": chunks_images}]
    if isinstance(chunks_text, str):
        chunks_text = [{"text": chunks_text}]

    return chunks_images, chunks_text


def run_post_processing_pipeline(
    *,
    out_dir: Path,
    pdf_path: Path,
    md_path: Path,
    manifest_path: Path,
    config: PostProcessingConfig,
) -> Tuple[str, Dict[str, Any], bool]:
    """Esegue la pipeline di post-processing ripartendo dal backup .processing.md."""
    _configure_pdf2tree_logger(_resolve_log_level(config.verbose, config.debug))

    def _save_markdown(md_content: str) -> str:
        """Salva il Markdown corrente su disco garantendo il newline finale."""

        final_text = md_content if md_content.endswith("\n") else f"{md_content}\n"
        safe_write_text(md_path, final_text)
        return final_text

    backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
    if not backup_path.exists():
        raise RuntimeError(f"Backup Markdown (.processing.md) not found: {backup_path}")

    try:
        shutil.copyfile(backup_path, md_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to restore Markdown from backup {backup_path}: {exc}") from exc

    normalized_md_text, toc_headings, toc_raw, toc_path, pdf_headings = normalize_markdown_file(
        pdf_path, md_path, out_dir, add_toc=not config.disable_toc
    )
    normalized_md_text = _save_markdown(normalized_md_text)

    toc_mismatch = False
    if config.skip_toc_validation:
        LOG.info(
            "TOC validation skipped in test mode for limited page range (set PDF2TREE_FORCE_TOC_VALIDATION=1 to enforce)."
        )
    else:
        toc_result = validate_markdown_toc_against_pdf(pdf_path, toc_headings)
        log_toc_validation_result(toc_result, verbose=config.verbose, debug=config.debug)
        toc_mismatch = not toc_result.ok

    images_dir = out_dir / "images"
    image_source_map: Dict[str, str] = {}
    if images_dir.exists():
        for path in sorted(images_dir.glob("*.png")):
            name = path.name
            if "-form-" in name:
                image_source_map[name] = "form-xobject"
            elif "-vector-" in name:
                image_source_map[name] = "vector-image"

    manifest_built, _, _ = build_manifest_from_outputs(
        pdf_path=pdf_path,
        md_path=md_path,
        out_dir=out_dir,
        images_dir=images_dir,
        tables_dir=out_dir / "tables",
        toc_path=toc_path,
        toc_root=build_toc_tree(toc_raw),
        image_source=image_source_map,
    )

    safe_write_text(manifest_path, json.dumps(manifest_built, ensure_ascii=False, indent=2) + "\n")

    updated_md = normalized_md_text
    updated_manifest = manifest_built
    pix2tex_executed = False

    if not config.disable_remove_small_images:
        updated_md, updated_manifest = remove_small_images_phase(
            manifest=updated_manifest, md_text=updated_md, out_dir=out_dir, config=config
        )
        updated_md = _save_markdown(updated_md)

    if config.enable_pix2tex and not config.disable_pix2tex:
        updated_md, updated_manifest = run_pix2tex_phase(updated_manifest, updated_md, out_dir, config)
        pix2tex_executed = True
        updated_md = _save_markdown(updated_md)
    elif config.verbose:
        if config.disable_pix2tex:
            LOG.info("Pix2Tex disabled via --disable-pic2tex flag.")
        else:
            LOG.info("Pix2Tex disabled by default (use --enable-pic2tex to activate)")

    if config.annotate_images or config.annotate_equations:
        updated_md, updated_manifest = run_annotation_phase(
            updated_manifest, updated_md, out_dir, config, pix2tex_executed=pix2tex_executed
        )
        updated_md = _save_markdown(updated_md)

    if not config.disable_cleanup:
        updated_md = cleanup_markdown(updated_md, pdf_headings)
        updated_md = _save_markdown(updated_md)
    elif config.verbose:
        LOG.info("Cleanup disabled via --disable-cleanup flag; Markdown markers preserved.")

    updated_manifest = generate_item_ids(updated_manifest)
    updated_manifest = referring_toc(updated_manifest)
    updated_manifest = referring_tables(updated_manifest)
    updated_manifest = referring_images(updated_manifest)
    updated_manifest = populate_tables(updated_manifest)
    updated_manifest = populate_images(updated_manifest)

    updated_manifest = set_toc_lines(updated_manifest, updated_md)
    updated_manifest = set_tables_lines(updated_manifest, updated_md)
    updated_manifest = set_images_lines(updated_manifest, updated_md)

    updated_manifest = cleanup_manifest(updated_manifest, config.enable_pdf_pages_ref)

    safe_write_text(manifest_path, json.dumps(updated_manifest, ensure_ascii=False, indent=2) + "\n")

    updated_md = _save_markdown(updated_md)

    return updated_md, updated_manifest, toc_mismatch


def find_existing_markdown(out_dir: Path, pdf_stem: str) -> Optional[Path]:
    """Trova un file Markdown esistente nella cartella di output per il PDF indicato."""

    candidate = out_dir / f"{slugify_filename(pdf_stem)}.md"
    if candidate.exists():
        return candidate
    md_files = sorted(out_dir.glob("*.md"))
    return md_files[0] if md_files else None


def run_processing_pipeline(
    *,
    args: argparse.Namespace,
    pdf_path: Path,
    out_dir: Path,
    post_processing_cfg: PostProcessingConfig,
    form_xobject_enabled: bool,
    vector_images_enabled: bool,
) -> Tuple[str, Dict[str, Any], Path, Path]:
    """Esegue la pipeline PDF→Markdown scrivendo gli artefatti principali."""
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)

    images_dir, tables_dir, debug_dir = processing_prepare_output_dirs(out_dir, bool(args.debug))

    start_phase("Loading PDF and metadata")
    layout_enabled = processing_setup_layout(bool(args.verbose))
    pymupdf4llm = processing_import_pymupdf4llm()

    base = slugify_filename(pdf_path.stem)
    md_out = out_dir / f"{base}.md"
    manifest_out = out_dir / f"{pdf_path.stem}.json"

    manifest_tables: List[Dict[str, Any]] = []
    manifest_images: List[Dict[str, Any]] = []
    image_page_map: Dict[str, int] = {}
    image_source: Dict[str, str] = {}

    LOG.info("Input:  %s", pdf_path)
    LOG.info("Output: %s", md_out)
    LOG.info("Images: %s", images_dir)
    LOG.info("Tables: %s", tables_dir)
    LOG.info("Layout mode: %s", "ON" if layout_enabled else "OFF (legacy)")

    doc_images, doc_text = processing_open_documents(pdf_path)

    (
        metadata,
        toc,
        toc_root,
        processed_page_count,
        page_offset,
        page_range_end,
        requested_limit,
        original_page_count,
        start_page,
    ) = processing_select_pages_and_toc(
        doc_images=doc_images,
        doc_text=doc_text,
        pdf_path=pdf_path,
        args=args,
    )

    ocr_ok = True
    if args.verbose and layout_enabled:
        LOG.info("OpenCV importable in the environment: YES")
    end_phase()

    chunks_images, chunks_text = processing_extract_chunks(
        doc_images=doc_images,
        doc_text=doc_text,
        images_dir=images_dir,
        args=args,
        layout_enabled=layout_enabled and ocr_ok,
        pymupdf4llm=pymupdf4llm,
    )

    page_count = processed_page_count
    final_md: Optional[str] = None
    debug_enabled = bool(args.debug)
    debug_report: Optional[Dict[str, Any]] = {"pages": []} if debug_enabled else None

    page_count = processed_page_count if processed_page_count > 0 else len(chunks_text)

    images_by_page: Dict[int, List[str]] = {}
    for i, page_dict in enumerate(chunks_images or []):
        page_no = i + 1
        actual_page_no = page_offset + page_no
        text_img = page_dict.get("text") or ""
        basenames_ordered = extract_image_basenames_in_order(text_img)
        rels = [f"images/{name}" for name in basenames_ordered]
        if rels:
            images_by_page[actual_page_no] = rels
            for name in basenames_ordered:
                image_page_map[name] = actual_page_no
                image_source.setdefault(name, "pymupdf")

    parts: List[str] = []
    parts.append(yaml_front_matter(metadata, pdf_path, page_count))
    if toc:
        parts.append(build_toc_markdown(toc))

    start_phase("Processing pages")
    if args.verbose:
        if vector_images_enabled:
            LOG.info("Vector extraction enabled; vector regions will be rendered when detected.")
        else:
            LOG.info("vector extraction disabled (use --enable-vector-images to activate)")
        if form_xobject_enabled:
            LOG.info("Form XObject extraction enabled; placements will be rasterized when present.")
        else:
            LOG.info("Form XObject extraction disabled (use --enable-form-xobject to activate)")
    for i, page_dict in enumerate(chunks_text):
        page_no = i + 1
        actual_page_no = page_offset + page_no
        page = doc_text[i]
        if args.verbose:
            detail = f"Processing PDF page {actual_page_no}" if actual_page_no != page_no else "Processing page"
            _log_verbose_progress("Pages", page_no, page_count, detail=detail)

        page_label = f"Page {actual_page_no}"

        raw_text = page_dict.get("text") or ""
        text = PAGE_END_MARKER_RE.sub("", raw_text).strip()

        if text:
            text = rewrite_image_links_to_images_subdir(text, subdir="images")

        entry: Optional[Dict[str, Any]] = None
        if debug_enabled:
            entry = {"page": actual_page_no, "form_placements": [], "vector_regions": [], "insert_method": "none"}

        rel_image_paths = images_by_page.get(actual_page_no, [])
        if args.verbose and rel_image_paths:
            LOG.info("%s: inserting %d raster images from extraction", page_label, len(rel_image_paths))
        text, method = inject_images_into_page_markdown(text, rel_image_paths)
        if entry is not None:
            entry["insert_method"] = method

        if args.verbose and vector_images_enabled:
            LOG.info("%s: vector extraction enabled", page_label)
        if vector_images_enabled:
            regions = find_vector_regions(page, debug=args.debug)
            if regions:
                if args.verbose:
                    LOG.info("%s: rendering %d vector region(s)", page_label, len(regions))
                saved_vectors = render_vector_regions(
                    doc=doc_text,
                    doc_page_index=i,
                    regions=regions,
                    images_dir=images_dir,
                    pdf_filename_prefix=pdf_path.name,
                    pdf_page_no=actual_page_no,
                    dpi=200,
                )
                rels_vec = [f"images/{name}" for name in saved_vectors]
                if rels_vec:
                    if args.verbose:
                        LOG.info("%s: injected %d vector image(s) into Markdown", page_label, len(rels_vec))
                    text, method = inject_images_into_page_markdown(text, rels_vec)
                    if entry is not None:
                        if entry["insert_method"] == "none":
                            entry["insert_method"] = method
                        entry["vector_regions"] = [
                            [float(r.x0), float(r.y0), float(r.x1), float(r.y1)] for r in regions
                        ]
                    for name in saved_vectors:
                        image_page_map[name] = actual_page_no
                        image_source[name] = "vector-image"
                text = rewrite_image_links_to_images_subdir(text, subdir="images")
            elif args.verbose:
                LOG.info("%s: no vector regions detected", page_label)
        if form_xobject_enabled:
            page_uncropped = doc_images[i]
            placements = find_form_placements_on_page(doc_images, page_uncropped, actual_page_no, debug=args.debug)
            if placements:
                if args.verbose:
                    LOG.info("%s: rendering %d Form XObject image(s)", page_label, len(placements))
                saved = render_and_save_form_images(
                    page=page_uncropped,
                    placements=placements,
                    images_dir=images_dir,
                    pdf_filename_prefix=pdf_path.name,
                    page_no=actual_page_no,
                    dpi=150,
                )
                rels = [f"images/{fname}" for _, fname in saved]

                text, method = inject_images_into_page_markdown(text, rels)
                if entry is not None:
                    entry["insert_method"] = entry["insert_method"] if entry["insert_method"] != "none" else method
                    entry["form_placements"] = [
                        {"name": placement.name, "xref": placement.xref, "saved_as": f"images/{fname}"} for placement, fname in saved
                    ]

                text = rewrite_image_links_to_images_subdir(text, subdir="images")
                for _, fname in saved:
                    image_page_map[fname] = actual_page_no
                    image_source[fname] = "form-xobject"
            elif args.verbose:
                LOG.info("%s: no Form XObject placements found", page_label)
        elif entry is not None and entry["insert_method"] == "none":
            entry["insert_method"] = "disabled"

        if entry is not None and debug_report is not None:
            debug_report["pages"].append(entry)

        page_image_basenames = extract_image_basenames_from_markdown(text)
        for base_name in page_image_basenames:
            image_page_map.setdefault(base_name, actual_page_no)

        fallback_tables = extract_tables_fallback(doc_text, i)
        table_reference_blocks: List[str] = []
        if fallback_tables:
            exported = export_tables_files(tables_dir, actual_page_no, fallback_tables)
            if args.verbose:
                LOG.info("%s: exported %d fallback table(s)", page_label, len(exported))
            context_titles = find_context_for_page(toc_root, actual_page_no)
            context_str, context_path = build_context_metadata(context_titles)
            title = context_titles[-1] if context_titles else ""
            for files in exported:
                manifest_tables.append(
                    {
                        "pdf_source_page": actual_page_no,
                        "title": title,
                        "context": context_str,
                        "context_path": context_path,
                        "files": [relative_to_output(p, out_dir) for p in files],
                    }
                )
            table_reference_blocks = format_table_references(exported, out_dir)

        if fallback_tables and looks_like_markdown_table(text):
            if table_reference_blocks:
                combined_refs = "\n".join(block.strip() for block in table_reference_blocks if block).strip()
                if combined_refs:
                    lines = text.splitlines()
                    last_table_line = -1
                    for idx, line in enumerate(lines):
                        if "|" in line:
                            last_table_line = idx
                    if last_table_line >= 0:
                        insert_at = last_table_line + 1
                        lines.insert(insert_at, "")
                        lines.insert(insert_at + 1, combined_refs)
                        text = "\n".join(lines)
                    else:
                        text = text.rstrip() + "\n\n" + combined_refs

        parts.append(f"\n--- start of page.page_number={actual_page_no} ---\n")
        parts.append(text if text else "_[Pagina senza testo estratto]_")

        if fallback_tables and not looks_like_markdown_table(text):
            parts.append("\n\n<!-- extracted_tables_fallback -->\n")
            parts.append(f"### Tabelle (fallback) – pagina {actual_page_no}\n")
            for t_idx, (table_md, table_rows) in enumerate(fallback_tables, start=1):
                if table_md.strip():
                    parts.append(f"\n#### Tabella {t_idx}\n")
                    parts.append(table_md.strip())
                elif table_rows:
                    parts.append(f"\n#### Tabella {t_idx}\n")
                    header = table_rows[0]
                    body = table_rows[1:]

                    def row_md(row: List[str]) -> str:
                        """Converte una riga di tabella in formato Markdown."""

                        return "| " + " | ".join(row) + " |"

                    parts.append(row_md(header))
                    parts.append("| " + " | ".join(["---"] * len(header)) + " |")
                    for row in body:
                        parts.append(row_md(row))

                if t_idx - 1 < len(table_reference_blocks):
                    ref_block = table_reference_blocks[t_idx - 1]
                    if ref_block:
                        parts.append("\n" + ref_block)
        elif args.verbose:
            LOG.info("%s: no fallback tables detected", page_label)

        parts.append(f"\n\n--- end of page.page_number={actual_page_no} ---\n")

    final_md = "\n".join(parts).strip() + "\n"
    end_phase()

    if final_md is None:
        try:
            doc_images.close()
        except Exception:
            pass
        try:
            doc_text.close()
        except Exception:
            pass
        raise RuntimeError("Unable to generate the final Markdown content.")

    referenced = extract_image_basenames_from_markdown(final_md)

    search_dirs = iter_search_dirs(pdf_path, out_dir)
    probable = collect_probable_generated_images(pdf_path, fmt="png", search_dirs=search_dirs)

    all_to_move = referenced | probable
    for name in all_to_move:
        image_source.setdefault(name, "pymupdf")
    if args.verbose:
        LOG.info(
            "Ensuring referenced/generated images are placed into %s (found: %d, probable: %d)",
            images_dir,
            len(referenced),
            len(probable),
        )
    moved = move_files_by_name(all_to_move, search_dirs=search_dirs, dest_dir=images_dir, verbose=bool(args.verbose))

    if args.verbose:
        LOG.info(
            "Referenced in Markdown: %d | probable generated: %d | moved to images/: %d",
            len(referenced),
            len(probable),
            moved,
        )

    start_phase("Writing artifacts")
    safe_write_text(md_out, final_md if final_md.endswith("\n") else f"{final_md}\n")

    if args.debug and debug_report:
        if debug_dir is not None:
            safe_write_text(debug_dir / "form_xobjects.json", json.dumps(debug_report, ensure_ascii=False, indent=2))
            safe_write_text(debug_dir / f"{base}.chunks.json", json.dumps(chunks_text, ensure_ascii=False, indent=2))

    try:
        doc_images.close()
    except Exception:
        pass
    try:
        doc_text.close()
    except Exception:
        pass

    LOG.info("Created: %s", md_out)
    end_phase()

    backup_path = md_out.with_suffix(md_out.suffix + ".processing.md")
    try:
        shutil.copyfile(md_out, backup_path)
        if args.verbose:
            LOG.info("Backup created: %s", backup_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to create backup Markdown {backup_path}: {exc}") from exc

    return final_md, md_out


def main() -> int:
    """Entry point CLI che orchestra parsing argomenti, processing e post-processing."""

    ap = argparse.ArgumentParser(description="PDF -> Markdown with images/tables and Form XObject fallback.")
    ap.add_argument("--from-file", help="Source PDF path")
    ap.add_argument("--to-dir", help="Output directory")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress logs")
    ap.add_argument("--debug", action="store_true", help="Debug logs + extra artifacts")
    ap.add_argument(
        "--version",
        "--ver",
        action="store_true",
        help="Print the program version and exit",
    )
    ap.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade the installed pdf2tree package and exit",
    )
    ap.add_argument("--header", type=float, default=0.0, help="Header margin in mm to ignore (default: 0)")
    ap.add_argument("--footer", type=float, default=0.0, help="Footer margin in mm to ignore (default: 0)")
    ap.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Page number to start processing from (default: 1)",
    )
    ap.add_argument(
        "--n-pages",
        type=int,
        help="Maximum number of pages to process starting from the selected start page",
    )
    ap.add_argument(
        "--enable-form-xobject",
        action="store_true",
        help="Enable detection, rasterization and insertion of Form XObject as images (disabled by default)",
    )
    ap.add_argument(
        "--enable-vector-images",
        action="store_true",
        help="Enable vector diagram extraction (disabled by default)",
    )
    ap.add_argument(
        "--post-processing",
        action="store_true",
        help="Run post-processing pipeline after conversion (e.g., Pix2Tex equations)",
    )
    ap.add_argument(
        "--post-processing-only",
        action="store_true",
        help="Skip conversion and run post-processing on an existing output directory",
    )
    ap.add_argument(
        "--enable-pic2tex",
        action="store_true",
        help="Enable Pix2Tex phase within post-processing (disabled by default)",
    )
    ap.add_argument(
        "--disable-pic2tex",
        action="store_true",
        help="Disable Pix2Tex phase within post-processing even when --enable-pic2tex is provided",
    )
    ap.add_argument(
        "--disable-remove-small-images",
        action="store_true",
        help="Disable the remove-small-images post-processing phase",
    )
    ap.add_argument(
        "--enable-pdf-pages-ref",
        action="store_true",
        help="Keep pdf_source_page fields in the final manifest instead of removing them during cleanup",
    )
    ap.add_argument(
        "--disable-cleanup",
        action="store_true",
        help="Disable the cleanup step that removes page markers before manifest enrichment",
    )
    ap.add_argument(
        "--disable-toc",
        action="store_true",
        help="Disable insertion of the Markdown TOC rebuilt from the PDF TOC during post-processing",
    )
    ap.add_argument(
        "--equation-min-len",
        type=int,
        default=5,
        help="Minimum length of Pix2Tex output to classify an image as equation (default: 5)",
    )
    ap.add_argument(
        "--min-size-x",
        type=int,
        default=100,
        help="Minimum width in pixels for remove-small-images (default: 100)",
    )
    ap.add_argument(
        "--min-size-y",
        type=int,
        default=100,
        help="Minimum height in pixels for remove-small-images (default: 100)",
    )
    ap.add_argument(
        "--disable-annotate-images",
        action="store_true",
        help="Disable Gemini-based annotation for images during post-processing (enabled by default)",
    )
    ap.add_argument(
        "--enable-annotate-equations",
        action="store_true",
        help="Enable Gemini-based annotation for equations during post-processing",
    )
    ap.add_argument(
        "--gemini-api-key",
        help="API key for Gemini used by annotation phase (fallback: GEMINI_API_KEY env var)",
    )
    ap.add_argument(
        "--gemini-model",
        default=GEMINI_DEFAULT_MODEL,
        help=f"Gemini model name for annotation (default: {GEMINI_DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--prompts",
        help="Path to a JSON file containing prompt_equation, prompt_non_equation, prompt_uncertain",
    )
    ap.add_argument(
        "--write-prompts",
        help="Write the default prompts JSON to the given path and exit",
    )
    args = ap.parse_args()

    if args.version:
        print(program_version())
        return 0

    if args.upgrade:
        return run_self_upgrade(package_name="pdf2tree")

    setup_logging(args.verbose, args.debug)

    if args.write_prompts:
        target = Path(args.write_prompts).expanduser().resolve()
        try:
            _write_prompts_file(target)
        except Exception as exc:
            LOG.error("Unable to write prompts file %s: %s", target, exc)
            return EXIT_INVALID_ARGS
        LOG.info("Default prompts written to %s", target)
        return 0

    if not args.from_file or not args.to_dir:
        ap.print_help()
        LOG.error("Options --from-file and --to-dir are required unless --write-prompts or --version/--ver is used")
        return EXIT_INVALID_ARGS

    if args.post_processing and args.post_processing_only:
        LOG.error("Options --post-processing and --post-processing-only are mutually exclusive")
        return EXIT_INVALID_ARGS
    if args.equation_min_len is None or args.equation_min_len <= 0:
        LOG.error("Invalid value for --equation-min-len: must be > 0")
        return EXIT_INVALID_ARGS
    if args.min_size_x is None or args.min_size_x <= 0:
        LOG.error("Invalid value for --min-size-x: must be > 0")
        return EXIT_INVALID_ARGS
    if args.min_size_y is None or args.min_size_y <= 0:
        LOG.error("Invalid value for --min-size-y: must be > 0")
        return EXIT_INVALID_ARGS
    if args.header is not None and args.header < 0:
        LOG.error("Invalid value for --header: must be >= 0")
        return EXIT_INVALID_ARGS
    if args.footer is not None and args.footer < 0:
        LOG.error("Invalid value for --footer: must be >= 0")
        return EXIT_INVALID_ARGS
    if args.start_page is None or args.start_page <= 0:
        LOG.error("Invalid value for --start-page: must be > 0")
        return EXIT_INVALID_ARGS
    if args.n_pages is not None and args.n_pages <= 0:
        LOG.error("Invalid value for --n-pages: must be > 0")
        return EXIT_INVALID_ARGS

    annotate_images_enabled = not args.disable_annotate_images
    annotate_equations_enabled = bool(args.enable_annotate_equations)
    post_processing_active = bool(args.post_processing or args.post_processing_only)
    effective_annotate_images = annotate_images_enabled if post_processing_active else False
    effective_annotate_equations = annotate_equations_enabled if post_processing_active else False
    form_xobject_enabled = bool(args.enable_form_xobject)
    vector_images_enabled = bool(args.enable_vector_images)

    prompts_cfg = dict(DEFAULT_PROMPTS)
    if args.prompts:
        prompts_path = Path(args.prompts).expanduser().resolve()
        if not prompts_path.exists() or not prompts_path.is_file():
            LOG.error("Prompts file not found: %s", prompts_path)
            return EXIT_INVALID_ARGS
        try:
            prompts_cfg = load_prompts_file(prompts_path)
        except ValueError as exc:
            LOG.error("%s", exc)
            return EXIT_INVALID_ARGS

    gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    gemini_module = os.environ.get("PDF2TREE_GEMINI_MODULE", "google.genai")

    pdf_path = Path(args.from_file).expanduser().resolve()
    out_dir = Path(args.to_dir).expanduser().resolve()
    force_toc_validation = _env_flag_enabled(os.environ.get("PDF2TREE_FORCE_TOC_VALIDATION"))
    skip_toc_validation = bool(
        is_test_mode()
        and (args.n_pages is not None or (args.start_page is not None and args.start_page != 1))
        and not force_toc_validation
    )

    post_processing_cfg = PostProcessingConfig(
        enable_pix2tex=bool(args.enable_pic2tex) and post_processing_active,
        disable_pix2tex=bool(args.disable_pic2tex),
        equation_min_len=int(args.equation_min_len),
        verbose=bool(args.verbose),
        debug=bool(args.debug),
        annotate_images=effective_annotate_images,
        annotate_equations=effective_annotate_equations,
        gemini_api_key=gemini_api_key,
        gemini_model=str(args.gemini_model or GEMINI_DEFAULT_MODEL),
        gemini_module=str(gemini_module or "google.genai"),
        test_mode=is_test_mode(),
        disable_remove_small_images=bool(args.disable_remove_small_images),
        disable_cleanup=bool(args.disable_cleanup),
        disable_toc=bool(args.disable_toc),
        enable_pdf_pages_ref=bool(getattr(args, "enable_pdf_pages_ref", False)),
        min_size_x=int(args.min_size_x),
        min_size_y=int(args.min_size_y),
        prompt_equation=prompts_cfg["prompt_equation"],
        prompt_non_equation=prompts_cfg["prompt_non_equation"],
        prompt_uncertain=prompts_cfg["prompt_uncertain"],
        skip_toc_validation=skip_toc_validation,
    )

    annotation_active = post_processing_cfg.annotate_images or post_processing_cfg.annotate_equations
    if annotation_active and not post_processing_cfg.gemini_api_key:
        LOG.error(
            "Gemini API key is required when annotation is enabled (image annotation enabled by default or --enable-annotate-equations requested)",
        )
        return EXIT_INVALID_ARGS

    if not args.post_processing_only:
        if not pdf_path.exists() or not pdf_path.is_file():
            LOG.error("Source file not found: %s", pdf_path)
            return 2

    if out_dir.exists():
        if not out_dir.is_dir():
            LOG.error("Output path is not a directory: %s", out_dir)
            return EXIT_OUTPUT_DIR
        if not args.post_processing_only and any(out_dir.iterdir()):
            LOG.error("Output directory must be empty: %s", out_dir)
            return EXIT_OUTPUT_DIR

    manifest_path = out_dir / f"{pdf_path.stem}.json"

    # CORE-DES-083: controllo versione dopo validazione e prima di avviare qualsiasi pipeline.
    # CORE-DES-084: su errore non stampare nulla e proseguire.
    maybe_print_new_version_notice(program_name="pdf2tree")

    if args.post_processing_only:
        if not pdf_path.exists() or not pdf_path.is_file():
            LOG.error("Source file not found for post-processing-only: %s", pdf_path)
            return EXIT_POSTPROC_ARTIFACT
        print_program_banner()
        print_parameter_summary(
            args=args,
            post_config=post_processing_cfg,
            pdf_path=pdf_path,
            out_dir=out_dir,
            post_processing_active=post_processing_active,
            post_processing_only=True,
            form_xobject_enabled=form_xobject_enabled,
            vector_images_enabled=vector_images_enabled,
        )
        # CORE-DES-042: esplicitare che l'esecuzione riprende direttamente dalla fase di post-processing.
        LOG.info("Post-processing-only mode detected: resuming pipeline directly at the post-processing stage.")
        LOG.info("Manifest JSON will be rebuilt from the restored Markdown and source PDF.")
        start_phase("Loading existing artifacts")
        if not out_dir.exists():
            LOG.error("Output directory not found for post-processing: %s", out_dir)
            return EXIT_POSTPROC_ARTIFACT
        md_path = find_existing_markdown(out_dir, pdf_path.stem)
        if not md_path or not md_path.exists():
            LOG.error("Markdown file not found in output directory: %s", out_dir)
            return EXIT_POSTPROC_ARTIFACT
        backup_path = md_path.with_suffix(md_path.suffix + ".processing.md")
        if not backup_path.exists():
            LOG.error("Backup Markdown (.processing.md) not found: %s", backup_path)
            return EXIT_POSTPROC_ARTIFACT
        end_phase()

        start_phase("Post-processing")
        try:
            updated_md, updated_manifest, toc_mismatch = run_post_processing_pipeline(
                out_dir=out_dir,
                pdf_path=pdf_path,
                md_path=md_path,
                manifest_path=manifest_path,
                config=post_processing_cfg,
            )
        except RuntimeError as exc:
            LOG.error("Post-processing failed: %s", exc)
            return EXIT_POSTPROC_DEP
        safe_write_text(md_path, updated_md if updated_md.endswith("\n") else f"{updated_md}\n")
        safe_write_text(manifest_path, json.dumps(updated_manifest, ensure_ascii=False, indent=2) + "\n")
        end_phase()
        LOG.info("Updated: %s", md_path)
        LOG.info("Updated: %s", manifest_path)
        return EXIT_POSTPROC_DEP if toc_mismatch else 0

    opencv_available = has_opencv()
    if not opencv_available:
        LOG.error("OpenCV (cv2) not installed or not importable: install opencv-python in the environment")
        return EXIT_OPENCV_MISSING

    print_program_banner()
    print_parameter_summary(
        args=args,
        post_config=post_processing_cfg,
        pdf_path=pdf_path,
        out_dir=out_dir,
        post_processing_active=post_processing_active,
        post_processing_only=False,
        form_xobject_enabled=form_xobject_enabled,
        vector_images_enabled=vector_images_enabled,
    )

    try:
        final_md, md_out = run_processing_pipeline(
            args=args,
            pdf_path=pdf_path,
            out_dir=out_dir,
            post_processing_cfg=post_processing_cfg,
            form_xobject_enabled=form_xobject_enabled,
            vector_images_enabled=vector_images_enabled,
        )
    except RuntimeError as exc:
        msg = str(exc)
        LOG.error("%s", msg)
        if "TOC not found" in msg:
            return 4
        if "Start page" in msg or "Requested page range" in msg:
            return EXIT_INVALID_ARGS
        if "Unable to import pymupdf4llm" in msg:
            return 3
        return 5

    if not args.post_processing:
        LOG.info("Post-processing flag not provided; execution stops after writing Markdown.")
        return 0

    start_phase("Post-processing")
    try:
        updated_md, updated_manifest, toc_mismatch = run_post_processing_pipeline(
            out_dir=out_dir,
            pdf_path=pdf_path,
            md_path=md_out,
            manifest_path=manifest_path,
            config=post_processing_cfg,
        )
    except RuntimeError as exc:
        LOG.error("Post-processing failed: %s", exc)
        return EXIT_POSTPROC_DEP
    safe_write_text(md_out, updated_md if updated_md.endswith("\n") else f"{updated_md}\n")
    safe_write_text(manifest_path, json.dumps(updated_manifest, ensure_ascii=False, indent=2) + "\n")
    end_phase()
    LOG.info("Updated: %s", md_out)
    LOG.info("Updated: %s", manifest_path)
    return EXIT_POSTPROC_DEP if toc_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
