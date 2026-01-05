r"""
Helper leggeri per la validazione LaTeX usati nella pipeline di post-processing Pix2Tex.
Il validatore esegue quattro stadi sequenziali:
1) controllo del bilanciamento dei delimitatori (inclusi \left/\right)
2) parsing con pylatexenc
3) verifica di compatibilità MathJax
4) tentativo di render con matplotlib (backend Agg)
Qualsiasi errore o eccezione segna la formula come non valida.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # garantisce il rendering headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import mathtext  # noqa: E402
from pylatexenc.latexwalker import LatexWalker, LatexWalkerError  # noqa: E402

LOG = logging.getLogger("pdf2tree.latex")


def _strip_markdown_delimiters(latex: str) -> str:
    """Rimuove i delimitatori Markdown esterni da una formula."""

    return latex.strip().strip("$ ").strip()


def _iter_chars(content: str) -> Iterable[str]:
    """Itera sui caratteri distinguendo token speciali \\left/\\right."""

    i = 0
    while i < len(content):
        if content.startswith(r"\left", i):
            yield r"\left"
            i += 5
            continue
        if content.startswith(r"\right", i):
            yield r"\right"
            i += 6
            continue
        yield content[i]
        i += 1


def _validate_delimiters(latex: str) -> bool:
    """Verifica il bilanciamento di parentesi e \\left/\\right."""

    mapping = {"{": "}", "[": "]", "(": ")"}
    stack: list[str] = []
    for token in _iter_chars(_strip_markdown_delimiters(latex)):
        if token == r"\left":
            stack.append("left-right")
            continue
        if token == r"\right":
            if not stack or stack.pop() != "left-right":
                return False
            continue
        if token in mapping:
            stack.append(mapping[token])
            continue
        if token in mapping.values():
            if not stack or token != stack.pop():
                return False
    return len(stack) == 0


def _validate_with_pylatexenc(latex: str) -> bool:
    """Esegue il parsing con pylatexenc per rilevare errori sintattici."""

    clean = _strip_markdown_delimiters(latex)
    try:
        LatexWalker(clean).get_latex_nodes()
        return True
    except LatexWalkerError as exc:
        LOG.debug("pylatexenc validation failed: %s", exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("Unexpected pylatexenc error: %s", exc)
        return False


FORBIDDEN_TEX_TOKENS = [r"\atop", r"\atopwithdelims", r"\overwithdelims"]
EMPTY_GROUP_RE = re.compile(r"\\(?:mathrm|mathit|mathbf|mathsf|mathtt|textrm|textbf|textit)\s*\{\s*\}")
BEGIN_ENV_RE = re.compile(r"\\(?:begin|end)\s*\{[^}]+\}", re.IGNORECASE)


def _validate_mathjax_compat(latex: str) -> bool:
    """Scarta formule problematiche per Markdown/MathJax prima del rendering."""
    formula = _strip_markdown_delimiters(latex)
    lowered = formula.lower()
    for token in FORBIDDEN_TEX_TOKENS:
        if token in lowered:
            LOG.debug("MathJax compatibility failed: forbidden token %s", token)
            return False
    if "{}" in formula:
        LOG.debug("MathJax compatibility failed: empty group detected")
        return False
    if EMPTY_GROUP_RE.search(formula):
        LOG.debug("MathJax compatibility failed: empty styled group detected")
        return False
    env_match = BEGIN_ENV_RE.search(formula)
    if env_match:
        LOG.debug(
            "MathJax compatibility failed: unsupported environment %s",
            env_match.group(0),
        )
        return False
    try:
        parser = mathtext.MathTextParser("path")
        parser.parse(f"${formula}$")
        return True
    except Exception as exc:
        LOG.debug("MathJax compatibility parse failed: %s", exc)
        return False


def _validate_with_matplotlib(latex: str) -> bool:
    """Tenta il rendering con matplotlib per individuare errori residui."""

    formula = _strip_markdown_delimiters(latex)
    try:
        plt.figure()
        plt.text(0, 0, f"${formula}$")
        plt.close()
        return True
    except Exception as exc:
        plt.close()
        LOG.debug("Matplotlib render failed: %s", exc)
        return False


def validate_latex_formula(latex: str) -> bool:
    """
    Valida una formula LaTeX tramite controlli sequenziali: delimitatori, parsing pylatexenc,
    compatibilità MathJax e render con matplotlib. Restituisce True solo se tutti gli stadi superano.
    """
    text = latex.strip()
    if not text:
        return False
    if not _validate_delimiters(text):
        return False
    if not _validate_with_pylatexenc(text):
        return False
    if not _validate_mathjax_compat(text):
        return False
    if not _validate_with_matplotlib(text):
        return False
    return True
