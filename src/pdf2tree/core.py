# -*- coding: utf-8 -*-
import argparse
import sys
import os
import pymupdf  # Importa la libreria PyMuPDF per la manipolazione di file PDF.
import pymupdf4llm # Importa un'estensione di PyMuPDF ottimizzata per l'estrazione di dati per LLM.
try:
    import pymupdf.layout  # Attiva il layout analyzer se disponibile.
    _HAS_PYMUPDF_LAYOUT = True
except Exception:
    _HAS_PYMUPDF_LAYOUT = False
import re
import json
import shutil
from pathlib import Path
import tempfile
import signal
import mimetypes
import importlib
import subprocess

# --- CONFIGURAZIONE GLOBALE ---
# Questi parametri influenzano il modo in cui il testo e gli elementi grafici vengono estratti.

# MARGINI_TESTO: Definisce un'area di esclusione (sinistra, sopra, destra, sotto) per l'estrazione del testo.
# Utile per ignorare intestazioni, piè di pagina e numeri di pagina.
TEXT_MARGINS = (0, 50, 0, 50)

# SALTO_VERTICALE_VETTORI: Percentuale della pagina da ignorare in alto e in basso durante la ricerca di diagrammi vettoriali.
# Previene che intestazioni e piè di pagina vengano inclusi nei diagrammi.
VECTOR_SKIP_Y_PERCENT = 0.15

# NUMERO_MINIMO_TRACCIATI_VETTORIALI: Un cluster di disegni vettoriali viene considerato un diagramma solo se
# contiene almeno questo numero di tracciati (linee, curve, ecc.). Filtra i piccoli elementi irrilevanti.
MIN_VECTOR_PATHS = 10

# RAPPORTO_MASSIMO_LARGHEZZA_SEPARATORE: Le linee orizzontali la cui larghezza supera questa percentuale
# della larghezza della pagina vengono considerate separatori e ignorate, non parte di un diagramma.
MAX_SEPARATOR_WIDTH_RATIO = 0.8

# TOLLERANZA_CLUSTERING: Distanza in pixel (orizzontale e verticale) entro la quale elementi grafici vicini
# vengono raggruppati insieme per formare un unico "cluster" o diagramma.
CLUSTER_X_TOLERANCE = 10
CLUSTER_Y_TOLERANCE = 10

# Nome file progress per la modalità resume (singola sorgente di verità)
PROGRESS_FILENAME = "progress_state.json"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
IMAGE_DESCRIPTION_PROMPT = (
    "Fornisci una descrizione dettagliata dell'immagine per indicizzare il contenuto in un vector DB. "
    "Includi soggetti, oggetti, testo visibile, colori, contesto, stile e relazioni spaziali. "
    "Rispondi in italiano con frasi complete."
)
MATH_LATEX_PROMPT = (
    "Estrarre la formula matematica presente nell'immagine e restituirla in puro LaTeX, "
    "senza testo extra, commenti o markdown aggiuntivo. Non aggiungere delimitatori."
)
MATH_PADDING = 6
_MATH_SYMBOLS = set("=+−-*/·×÷√∑∏≈≠≤≥∞∫∇∂πµσθλΩΦΨαβγδ∆∈∪∩∀∃∧∨⇒⇔→←∞∅^_{}[]()<>|")
_LATEX_TOKEN_RE = re.compile(r"\\[A-Za-z]+")
_MATH_DELIMS = ("\\[", "\\]", "$", "\\(", "\\)")


def check_dependencies():
    """
    Verifica che la versione della libreria PyMuPDF installata sia sufficientemente recente.
    La funzione 'cluster_drawings', cruciale per l'analisi vettoriale, è disponibile solo dalla versione 1.24.2.
    Se la versione è troppo vecchia, lo script si interrompe con un messaggio di errore.
    """
    try:
        # Estrae la versione come stringa (es. '1.24.5')
        v_str = pymupdf.__version__
        # Converte i numeri di versione in interi per il confronto
        major, minor, patch = map(int, v_str.split('.')[:3])

        # pymupdf4llm e pymupdf_layout richiedono PyMuPDF >= 1.26.6.
        if (major, minor, patch) < (1, 26, 6):
            print(f"❌ ERRORE CRITICO: La tua versione di PyMuPDF è obsoleta ({v_str}).")
            print("   Questo tool richiede almeno la versione 1.26.6.")
            print("   Per aggiornare, esegui: pip install --upgrade pymupdf")
            sys.exit(1) # Interrompe l'esecuzione.

    except Exception as e:
        print(f"⚠️  Attenzione: Impossibile verificare la versione di PyMuPDF. Causa: {e}")


def log(msg, is_debug):
    """
    Funzione di utilità per stampare messaggi di debug.
    Il messaggio viene mostrato solo se la modalità debug è attiva.

    Args:
        msg (str): Il messaggio da stampare.
        is_debug (bool): Flag che indica se la modalità debug è attiva.
    """
    if is_debug:
        print(f"[DEBUG] {msg}")


def sanitize_filename(name):
    """
    Pulisce una stringa per renderla un nome di file/directory valido e sicuro.
    - Rimuove caratteri speciali non consentiti.
    - Sostituisce spazi multipli con uno singolo.
    - Tronca la stringa a 50 caratteri per evitare nomi troppo lunghi.

    Args:
        name (str): La stringa da "sanificare".

    Returns:
        str: La stringa pulita e abbreviata.
    """
    # Rimuove tutti i caratteri che non sono lettere, numeri, spazi o trattini.
    clean = re.sub(r'[^\w\s-]', '', name).strip()
    # Sostituisce sequenze di spazi con un singolo spazio.
    clean = re.sub(r'\s+', ' ', clean)
    # Se la stringa pulita è vuota, restituisce un nome di default. Altrimenti la tronca.
    return clean[:50] if clean else "Sezione_Senza_Titolo"


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def _init_gemini_model(api_key: str, model_name: str):
    stub_path = os.environ.get("PDF2TREE_GEMINI_STUB_PATH")
    if stub_path:
        sys.path.insert(0, stub_path)

    module_path = os.environ.get("PDF2TREE_GEMINI_MODULE", "google.generativeai")

    try:
        genai = importlib.import_module(module_path)
    except ImportError:
        sys.exit(f"❌ Errore: modulo '{module_path}' non installato. Installa con 'pip install google-generativeai' oppure specifica un modulo alternativo.")

    try:
        if hasattr(genai, "configure"):
            genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as exc:
        sys.exit(f"❌ Errore nella configurazione del modello Gemini '{model_name}': {exc}")


def _describe_image_with_gemini(model, image_path: Path, is_debug: bool) -> str:
    try:
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        raise RuntimeError(f"Impossibile leggere l'immagine {image_path}: {exc}") from exc

    mime_type = _guess_mime_type(image_path)
    try:
        response = model.generate_content([IMAGE_DESCRIPTION_PROMPT, {"mime_type": mime_type, "data": image_bytes}])
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise RuntimeError("Risposta vuota da Gemini")
        return text.strip()
    except Exception as exc:
        log(f"Errore durante l'annotazione immagine {image_path}: {exc}", is_debug)
        hint = ""
        if "not found" in str(exc).lower():
            hint = f" Modello non disponibile; riprova con --gemini-model {GEMINI_DEFAULT_MODEL}."
        raise RuntimeError(f"Annotazione immagine fallita: {exc}.{hint}") from exc


def annotate_images_in_markdown(markdown: str, chapter_dir: Path, model, is_debug: bool, verbose: bool, dry_run: bool) -> str:
    """
    Inserisce descrizioni generate dal modello Gemini accanto ai link delle immagini.
    """
    if dry_run:
        if verbose or is_debug:
            print("[VERBOSE] (dry-run) annotazione immagini saltata")
        return markdown

    pattern = re.compile(r"!\[[^\]]*]\((assets/[^\)]+)\)")
    seen = set()

    def _replace(match: re.Match) -> str:
        rel_path = match.group(1)
        if "math_formula" in rel_path:
            return match.group(0)
        if rel_path in seen:
            return match.group(0)
        seen.add(rel_path)

        image_path = (chapter_dir / rel_path).resolve()
        if not image_path.exists():
            log(f"Immagine non trovata per annotazione: {image_path}", is_debug)
            return match.group(0)

        if verbose or is_debug:
            print(f"[VERBOSE] Annotazione immagine: {image_path}")

        description = _describe_image_with_gemini(model, image_path, is_debug)
        return f"{match.group(0)}\n\n> {description}\n"

    return pattern.sub(_replace, markdown)


def _describe_math_with_gemini(model, image_path: Path, is_debug: bool) -> str:
    try:
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        raise RuntimeError(f"Impossibile leggere l'immagine {image_path}: {exc}") from exc

    mime_type = _guess_mime_type(image_path)
    try:
        response = model.generate_content([MATH_LATEX_PROMPT, {"mime_type": mime_type, "data": image_bytes}])
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise RuntimeError("Risposta vuota da Gemini per formula")
        return text.strip()
    except Exception as exc:
        log(f"Errore durante l'estrazione LaTeX per {image_path}: {exc}", is_debug)
        raise RuntimeError(f"Annotazione formula fallita: {exc}") from exc


def _is_math_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(d in stripped for d in _MATH_DELIMS):
        return True
    latex_hits = len(_LATEX_TOKEN_RE.findall(stripped))
    math_hits = sum(1 for ch in stripped if ch in _MATH_SYMBOLS)
    digit_hits = sum(1 for ch in stripped if ch.isdigit())
    if latex_hits:
        return True
    if math_hits >= 2 and (math_hits / max(len(stripped), 1)) >= 0.2:
        return True
    if ("=" in stripped or "≤" in stripped or "≥" in stripped) and digit_hits:
        return True
    return False


def _collect_math_spans(page, is_debug: bool):
    spans = []
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception as exc:
        log(f"Impossibile estrarre testo per math detection: {exc}", is_debug)
        return spans

    for block in blocks:
        for line in block.get("lines", []):
            line_rect = pymupdf.Rect(line.get("bbox", (0, 0, 0, 0)))
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not _is_math_text(text):
                    continue
                rect = pymupdf.Rect(span.get("bbox", (0, 0, 0, 0)))
                # Include full line height to avoid vertical clipping
                rect.y0 = min(rect.y0, line_rect.y0)
                rect.y1 = max(rect.y1, line_rect.y1)
                spans.append({"text": text.strip(), "rect": rect})

    merged = []
    for span in spans:
        if merged:
            prev = merged[-1]
            same_line = abs(prev["rect"].y0 - span["rect"].y0) < 5 and abs(prev["rect"].y1 - span["rect"].y1) < 10
            close_x = (span["rect"].x0 - prev["rect"].x1) < 24
            overlap_x = not (span["rect"].x0 > prev["rect"].x1 or span["rect"].x1 < prev["rect"].x0)
            if (same_line and close_x) or overlap_x:
                prev["rect"].x0 = min(prev["rect"].x0, span["rect"].x0)
                prev["rect"].y0 = min(prev["rect"].y0, span["rect"].y0)
                prev["rect"].x1 = max(prev["rect"].x1, span["rect"].x1)
                prev["rect"].y1 = max(prev["rect"].y1, span["rect"].y1)
                prev["text"] = f"{prev['text']} {span['text']}".strip()
                continue
        merged.append(span)
    return merged


def _has_math_styled_spans(page) -> bool:
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return False

    def _non_alnum_ratio(txt: str) -> float:
        if not txt:
            return 0.0
        non_alnum = sum(1 for ch in txt if not (ch.isalnum() or ch.isspace()))
        return non_alnum / max(len(txt), 1)

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                font = span.get("font", "").lower()
                if (
                    _is_math_text(text)
                    or "math" in font
                    or "italic" in font
                    or _non_alnum_ratio(text) > 0.3
                    or any(ch in text for ch in "[ ]=±∑∞∫")
                    or "\\" in text
                ):
                    return True
    return False


def _render_math_formulas(doc, page_num: int, assets_path: Path, is_debug: bool, annotate_math: bool, gemini_model, verbose: bool):
    formulas = []
    page = doc.load_page(page_num - 1)
    candidates = _collect_math_spans(page, is_debug)
    if not candidates:
        return formulas

    for idx, cand in enumerate(candidates, start=1):
        rect = cand["rect"]
        # Safety expansion before padding to avoid clipped tall symbols
        expand = max(4, int(rect.height * 0.10))
        rect.x0 = max(0, rect.x0 - expand)
        rect.y0 = max(0, rect.y0 - expand)
        rect.x1 = rect.x1 + expand
        rect.y1 = rect.y1 + expand

        pad = max(MATH_PADDING, int(rect.height * 0.35), 12)
        rect.x0 = max(0, rect.x0 - pad)
        rect.y0 = max(0, rect.y0 - pad)
        rect.x1 = min(page.rect.x1, rect.x1 + pad)
        rect.y1 = min(page.rect.y1, rect.y1 + pad)
        fname = f"math_formula_p{page_num}_{idx}.png"
        img_path = assets_path / fname
        try:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=rect)
            pix.save(img_path)
            if verbose or is_debug:
                print(f"[VERBOSE] Salvata formula p.{page_num} #{idx} bbox={rect} pad={pad} -> {img_path}")
        except Exception as exc:
            log(f"Errore salvataggio formula p.{page_num} #{idx}: {exc}", is_debug)
            continue

        latex_text = None
        if annotate_math and gemini_model is not None:
            try:
                latex_text = _describe_math_with_gemini(gemini_model, img_path, is_debug)
            except Exception as exc:
                log(f"Annotazione LaTeX formula p.{page_num} #{idx} fallita: {exc}", is_debug)

        formulas.append(
            {
                "text": cand["text"],
                "image_rel": f"assets/{fname}",
                "latex": latex_text,
                "inserted": False,
            }
        )
    return formulas


def _render_math_page_image(doc, page_num: int, assets_path: Path, is_debug: bool, verbose: bool):
    page = doc.load_page(page_num - 1)
    fname = f"math_page_p{page_num}.png"
    out_path = assets_path / fname
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
        pix.save(out_path)
        if verbose or is_debug:
            print(f"[VERBOSE] Salvata pagina matematica p.{page_num} -> {out_path}")
        return fname
    except Exception as exc:
        log(f"Errore salvataggio pagina matematica p.{page_num}: {exc}", is_debug)
        return None


def _try_pic2tex(image_path: Path, is_debug: bool):
    if shutil.which("pic2tex") is None:
        return None
    try:
        proc = subprocess.run(
            ["pic2tex", str(image_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        latex = proc.stdout.strip()
        return latex or None
    except Exception as exc:
        log(f"pic2tex fallito per {image_path}: {exc}", is_debug)
        return None


def _inject_math_images(text: str, formulas: list, append_unused: bool) -> str:
    result = text
    normalized = re.sub(r"\s+", " ", result)
    for formula in formulas:
        if formula.get("inserted"):
            continue
        needle = formula.get("text", "").strip()
        if not needle:
            continue
        simplified = re.sub(r"\s+", " ", needle)
        pattern_exact = re.compile(re.escape(needle))
        pattern_simple = re.compile(re.escape(simplified).replace("\\ ", r"\\s+"))
        replacement = f"![Formula]({formula['image_rel']})"
        if formula.get("latex"):
            replacement = f"{replacement}\n\n$${formula['latex']}$$"
        for pat in (pattern_exact, pattern_simple):
            new_result, count = pat.subn(replacement, result, count=1)
            if count:
                result = new_result
                formula["inserted"] = True
                break

    # Line-based fallback: replace any remaining math-like lines with unused formulas.
    if any(not f.get("inserted") for f in formulas):
        lines = result.splitlines()
        out_lines = []
        for line in lines:
            pending = next((f for f in formulas if not f.get("inserted")), None)
            if pending and not line.strip().startswith("![Formula]") and _is_math_text(line):
                replacement = f"![Formula]({pending['image_rel']})"
                if pending.get("latex"):
                    replacement = f"{replacement}\n\n$${pending['latex']}$$"
                out_lines.append(replacement)
                pending["inserted"] = True
            else:
                out_lines.append(line)
        result = "\n".join(out_lines)

    if append_unused:
        extras = []
        for formula in formulas:
            if not formula.get("inserted"):
                replacement = f"![Formula]({formula['image_rel']})"
                if formula.get("latex"):
                    replacement = f"{replacement}\n\n$${formula['latex']}$$"
                extras.append(replacement)
                formula["inserted"] = True
        if extras:
            result = result.rstrip() + "\n\n" + "\n\n".join(extras) + "\n"

    # Remove lingering math-like lines if we processed formulas
    if formulas:
        cleaned = []
        for line in result.splitlines():
            if _is_math_text(line):
                continue
            cleaned.append(line)
        result = "\n".join(cleaned)
    return result


def get_smart_vector_crop(doc, page_num, is_debug):
    """
    Utilizza la funzione avanzata `cluster_drawings` di PyMuPDF (v1.24+)
    per identificare e ritagliare in modo intelligente i diagrammi vettoriali in una pagina.

    Args:
        doc (pymupdf.Document): L'oggetto documento PDF.
        page_num (int): Il numero della pagina da analizzare (partendo da 1).
        is_debug (bool): Flag per attivare i log di debug.

    Returns:
        pymupdf.Rect or None: Il rettangolo (bbox) che contiene il diagramma più grande trovato,
                              oppure None se non viene trovato nessun diagramma significativo.
    """
    try:
        # Carica la pagina specificata (gli indici sono a base 0).
        page = doc.load_page(page_num - 1)

        # --- Fase 1: Pre-filtraggio manuale ---
        # Si analizzano i disegni "grezzi" per escludere elementi comuni che non sono diagrammi,
        # come linee separatrici, intestazioni e piè di pagina.
        raw_drawings = page.get_drawings()
        clean_drawings = [] # Lista per contenere solo i disegni "buoni".

        page_w = page.rect.width
        page_h = page.rect.height

        for d in raw_drawings:
            r = d["rect"] # Bounding box del disegno
            # Ignora elementi nelle aree di intestazione/piè di pagina.
            if r.y1 < (page_h * VECTOR_SKIP_Y_PERCENT) or r.y0 > (page_h * (1 - VECTOR_SKIP_Y_PERCENT)):
                continue
            # Ignora linee orizzontali molto larghe, probabilmente separatori.
            if r.width > (page_w * MAX_SEPARATOR_WIDTH_RATIO) and r.height < 20:
                continue
            # Ignora rettangoli molto grandi, probabilmente sfondi di pagina.
            if r.width > (page_w * 0.9) and r.height > (page_h * 0.9):
                continue
            clean_drawings.append(d)

        if not clean_drawings: return None # Nessun disegno interessante trovato.

        # --- Fase 2: Clustering Nativo ---
        # Si usa la funzione di PyMuPDF per raggruppare i disegni vicini.
        try:
            clusters = page.cluster_drawings(
                drawings=clean_drawings,
                x_tolerance=CLUSTER_X_TOLERANCE,
                y_tolerance=CLUSTER_Y_TOLERANCE
            )
        except AttributeError:
            # Questo errore si verifica se la versione di PyMuPDF è troppo vecchia.
            log(f"⚠️  Errore: Il metodo 'cluster_drawings' non esiste. Aggiorna PyMuPDF!", True)
            return None

        # --- Fase 3: Selezione del cluster migliore ---
        # Si analizzano i cluster trovati per scegliere il più promettente.
        valid_clusters = []
        for rect in clusters:
            # Ignora cluster troppo piccoli per essere un diagramma.
            if rect.width < 100 or rect.height < 100: continue

            # Controlla la "densità" del cluster: quanti tracciati contiene?
            # È una stima approssimativa per verificare che non sia un'area vuota.
            count = sum(1 for d in clean_drawings if d["rect"].intersects(rect))
            if count >= MIN_VECTOR_PATHS:
                valid_clusters.append(rect)

        if not valid_clusters: return None # Nessun cluster valido trovato.

        # Se ci sono più cluster validi, si sceglie quello con l'area maggiore.
        best_rect = max(valid_clusters, key=lambda r: r.width * r.height)

        # Aggiunge un piccolo margine (padding) attorno al rettangolo trovato
        # per assicurarsi di non tagliare i bordi del diagramma.
        best_rect.x0 = max(0, best_rect.x0 - 10)
        best_rect.y0 = max(0, best_rect.y0 - 10)
        best_rect.x1 = min(page_w, best_rect.x1 + 10)
        best_rect.y1 = min(page_h, best_rect.y1 + 10)

        log(f"📸 [Cluster] Trovato diagramma a pag.{page_num}: {best_rect}", is_debug)
        return best_rect

    except Exception as e:
        log(f"Errore durante l'analisi vettoriale della pagina {page_num}: {e}", is_debug)
        return None


def process_chapter_chunks(doc, start, end, assets_path, is_debug, vectors_on, pages=None, annotate_math=False, gemini_model=None, verbose=False, dry_run=False):
    """
    Estrae il contenuto (testo e immagini) di un capitolo, definito da un range di pagine,
    e lo converte in formato Markdown. Opzionalmente, cerca ed estrae anche diagrammi vettoriali.

    Args:
        doc (pymupdf.Document): L'oggetto documento PDF.
        start (int): La pagina iniziale del capitolo (indice a base 0).
        end (int): La pagina finale del capitolo (non inclusa).
        assets_path (Path): La cartella dove salvare le immagini estratte.
        is_debug (bool): Flag per il debug.
        vectors_on (bool): Flag per attivare l'estrazione di diagrammi vettoriali.

    Returns:
        str: Il contenuto del capitolo in formato Markdown.
    """
    # `pages` può essere fornito come lista di indici (base 0). Altrimenti
    # si elabora l'intero intervallo [start, end).
    if pages is None:
        pages = list(range(start, end))
    else:
        # Assicuriamoci che sia una lista di interi
        pages = list(pages)

    if not pages:
        return "", set()

    log(f"Analisi range di pagine: {start+1} -> {end} ({len(pages)} pagine)", is_debug)

    try:
        # Usa pymupdf4llm per convertire le pagine in Markdown.
        # Questa funzione estrae testo e immagini raster in un colpo solo.
        data = pymupdf4llm.to_markdown(
            doc,
            pages=pages,
            page_chunks=True,     # Divide il contenuto in blocchi per pagina.
            write_images=not dry_run,    # Evita scritture in dry-run.
            image_path=str(assets_path), # Specifica la cartella per le immagini.
            image_format="png",   # Formato delle immagini.
            margins=TEXT_MARGINS  # Applica i margini per ignorare header/footer.
        )

        page_numbers = []
        page_last_idx = {}
        for idx, chunk in enumerate(data):
            actual_page = pages[idx] + 1
            page_numbers.append(actual_page)
            page_last_idx[actual_page] = idx

        formulas_by_page = {}
        math_page_flags = {}
        math_page_images = {}
        math_page_latex = {}
        for page_num in sorted(page_last_idx.keys()):
            if dry_run:
                formulas_by_page[page_num] = []
                math_page_flags[page_num] = False
            else:
                page_obj = doc.load_page(page_num - 1)
                math_page_flags[page_num] = _has_math_styled_spans(page_obj)
                formulas_by_page[page_num] = _render_math_formulas(doc, page_num, assets_path, is_debug, annotate_math, gemini_model, verbose)
                if math_page_flags[page_num]:
                    img_name = _render_math_page_image(doc, page_num, assets_path, is_debug, verbose)
                    if img_name:
                        math_page_images[page_num] = img_name
                        latex_text = _try_pic2tex(assets_path / img_name, is_debug)
                        if latex_text:
                            math_page_latex[page_num] = latex_text
            if verbose:
                count = len(formulas_by_page[page_num])
                print(f"[VERBOSE] Formule rilevate pag.{page_num}: {count} (pagina matematica: {math_page_flags.get(page_num, False)})")

        md_output = ""
        processed_vector_pages = set() # Tiene traccia delle pagine già analizzate per vettori.
        processed_pages = set()

        for idx, chunk in enumerate(data):
            text = chunk.get("text", "")
            page_num = page_numbers[idx] # Numero di pagina (partendo da 1)
            processed_pages.add(page_num)

            formulas = formulas_by_page.get(page_num, [])
            append_unused = idx == page_last_idx.get(page_num)
            if formulas and not math_page_flags.get(page_num):
                before = text
                text = _inject_math_images(text, formulas, append_unused)
                if verbose and text != before:
                    injected = sum(1 for f in formulas if f.get("inserted"))
                    print(f"[VERBOSE] Inserite {injected}/{len(formulas)} formule in pag.{page_num}")
            elif math_page_flags.get(page_num):
                # Rimuovi righe riconosciute come matematiche e inserisci blocco pagina
                kept = []
                for line in text.splitlines():
                    if _is_math_text(line):
                        continue
                    kept.append(line)
                math_block = ""
                img_name = math_page_images.get(page_num)
                if img_name:
                    math_block = f"![Formula pagina {page_num}](assets/{img_name})"
                latex_block = math_page_latex.get(page_num)
                if latex_block:
                    math_block = f"{math_block}\n\n$${latex_block}$$" if math_block else f"$${latex_block}$$"
                if math_block:
                    kept.append("")
                    kept.append(math_block)
                text = "\n".join(kept)

            vector_add = "" # Contenuto Markdown aggiuntivo per il diagramma vettoriale.

            # Se l'opzione è attiva e la pagina non è già stata processata per i vettori...
            if vectors_on and page_num not in processed_vector_pages and not dry_run:
                bbox = get_smart_vector_crop(doc, page_num, is_debug)
                if bbox: # Se è stato trovato un diagramma...
                    try:
                        page_obj = doc.load_page(page_num - 1)
                        # Renderizza solo l'area del diagramma (clip=bbox) ad alta risoluzione (Matrix(2,2)).
                        pix = page_obj.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=bbox)
                        fname = f"vector_diagram_p{page_num}.png"
                        output_img = assets_path / fname
                        pix.save(output_img) # Salva l'immagine.

                        # Crea il link Markdown per l'immagine.
                        vector_add = f"\n\n![Diagramma Vettoriale (Pag. {page_num})](assets/{fname})\n> *Fonte: Diagramma vettoriale estratto tramite clustering.*\n\n"
                        processed_vector_pages.add(page_num)
                        if verbose or is_debug:
                            print(f"[VERBOSE] Salvato diagramma vettoriale p.{page_num} -> {output_img}")
                    except Exception as e:
                        log(f"Errore durante il salvataggio dell'immagine vettoriale: {e}", is_debug)

            # Aggiunge il testo e l'eventuale diagramma al risultato finale.
            if text.strip() or vector_add:
                md_output += f"{text}\n{vector_add}\n\n"

        return md_output, processed_pages

    except Exception as e:
        return f"> Errore durante l'elaborazione del capitolo: {e}", set()


def main():
    """
    Funzione principale che orchestra l'intero processo:
    1. Legge gli argomenti dalla riga di comando.
    2. Apre il PDF e ne legge l'indice (Table of Contents).
    3. Itera sull'indice per creare una struttura di cartelle che rispecchia i capitoli.
    4. Per ogni capitolo, estrae il contenuto e lo salva in un file 'content.md'.
    5. Genera un file 'project_manifest.json' che riassume la struttura creata.
    """
    check_dependencies() # Verifica subito le dipendenze.

    # Configurazione del parser per gli argomenti da riga di comando.
    parser = argparse.ArgumentParser(description="Converte un PDF in una struttura di cartelle e file Markdown basata sull'indice.")
    parser.add_argument("--from-file", required=True, help="Percorso del file PDF di input.")
    parser.add_argument("--to-dir", required=True, help="Percorso della cartella di output (non deve esistere).")
    parser.add_argument("--debug", action="store_true", help="Attiva i messaggi di debug dettagliati.")
    parser.add_argument("--disable-vector-images", action="store_true", help="Disattiva l'estrazione di diagrammi vettoriali (attiva per impostazione predefinita).")
    parser.add_argument("--force-restart", action="store_true", help="Ignora la cartella di output esistente e ricomincia da zero.")
    parser.add_argument("--dry-run", action="store_true", help="Esegue una prova senza scrivere file o modificare la cartella di output.")
    parser.add_argument("--verbose", action="store_true", help="Mostra informazioni aggiuntive sulle azioni eseguite.")
    parser.add_argument("--annotate-images", action="store_true", help="Attiva l'annotazione automatica delle immagini tramite Gemini.")
    parser.add_argument("--gemini-model", default=GEMINI_DEFAULT_MODEL, help="Nome del modello Gemini da usare per l'annotazione delle immagini.")

    args = parser.parse_args()

    stop_requested = False

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n🛑 Interruzione richiesta (Ctrl+C). Procedo con stop soft...")

    signal.signal(signal.SIGINT, _handle_sigint)

    input_path = Path(args.from_file)
    output_path = Path(args.to_dir)
    debug_mode = args.debug
    dry_run = args.dry_run
    verbose_mode = args.verbose
    annotate_images = args.annotate_images
    gemini_model_name = args.gemini_model
    # Logica invertita: i vettori sono attivi di default, disattivati dal nuovo flag.
    vectors_on = not args.disable_vector_images
    gemini_model = None

    def vlog(msg):
        """Verbose logger: prints when `--verbose` or `--debug` are active."""
        if debug_mode or verbose_mode:
            print(f"[VERBOSE] {msg}")

    # Controlli di base sui percorsi.
    if not input_path.exists():
        sys.exit(f"❌ Errore: Il file di input '{input_path}' non è stato trovato.")

    if annotate_images:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("❌ Errore: GEMINI_API_KEY non impostata. Necessaria per l'annotazione immagini con Gemini.")
        gemini_model = _init_gemini_model(api_key, gemini_model_name)
        if verbose_mode or debug_mode:
            print(f"[VERBOSE] Annotazione immagini attiva con modello '{gemini_model_name}'")

    # Resume mode: se la cartella di output esiste, normalmente entriamo
    # nella modalità di ripresa. Se però viene passato `--force-restart`,
    # cancelliamo la cartella di output e ricominciamo da zero.
    resume_mode = False
    existing_manifest = []
    manifest_path = output_path / "project_manifest.json"

    if output_path.exists():
        if args.force_restart:
            print(f"⚠️  --force-restart: rimuovo la cartella di output esistente '{output_path}' e ricomincio da zero.")
            if dry_run:
                vlog(f"(dry-run) would remove '{output_path}' and recreate it")
                manifest = []
                resume_mode = False
            else:
                try:
                    shutil.rmtree(output_path)
                except Exception as e:
                    print(f"⚠️ Impossibile rimuovere '{output_path}': {e}")
                    sys.exit(1)
                output_path.mkdir(parents=True, exist_ok=True)
                manifest = []
                resume_mode = False
        else:
            resume_mode = True
            print(f"🔁 Modalità ripresa attivata: la cartella di output '{output_path}' esiste.")

            # Se esiste un manifest precedente, caricalo per determinare i capitoli completati.
            if manifest_path.exists():
                try:
                    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"⚠️ Impossibile leggere project_manifest.json: {e}. Verrà ricostruito.")
                    existing_manifest = []

            # Determina quali indici capitolo sono stati effettivamente completati,
            # sia leggendo il manifest esistente che analizzando le cartelle già create.
            completed_indices = set()
            for entry in existing_manifest:
                try:
                    rel = Path(entry.get("path", ""))
                    chapter_dir = rel.parts[0] if rel.parts else ""
                    idx_str = chapter_dir.split("_")[0]
                    idx = int(idx_str)
                    # Verifica che il file content.md esista effettivamente
                    full_path = output_path / rel
                    if full_path.exists() and full_path.is_file() and full_path.stat().st_size > 0:
                        completed_indices.add(idx)
                except Exception:
                    continue

            # Integra con le cartelle trovate sul filesystem (utile se manifest mancante/incompleto).
            for p in output_path.iterdir():
                if p.is_dir() and re.match(r"^[0-9]{2}_", p.name):
                    try:
                        idx = int(p.name.split("_")[0])
                        content_file = p / "content.md"
                        if content_file.exists() and content_file.is_file() and content_file.stat().st_size > 0:
                            completed_indices.add(idx)
                    except Exception:
                        continue

            if completed_indices:
                # L'ultimo capitolo realmente completato
                last_completed = max(completed_indices)
                # Per rispettare la richiesta: cancellare l'ultima sezione processata
                # e riprendere dall'indice di quella stessa sezione.
                resume_from = last_completed
                print(f"   Capitoli completati rilevati: {sorted(completed_indices)}")
                print(f"   Ultima sezione completata: {last_completed}. Verrà cancellata e verrà ripresa da qui.")
            else:
                resume_from = 1
                print(f"   Capitoli completati rilevati: nessuno")
                print(f"   Riprendo dall'indice capitolo: {resume_from}")

            # Rimuove eventuali cartelle parziali a partire da resume_from (inclusa l'ultima completata)
            for p in output_path.iterdir():
                if p.is_dir() and re.match(r"^[0-9]{2}_", p.name):
                    try:
                        idx = int(p.name.split("_")[0])
                        if idx >= resume_from:
                            print(f"   Rimuovo dati (incluso parziali/completati da rielaborare): {p}")
                            if dry_run:
                                vlog(f"(dry-run) would remove folder: {p}")
                            else:
                                shutil.rmtree(p)
                    except Exception:
                        continue

            # Ricostruisci il manifest mantenendo solo gli elementi con indice < resume_from
            manifest = [m for m in existing_manifest if int(Path(m.get("path", "")).parts[0].split("_")[0]) < resume_from]
    else:
        if dry_run:
            vlog(f"(dry-run) would create output folder: {output_path}")
            manifest = []
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            manifest = []

    log(f"Apertura del file PDF: {input_path}", debug_mode)

    try:
        # Apre il documento PDF. 'pymupdf.open' è la sintassi moderna.
        doc = pymupdf.open(input_path)
    except Exception as e:
        sys.exit(f"❌ Errore critico durante l'apertura del PDF: {e}")

    # Estrae l'indice (Table of Contents). Se non c'è, usa una lista vuota.
    toc = doc.get_toc() or []
    if not toc:
        print("⚠️ Attenzione: il PDF non ha un indice (TOC). L'intero documento sarà trattato come un unico capitolo.")
        # Simula un indice con un solo capitolo che copre tutto il documento.
        toc = [[1, input_path.stem, 1]]


    # `manifest` è già inizializzato sopra (può contenere voci esistenti in resume mode).
    # Stack per gestire la gerarchia delle cartelle (capitoli, sottocapitoli, etc.).
    path_stack = [{'level': 0, 'path': output_path, 'titles': []}]

    # Aggiunge un elemento fittizio alla fine dell'indice per garantire che l'ultimo capitolo venga processato correttamente.
    # Questo elemento segna la "fine" del documento.
    toc.append([1, "END_OF_DOCUMENT", doc.page_count + 1])
    
    total_chapters = len(toc) - 1

    print(f"🚀 Avvio elaborazione (Versione Core Moderno)...")
    if vectors_on:
        print("💡 Modalità Estrazione Vettoriale ATTIVA (sperimentale).")
    else:
        print("     Modalità Estrazione Vettoriale disattivata.")

    # Percorso del file di progresso che tiene traccia delle pagine già processate
    progress_path = output_path / PROGRESS_FILENAME
    processed_pages = set()
    if resume_mode and progress_path.exists():
        try:
            pdata = json.loads(progress_path.read_text(encoding="utf-8") or "{}")
            processed_pages = set(pdata.get("processed_pages", []))
            print(f"   Ripristinato progresso: {sorted(list(processed_pages))}")
        except Exception as e:
            print(f"⚠️ Impossibile leggere {PROGRESS_FILENAME}: {e}. Parto da zero per le pagine già processate.")

    def _atomic_write_json(path: Path, obj: object, encoding: str = "utf-8"):
        """Write JSON to `path` atomically by writing to a temp file and renaming.

        Args:
            path (Path): destination file path
            obj (object): serializable object
            encoding (str): file encoding
        """
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".tmp_progress_", dir=str(path.parent))
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(json.dumps(obj, ensure_ascii=False))
            # Atomic replace
            os.replace(tmp, str(path))
        except Exception:
            # Cleanup temp file on failure
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise


    # Itera su ogni voce dell'indice.
    for i in range(total_chapters):
        if stop_requested:
            print("   Interruzione soft: blocco ciclo capitoli e salvo stato.")
            break
        lvl, title, page_num = toc[i]

        # Stampa lo stato di avanzamento anche senza debug.
        progress_indicator = f"[{i+1}/{total_chapters}]"
        print(f"{progress_indicator} Elaborazione Capitolo: '{title}'")

        # Determina il range di pagine per il capitolo corrente (indici base 0).
        start = max(0, page_num - 1)
        next_start = toc[i+1][2]
        raw_end = max(0, next_start - 1)
        end = max(start + 1, raw_end)
        if end > doc.page_count: end = doc.page_count

        pages = list(range(start, end))
        log(f"Pagine: {start+1}-{end}", debug_mode)

        # Logica di ripresa: mostra informazioni utili su quali pagine esistono
        # e quali sono già state processate (più dettagli se in modalità debug).
        if resume_mode:
            skipped = [p + 1 for p in pages if (p + 1) in processed_pages]
            if debug_mode:
                print(f"   [Resume][DEBUG] Capitolo '{title}' pagine totali: {[p+1 for p in pages]}")
                print(f"   [Resume][DEBUG] Pagine già processate globalmente: {sorted(list(processed_pages))}")
                print(f"   [Resume][DEBUG] Pagine saltate per questo capitolo: {skipped}")
            else:
                if skipped:
                    print(f"   [Resume] Skipping {len(skipped)} already-processed page(s) for this chapter")

        # Gestisce la gerarchia: risale lo stack finché non trova il genitore corretto.
        while path_stack[-1]['level'] >= lvl:
            path_stack.pop()

        parent = path_stack[-1]
        safe_title = sanitize_filename(title)
        folder_name = f"{i+1:02d}_{safe_title}"
        current_dir = parent['path'] / folder_name
        current_titles = parent['titles'] + [title]
        breadcrumb = " > ".join(current_titles)

        if dry_run:
            vlog(f"(dry-run) would create chapter folder: {current_dir}")
        else:
            current_dir.mkdir(parents=True, exist_ok=True)
        path_stack.append({'level': lvl, 'path': current_dir, 'titles': current_titles})

        assets_dir = current_dir / "assets"
        if dry_run:
            vlog(f"(dry-run) would create assets folder: {assets_dir}")
        else:
            assets_dir.mkdir(exist_ok=True)

        # Se siamo in resume mode, calcoliamo quali pagine del capitolo NON sono già state processate.
        pages_to_process = pages
        if resume_mode and processed_pages:
            pages_to_process = [p for p in pages if (p+1) not in processed_pages]
            if not pages_to_process:
                print(f"   Capitolo già completo, salto: '{title}'")
                # Aggiungi voce al manifest se manca
                content_file = current_dir / "content.md"
                if content_file.exists():
                    rel = str(content_file.relative_to(output_path)).replace("\\", "/")
                    manifest.append({"title": title, "path": rel})
                continue
            else:
                if debug_mode:
                    print(f"   [Resume][DEBUG] Pages to process for '{title}': {[p+1 for p in pages_to_process]}")
                else:
                    print(f"   [Resume] Will process {len(pages_to_process)} page(s) for '{title}'")

        # Processa solo le pagine rimanenti per questo capitolo.
        content_body, newly_processed = process_chapter_chunks(
            doc,
            start,
            end,
            assets_dir,
            debug_mode,
            vectors_on,
            pages=pages_to_process,
            annotate_math=annotate_images,
            gemini_model=gemini_model,
            verbose=verbose_mode,
            dry_run=dry_run,
        )

        if stop_requested:
            print("   Interruzione soft: fermo elaborazione del capitolo corrente dopo il blocco attuale.")

        # Se la cartella 'assets' è vuota, la cancella.
        if assets_dir.exists() and not any(assets_dir.iterdir()):
            try:
                if dry_run:
                    vlog(f"(dry-run) would remove empty assets folder: {assets_dir}")
                else:
                    assets_dir.rmdir()
            except OSError:
                pass
        else:
            content_body = content_body.replace(str(assets_dir), "assets").replace("\\", "/")

        if annotate_images:
            content_body = annotate_images_in_markdown(content_body, current_dir, gemini_model, debug_mode, verbose_mode, dry_run)

        content_file = current_dir / "content.md"
        if content_file.exists():
            # Append solo il corpo (senza frontmatter/header) per non duplicare meta.
            if dry_run:
                vlog(f"(dry-run) would append {len(content_body)} bytes to: {content_file}")
            else:
                content_file.write_text(content_file.read_text(encoding="utf-8") + "\n" + content_body, encoding="utf-8")
        else:
            md_content = f"---\ntitle: \"{title}\"\ncontext: \"{breadcrumb}\"\n---\n\n# {title}\n\n{content_body}"
            if dry_run:
                vlog(f"(dry-run) would create content file: {content_file} ({len(md_content)} bytes)")
            else:
                content_file.write_text(md_content, encoding="utf-8")

        # Aggiorna manifest
        rel_path = str((current_dir / "content.md").relative_to(output_path)).replace("\\", "/")
        manifest.append({"title": title, "path": rel_path})

        # Aggiorna file di progresso con le nuove pagine processate
        if newly_processed:
            processed_pages.update(newly_processed)
            try:
                if dry_run:
                    vlog(f"(dry-run) would update {PROGRESS_FILENAME} with pages: {sorted(list(processed_pages))}")
                else:
                    _atomic_write_json(progress_path, {"processed_pages": sorted(list(processed_pages))})
                    print(f"   [Resume] Aggiornato {PROGRESS_FILENAME} ({len(processed_pages)} total pages recorded).")
                if debug_mode:
                    print(f"   [Resume][DEBUG] Nuove pagine processate in questo capitolo: {sorted(list(newly_processed))}")
            except Exception as e:
                print(f"⚠️ Impossibile aggiornare {PROGRESS_FILENAME}: {e}")
        else:
            print(f"   [Resume] Nessuna nuova pagina processata per '{title}'.")

    # Dopo l'elaborazione, ricostruisci il manifest finale analizzando
    # l'intero albero di output: questo assicura che il JSON rifletta
    # lo stato reale del ramo radice sul filesystem.
    final_manifest = []
    if output_path.exists():
        for p in sorted([d for d in output_path.iterdir() if d.is_dir() and re.match(r"^[0-9]{2}_", d.name)],
                        key=lambda d: int(d.name.split("_")[0])):
            try:
                content_file = p / "content.md"
                if content_file.exists() and content_file.is_file():
                    # Prova a leggere il titolo dal frontmatter, altrimenti usa il nome della cartella.
                    title = p.name.split("_", 1)[1] if "_" in p.name else p.name
                    try:
                        txt = content_file.read_text(encoding="utf-8")
                        m = re.search(r"^title:\s*\"(.+?)\"", txt, flags=re.MULTILINE)
                        if m:
                            title = m.group(1)
                    except Exception:
                        pass

                    rel_path = str(content_file.relative_to(output_path)).replace("\\", "/")
                    final_manifest.append({"title": title, "path": rel_path})
            except Exception:
                continue

    # Se durante l'esecuzione abbiamo aggiunto nuove voci a `manifest`, possono essere già
    # presenti tra le cartelle; sostituiamo con `final_manifest` per coerenza.
    if dry_run:
        vlog(f"(dry-run) would write final project_manifest.json with {len(final_manifest)} entries to {output_path / 'project_manifest.json'}")
    else:
        (output_path / "project_manifest.json").write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if stop_requested:
        print("\n✅ Interruzione soft completata. Stato salvato.")
    else:
        print(f"\n✅ Elaborazione completata con successo.")
    print(f"   L'output è stato salvato in: {output_path}")

# Esegue la funzione main solo se lo script è lanciato direttamente.
if __name__ == "__main__":
    main()
