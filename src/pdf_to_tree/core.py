# -*- coding: utf-8 -*-
import argparse
import sys
import os
import pymupdf  # Importa la libreria PyMuPDF per la manipolazione di file PDF.
import pymupdf4llm # Importa un'estensione di PyMuPDF ottimizzata per l'estrazione di dati per LLM.
import re
import json
import shutil
from pathlib import Path
import tempfile
import signal

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

        # La funzione 'cluster_drawings' richiede la versione 1.24.2 o superiore.
        if (major, minor, patch) < (1, 24, 2):
            print(f"❌ ERRORE CRITICO: La tua versione di PyMuPDF è obsoleta ({v_str}).")
            print("   La funzione 'cluster_drawings' richiede almeno la versione 1.24.2.")
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


def process_chapter_chunks(doc, start, end, assets_path, is_debug, vectors_on, pages=None):
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
            write_images=True,    # Abilita il salvataggio delle immagini.
            image_path=str(assets_path), # Specifica la cartella per le immagini.
            image_format="png",   # Formato delle immagini.
            margins=TEXT_MARGINS  # Applica i margini per ignorare header/footer.
        )

        md_output = ""
        processed_vector_pages = set() # Tiene traccia delle pagine già analizzate per vettori.
        processed_pages = set()

        for chunk in data:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            page_num = meta.get("page", 0) + 1 # Numero di pagina (partendo da 1)
            processed_pages.add(page_num)

            vector_add = "" # Contenuto Markdown aggiuntivo per il diagramma vettoriale.

            # Se l'opzione è attiva e la pagina non è già stata processata per i vettori...
            if vectors_on and page_num not in processed_vector_pages:
                bbox = get_smart_vector_crop(doc, page_num, is_debug)
                if bbox: # Se è stato trovato un diagramma...
                    try:
                        page_obj = doc.load_page(page_num - 1)
                        # Renderizza solo l'area del diagramma (clip=bbox) ad alta risoluzione (Matrix(2,2)).
                        pix = page_obj.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=bbox)
                        fname = f"vector_diagram_p{page_num}.png"
                        pix.save(assets_path / fname) # Salva l'immagine.

                        # Crea il link Markdown per l'immagine.
                        vector_add = f"\n\n![Diagramma Vettoriale (Pag. {page_num})](assets/{fname})\n> *Fonte: Diagramma vettoriale estratto tramite clustering.*\n\n"
                        processed_vector_pages.add(page_num)
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
    # Logica invertita: i vettori sono attivi di default, disattivati dal nuovo flag.
    vectors_on = not args.disable_vector_images

    def vlog(msg):
        """Verbose logger: prints when `--verbose` or `--debug` are active."""
        if debug_mode or verbose_mode:
            print(f"[VERBOSE] {msg}")

    # Controlli di base sui percorsi.
    if not input_path.exists():
        sys.exit(f"❌ Errore: Il file di input '{input_path}' non è stato trovato.")

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
        content_body, newly_processed = process_chapter_chunks(doc, start, end, assets_dir, debug_mode, vectors_on, pages=pages_to_process)

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