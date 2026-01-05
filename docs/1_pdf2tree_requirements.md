---
title: "Requisiti pdf2tree"
description: Specifica dei requisiti software
date: "2026-01-05"
author: "Codex"
scope:
  paths:
    - "**/*.py"
    - "**/*.ipynb"
    - "**/*.c"
    - "**/*.h"
    - "**/*.cpp"
  excludes:
    - ".*/**"
visibility: "draft"
tags: ["markdown", "requirements", "example"]
---

# Requisiti pdf2tree
**Versione**: 0.70
**Autore**: Codex  
**Data**: 2026-01-05

## Indice
<!-- TOC -->
- [Requisiti pdf2tree](#requisiti-pdf2tree)
  - [Indice](#indice)
  - [Storico revisioni](#storico-revisioni)
  - [1. Introduzione](#1-introduzione)
    - [1.1 Regole del documento](#11-regole-del-documento)
    - [1.2 Scopo del progetto](#12-scopo-del-progetto)
  - [2. Requisiti di progetto](#2-requisiti-di-progetto)
    - [2.1 Funzioni di progetto](#21-funzioni-di-progetto)
    - [2.2 Vincoli di progetto](#22-vincoli-di-progetto)
    - [2.3 Componenti, librerie e struttura](#23-componenti-librerie-e-struttura)
  - [3. Requisiti](#3-requisiti)
    - [3.1 Progettazione e implementazione](#31-progettazione-e-implementazione)
    - [3.2 Funzioni](#32-funzioni)
  - [4. Verifica](#4-verifica)
<!-- TOC -->

## Storico revisioni
| Data | Versione | Motivo e descrizione della modifica |
|------|----------|-------------------------------------|
| 2026-01-02 | 0.1 | Bozza iniziale basata sul sorgente |
| 2026-01-02 | 0.13 | Integrazione pipeline pdf_to_markdown in core.py con gestione immagini/tabelle e logging debug |
| 2026-01-02 | 0.14 | Aggiunta requisito test pdf_sample post-modifica e errore CLI su PDF senza indice |
| 2026-01-02 | 0.15 | Aggiunta manifest JSON con TOC, tabelle e immagini generate |
| 2026-01-02 | 0.16 | Aggiunta opzioni CLI per escludere header e footer tramite margini in millimetri |
| 2026-01-02 | 0.17 | Aggiunta opzione CLI per disabilitare esportazione Form XObject |
| 2026-01-02 | 0.18 | Aggiornato default CLI header/footer a 0 mm e disabilitato crop quando impostati a 0 |
| 2026-01-02 | 0.19 | Aggiunti controlli di avvio su directory di output non vuota e assenza di OpenCV |
| 2026-01-02 | 0.20 | Aggiunta estrazione diagrammi vettoriali abilitata di default e opzione CLI di disattivazione |
| 2026-01-02 | 0.21 | Il manifest JSON include per ogni immagine i campi `source` e `type` |
| 2026-01-02 | 0.22 | Aggiunti requisiti per messaggi CLI in inglese, logging verbose con avanzamento e commenti in inglese |
| 2026-01-02 | 0.23 | Aggiunte stampe iniziali di nome/versione e marcatori di fase con completamento |
| 2026-01-02 | 0.24 | Aggiunti comandi di post-processing con pipeline Pix2Tex e parametri CLI dedicati |
| 2026-01-02 | 0.25 | Aggiunta fase opzionale di annotazione immagini/equazioni tramite Gemini con flag CLI dedicati |
| 2026-01-02 | 0.26 | Aggiunto framework di validazione LaTeX e integrazione nel post-processing Pix2Tex |
| 2026-01-02 | 0.27 | Logging verbose/debug per Pix2Tex con riferimenti posizionali e output grezzo |
| 2026-01-02 | 0.28 | Rimosso il fallback `google.generativeai` e vincolato Gemini all'SDK python-genai |
| 2026-01-04 | 0.29 | Aggiunto requisito su risposta Gemini in Markdown adatta alla RAG |
| 2026-01-04 | 0.30 | Richiesta esplicita di restituzione della formula LaTeX in Markdown/MathJax per le annotazioni delle equazioni |
| 2026-01-04 | 0.31 | Formattazione delle formule Pix2Tex e delle annotazioni Gemini con marcatori Start/End nel Markdown |
| 2026-01-04 | 0.32 | Aggiunto controllo di compatibilità MathJax nel framework di validazione LaTeX |
| 2026-01-04 | 0.33 | Il manifest descrive il contesto delle immagini/tabelle con stringhe gerarchiche e breadcrumb `context_path` per la RAG |
| 2026-01-04 | 0.35 | Abilita l'annotazione immagini di default e introduce `--disable-annotate-images` con `--enable-annotate-equations` |
| 2026-01-04 | 0.34 | Rinomina il campo `page` del manifest in `pdf_source_page` per identificare la pagina di origine del PDF |
| 2026-01-04 | 0.36 | Aggiunta l'opzione CLI `--n-pages` per limitare il numero massimo di pagine processate |
| 2026-01-04 | 0.37 | Introdotti requisiti per modalità test con risposte proforma Pix2Tex e Gemini |
| 2026-01-04 | 0.38 | Imposto l'uso obbligatorio di `--n-pages 1` per gli unit test |
| 2026-01-04 | 0.41 | Aggiunte opzione `--start-page` e vincoli di test su range pagine e compilazioni condivise |
| 2026-01-04 | 0.42 | Aggiornate opzioni CLI Form/Vector e chiarita pipeline di post-processing |
| 2026-01-04 | 0.43 | Introdotta fase di post-processing "remove-small-images" con parametri CLI dedicati |
| 2026-01-04 | 0.44 | Resa esplicita l'attivazione Pix2Tex tramite `--enable-pic2tex` e aggiornati i requisiti della pipeline CLI |
| 2026-01-04 | 0.45 | Introdotto il riepilogo dei parametri validati subito dopo la stampa del banner CLI |
| 2026-01-05 | 0.48 | Rifattorizzate le pipeline di processing e post-processing con backup .processing.md e ordine fasi manifest/remove-small |
| 2026-01-04 | 0.47 | Aggiornati i prompt Gemini di default con nuovi template RAG per equazioni, non-equazioni e casi incerti |
| 2026-01-04 | 0.46 | Aggiunta gestione file di prompt Gemini (lettura/scrittura) e selezione dinamica dei prompt per le annotazioni |
| 2026-01-05 | 0.50 | Introdotto requisito per commenti e docstring obbligatori in italiano su tutte le funzioni e blocchi critici |
| 2026-01-05 | 0.49 | Raffinati i requisiti di post-processing-only per richiedere il backup .processing.md del Markdown |
| 2026-01-05 | 0.51 | Aggiunti requisiti su lingua dei log/output, barra di avanzamento in verbose e contenuti di debug riservati a --debug |
| 2026-01-05 | 0.52 | Rafforzata la pipeline di post-processing con ricostruzione manifest come secondo step, dipendenza esplicita dal PDF origine e scomposizione della pipeline Markdown in funzioni dedicate |
| 2026-01-05 | 0.53 | Introdotta generazione del file .toc Markdown dal .md, validazione obbligatoria con la TOC del PDF e separazione della pipeline di creazione manifest in step dedicati invocati da run_post_processing_pipeline |
| 2026-01-05 | 0.54 | Disabilitata la validazione TOC in modalità test su range di pagine limitato salvo forzatura via env, introdotti casi di test dedicati per convalida completa e parziale |
| 2026-01-05 | 0.55 | Resa la validazione TOC robusta a differenze di formattazione stilistica confrontando titoli normalizzati tra PDF e Markdown |
| 2026-01-05 | 0.56 | Aggiunto requisito di inserire nel Markdown link ai file delle tabelle salvati in tables/ (Markdown e CSV) |
| 2026-01-05 | 0.57 | Gestione mismatch TOC con log di dettaglio (verbose/debug) e prosecuzione della pipeline con errore finale |
| 2026-01-05 | 0.58 | Centralizzazione della costruzione del manifest in una funzione dedicata usata da processing e post-processing subito dopo la validazione TOC |
| 2026-01-05 | 0.59 | Aggiunto requisito per normalizzare il Markdown post-processing rimuovendo indici duplicati e riallineando le intestazioni/TOC prima della ricostruzione del manifest |
| 2026-01-05 | 0.61 | Raffinati CORE-DES-051 e CORE-REQ-013 per spiegare che `--post-processing-only` verifica solo Markdown e backup prima di ricostruire il manifest |
| 2026-01-05 | 0.60 | Chiarito che `--post-processing-only` richiede solo il Markdown e il backup `.processing.md` esistenti perché il manifest viene ricostruito dalla pipeline di post-processing |
| 2026-01-05 | 0.62 | Introdotta la funzione dedicata `normalize_markdown_file` invocata dal post-processing per riallineare intestazioni al TOC del PDF e rigenerare il `.toc` prima della validazione |
| 2026-01-05 | 0.63 | Rifattorizzata la costruzione del contesto nel manifest con funzione `find_context`, uso del `.toc` generato e allineamento tra `context` e `context_path` |
| 2026-01-05 | 0.64 | Introdotta pulizia delle intestazioni Markdown non presenti nella TOC del PDF convertendole in testo maiuscolo in grassetto |
| 2026-01-05 | 0.65 | Aggiunta numerazione di riferimento incrociato nel manifest con ID univoci e relazioni parent/prev/next tra TOC, tabelle e immagini |
| 2026-01-05 | 0.66 | Il requisito di remove-small-images deve eliminare riferimenti da manifest/Markdown senza cancellare i file PNG su disco |
| 2026-01-05 | 0.67 | CORE-DES-069 richiede che `prev_id` segua l'ordine di lettura inverse di `next_id` invece di puntare al `parent_id` |
| 2026-01-05 | 0.68 | Introduzione di `normalize_markdown_format` per standardizzare i tag HTML `<br>` in newline prima della rimozione dell'indice duplicato nella normalizzazione Markdown |
| 2026-01-05 | 0.69 | Rinomina il campo `page` dei nodi `toc_tree` in `pdf_source_page` e aggiunge il requisito/test che ne verifica la presenza nel manifest |
| 2026-01-05 | 0.70 | Aggiunge il requisito di marcare righe e byte del Markdown per ogni nodo TOC, tabella e immagine nel manifest |


## 1. Introduzione
Questo documento definisce i requisiti del progetto pdf2tree, un tool CLI che converte file PDF in una struttura di cartelle con contenuto Markdown e asset estratti.

### 1.1 Regole del documento
Questo documento deve sempre seguire queste regole:
- Questo documento deve essere scritto in italiano.
- Formattare i requisiti come lista puntata, utilizzando le parole chiave "deve" o "devono" per indicare azioni obbligatorie.
- Ogni ID requisito (per esempio, **CORE-PRJ-001**, **CORE-PRJ-002**,.. **CTN-001**, **CORE-CTN-002**,.. **DES-001**, **CORE-DES-002**,.. **REQ-001**, **CORE-REQ-002**,..) deve essere unico; non assegnare lo stesso ID a requisiti diversi.
- Ogni ID requisito deve iniziare con la stringa che identifica il gruppo di requisiti:
  * I requisiti di funzione di progetto iniziano con **CORE-PRJ-**
  * I requisiti di vincolo di progetto iniziano con **CORE-CTN-**
  * I requisiti di progettazione e implementazione iniziano con **CORE-DES-**
  * I requisiti di funzione iniziano con **CORE-REQ-**
- Ogni requisito deve essere identificabile, verificabile e testabile.
- A ogni modifica di questo documento, aggiornare il numero di versione e aggiungere una nuova riga in fondo allo storico revisioni.
- A ogni modifica di questo documento, aggiornare la data del documento e la data contentuta negli header di questo documento con la data odierna.
- 
### 1.2 Scopo del progetto
Il progetto converte un PDF in una gerarchia di cartelle e file Markdown basata sull'indice del documento, estraendo testo e immagini (incluse figure vettoriali) e producendo un manifest JSON dell'output.

## 2. Requisiti di progetto

### 2.1 Funzioni di progetto
- **CORE-PRJ-001**: Il progetto deve convertire un PDF in una struttura di cartelle e file Markdown basata sull'indice del documento.
- **CORE-PRJ-002**: Il progetto deve estrarre testo e immagini raster dal PDF e salvarle come asset collegati al Markdown.
- **CORE-PRJ-003**: Il progetto deve supportare l'estrazione opzionale di diagrammi vettoriali dal PDF.
- **CORE-PRJ-004**: Il progetto deve offrire un'interfaccia CLI per l'esecuzione locale e la gestione delle opzioni di conversione.
- **CORE-PRJ-005**: Il progetto deve generare un unico file Markdown che rappresenti l'intero documento convertito secondo la TOC.

### 2.2 Vincoli di progetto
- **CORE-CTN-001**: Il progetto deve essere eseguibile con Python >= 3.11.
- **CORE-CTN-002**: Il progetto deve dipendere da PyMuPDF con versione minima 1.26.6.
- **CORE-CTN-003**: Il progetto deve richiedere una directory di output non esistente o gestita in modalita' ripresa.
- **CORE-CTN-004**: La CLI deve terminare con errore se la directory di output esiste ed è non vuota al momento dell'avvio.
- **CORE-CTN-005**: La CLI deve terminare con errore se la libreria OpenCV (`cv2`) non è disponibile nell'ambiente di esecuzione.


### 2.3 Componenti, librerie e struttura

Struttura del progetto (esclusi i percorsi che iniziano con punto):
```
.
├── CHANGELOG.md
├── LICENSE
├── README.md
├── TODO.md
├── docs
│   ├── 1_pdf2tree_requirements.md
│   └── 2_pdf_sample_requirements.md
├── pdf2tree.sh
├── pdf_sample
│   ├── pdf_sample.pdf
│   ├── pdf_sample.tex
│   └── pdf_sample.toc
├── pyproject.toml
├── requirements.txt
├── src
│   └── pdf2tree
│       ├── __init__.py
│       ├── __main__.py
│       ├── core.py
│       └── latex.py
├── tech
├── temp
├── tests
│   └── test_cli_venv.py
└── venv.sh
```

Organizzazione dei componenti e relazioni:
- Il package `pdf2tree` contiene l'implementazione principale della CLI; la funzione `main` orchestra lettura PDF, analisi indice, estrazione contenuti e scrittura dell'output.
- Il package `pdf2tree` fa da wrapper e re-esporta `main` per consentire l'esecuzione come modulo o entrypoint CLI.

Interfaccia testuale/GUI:
- Interfaccia testuale CLI con opzioni di debug/verbose e messaggi di stato a console.
- Nessuna GUI interattiva.

Componenti e librerie utilizzati:
- Librerie esterne: `pymupdf`, `pymupdf4llm`, `pymupdf_layout`.
- Librerie di test: `pytest`.
- Standard library Python: `argparse`, `json`, `pathlib`, `tempfile`, `signal`, `shutil`, `os`, `sys`, `re`.
- Script di supporto: `pdf2tree.sh` per creare un virtualenv ed eseguire la CLI.

## 3. Requisiti

### 3.1 Progettazione e implementazione
- **CORE-DES-001**: Il package `pdf2tree` deve esporre la funzione `main` come wrapper di `pdf2tree.core`.
- **CORE-DES-002**: La funzione `main` deve implementare il parsing degli argomenti CLI per `--from-file` e `--to-dir` come parametri obbligatori.
- **CORE-DES-003**: La CLI deve supportare le opzioni `--verbose` e `--debug` per controllare il livello di log e l'output di artefatti aggiuntivi.
- **CORE-DES-004**: La conversione deve generare un front matter YAML con metadati del PDF e, se disponibile, includere la TOC Markdown in testa al documento.
- **CORE-DES-005**: L'inizializzazione deve importare `pymupdf.layout` prima di `pymupdf4llm`, loggando il fallback legacy in caso di assenza.
- **CORE-DES-006**: Il codice deve salvare artefatti di debug (chunk JSON, mappa dei Form XObject) nella sottocartella `debug` quando l'opzione `--debug` è attivata.
- **CORE-DES-007**: Dopo ogni nuova implementazione o modifica al codice sorgente deve essere eseguito il test automatico basato su `pdf_sample` per verificare la conversione con e senza TOC.
- **CORE-DES-008**: La conversione deve generare un file manifest JSON nella directory di output, strutturato con le sezioni `markdown` (percorso del file Markdown unico e struttura ad albero completa della TOC), `tables` (una voce per ogni tabella estratta con il campo `pdf_source_page` e il contesto gerarchico) e `images` (una voce per ogni immagine generata con il campo `pdf_source_page` e il contesto gerarchico). Ogni voce deve includere `context_path`, l'array dei titoli TOC attraversati, e `context`, la stessa sequenza unita con il separatore ` > ` (senza prefissi o etichette aggiuntive).
- **CORE-DES-009**: La CLI deve fornire le opzioni facoltative `--header` e `--footer` in millimetri (default 0 mm) per escludere rispettivamente l'area superiore e inferiore della pagina da ogni forma di estrazione (testo, formule, immagini, tabelle, processamento immagini) senza interferire con gli altri processi di estrazione; se un margine è impostato a 0 mm, il crop corrispondente non deve essere applicato.
- **CORE-DES-010**: La CLI deve offrire l'opzione `--enable-form-xobject` per attivare l'individuazione, la rasterizzazione e l'inserimento dei Form XObject nelle immagini generate e nel Markdown; in assenza del flag tale funzionalità deve rimanere disattivata.
- **CORE-DES-011**: La CLI deve offrire l'opzione `--enable-vector-images` per abilitare l'estrazione dei diagrammi vettoriali, mantenendola disattivata per impostazione predefinita e applicandola dopo l'eventuale esclusione di header e footer.
- **CORE-DES-012**: Quando `--enable-vector-images` è attivo, l'estrazione dei diagrammi vettoriali deve filtrare tracciati per posizione, dimensioni minime e densità prima del clustering, salvare i ritagli con suffisso `-vector` nella sottocartella `images/` e inserirli nel Markdown unico.
- **CORE-DES-013**: Il manifest JSON deve includere per ogni immagine i campi obbligatori `source` (valori ammessi: `pymupdf` di default, `form-xobject` per immagini dai Form XObject, `vector-image` per immagini vettoriali) e `type` con valore di default `image`, poi aggiornato in seguito.
- **CORE-DES-014**: Tutti i messaggi a console (info, warning, error) devono essere emessi in lingua inglese.
- **CORE-DES-015**: Con l'opzione `--verbose` la CLI deve mostrare l'avanzamento dell'elaborazione (per esempio progress bar o contatore pagine), indicando la procedura in corso e gli asset o immagini processati per dare evidenza dello stato.
- **CORE-DES-016**: La modalità `--verbose` deve riportare informazioni operative di dettaglio, incluse operazioni su file (spostamenti o cancellazioni) e passaggi critici come estrazione immagini vettoriali, Form XObject ed esportazione tabelle, senza richiedere l'attivazione di `--debug`.
- **CORE-DES-017**: Il codice Python deve contenere commenti in lingua italiana che descrivano le funzionalità principali e le funzioni complesse responsabili del comportamento di alto livello del programma.
- **CORE-DES-018**: Dopo la validazione dei parametri di input e prima di avviare qualsiasi fase di elaborazione, la CLI deve stampare a console il banner `*** pdf2tree (<versione>) ***`.
- **CORE-DES-044**: Subito dopo la stampa del banner iniziale, la CLI deve mostrare in inglese un riepilogo compatto dei parametri validati, indicando ON/OFF per i flag booleani e i valori effettivi per i parametri numerici o stringa, indipendentemente dal livello di log.
- **CORE-DES-045**: La CLI deve offrire le opzioni `--prompts <file>` per caricare i prompt Gemini da un file di configurazione e `--write-prompts <file>` per generare e salvare i prompt di default; il comando `--write-prompts` deve scrivere il file (creando le cartelle se mancanti) e terminare l'esecuzione senza avviare ulteriori fasi.
- **CORE-DES-046**: Il file di prompt deve essere in formato JSON e contenere le chiavi obbligatorie `prompt_equation`, `prompt_non_equation`, `prompt_uncertain` con valori stringa non vuoti; se il caricamento o la validazione del file passato con `--prompts` fallisce, la CLI deve terminare con errore di argomenti non validi.
- **CORE-DES-047**: Le annotazioni Gemini devono usare prompt configurabili secondo queste regole: se la fase Pix2Tex è stata eseguita, le immagini non classificate come equazioni usano `prompt_non_equation` e le equazioni (abilitate con `--enable-annotate-equations`) usano `prompt_equation`; se la Pix2Tex non è stata eseguita, tutte le annotazioni devono usare `prompt_uncertain`.
- **CORE-DES-048**: I prompt di default per `prompt_equation`, `prompt_non_equation` e `prompt_uncertain` devono corrispondere esattamente ai template English RAG forniti (sezioni e testi inclusi) e risultare hardcodati nel sorgente.
- **CORE-DES-049**: La pipeline di processing deve essere incapsulata nella funzione `run_processing_pipeline`, che gestisce l'intero flusso di conversione PDF→Markdown e deve chiudersi copiando il file Markdown finale in un backup con estensione `.processing.md` nella stessa directory di output come ultima attività della pipeline.
- **CORE-DES-050**: La pipeline di post-processing deve essere incapsulata nella funzione `run_post_processing_pipeline`, che deve iniziare copiando il backup `.processing.md` sul file `.md` e, come seconda attività, ricreare e scrivere su disco il manifest JSON tramite una funzione dedicata che utilizza il PDF originale e il Markdown ripristinato prima di eseguire `remove_small_images_phase`, Pix2Tex e annotazioni.
- **CORE-DES-051**: Non devono essere eseguite attività intermedie tra l'ultima operazione di `run_processing_pipeline` (copia del `.processing.md`) e la prima operazione di `run_post_processing_pipeline` (ripristino del `.md`); l'opzione `--post-processing-only` deve avviare direttamente `run_post_processing_pipeline` senza rieseguire la pipeline di processing, verificando solo la presenza del `.md` e del backup `.processing.md` perché il manifest JSON viene ricostruito come secondo step della pipeline stessa e nessun controllo o test deve assumere la sua esistenza prima dell'avvio della fase di post-processing.
- **CORE-DES-052**: Tutti i commenti e le docstring del codice Python devono essere scritti in italiano e coprire ogni funzione e i blocchi logici critici del flusso di esecuzione; le descrizioni devono essere concise e professionali, con spiegazioni più estese solo dove la logica è complessa. Qualsiasi commento o docstring esistente in lingua diversa deve essere sostituito per rispettare questo vincolo.
- **CORE-DES-053**: Tutti i messaggi, log e output rivolti all'utente devono essere in inglese; in modalità predefinita l'output deve rimanere essenziale; con `--verbose` deve essere mostrata una barra di avanzamento visiva per le attività lunghe oltre a aggiornamenti chiari sul task in corso; le informazioni di basso livello (payload API, oggetti JSON grezzi, stack trace o stato interno) devono essere visibili solo con `--debug`.
- **CORE-DES-054**: La modalità `--post-processing-only` deve richiedere sia `--to-dir` sia `--from-file` e deve terminare con errore se il PDF di origine non esiste o non è leggibile, poiché il PDF è necessario per completare il post-processing.
- **CORE-DES-055**: La generazione del file Markdown deve essere organizzata come pipeline di step chiaramente identificati, ciascuno implementato in una funzione dedicata (ad esempio preparazione output, caricamento PDF/metadata, estrazione Markdown/asset, processamento pagine, scrittura artefatti/backup), mantenendo il comportamento corrente ma rendendo esplicita la separazione delle fasi.
- **CORE-DES-019**: Ogni fase principale del processing deve iniziare con una stampa del tipo `\n--- <descrizione della fase che viene eseguita> ---` e terminare con `done.`, separando chiaramente le fasi di preparazione output, caricamento PDF/metadata, estrazione contenuti, processamento pagine e scrittura artefatti finali.
- **CORE-DES-020**: La CLI deve offrire le opzioni `--post-processing` e `--post-processing-only`, convalidando le combinazioni tra loro e richiedendo sempre `--to-dir` e `--from-file`; se `--post-processing` non è fornito la pipeline deve terminare dopo la scrittura del Markdown senza avviare altre fasi, mentre `--post-processing-only` deve saltare la conversione PDF→Markdown e operare sugli artefatti già presenti, terminando con errore se manca il file Markdown.
- **CORE-DES-021**: Il post-processing deve essere organizzato come pipeline estendibile di fasi sequenziali, configurabile tramite flag, includendo `--enable-pic2tex` per attivare la fase Pix2Tex (disabilitata di default), `--disable-pic2tex` per forzarne l'esclusione e `--equation-min-len` (default 5, valore minimo >0) per impostare la soglia di classificazione.
- **CORE-DES-022**: Il post-processing deve includere una fase opzionale di annotazione eseguita dopo Pix2Tex; l'annotazione delle immagini è abilitata per impostazione predefinita e può essere disattivata con `--disable-annotate-images`, mentre l'annotazione delle equazioni si attiva esplicitamente con `--enable-annotate-equations`.
- **CORE-DES-023**: Quando la fase di annotazione è richiesta, la CLI deve verificare la presenza di una chiave valida per l'API Gemini e terminare con errore prima dell'esecuzione se la chiave manca o è invalida.
- **CORE-DES-024**: La fase di annotazione deve interrogare l'API Google Gemini per ogni immagine ed eventuale equation nel manifest, inserendo il testo di annotazione immediatamente sotto l'immagine nel Markdown (separato da una riga vuota e marcato in modo riconoscibile) e aggiungendo il campo `annotation` nel manifest.
- **CORE-DES-025**: In modalità `--verbose` la fase di annotazione deve riportare per ogni immagine/equation il nome del file e l'esito della chiamata API; con `--debug` deve anche mostrare il contenuto dell'annotazione ottenuta.
- **CORE-DES-026**: Il progetto deve includere un modulo `latex.py` con un framework di validazione delle formule LaTeX a tre stadi sequenziali (bilanciamento delimitatori, parsing con `pylatexenc`, rendering con `matplotlib`) che restituisce esito positivo solo al superamento di tutti gli stadi; errori o eccezioni devono interrompere la pipeline e marcare la formula come non valida.
- **CORE-DES-027**: Durante il post-processing Pix2Tex la modalità `--verbose` deve stampare per ogni immagine il riferimento posizionale nel manifest, il nome del file e l'esito della validazione LaTeX; con `--debug` deve essere mostrato anche l'output integrale restituito da Pix2Tex.
- **CORE-DES-028**: L'inizializzazione del modello Gemini deve avvenire esclusivamente tramite l'SDK python-genai (`google.genai`), senza fallback o compatibilità con `google.generativeai`; in assenza di supporto nell'SDK selezionato l'esecuzione deve fallire con errore esplicito.
- **CORE-DES-029**: Le richieste di annotazione verso l'API Gemini devono includere istruzioni esplicite a produrre una risposta in Markdown adatta a flussi RAG, descrivendo in modo completo formule/equazioni, grafici matematici, processi, flussi/flow chart, diagrammi temporali, mappe mentali/grafi e tabelle.
- **CORE-DES-030**: Per le annotazioni delle equazioni (`--enable-annotate-equations`), la richiesta verso l'API Gemini deve richiedere esplicitamente di includere nella risposta la formula in formato LaTeX resa in Markdown/MathJax.
- **CORE-DES-031**: L'inserimento nel Markdown delle formule LaTeX validate da Pix2Tex deve racchiudere la formula tra le righe `----- Start of equation: <nome immagine> -----` e `----- End of equation: <nome immagine> -----`, usando il nome del file immagine associato.
- **CORE-DES-032**: L'inserimento nel Markdown delle annotazioni generate tramite Gemini deve racchiudere il testo tra le righe `----- Start of annotation: <nome immagine> -----` e `----- End of annotation: <nome immagine> -----`, sostituendo il precedente formato `> [annotation:<nome>]:`.
- **CORE-DES-033**: Il framework di validazione LaTeX deve includere uno stadio di compatibilità Markdown/MathJax successivo al parsing con `pylatexenc` e precedente al rendering con `matplotlib`, che rifiuti formule contenenti costrutti TeX di basso livello (ad esempio `\atop`, `\atopwithdelims`, `\overwithdelims`) o gruppi vuoti che non producono output (ad esempio `\mathrm{}` o `{}`), considerandole non valide ai fini del post-processing Pix2Tex.
- **CORE-DES-034**: Il manifest delle immagini e delle tabelle deve includere l'array `context_path` con i titoli TOC che portano all'asset, ordinati dal livello più alto (capitolo) a quello più basso (sottosezione); il campo `context` deve essere la stessa sequenza serializzata con ` > `.
- **CORE-DES-073**: Il manifest `markdown.toc_tree` deve esporre lo stesso riferimento alla pagina PDF tramite il campo `pdf_source_page` per ciascun nodo (sostituendo il precedente `page`) in modo che la struttura serializzata mantenga la provenienza originale dei titoli.
- **CORE-DES-066**: La costruzione del contesto per immagini, figure e tabelle deve essere centralizzata in una funzione dedicata `find_context` che utilizza il file `.toc` generato per risalire ai titoli TOC che precedono l'asset; in assenza di match deve ripiegare sulla TOC del PDF e sul numero di pagina inferito. `context` e `context_path` devono restare coerenti.
- **CORE-DES-035**: La CLI deve accettare il flag `--n-pages` che consente di limitare il numero massimo di pagine consecutive, a partire dalla prima, da processare tramite PyMuPDF/PyMuPDF4LLM; il valore deve essere un intero positivo e deve essere validato restituendo errore in caso contrario.
- **CORE-DES-036**: Durante l'esecuzione degli unit test la pipeline deve attivare automaticamente una modalità di test (abilitata tramite `PDF2TREE_TEST_MODE` o rilevando `PYTEST_CURRENT_TEST`) che impedisca chiamate reali all'API Gemini e produca una risposta proforma deterministica riutilizzata per aggiornare Markdown e manifest con le annotazioni.
- **CORE-DES-037**: In modalità di test la fase Pix2Tex non deve inizializzare il modello reale, ma restituire una formula LaTeX predefinita (sovrascrivibile via variabile d'ambiente) che superi il framework `latex.py` e che venga usata per contrassegnare e inserire le equation nel Markdown e nel manifest.
- **CORE-DES-038**: Gli unit test ufficiali devono invocare la CLI con l'opzione `--n-pages 1` per limitare l'elaborazione alla sola prima pagina, garantendo tempi ridotti e un output deterministico in modalità di test.
- **CORE-DES-039**: La CLI deve offrire l'opzione `--start-page` (intero >=1) che, usata da sola o assieme a `--n-pages`, definisce l'intervallo di pagine consecutive passato a PyMuPDF/PyMuPDF4LLM; il codice deve validare il range rispetto al numero totale di pagine del PDF e mantenere nei log, nel Markdown i numeri di pagina originali del documento.
- **CORE-DES-040**: Gli unit test devono operare sempre su una singola pagina combinando `--n-pages 1` con l'opzione `--start-page`, includendo casi con `--start-page 2` per validare l'elaborazione di pagine interne del PDF di esempio.
- **CORE-DES-041**: L'intera suite di unit test deve generare i PDF di esempio con una singola sequenza di compilazione LaTeX condivisa (prima passata senza TOC, seconda con TOC) riutilizzando i file prodotti, senza ricompilazioni supplementari; il test sulla mancanza di TOC deve basarsi sul PDF privo di indice ottenuto dalla prima compilazione.
- **CORE-DES-042**: Il codice deve evidenziare con messaggi di log e commenti il punto in cui l'elaborazione si arresta per mancanza del flag `--post-processing` e il punto da cui riprende quando viene usato `--post-processing-only`, in modo che il comportamento sia immediatamente distinguibile durante l'esecuzione.
- **CORE-DES-043**: Nella pipeline di post-processing esiste una fase `remove-small-images`, abilitata di default (disattivabile con `--disable-remove-small-images`), che scansiona le immagini elencate nel manifest durante `--post-processing` e `--post-processing-only`, misura le dimensioni effettive delle immagini e, quando entrambe le dimensioni risultano inferiori alle soglie configurate, rimuove le voci corrispondenti dal manifest e i riferimenti dal Markdown (inclusi i blocchi Pix2Tex e le annotazioni) indicando in modalità `--verbose` l'esito (KEEP/REMOVE) e le dimensioni, ma lascia i file PNG originali sul disco affinché siano riutilizzabili o cancellabili manualmente.
- **CORE-DES-056**: La creazione del manifest nel post-processing deve essere suddivisa in step distinti ciascuno implementato in una funzione dedicata e orchestrato da `run_post_processing_pipeline`, includendo almeno: ripristino del Markdown dal backup `.processing.md`, normalizzazione tramite `normalize_markdown_file` che rimuove l'indice duplicato, riallinea le intestazioni alla TOC del PDF e rigenera il file `.toc` in formato Markdown, validazione della `.toc` contro la TOC del PDF, costruzione del manifest e scrittura su disco.
- **CORE-DES-057**: La generazione del file `.toc` deve estrarre dal Markdown tutte le intestazioni (livelli `#`, `##`, `###`, ...), tutti i riferimenti a immagini in `images/` e tutti i riferimenti a tabelle in `tables/` applicando una lista di espressioni regolari, scrivendo il risultato in un file `.toc` (formato Markdown) nella cartella di output.
- **CORE-DES-058**: La TOC estratta dal file `.toc` deve essere validata confrontandola con la TOC del PDF (`doc.get_toc()`); il confronto deve avvenire su numero di voci e titoli normalizzati (rimozione di numerazione, enfasi, spaziatura e virgolette tipografiche), ignorando le differenze di livello. Se il conteggio o i titoli normalizzati non coincidono la pipeline di post-processing deve comunque proseguire tutte le fasi successive (costruzione manifest, remove-small-images, Pix2Tex, annotazioni).
- **CORE-DES-059**: La fase `remove-small-images` deve essere eseguita solo dopo il completamento della pipeline di normalizzazione del Markdown (inclusi rimozione indice duplicato, riallineamento intestazioni alla TOC del PDF e rigenerazione `.toc`) e di creazione/validazione del manifest.
- **CORE-DES-060**: In modalità test, quando viene usato un intervallo di pagine limitato (`--n-pages` o `--start-page` diverso da 1), la validazione della TOC deve essere disabilitata automaticamente; l'ambiente `PDF2TREE_FORCE_TOC_VALIDATION=1` deve riattivarla esplicitamente sia per esecuzioni su PDF completo (che devono passare) sia per esecuzioni parziali (che devono fallire segnalando la mancata corrispondenza).
- **CORE-DES-061**: Quando la validazione TOC rileva una non corrispondenza tra PDF e `.toc` Markdown segnala un errore senza interrompere l'esecuzione.
- **CORE-DES-062**: In caso di mismatch TOC, la modalità `--verbose` deve stampare l'intera TOC confrontata (PDF vs Markdown) evidenziando per ogni voce l'esito dei controlli (OK/FAIL) e indicando eventuali differenze di conteggio.
- **CORE-DES-063**: In caso di mismatch TOC, la modalità `--debug` deve stampare informazioni tecniche di diagnosi (liste normalizzate, indici dei primi mismatch, conteggi PDF/Markdown) oltre al riepilogo del mismatch già visibile senza debug.
- **CORE-DES-064**: La costruzione del manifest deve essere centralizzata in una funzione dedicata invocata da `run_post_processing_pipeline` immediatamente dopo la validazione della TOC, evitando duplicazioni e garantendo la conservazione dei metadati (`source`, `type`, `equation`, `annotation`).
- **CORE-DES-065**: La funzione `normalize_markdown_file` deve essere invocata da `run_post_processing_pipeline` subito dopo il ripristino del Markdown dal backup `.processing.md`, utilizzare la TOC nativa del PDF per normalizzare le intestazioni del Markdown, rigenerare il file `.toc` coerente e restituire il Markdown normalizzato insieme alle intestazioni da validare; la funzione deve anche salvare su disco sia il Markdown normalizzato sia il `.toc` generato.
- **CORE-DES-067**: La funzione `normalize_markdown_file` deve eseguire un passaggio `clean_markdown_headings` che individua tutte le intestazioni Markdown (`#`, `##`, `###`, etc.) non presenti nella TOC del PDF (inclusa "Index") e le converte in testo maiuscolo in grassetto, rimuovendo il qualificatore di titolo; le intestazioni presenti nella TOC devono rimanere inalterate.
- **CORE-DES-072**: La funzione `normalize_markdown_file` deve invocare `normalize_markdown_format` prima della rimozione dell'indice duplicato; `normalize_markdown_format` deve convertire i tag HTML `<br>`, `<br/>` e `<br />` in newline per garantire un Markdown coerente nella normalizzazione.
- **CORE-DES-068**: La pipeline di post-processing deve assegnare ID univoci e stabili a tutte le entità del manifest (nodi TOC, tabelle, immagini) e salvarli nel JSON finale.
- **CORE-DES-069**: Ogni nodo della TOC serializzata nel manifest deve includere `parent_id`, `next_id` e `prev_id`: `parent_id` punta al nodo padre se presente; `next_id` punta al nodo visitato immediatamente successivo nell'ordine di lettura DFS (fratello o primo discendente del successivo parallelo); `prev_id` punta al nodo visitato immediatamente precedente nell'ordine di lettura DFS, cioè l'ultimo discendente dell'elemento precedente, senza riaffidarsi al `parent_id`.
- **CORE-DES-070**: Ogni tabella nel manifest deve includere `id`, `parent_id`, `next_id` e `prev_id`; `parent_id` punta al nodo TOC della sezione Markdown che contiene la tabella; `next_id` e `prev_id` puntano solo alle tabelle sorelle se esistono (altrimenti sono omessi); il nodo TOC padre deve esporre un array `tables` con gli ID delle tabelle figlie.
- **CORE-DES-071**: Ogni immagine nel manifest deve includere `id`, `parent_id`, `next_id` e `prev_id`; `parent_id` punta al nodo TOC della sezione Markdown che contiene l'immagine; `next_id` e `prev_id` puntano solo alle immagini sorelle se esistono (altrimenti sono omessi); il nodo TOC padre deve esporre un array `images` con gli ID delle immagini figlie.
- **CORE-DES-074**: Il manifest JSON deve annotare ogni nodo `markdown.toc_tree`, ogni voce `tables` e ogni voce `images` con i campi `start_line`, `end_line`, `start_byte` e `end_byte` che delimitano il blocco Markdown corrispondente, seguendo l'ordine di lettura del TOC (saltando l'indice duplicato iniziale), includendo nei blocchi delle tabelle i riferimenti `[Markdown](tables/...)` e `[CSV](tables/...)` e nelle immagini le righe `----- Start/End of equation -----`, il link all'immagine e i blocchi `----- Start/End of annotation -----` quando presenti per coprire l'intera sezione.

### 3.2 Funzioni
- **CORE-REQ-001**: Il tool deve terminare con errore se il file PDF di input non esiste.
- **CORE-REQ-002**: Il tool deve generare un unico file Markdown nella cartella di output che contenga l'intero contenuto del PDF strutturato secondo il TOC.
- **CORE-REQ-003**: Il file Markdown unico deve includere l'intero testo estratto e i link alle immagini.
- **CORE-REQ-004**: Il tool deve salvare tutte le immagini estratte (raster e vettoriali) nella sotto cartella `images` nella cartella del file Markdown, aggiornando i link di conseguenza.
- **CORE-REQ-005**: Quando viene passato `--enable-form-xobject`, il tool deve individuare e rasterizzare i Form XObject vettoriali per aggiungerli come immagini nella sottocartella `images` e inserirli nel Markdown preservando eventuali etichette testuali; in assenza del flag la funzionalità deve restare inattiva.
- **CORE-REQ-006**: Il tool deve esportare le tabelle rilevate tramite fallback nella sottocartella `tables` e inserirle nel Markdown quando non sono già presenti.
- **CORE-REQ-007**: Il tool deve normalizzare e spostare nel percorso `images/` anche i file immagine generati con naming non standard, aggiornando i link nel Markdown.
- **CORE-REQ-008**: La CLI deve terminare con errore se il PDF di input è privo di indice/TOC.
- **CORE-REQ-009**: Il tool deve creare nella directory di output un file JSON con il nome del PDF di ingresso che elenca tutti i file prodotti su disco (Markdown unico, file generati in `tables/` raggruppati per tabella, file generati in `images/`) includendo percorsi relativi e metadati.
- **CORE-REQ-010**: In assenza dell'opzione `--enable-form-xobject`, la conversione deve saltare l'esportazione dei Form XObject: non devono essere creati file immagine dedicati né inseriti riferimenti nel Markdown.
- **CORE-REQ-011**: Quando viene passato `--enable-vector-images`, il tool deve estrarre diagrammi vettoriali per ogni pagina, salvarli come PNG con suffisso `-vector` sotto `images/`, includerli nel Markdown.
- **CORE-REQ-012**: Quando `--post-processing` o `--post-processing-only` vengono eseguiti insieme al flag `--enable-pic2tex` (e senza `--disable-pic2tex`), il tool deve eseguire il riconoscimento LaTeX sulle immagini elencate nel manifest e, se l'output supera la soglia `equation-min-len`, aggiornare il campo `type` a `equation`, inserire la formula LaTeX nel Markdown immediatamente prima dell'immagine e aggiornare il manifest di conseguenza.
- **CORE-REQ-013**: Con l'opzione `--post-processing-only` il tool deve verificare la presenza del file Markdown e del backup `.md.processing.md` nella cartella di output, ripristinare il Markdown dal backup prima di avviare la pipeline direttamente dalla fase di post-processing (saltando la conversione) e riscrivere i file aggiornati, terminando con errore in caso di assenza o dati non validi; il manifest JSON non deve essere richiesto a priori perché viene generato durante `run_post_processing_pipeline` subito dopo il ripristino del Markdown e i test automatizzati devono fornire esclusivamente il Markdown e il backup `.processing.md`, controllando il manifest soltanto dopo l'esecuzione della pipeline di post-processing-only.
- **CORE-REQ-014**: In modalità `--verbose` il post-processing deve riportare per ogni immagine la fase di analisi, se è stata identificata come equation e la formula LaTeX ricavata.
- **CORE-REQ-015**: Nel post-processing Pix2Tex attivato tramite `--enable-pic2tex`, solo le formule che superano la soglia `equation-min-len` e vengono validate dal framework LaTeX devono aggiornare il Markdown e il manifest, impostando `type` a `equation` e aggiungendo il campo `equation` con la formula; in caso di validazione fallita la voce deve restare `image` senza modifiche al Markdown.
- **CORE-REQ-016**: Quando viene passato `--n-pages`, il tool deve elaborare esclusivamente le prime *N* pagine del PDF e produrre artefatti (Markdown, immagini, tabelle, logging) coerenti con tale sottoinsieme, aggiornando contatori e metadati alla nuova durata.
- **CORE-REQ-017**: La CLI deve offrire i parametri `--min-size-x` e `--min-size-y` (valori interi positivi, default 100 px) da applicare alla fase `remove-small-images`; l’immagine viene rimossa solo se entrambe le dimensioni sono inferiori alle soglie impostate e, in tal caso, l’eliminazione deve riflettersi sia nel manifest sia nel Markdown generato o ricaricato in modalità `--post-processing-only`.
- **CORE-REQ-018**: Quando viene passato `--prompts <file>`, il tool deve caricare i tre prompt richiesti e, in caso di file mancante o privo di una delle chiavi obbligatorie, terminare con errore senza avviare la conversione o il post-processing.
- **CORE-REQ-019**: Il Markdown generato deve includere, per ogni tabella estratta e salvata in `tables/`, link espliciti ai file prodotti (almeno il `.md` e il `.csv` quando presenti) in modo che l’utente possa raggiungerli dal punto del documento in cui la tabella è referenziata.
- **CORE-REQ-020**: Il post-processing deve normalizzare il Markdown ripristinato eliminando eventuali indici duplicati importati dal PDF, uniformare tutte le intestazioni ai livelli `#`/`##`/`###` corrispondenti alla TOC nativa del PDF tramite `normalize_markdown_file` e rigenerare il file `.toc` con link coerenti verso le nuove intestazioni prima di procedere con la ricostruzione del manifest.
- **CORE-REQ-021**: Gli unit test devono verificare che ogni nodo presente in `markdown.toc_tree` nel manifest includa il campo `pdf_source_page` e che il campo `page` non sia più presente.

## 4. Verifica
Sintesi dei test presenti:
- Il test CLI compila un PDF di esempio tramite `pdflatex` quando disponibile.
- Il test esegue `pdf2tree.core:main` in un virtualenv locale e verifica che il processo termini con successo.
- Il test verifica la creazione del file Markdown unico nella directory di output.
- Eseguo python usando sempre il virtual environment presente in `.venv`
