---
title: "Requisiti pdf2tree"
description: Specifica dei requisiti software
date: "2026-01-02"
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
**Versione**: 0.1
**Autore**: Codex  
**Data**: 2026-01-02

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
- A ogni modifica di questo documento, aggiornare il numero di versione e aggiungere una nuova riga nello storico revisioni.

### 1.2 Scopo del progetto
Il progetto converte un PDF in una gerarchia di cartelle e file Markdown basata sull'indice del documento, estraendo testo e immagini (incluse figure vettoriali) e producendo un manifest JSON dell'output.

## 2. Requisiti di progetto

### 2.1 Funzioni di progetto
- **CORE-PRJ-001**: Il progetto deve convertire un PDF in una struttura di cartelle e file Markdown basata sull'indice del documento.
- **CORE-PRJ-002**: Il progetto deve estrarre testo e immagini raster dal PDF e salvarle come asset collegati al Markdown.
- **CORE-PRJ-003**: Il progetto deve supportare l'estrazione opzionale di diagrammi vettoriali dal PDF.
- **CORE-PRJ-004**: Il progetto deve offrire un'interfaccia CLI per l'esecuzione locale e la gestione delle opzioni di conversione.
- **CORE-PRJ-005**: Il progetto deve generare un file `project_manifest.json` che riassuma i capitoli convertiti.

### 2.2 Vincoli di progetto
- **CORE-CTN-001**: Il progetto deve essere eseguibile con Python >= 3.11.
- **CORE-CTN-002**: Il progetto deve dipendere da PyMuPDF con versione minima 1.26.6.
- **CORE-CTN-003**: Il progetto deve richiedere una directory di output non esistente o gestita in modalita' ripresa.
- **CORE-CTN-004**: Il progetto deve limitare l'estrazione testo ignorando intestazioni e pie' di pagina tramite margini configurati.

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
│       └── core.py
├── tech
├── temp
├── tests
│   └── test_cli_venv.py
└── venv.sh
```

Organizzazione dei componenti e relazioni:
- Il package `pdf2tree` contiene l'implementazione principale della CLI; la funzione `main` orchestra lettura PDF, analisi indice, estrazione contenuti e scrittura dell'output.
- Il package `pdf2tree` fa da wrapper e re-esporta `main` per consentire l'esecuzione come modulo o entrypoint CLI.
- `process_chapter_chunks` delega l'estrazione di testo/immagini a `pymupdf4llm` e integra l'estrazione vettoriale tramite `get_smart_vector_crop`.
- La gerarchia di cartelle di output viene determinata dalla struttura dell'indice (TOC) del PDF e mantenuta tramite uno stack di percorsi.

Ottimizzazioni e miglioramenti prestazionali presenti:
- Modalita' ripresa che evita di rielaborare pagine gia' processate tramite `progress_state.json` e controllo di file esistenti.
- Pre-filtraggio dei disegni vettoriali e clustering con soglie per ridurre falsi positivi e operazioni superflue.
- Scrittura atomica del file di progresso per ridurre il rischio di corruzione in caso di interruzione.

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
- **CORE-DES-003**: La funzione `main` deve gestire la modalita' di ripresa quando la cartella di output esiste senza `--force-restart`.
- **CORE-DES-004**: Il sistema deve salvare lo stato di avanzamento in un file `progress_state.json` usando una scrittura atomica.
- **CORE-DES-005**: L'estrazione di diagrammi vettoriali deve applicare filtri su dimensioni, posizione e densita' dei tracciati prima del clustering.
- **CORE-DES-006**: Il contenuto Markdown dei capitoli deve includere frontmatter con `title` e `context`.

### 3.2 Funzioni
- **CORE-REQ-001**: Il tool deve terminare con errore se il file PDF di input non esiste.
- **CORE-REQ-002**: Il tool deve creare una cartella per ogni capitolo dell'indice PDF, con prefisso numerico a due cifre.
- **CORE-REQ-003**: Il tool deve creare un file `content.md` per ogni capitolo con il testo estratto e i link alle immagini.
- **CORE-REQ-004**: Il tool deve salvare le immagini raster estratte nella sottocartella `assets` del capitolo.
- **CORE-REQ-005**: Il tool deve estrarre diagrammi vettoriali quando l'opzione `--disable-vector-images` non e' attiva.
- **CORE-REQ-006**: Il tool deve creare un file `project_manifest.json` con l'elenco dei capitoli convertiti.
- **CORE-REQ-007**: Il tool deve supportare la modalita' `--dry-run` senza scrivere file su disco.
- **CORE-REQ-008**: Il tool deve permettere l'interruzione soft tramite segnale `SIGINT` salvando lo stato di avanzamento.
- **CORE-REQ-009**: Il tool deve supportare messaggi di debug con l'opzione `--debug`.
- **CORE-REQ-010**: Il tool deve supportare messaggi di dettaglio con l'opzione `--verbose`.

## 4. Verifica
Sintesi dei test presenti:
- Il test CLI compila un PDF di esempio tramite `pdflatex` quando disponibile.
- Il test esegue `pdf2tree.core:main` in un virtualenv locale e verifica che il processo termini con successo.
- Il test verifica la creazione di `project_manifest.json` nella directory di output.
