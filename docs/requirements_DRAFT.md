---
title: "Requisiti pdf_sample"
description: Specifica dei requisiti software
date: "2026-01-02"
author: "Codex"
scope:
  paths:
    - "**/*.tex"
  excludes:
    - ".*/**"
visibility: "draft"
tags: ["markdown", "requirements", "example"]
---

# Requisiti pdf_sample
**Versione**: 0.1
**Autore**: Codex  
**Data**: 2026-01-02

## Indice
<!-- TOC -->
- [Requisiti pdf_sample](#requisiti-pdf_sample)
  - [Indice](#indice)
  - [Storico revisioni](#storico-revisioni)
  - [1. Introduzione](#1-introduzione)
    - [1.1 Regole del documento](#11-regole-del-documento)
    - [1.2 Scopo del progetto](#12-scopo-del-progetto)
  - [2. Requisiti di progetto](#2-requisiti-di-progetto)
    - [2.1 Funzioni di progetto](#21-funzioni-di-progetto)
    - [2.2 Vincoli di progetto](#22-vincoli-di-progetto)
  - [3. Requisiti](#3-requisiti)
    - [3.1 Progettazione e implementazione](#31-progettazione-e-implementazione)
    - [3.2 Funzioni](#32-funzioni)
<!-- TOC -->

## Storico revisioni
| Data | Versione | Motivo e descrizione della modifica |
|------|----------|-------------------------------------|
| 2026-01-02 | 0.1 | Bozza iniziale basata sul sorgente |

## 1. Introduzione
Questo documento definisce i requisiti del progetto pdf_sample. L'obiettivo e' descrivere il comportamento atteso del sorgente LaTeX che genera un PDF dimostrativo.

### 1.1 Regole del documento
Questo documento deve sempre seguire queste regole:
- Questo documento deve essere scritto in italiano.
- Formattare i requisiti come lista puntata, utilizzando le parole chiave "deve" o "devono" per indicare azioni obbligatorie.
- Ogni ID requisito (per esempio, **PRJ-001**, **PRJ-002**,.. **CTN-001**, **CTN-002**,.. **DES-001**, **DES-002**,.. **REQ-001**, **REQ-002**,..) deve essere unico; non assegnare lo stesso ID a requisiti diversi.
- Ogni ID requisito deve iniziare con la stringa che identifica il gruppo di requisiti:
  * I requisiti di funzione di progetto iniziano con **PRJ-**
  * I requisiti di vincolo di progetto iniziano con **CTN-**
  * I requisiti di progettazione e implementazione iniziano con **DES-**
  * I requisiti di funzione iniziano con **REQ-**
- Ogni requisito deve essere identificabile, verificabile e testabile.
- A ogni modifica di questo documento, aggiornare il numero di versione e aggiungere una nuova riga nello storico revisioni.

### 1.2 Scopo del progetto
Il progetto produce un documento PDF dimostrativo con capitoli, sezioni, figure, tabelle e formule, compilato da un sorgente LaTeX.

## 2. Requisiti di progetto

### 2.1 Funzioni di progetto
- **PRJ-001**: Il progetto deve fornire il sorgente LaTeX `pdf_sample.tex` per generare un documento PDF dimostrativo.
- **PRJ-002**: Il progetto deve includere un indice automatico che elenchi capitoli e sezioni.
- **PRJ-003**: Il progetto deve presentare capitoli e sezioni con contenuti misti (tabelle, figure raster, figure vettoriali e formule matematiche).
- **PRJ-004**: Il progetto deve impostare intestazioni e piu' di pagina con autore, titolo e capitolo corrente.

Struttura del progetto (esclusi i percorsi che iniziano con punto):
```
.
├── docs
│   ├── requirements.md
│   └── requirements_DRAFT.md
├── pdf_sample.pdf
└── pdf_sample.tex
```

Organizzazione e relazioni dei componenti:
- Il documento LaTeX e' organizzato in capitoli; ogni capitolo contiene sezioni e, quando necessario, tabelle, figure e formule.
- Le figure raster sono incluse tramite `graphicx` con file di esempio; le figure vettoriali e il flow chart sono definiti in TikZ.
- Le intestazioni e i pie di pagina sono gestiti da `fancyhdr` e applicati sia alle pagine standard sia alle pagine con stile `plain`.

Componenti e librerie utilizzati:
- Classe LaTeX `report`.
- Pacchetti: `inputenc`, `fontenc`, `babel`, `amsmath`, `amssymb`, `graphicx`, `tikz`, `booktabs`, `mwe`, `geometry`, `hyperref`, `lipsum`, `fancyhdr`.
- Librerie TikZ: `shapes.geometric`, `arrows.meta`, `positioning`.

Interfaccia testuale/GUI:
- Il PDF fornisce una UI testuale tramite indice cliccabile, intestazioni e numerazione delle pagine; non sono presenti GUI interattive.

Ottimizzazioni e miglioramenti prestazionali:
- Non sono presenti ottimizzazioni specifiche nel sorgente LaTeX.

### 2.2 Vincoli di progetto
- **CTN-001**: Il progetto deve usare LaTeX con classe `report` e formato pagina A4.
- **CTN-002**: Il progetto deve dichiarare la lingua italiana nel sorgente LaTeX.

## 3. Requisiti

### 3.1 Progettazione e implementazione
- **DES-001**: Il documento deve configurare margini sinistro e destro di 1 cm, margine superiore di 1.5 cm e margine inferiore di 2 cm tramite il pacchetto `geometry`.
- **DES-002**: Il documento deve includere le librerie TikZ `shapes.geometric`, `arrows.meta` e `positioning` per i diagrammi.
- **DES-003**: L'intestazione deve mostrare autore a sinistra, titolo al centro e capitolo corrente a destra, con una linea orizzontale sotto l'intestazione.
- **DES-004**: Il documento deve rendere l'indice cliccabile tramite il pacchetto `hyperref`.

### 3.2 Funzioni
- **REQ-001**: Il documento deve includere i capitoli nell'ordine: "Introduzione Stravagante", "Dati Senza Senso", "Immagini Misteriose", "Figure Vettoriali e Geometria", "Formule Matematiche Casuali", "Flow Chart Casuale", "Conti Matriciali Sconclusionati", "Conclusioni Inutili".
- **REQ-002**: Il documento deve includere almeno una tabella con intestazioni e didascalia.
- **REQ-003**: Il documento deve includere almeno due immagini raster di esempio.
- **REQ-004**: Il documento deve includere almeno una figura vettoriale generata con TikZ.
- **REQ-005**: Il documento deve includere un flow chart a pagina intera generato con TikZ.
- **REQ-006**: Il documento deve includere formule matematiche che coprano integrali, sistemi di equazioni e matrici.
