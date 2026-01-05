#!/bin/bash

./pdf2tree.sh

# Verifica se la variabile GEMINI_API_KEY è definita e non vuota
if [ -n "$GEMINI_API_KEY" ]; then
    echo "[INFO] GEMINI_API_KEY trovata. Esecuzione di pippo con opzione --pluto."
    echo "./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --verbose --debug --header 8 --footer 10"
    ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --verbose --debug --header 8 --footer 10
else
    echo "[WARNING] GEMINI_API_KEY non definita. Esecuzione di pippo semplice."
    echo "./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --disable-annotate-images --verbose --debug --header 8 --footer 10"
    ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --post-processing-only --disable-annotate-images --verbose --debug --header 8 --footer 10
fi
