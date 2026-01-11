#!/bin/bash

rm -rf temp/pdf_sample_test/

./pdf2tree.sh

# Verifica se la variabile GEMINI_API_KEY è definita e non vuota
if [ -n "$GEMINI_API_KEY" ]; then
    echo "[INFO] GEMINI_API_KEY=$GEMINI_API_KEY."
    echo ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --verbose --debug --header 8 --footer 12
    ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --verbose --debug --header 8 --footer 12
else
    echo '[WARNING] GEMINI_API_KEY non definita. Aggiungerla con GEMINI_API_KEY=$(cat .gemini-api-key) ./pdf2tree.sh ...'
    echo ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --verbose --debug --header 8 --footer 12
    ./pdf2tree.sh --from-file pdf_sample/pdf_sample.pdf --to-dir temp/pdf_sample_test/ --verbose --debug --header 8 --footer 12
fi



