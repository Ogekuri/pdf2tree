#!/bin/bash

rm -rf temp/datasheet/

./pdf2tree.sh

# Verifica se la variabile GEMINI_API_KEY è definita e non vuota
if [ -n "$GEMINI_API_KEY" ]; then
    echo "[INFO] GEMINI_API_KEY=$GEMINI_API_KEY."
    echo "./pdf2tree.sh --from-file examples/datasheet.pdf --to-dir temp/datasheet/ --enable-form-xobject --enable-vector-images --post-processing --enable-pic2tex --verbose --debug --header 8 --footer 10"
    ./pdf2tree.sh --from-file examples/datasheet.pdf --to-dir temp/datasheet/ --post-processing --verbose --debug --header 8 --footer 10
else
    echo '[WARNING] GEMINI_API_KEY non definita. Aggiungerla con GEMINI_API_KEY=$(cat .gemini-api-key) ./pdf2tree.sh ...'
    echo "./pdf2tree.sh --from-file examples/datasheet.pdf --to-dir temp/datasheet/ --enable-form-xobject --enable-vector-images --post-processing --enable-pic2tex --disable-annotate-images --verbose --debug --header 8 --footer 10"
    ./pdf2tree.sh --from-file examples/datasheet.pdf --to-dir temp/datasheet/ --post-processing --disable-annotate-images --verbose --debug --header 8 --footer 10
fi



