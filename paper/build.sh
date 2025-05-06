#!/bin/bash

# Script to compile a LaTeX document with BibTeX
# Usage: ./build.sh [filename.tex] (default: apssamp.tex)

TEXFILE="${1:-apssamp.tex}"
BASENAME="${TEXFILE%.tex}"

pdflatex -interaction=nonstopmode  "$TEXFILE"        || true
bibtex "$BASENAME"                                   || true
pdflatex -interaction=nonstopmode  "$TEXFILE"        || true
pdflatex -interaction=nonstopmode  "$TEXFILE"        || true

echo "✅ Build complete: ${BASENAME}.pdf"

