#!/usr/bin/env bash
# Build the arXiv preprint (also the version presented at IASEAI '27). Run from the repository root:
#   ./4_writeups/paper/Arxiv/build.sh          -> 4_writeups/paper/Arxiv/main_arxiv_final.pdf
#   ./4_writeups/paper/Arxiv/build.sh bundle   -> also 4_writeups/paper/Arxiv/arxiv_submission.zip
#
# Paths inside the source (style file, figures, auto-generated tables, bibliography) are
# relative to the repository root, so pdflatex runs from there and writes into this folder.
# The bundle keeps that layout and ships the .bbl, since arXiv does not run BibTeX on .bib
# files it is not given.
set -euo pipefail

JOB="main_arxiv_final"
OUTDIR="./4_writeups/paper/Arxiv"
SRC="$OUTDIR/$JOB.tex"

latex_pass() {
    pdflatex -interaction=nonstopmode -halt-on-error -recorder \
             -output-directory="$OUTDIR" -jobname="$JOB" "$SRC" > /dev/null
}

latex_pass                                    # generate .aux
cp "$OUTDIR/$JOB.aux" "./$JOB.aux"            # BibTeX resolves \bibliography{} from the root
bibtex "$JOB" > /dev/null
mv "$JOB.bbl" "$OUTDIR/"
rm -f "./$JOB.aux" "./$JOB.blg"
latex_pass                                    # resolve citations
latex_pass                                    # resolve cross-references

echo "Built $OUTDIR/$JOB.pdf"
# grep -c exits 1 on zero matches, which set -e would treat as a failure.
echo "Undefined citations: $(grep -c "Citation.*undefined" "$OUTDIR/$JOB.log" || true)"
echo "Undefined references: $(grep -c "Reference.*undefined" "$OUTDIR/$JOB.log" || true)"

if [[ "${1:-}" == "bundle" ]]; then
    # arXiv compiles the top-level .tex at the archive root, so the main file goes there and
    # every \input / \includegraphics / \usepackage path stays repository-relative beside it.
    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' EXIT
    cp "$SRC" "$STAGE/$JOB.tex"
    cp "$OUTDIR/$JOB.bbl" "$STAGE/$JOB.bbl"
    mkdir -p "$STAGE/4_writeups/paper/Arxiv"
    cp "$OUTDIR/arxiv.sty" "$STAGE/4_writeups/paper/Arxiv/"
    # Every figure and table the source pulls in, from the file list pdflatex -recorder writes
    # (the .log wraps long paths, so it cannot be grepped reliably).
    grep -oE '^INPUT (\./)?3_outputs/.+\.(pdf|tex)$' "$OUTDIR/$JOB.fls" | sed -E 's/^INPUT (\.\/)?//' \
        | sort -u | while read -r f; do
        mkdir -p "$STAGE/$(dirname "$f")"
        cp "$f" "$STAGE/$f"
    done
    rm -f "$OUTDIR/arxiv_submission.zip"
    (cd "$STAGE" && zip -qr - .) > "$OUTDIR/arxiv_submission.zip"
    echo "Bundle: $OUTDIR/arxiv_submission.zip ($(unzip -l "$OUTDIR/arxiv_submission.zip" | tail -1 | awk '{print $2}') files)"
fi
