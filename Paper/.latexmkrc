# Latexmk configuration for Overleaf
# This tells Overleaf which file to compile

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
@default_files = ('main.tex');
