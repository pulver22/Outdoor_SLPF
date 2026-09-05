.PHONY : all latex bibtex view clean distclean small smaller tiny nonstop
# Change "paper" in the next line to the name of your document.
TARGET=paper
SOURCE=$(TARGET).tex
GENERATED=generated/icra_main_table.tex generated/icra_gnss_stress_table.tex generated/icra_operational_table.tex generated/icra_ablation_table.tex generated/icra_robustness_table.tex

all: $(TARGET).pdf

$(TARGET).pdf: $(SOURCE) $(GENERATED) glorified.bib new.bib stachnisslab-math.tex stachnisslab-latex.tex
	latexmk -pdf -halt-on-error $(SOURCE)

nonstop: $(SOURCE) glorified.bib new.bib stachnisslab-math.tex stachnisslab-latex.tex
	pdflatex -interaction=nonstopmode $(SOURCE)
	bibtex $(TARGET)
	pdflatex -interaction=nonstopmode $(SOURCE)
	pdflatex -interaction=nonstopmode $(SOURCE)

latex:
	latexmk -pdf -halt-on-error $(SOURCE)

bibtex:
	bibtex $(TARGET)

view:
	open $(TARGET).pdf &

small: $(TARGET).pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(TARGET)-small.pdf $(TARGET).pdf

smaller:$(TARGET).pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(TARGET)-smaller.pdf $(TARGET).pdf

tiny: $(TARGET).pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(TARGET)-tiny.pdf $(TARGET).pdf

clean:
	latexmk -c $(SOURCE)

distclean:clean
	rm -f $(TARGET).pdf $(TARGET)-small*.pdf $(TARGET)-tiny.pdf 


teaser.png: $(TARGET).pdf
	pdftoppm -r 15 -f 1 -l 8 $(TARGET).pdf thumbnail
	convert +append thumbnail*.ppm teaser.png
	rm thumbnail*.ppm

guidelines: guidelines.tex
		pdflatex -halt-on-error guidelines.tex
		bibtex guidelines
		pdflatex -halt-on-error guidelines.tex
		pdflatex -halt-on-error guidelines.tex
