.PHONY : all response_letter clean distclean
# Change "paper" in the next line to the name of your document.


all: response_letter.pdf

response_letter.pdf: response_letter.tex ../glorified.bib ../new.bib ../stachnisslab-math.tex ../stachnisslab-latex.tex
	pdflatex -halt-on-error response_letter.tex

bibtex:
	bibtex response_letter

clean:
	rm -f response_letter.aux response_letter.bbl response_letter.blg response_letter.log response_letter.out

distclean: clean
	rm response_letter.pdf

view:
	open response_letter.pdf &


