COMMON_DIR := ../common

VPATH      := $(COMMON_DIR)

TEXINPUTS := .:$(COMMON_DIR):

COMMON := $(wildcard $(COMMON_DIR)/*.tex $(COMMON_DIR)/*.bib)

$(DOC).pdf:$(DOC).tex $(COMMON) $(IMAGES)
	TEXINPUTS=$(TEXINPUTS) pdflatex $(DOC).tex
	BIBINPUTS=$(TEXINPUTS) bibtex $(DOC).aux
	TEXINPUTS=$(TEXINPUTS) pdflatex $(DOC).tex
	TEXINPUTS=$(TEXINPUTS) pdflatex $(DOC).tex

clean:
	rm -f $(DOC).{pdf,log,bbl,blg,vrb,aux,out,snm,nav,toc} $(IMAGES)

.PHONY: clean
