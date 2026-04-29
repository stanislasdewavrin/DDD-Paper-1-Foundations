# Paper I — Discrete Drainage Dynamics: Foundations
# ----------------------------------------------------
# Reproduces the manuscript and the four internal-consistency tests.

PAPER = paper
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
PYTHON = python3

.PHONY: all paper figures tests clean distclean

all: figures paper

paper: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

figures: tests
	$(PYTHON) code/06_master_figure.py

tests:
	$(PYTHON) code/01_conservation.py
	$(PYTHON) code/02_interface_identity.py
	$(PYTHON) code/03_direction_emergence.py
	$(PYTHON) code/04_inertial_relaxation.py
	$(PYTHON) code/05_sr_recovery.py

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).out $(PAPER).toc

distclean: clean
	rm -f $(PAPER).pdf
	rm -f figures/*.pdf figures/*.png
	rm -f data/*.npz data/*.json
