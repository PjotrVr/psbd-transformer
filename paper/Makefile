# Build the paper. `make` compiles paper/main.tex to paper/main.pdf with tectonic,
# `make tables` regenerates every table and figure from results first, `make watch`
# recompiles on every change. tectonic lives in ~/.local/bin and fetches its TeX
# packages through the cluster proxy on first use.
TECTONIC ?= $(HOME)/.local/bin/tectonic
RESULTS ?= $(abspath ..)/results
export http_proxy ?= http://10.150.1.1:3128
export https_proxy ?= http://10.150.1.1:3128

.PHONY: all pdf tables watch clean

all: pdf

pdf:
	$(TECTONIC) -X compile main.tex --keep-logs

tables:
	cd .. && PYTHONPATH=. .venv/bin/python scripts/paper/build_all.py --results-dir $(RESULTS) --paper-dir paper

watch:
	$(TECTONIC) -X watch

clean:
	rm -f main.pdf main.log
