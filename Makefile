cudaLibObjs = $(patsubst src/%.cu, obj/cuda/%.o, $(wildcard src/*.cu))
cLibObjs = $(patsubst src/%.c, obj/c/%.o, $(wildcard src/*.c))

bins = $(patsubst test/%.c, bin/%, $(wildcard test/*.c))
figs = $(patsubst doc/%.gpi, obj/%.tex, $(wildcard doc/*.gpi))
cmprFigs = obj/n8192-d2-k32.png obj/n32768-d2-k64.png
secondaryFigs = obj/speedup.eps

ccTool = gcc
ccFlags = -O3 -Wall -std=iso9899:1999

# -lm for math
# -rt for real time clock
# -lpthread for cpu- parallel  code
# -cuda, lcudart -lstdc++ for linking with nvcc output
ccLibs = -L/usr/local/cuda/lib64 -lm -lrt -lpthread -lcuda -lcudart -lstdc++

nvccTool = nvcc
nvccFlags = -O3
nvccLibs =

.PHONY: all clean
.PRECIOUS: obj/test/%.o obj/%.dat obj/%-summary.dat obj/%.tex obj/%.eps

# -----------------------------------------------------------------------------
# Top Level Targets
# -----------------------------------------------------------------------------

all: $(bins)

clean:
	rm -rf obj
	rm -rf bin

# -----------------------------------------------------------------------------
# Paper Targets
# -----------------------------------------------------------------------------

bin/document.pdf: doc/document.tex $(figs) $(secondaryFigs) $(cmprFigs) | bin
	pdflatex --output-directory=bin doc/document.tex
	bibtex bin/document.aux
	pdflatex --output-directory=bin doc/document.tex
	pdflatex --output-directory=bin doc/document.tex

obj/%.eps: analysis/%.py $(figs) | obj
	python $<

obj/%.png: res/%.dat analysis/compareVisualResults.py $(bins) | obj
	python analysis/compareVisualResults.py $< $@

obj/%.tex: obj/%-summary.dat doc/%.gpi | obj
	gnuplot -e "argInput='$<'; argOutput='$@'" $(word 2, $^)

obj/%-summary.dat: analysis/%.py bin/% | obj
	python $< > $@

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

bin/%: obj/test/%.o obj/c-lib.o obj/cuda-lib.o | bin
	$(ccTool) $(ccFlaggs) $< obj/c-lib.o obj/cuda-lib.o -o $@ $(ccLibs)

obj/test/%.o: test/%.c | obj/test
	$(ccTool) $(ccFlags) -I./src -c $< -o $@ $(ccLibs)

obj/test: | obj
	mkdir -p ./obj/test

obj:
	mkdir -p ./obj

bin:
	mkdir -p ./bin

# -----------------------------------------------------------------------------
# C Library
# -----------------------------------------------------------------------------

obj/c-lib.o: $(cLibObjs) | obj
	ld -r $(cLibObjs) -o obj/c-lib.o

obj/c/%.o: src/%.c | obj/c
	$(ccTool) $(ccFlags) -I./src -c $< -o $@ $(ccLibs)

obj/c: | obj
	mkdir -p ./obj/c

# -----------------------------------------------------------------------------
# CUDA Library
# -----------------------------------------------------------------------------

obj/cuda-lib.o: $(cudaLibObjs) | obj
	$(nvccTool) -lib $(cudaLibObjs) -o obj/cuda-lib.o

obj/cuda/%.o: src/%.cu | obj/cuda
	$(nvccTool) $(nvccFlags) -I./src -c $< -o $@ $(nvccLibs)

obj/cuda: | obj
	mkdir -p ./obj/cuda
