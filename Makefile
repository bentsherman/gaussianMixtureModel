OBJS_CUDA = $(patsubst src/%.cu, obj/cuda/%.o, $(wildcard src/*.cu))
OBJS_C = $(patsubst src/%.c, obj/c/%.o, $(wildcard src/*.c))

BINS = $(patsubst test/%.c, bin/%, $(wildcard test/*.c))
FIGS = $(patsubst doc/%.gpi, obj/%.tex, $(wildcard doc/*.gpi))
FIGS_SECONDARY = obj/n8192-d2-k32.png obj/n32768-d2-k64.png obj/speedup.eps

CC = gcc
CCFLAGS = -O3 -Wall -std=iso9899:1999

NVCC = nvcc
NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets

# -lm for math
# -lrt for real time clock
# -lpthread for cpu- parallel  code
# -lcuda, lcudart -lstdc++ for linking with nvcc output
LIBS = -L/usr/local/cuda/lib64 -lm -lrt -lpthread -lcuda -lcudart -lstdc++

.PHONY: all clean
.PRECIOUS: obj/test/%.o obj/%.dat obj/%-summary.dat obj/%.tex obj/%.eps

# -----------------------------------------------------------------------------
# Top Level Targets
# -----------------------------------------------------------------------------

all: $(BINS)

clean:
	rm -rf obj
	rm -rf bin

# -----------------------------------------------------------------------------
# Paper Targets
# -----------------------------------------------------------------------------

bin/document.pdf: doc/document.tex $(FIGS) $(FIGS_SECONDARY) | bin
	pdflatex --output-directory=bin doc/document.tex
	bibtex bin/document.aux
	pdflatex --output-directory=bin doc/document.tex
	pdflatex --output-directory=bin doc/document.tex

obj/%.eps: analysis/%.py $(FIGS) | obj
	python $<

obj/%.png: res/%.dat analysis/compareVisualResults.py $(BINS) | obj
	python analysis/compareVisualResults.py $< $@

obj/%.tex: obj/%-summary.dat doc/%.gpi | obj
	gnuplot -e "argInput='$<'; argOutput='$@'" $(word 2, $^)

obj/%-summary.dat: analysis/%.py bin/% | obj
	python $< > $@

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

bin/%: obj/test/%.o obj/c-lib.o obj/cuda-lib.o | bin
	$(CC) $(CCFLAGS) -o $@ $^ $(LIBS)

obj/test/%.o: test/%.c | obj/test
	$(CC) -c $(CCFLAGS) -I./src -o $@ $<

obj/test: | obj
	mkdir -p ./obj/test

obj:
	mkdir -p ./obj

bin:
	mkdir -p ./bin

# -----------------------------------------------------------------------------
# C Library
# -----------------------------------------------------------------------------

obj/c-lib.o: $(OBJS_C) | obj
	ld -r -o obj/c-lib.o $(OBJS_C)

obj/c/%.o: src/%.c | obj/c
	$(CC) -c $(CCFLAGS) -I./src -o $@ $<

obj/c: | obj
	mkdir -p ./obj/c

# -----------------------------------------------------------------------------
# CUDA Library
# -----------------------------------------------------------------------------

obj/cuda-lib.o: $(OBJS_CUDA) | obj
	$(NVCC) -lib $(NVCCFLAGS) -o obj/cuda-lib.o $(OBJS_CUDA)

obj/cuda/%.o: src/%.cu | obj/cuda
	$(NVCC) -c $(NVCCFLAGS) -I./src -o $@ $<

obj/cuda: | obj
	mkdir -p ./obj/cuda
