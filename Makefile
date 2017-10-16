OBJS_CUDA = $(patsubst src/%.cu, obj/cuda/%.o, $(wildcard src/*.cu))
OBJS_CXX = $(patsubst src/%.cpp, obj/cxx/%.o, $(wildcard src/*.cpp))

BINS = $(patsubst test/%.cpp, bin/%, $(wildcard test/*.cpp))
FIGS = $(patsubst doc/%.gpi, obj/%.tex, $(wildcard doc/*.gpi))
FIGS_SECONDARY = obj/n8192-d2-k32.png obj/n32768-d2-k64.png obj/speedup.eps

CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3 -Wno-deprecated-gpu-targets

# -lm for math
# -lrt for real time clock
# -lpthread for cpu- parallel  code
# -lcuda, lcudart for CUDA runtime
LIBS = -L/usr/local/cuda/lib64 -lm -lrt -lpthread -lcuda -lcudart

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

bin/%: obj/test/%.o obj/libgmm.o obj/libgmm-cuda.o | bin
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

obj/test/%.o: test/%.cpp | obj/test
	$(CXX) -c $(CXXFLAGS) -I./src -o $@ $<

obj/test: | obj
	mkdir -p ./obj/test

obj:
	mkdir -p ./obj

bin:
	mkdir -p ./bin

# -----------------------------------------------------------------------------
# C++ Library
# -----------------------------------------------------------------------------

obj/libgmm.o: $(OBJS_CXX) | obj
	ld -r -o obj/libgmm.o $(OBJS_CXX)

obj/cxx/%.o: src/%.cpp | obj/cxx
	$(CXX) -c $(CXXFLAGS) -I./src -o $@ $<

obj/cxx: | obj
	mkdir -p ./obj/cxx

# -----------------------------------------------------------------------------
# CUDA Library
# -----------------------------------------------------------------------------

obj/libgmm-cuda.o: $(OBJS_CUDA) | obj
	$(NVCC) -lib $(NVCCFLAGS) -o obj/libgmm-cuda.o $(OBJS_CUDA)

obj/cuda/%.o: src/%.cu | obj/cuda
	$(NVCC) -c $(NVCCFLAGS) -I./src -o $@ $<

obj/cuda: | obj
	mkdir -p ./obj/cuda
