# VCSMC main Makefile
# assumes GNU make 3.80 or up
export CC=clang
export CXX=clang++
# Note that right now all the third_party/ builds have hard-coded CFLAGS
# of their own and don't follow this variable.
export CFLAGS=-std=c++11 -march=native -Wall -Wextra -Werror -fPIC
export LDFLAGS=
export LIBS=-lstdc++ -ldl
export OUT=$(CURDIR)/out
export UNAME:=$(shell uname)

ifeq ($(UNAME), Darwin)
export DYLIB=dylib
endif

ifeq ($(UNAME), Linux)
export DYLIB=so
export NVCC=nvcc -Wno-deprecated-gpu-targets
endif

export GPERF_LIB=$(OUT)/gperftools/lib/libprofiler.a
export GPERF_INCLUDE=$(OUT)/gperftools/include
export GTEST_INCLUDE=$(CURDIR)/third_party/googletest/googletest/include
export GTEST_LIB=$(OUT)/gtest/libgtest.a
export GFLAGS_INCLUDE=$(OUT)/gflags/include
export GFLAGS_LIB=$(OUT)/gflags/lib/libgflags.a
export HALIDE_DIR=$(CURDIR)/third_party/Halide
export HALIDE_INCLUDE=$(OUT)/Halide/include
export HALIDE_LIB=$(OUT)/Halide/lib
export LIBYAML_INCLUDE=$(CURDIR)/third_party/libyaml/include
export LIBYAML_LIB=$(OUT)/libyaml/libyaml_static.a
export LIBZ26_INCLUDE=$(CURDIR)/third_party/libz26/include
export LIBZ26_LIB=$(OUT)/libz26/libz26.o
export TBB_INCLUDE=$(CURDIR)/third_party/tbb/include
export XXHASH_LIB=$(OUT)/xxHash/libxxhash.a
export XXHASH_INCLUDE=$(CURDIR)/third_party/xxHash

all: export OPT=-O3
all: depends vcsmc tests

debug: export OPT=-O0 -g
debug: depends vcsmc tests

depends: | $(OUT)/
	$(MAKE) -C third_party/ all

vcsmc: depends | $(OUT)/
	$(MAKE) -C src/ vcsmc

tests: depends vcsmc | $(OUT)/
	$(MAKE) -C src/ tests

$(OUT)/:
	mkdir -p out/

.PHONY: clean
clean:
	$(MAKE) -C src/ clean

.PHONY: cleandeps
cleandeps: clean
	$(MAKE) -C third_party/ clean
