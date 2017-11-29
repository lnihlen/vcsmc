# VCSMC main Makefile
# assumes GNU make 3.80 or up
export CC=clang
# If changing CFLAGS it is good to change them in the .ycm_extra_conf.py file
# (also versioned) so that YCM will pick up the correct flags.
# Also note that right now all the third_party/ builds have hard-coded CFLAGS
# of their own and don't follow this variable.
export CFLAGS=-std=c++11 -Wall -Wextra -Werror -O2
export LDFLAGS=
export LIBS=-lstdc++
export OUT=$(CURDIR)/out
export UNAME:=$(shell uname)

ifeq ($(UNAME), Darwin)
export DYLIB=dylib
export NVCC=/Developer/NVIDIA/CUDA-8.0/bin/nvcc -Wno-deprecated-gpu-targets
endif

ifeq ($(UNAME), Linux)
export DYLIB=so
export NVCC=/opt/cuda/bin/nvcc -Wno-deprecated-gpu-targets
endif

export FARMHASH_LIB=$(OUT)/farmhash/lib/libfarmhash.a
export FARMHASH_INCLUDE=$(OUT)/farmhash/include
export GPERF_LIB=$(OUT)/gperftools/lib/libprofiler.a
export GPERF_INCLUDE=$(OUT)/gperftools/include
export GTEST_INCLUDE=$(CURDIR)/third_party/googletest/googletest/include
export GTEST_LIB=$(OUT)/gtest/libgtest.a
export GFLAGS_INCLUDE=$(OUT)/gflags/include
export GFLAGS_LIB=$(OUT)/gflags/lib/libgflags.a
export LIBYAML_INCLUDE=$(CURDIR)/third_party/libyaml/include
export LIBYAML_LIB=$(OUT)/libyaml/libyaml.a
export LIBZ26_INCLUDE=$(CURDIR)/third_party/libz26/include
export LIBZ26_LIB=$(OUT)/libz26/libz26.o
export TBB_INCLUDE=$(CURDIR)/third_party/tbb/include

all: depends vcsmc tests

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
