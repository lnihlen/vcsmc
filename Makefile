# VCSMC main Makefile
# assumes GNU make 3.80 or up
export CC=clang
# If chaning CFLAGS it is good to change them in the .ycm_extra_conf.py file
# (also versioned) so that YCM will pick up the correct flags.
export CFLAGS=-std=c++11 -Wall -Wextra -Werror -O0 -g
export LDFLAGS=
export LIBS=-lstdc++
export OUT=$(CURDIR)/out
export GTEST_INCLUDE=$(CURDIR)/third_party/googletest/googletest/include
export GTEST_LIB=$(OUT)/gtest/gtest.a
export GFLAGS_INCLUDE=$(OUT)/gflags/include
export GFLAGS_LIB=$(OUT)/gflags/lib/libgflags.a
export LIBZ26_INCLUDE=$(CURDIR)/third_party/libz26/include
export LIBZ26_LIB=$(OUT)/libz26/libz26.o

all: depends vcsmc tests

depends: | $(OUT)/
	$(MAKE) -C third_party/ all

vcsmc: depends | $(OUT)/
	$(MAKE) -C src/ all

tests: gtest picc | $(OUT)/
	$(MAKE) -C src/ tests

$(OUT)/:
	mkdir -p out/

.PHONY: clean
clean:
	$(MAKE) -C src/ clean
	$(MAKE) -C third_party/ clean
