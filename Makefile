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

all: picc tests

picc: gflags libz26 | $(OUT)/
	$(MAKE) -C src/ all

tests: gtest
	$(MAKE) -C src/ tests

gtest:
	$(MAKE) -C third_party/ gtest

gflags:
	$(MAKE) -C third_party/ gflags

libz26:
	$(MAKE) -C third_party/ libz26

$(OUT)/:
	mkdir -p out/

.PHONY: clean
clean:
	$(MAKE) -C src/ clean
	$(MAKE) -C third_party/ clean
