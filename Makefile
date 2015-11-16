# VCSMC main Makefile
# assumes GNU make 3.80 or up
export CC=clang
export CFLAGS=-std=c++11 -Wall
export LDFLAGS=
export LIBS=-lstdc++
export OUT=$(CURDIR)/out
export GTEST_INCLUDE=$(CURDIR)/third_party/googletest/googletest/include
export GTEST_LIB=$(OUT)/gtest/gtest.a

all: picc tests

picc: | $(OUT)/
	$(MAKE) -C src/ all

tests: gtest
	$(MAKE) -C src/ tests

gtest: | $(OUT)/
	$(MAKE) -C third_party/ gtest

.PHONY: clean
clean:
	$(MAKE) -C src/ clean
	$(MAKE) -C third_party/ clean

$(OUT)/:
	mkdir -p out/
	mkdir -p out/parts/


