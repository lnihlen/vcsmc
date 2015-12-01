export GTEST_OUT=$(OUT)/gtest
export GFLAGS_OUT=$(OUT)/gflags
export LIBZ26_OUT=$(OUT)/libz26

all: gtest gflags libz26

gtest: export CXX=$(CC)
gtest: export CXXFLAGS=-std=c++11
gtest: export GTEST_DIR=googletest/googletest
gtest: | $(GTEST_OUT)
	$(MAKE) -C $(GTEST_DIR)/make gtest.a
	mv $(GTEST_DIR)/make/gtest.a $(GTEST_OUT)
	rm -f $(GTEST_DIR)/make/gtest-all.o

gflags: $(GFLAGS_OUT)/lib/gflags.a | $(GFLAGS_OUT)

$(GFLAGS_OUT)/lib/gflags.a: export CFLAGS=-Wall
$(GFLAGS_OUT)/lib/gflags.a: export CXX=$(CC)
$(GFLAGS_OUT)/lib/gflags.a:
	cmake -B$(GFLAGS_OUT) -Hgflags -G"Unix Makefiles"
	$(MAKE) -C $(GFLAGS_OUT)

libz26: | $(LIBZ26_OUT) $(LIBZ26_OUT)/libz26.o

$(LIBZ26_OUT)/libz26.o:	libz26/
	$(MAKE) -C libz26/

$(GTEST_OUT):
	mkdir -p $(GTEST_OUT)

$(GFLAGS_OUT):
	mkdir -p $(GFLAGS_OUT)

$(LIBZ26_OUT):
	mkdir -p $(LIBZ26_OUT)
	mkdir -p $(LIBZ26_OUT)/m4

.PHONY: clean
clean:
	rm -f $(GTEST_OUT)
	rm -f $(GFLAGS_OUT)
#	$(MAKE) -C libstella/ clean
