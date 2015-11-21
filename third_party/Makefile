export GTEST_OUT=$(OUT)/gtest
export GFLAGS_OUT=$(OUT)/gflags
export LIBSTELLA_OUT=$(OUT)/libstella

all: gtest gflags libstella

gtest: export CXX=$(CC)
gtest: export CXXFLAGS=$(CFLAGS)
gtest: export GTEST_DIR=googletest/googletest
gtest: | $(GTEST_OUT)
	$(MAKE) -C $(GTEST_DIR)/make gtest.a
	mv $(GTEST_DIR)/make/gtest.a $(GTEST_OUT)
	rm -f $(GTEST_DIR)/make/gtest-all.o

gflags: | $(GFLAGS_OUT) $(GFLAGS_OUT)/lib/gtest.a

$(GFLAGS_OUT)/lib/gtest.a: export CFLAGS=-Wall
$(GFLAGS_OUT)/lib/gtest.a: export CXX=$(CC)
$(GFLAGS_OUT)/lib/gtest.a:
	env
	cmake -B$(GFLAGS_OUT) -Hgflags -G"Unix Makefiles"
	$(MAKE) -C $(GFLAGS_OUT)

libstella:
	$(MAKE) -C libstella/

$(GTEST_OUT):
	mkdir -p $(GTEST_OUT)

$(GFLAGS_OUT):
	mkdir -p $(GFLAGS_OUT)

$(LIBSTELLA_OUT):
	mkdir -p $(LIBSTELLA_OUT)

.PHONY: clean
clean:
	rm -f $(GTEST_OUT)
	rm -f $(GFLAGS_OUT)
#	$(MAKE) -C libstella/ clean

