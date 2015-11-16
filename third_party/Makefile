export GTEST_OUT=$(OUT)/gtest
export LIBSTELLA_OUT=$(OUT)/libstella

all: gtest

gtest: export CXX=$(CC)
gtest: export CXXFLAGS=$(CFLAGS)
gtest: export GTEST_DIR=googletest/googletest
gtest: | $(GTEST_OUT)
	$(MAKE) -C $(GTEST_DIR)/make gtest.a
	mv $(GTEST_DIR)/make/gtest.a $(GTEST_OUT)
	rm -f $(GTEST_DIR)/make/gtest-all.o

libstella:
	$(MAKE) -C libstella/

$(GTEST_OUT):
	mkdir -p $(OUT)/gtest

$(LIBSTELLA_OUT):
	mkdir -p $(OUT)/libstella

.PHONY: clean
clean:
	rm -f $(GTEST_OUT)/gtest.a
#	$(MAKE) -C libstella/ clean

