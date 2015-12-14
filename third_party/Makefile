export FARMHASH_OUT=$(OUT)/farmhash
export FARMHASH_DIR=$(CURDIR)/farmhash
export GTEST_OUT=$(OUT)/gtest
export GFLAGS_OUT=$(OUT)/gflags
export LIBYAML_OUT=$(OUT)/libyaml
export LIBZ26_OUT=$(OUT)/libz26

all: farmhash gtest gflags libyaml libz26

farmhash: $(FARMHASH_OUT)/lib/libfarmhash.a | $(FARMHASH_OUT)

$(FARMHASH_OUT)/lib/libfarmhash.a: export CFLAGS=-Wall
$(FARMHASH_OUT)/lib/libfarmhash.a: | $(FARMHASH_OUT)
	cd $(FARMHASH_OUT); $(FARMHASH_DIR)/configure --prefix=$(FARMHASH_OUT)
	$(MAKE) -C $(FARMHASH_OUT) install

gtest: $(GTEST_OUT)/libgtest.a | $(GTEST_OUT)

$(GTEST_OUT)/libgtest.a: export CFLAGS=-Wall
$(GTEST_OUT)/libgtest.a: export CXX=$(CC)
$(GTEST_OUT)/libgtest.a:
	cmake -B$(GTEST_OUT) -Hgoogletest/googletest -G"Unix Makefiles"
	$(MAKE) -C $(GTEST_OUT)

gflags: $(GFLAGS_OUT)/lib/libgflags.a | $(GFLAGS_OUT)

$(GFLAGS_OUT)/lib/libgflags.a: export CFLAGS=-Wall
$(GFLAGS_OUT)/lib/libgflags.a: export CXX=$(CC)
$(GFLAGS_OUT)/lib/libgflags.a:
	cmake -B$(GFLAGS_OUT) -Hgflags -G"Unix Makefiles"
	$(MAKE) -C $(GFLAGS_OUT)

libyaml: $(LIBYAML_OUT)/libyaml.a | $(LIBYAML_OUT)

$(LIBYAML_OUT)/libyaml.a: export CFLAGS=-Wall
$(LIBYAML_OUT)/libyaml.a: export CXX=$(CC)
$(LIBYAML_OUT)/libyaml.a:
	cmake -B$(LIBYAML_OUT) -Hlibyaml -G"Unix Makefiles"
	$(MAKE) -C $(LIBYAML_OUT)

libz26: | $(LIBZ26_OUT) $(LIBZ26_OUT)/libz26.o

$(LIBZ26_OUT)/libz26.o:	libz26/
	$(MAKE) -C libz26/

$(FARMHASH_OUT):
	mkdir -p $(FARMHASH_OUT)

$(GTEST_OUT):
	mkdir -p $(GTEST_OUT)

$(GFLAGS_OUT):
	mkdir -p $(GFLAGS_OUT)

$(LIBYAML_OUT):
	mkdir -p $(LIBYAML_OUT)

$(LIBZ26_OUT):
	mkdir -p $(LIBZ26_OUT)
	mkdir -p $(LIBZ26_OUT)/m4

.PHONY: clean
clean:
	rm -f $(FARMHASH_OUT)
	rm -f $(GTEST_OUT)
	rm -f $(GFLAGS_OUT)
	rm -f $(LIBYAML_OUT)
	rm -f $(LIBZ26_OUT)
