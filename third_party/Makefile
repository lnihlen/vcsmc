export GPERF_OUT=$(OUT)/gperftools
export GPERF_DIR=$(CURDIR)/gperftools
export GTEST_OUT=$(OUT)/gtest
export GFLAGS_OUT=$(OUT)/gflags
export HALIDE_OUT=$(OUT)/halide
export LIBYAML_OUT=$(OUT)/libyaml
export LIBZ26_OUT=$(OUT)/libz26
export TBB_OUT=$(OUT)/tbb
export XXHASH_OUT=$(OUT)/xxHash
export XXHASH_DIR=$(CURDIR)/xxHash

all: gperftools gtest gflags halide libyaml libz26 tbb xxhash

gperftools: $(GPERF_DIR)/configure $(GPERF_OUT)/lib/libprofiler.a | $(GPERF_OUT)

$(GPERF_DIR)/configure:
	cd $(GPERF_DIR); ./autogen.sh

$(GPERF_OUT)/lib/libprofiler.a: export CFLAGS=-Wall -fPIC $(OPT)
$(GPERF_OUT)/lib/libprofiler.a: $(GPERF_DIR)/configure | $(GPERF_OUT)
	cd $(GPERF_OUT); $(GPERF_DIR)/configure --prefix=$(GPERF_OUT)
	$(MAKE) -C $(GPERF_OUT) install

gtest: $(GTEST_OUT)/libgtest.a | $(GTEST_OUT)

$(GTEST_OUT)/libgtest.a: export CFLAGS=-Wall -fPIC $(OPT)
$(GTEST_OUT)/libgtest.a: export CXXFLAGS=-Wall -fPIC $(OPT)
$(GTEST_OUT)/libgtest.a: export CXX=$(CC)
$(GTEST_OUT)/libgtest.a:
	cmake -B$(GTEST_OUT) -Hgoogletest/googletest -G"Unix Makefiles"
	$(MAKE) -C $(GTEST_OUT)

gflags: $(GFLAGS_OUT)/lib/libgflags.a | $(GFLAGS_OUT)

$(GFLAGS_OUT)/lib/libgflags.a: export CFLAGS=-Wall -fPIC $(OPT)
$(GFLAGS_OUT)/lib/libgflags.a: export CXX=$(CC)
$(GFLAGS_OUT)/lib/libgflags.a:
	cmake -B$(GFLAGS_OUT) -Hgflags -G"Unix Makefiles"
	$(MAKE) -C $(GFLAGS_OUT)

halide: $(HALIDE_OUT)/lib/libHalide.so | $(HALIDE_OUT)

$(HALIDE_OUT)/lib/libHalide.so: export CC=
$(HALIDE_OUT)/lib/libHalide.so: export CFLAGS=
$(HALIDE_OUT)/lib/libHalide.so:
	cd $(HALIDE_OUT);                                                            \
  cmake -DBUILD_AOT_TUTORIAL:BOOL="0" -DHALIDE_SHARED_LIBRARY:BOOL="1"         \
  -DWITH_APPS:BOOL="0" -DWITH_TESTS:BOOL="0" -DWITH_TUTORIALS:BOOL="0"         \
  $(HALIDE_DIR);                                                               \
  make

libyaml: $(LIBYAML_OUT)/libyaml_static.a | $(LIBYAML_OUT)

$(LIBYAML_OUT)/libyaml_static.a: export CFLAGS=-Wall -fPIC $(OPT)
$(LIBYAML_OUT)/libyaml_static.a: export CXX=$(CC)
$(LIBYAML_OUT)/libyaml_static.a:
	cmake -B$(LIBYAML_OUT) -Hlibyaml -G"Unix Makefiles"
	$(MAKE) -C $(LIBYAML_OUT)

libz26: | $(LIBZ26_OUT) $(LIBZ26_OUT)/libz26.o

$(LIBZ26_OUT)/libz26.o: export CFLAGS=-Wall -fPIC $(OPT)
$(LIBZ26_OUT)/libz26.o:	libz26/
	$(MAKE) -C libz26/

TBB_TARGET:=$(OUT)/libtbb.$(DYLIB)

tbb: | $(TBB_OUT) $(TBB_TARGET)

$(TBB_TARGET): export tbb_build_dir=$(TBB_OUT)
$(TBB_TARGET): export tbb_build_prefix=tbb
$(TBB_TARGET):
	$(MAKE) -C tbb/ compiler=clang stdver=c++11 tbb
	cp $(TBB_OUT)/tbb_release/*.$(DYLIB)* $(OUT)

xxhash: $(XXHASH_OUT)/libxxhash.a | $(XXHASH_OUT)

$(XXHASH_OUT)/libxxhash.a: export CFLAGS=-Wall -fPIC $(OPT)
$(XXHASH_OUT)/libxxhash.a: | $(XXHASH_OUT)
	cd $(XXHASH_DIR); make libxxhash.a; mv libxxhash.a $(XXHASH_OUT); rm *.o

$(XXHASH_OUT):
	mkdir -p $(XXHASH_OUT)

$(GPERF_OUT):
	mkdir -p $(GPERF_OUT)

$(GTEST_OUT):
	mkdir -p $(GTEST_OUT)

$(GFLAGS_OUT):
	mkdir -p $(GFLAGS_OUT)

$(HALIDE_OUT):
	mkdir -p $(HALIDE_OUT)

$(LIBYAML_OUT):
	mkdir -p $(LIBYAML_OUT)

$(LIBZ26_OUT):
	mkdir -p $(LIBZ26_OUT)
	mkdir -p $(LIBZ26_OUT)/m4

$(TBB_OUT):
	mkdir -p $(TBB_OUT)
	echo $(TBB_TARGET)

.PHONY: clean
clean:
	rm -rf $(DCMT_OUT)
	rm -rf $(FARMHASH_OUT)
	cd $(GPERF_DIR); git clean -dxf
	rm -rf $(GPERF_OUT)
	rm -rf $(GTEST_OUT)
	rm -rf $(GFLAGS_OUT)
	rm -rf $(HALIDE_OUT)
	rm -rf $(LIBYAML_OUT)
	rm -rf $(LIBZ26_OUT)
	rm -rf $(TBB_OUT)
	rm -rf $(XXHASH_OUT)
	rm -f $(OUT)/libtbb*
