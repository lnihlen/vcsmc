export DCMT_OUT=$(OUT)/dcmt
export DCMT_DIR=$(CURDIR)/dcmt
export FARMHASH_OUT=$(OUT)/farmhash
export FARMHASH_DIR=$(CURDIR)/farmhash
export GPERF_OUT=$(OUT)/gperftools
export GPERF_DIR=$(CURDIR)/gperftools
export GTEST_OUT=$(OUT)/gtest
export GFLAGS_OUT=$(OUT)/gflags
export LIBYAML_OUT=$(OUT)/libyaml
export LIBZ26_OUT=$(OUT)/libz26
export TBB_OUT=$(OUT)/tbb

all: dcmt farmhash gperftools gtest gflags libyaml libz26 tbb

dcmt: | $(DCMT_OUT) $(DCMT_OUT)/libdcmt.a

$(DCMT_OUT)/libdcmt.a:
	$(MAKE) -C $(DCMT_DIR)/lib
	$(MAKE) -C $(DCMT_DIR)/lib oclean
	mv $(DCMT_DIR)/lib/libdcmt.a $(DCMT_OUT)

farmhash: $(FARMHASH_OUT)/lib/libfarmhash.a | $(FARMHASH_OUT)

$(FARMHASH_OUT)/lib/libfarmhash.a: export CFLAGS=-Wall -O2 -g
$(FARMHASH_OUT)/lib/libfarmhash.a: | $(FARMHASH_OUT)
	cd $(FARMHASH_OUT); $(FARMHASH_DIR)/configure --prefix=$(FARMHASH_OUT)
	$(MAKE) -C $(FARMHASH_OUT) install

gperftools: $(GPERF_DIR)/configure $(GPERF_OUT)/lib/libprofiler.a | $(GPERF_OUT)

$(GPERF_DIR)/configure:
	cd $(GPERF_DIR); ./autogen.sh

$(GPERF_OUT)/lib/libprofiler.a: export CFLAGS=-Wall -O2 -g
$(GPERF_OUT)/lib/libprofiler.a: $(GPERF_DIR)/configure | $(GPERF_OUT)
	cd $(GPERF_OUT); $(GPERF_DIR)/configure --prefix=$(GPERF_OUT)
	$(MAKE) -C $(GPERF_OUT) install

gtest: $(GTEST_OUT)/libgtest.a | $(GTEST_OUT)

$(GTEST_OUT)/libgtest.a: export CFLAGS=-Wall -O2 -g
$(GTEST_OUT)/libgtest.a: export CXX=$(CC)
$(GTEST_OUT)/libgtest.a:
	cmake -B$(GTEST_OUT) -Hgoogletest/googletest -G"Unix Makefiles"
	$(MAKE) -C $(GTEST_OUT)

gflags: $(GFLAGS_OUT)/lib/libgflags.a | $(GFLAGS_OUT)

$(GFLAGS_OUT)/lib/libgflags.a: export CFLAGS=-Wall -O2 -g
$(GFLAGS_OUT)/lib/libgflags.a: export CXX=$(CC)
$(GFLAGS_OUT)/lib/libgflags.a:
	cmake -B$(GFLAGS_OUT) -Hgflags -G"Unix Makefiles"
	$(MAKE) -C $(GFLAGS_OUT)

libyaml: $(LIBYAML_OUT)/libyaml.a | $(LIBYAML_OUT)

$(LIBYAML_OUT)/libyaml.a: export CFLAGS=-Wall -O2 -g
$(LIBYAML_OUT)/libyaml.a: export CXX=$(CC)
$(LIBYAML_OUT)/libyaml.a:
	cmake -B$(LIBYAML_OUT) -Hlibyaml -G"Unix Makefiles"
	$(MAKE) -C $(LIBYAML_OUT)

libz26: | $(LIBZ26_OUT) $(LIBZ26_OUT)/libz26.o

$(LIBZ26_OUT)/libz26.o:	libz26/
	$(MAKE) -C libz26/

TBB_TARGET:=$(OUT)/libtbb.$(DYLIB)

tbb: | $(TBB_OUT) $(TBB_TARGET)

$(TBB_TARGET): export tbb_build_dir=$(TBB_OUT)
$(TBB_TARGET): export tbb_build_prefix=tbb
$(TBB_TARGET):
	$(MAKE) -C tbb/
	cp $(TBB_OUT)/tbb_debug/*.$(DYLIB)* $(OUT)

$(DCMT_OUT):
	mkdir -p $(DCMT_OUT)

$(FARMHASH_OUT):
	mkdir -p $(FARMHASH_OUT)

$(GPERF_OUT):
	mkdir -p $(GPERF_OUT)

$(GTEST_OUT):
	mkdir -p $(GTEST_OUT)

$(GFLAGS_OUT):
	mkdir -p $(GFLAGS_OUT)

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
	rm -f $(DCMT_OUT)
	rm -f $(FARMHASH_OUT)
	cd $(GPERF_DIR); git clean -dxf
	rm -f $(GPERF_OUT)
	rm -f $(GTEST_OUT)
	rm -f $(GFLAGS_OUT)
	rm -f $(LIBYAML_OUT)
	rm -f $(LIBZ26_OUT)
	rm -f $(TBB_OUT)
