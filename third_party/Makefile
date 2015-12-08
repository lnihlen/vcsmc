export GTEST_OUT=$(OUT)/gtest
export GFLAGS_OUT=$(OUT)/gflags
export LIBYAML_OUT=$(OUT)/libyaml
export LIBZ26_OUT=$(OUT)/libz26

all: gtest gflags libyaml libz26

gtest: $(GTEST_OUT)/libgtest.a | $(GTEST_OUT)

$(GTEST_OUT)/libgtest.a: export CFLAGS=-Wall
$(GTEST_OUT)/libgtest.a: export CXX=$(CC)
$(GTEST_OUT)/libgtest.a:
	cmake -B$(GTEST_OUT) -Hgoogletest/googletest -G"Unix Makefiles"
	$(MAKE) -C $(GTEST_OUT)

gflags: $(GFLAGS_OUT)/lib/gflags.a | $(GFLAGS_OUT)

$(GFLAGS_OUT)/lib/gflags.a: export CFLAGS=-Wall
$(GFLAGS_OUT)/lib/gflags.a: export CXX=$(CC)
$(GFLAGS_OUT)/lib/gflags.a:
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
	rm -f $(GTEST_OUT)
	rm -f $(GFLAGS_OUT)
	rm -f $(LIBYAML_OUT)
	rm -f $(LIBZ26_OUT)
