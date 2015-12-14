# VCSMC src/ Makefile.
# Dependencies: libpng, clang, OpenCL
# tools dependencies: python colormath, Wand, ImageMagick
CFLAGS+=-I. -I$(GFLAGS_INCLUDE) -I$(LIBZ26_INCLUDE) -I$(LIBYAML_INCLUDE)
LIBS+=-lm -lpng $(LIBYAML_LIB) $(GFLAGS_LIB)
OUTTESTS=$(OUT)/tests

UNAME:=$(shell uname)

# OS X-specific build flags
ifeq ($(UNAME), Darwin)
CFLAGS+=-I/usr/local/include
LDFLAGS+=-L/usr/local/lib
endif

DEPS=constants.h image_file.h types.h
OBJ=$(OUT)/assembler.o                                                         \
    $(OUT)/color_table.o                                                       \
    $(OUT)/image.o                                                             \
    $(OUT)/image_file.o                                                        \
    $(OUT)/job_queue.o                                                         \
    $(OUT)/kernel.o                                                            \
    $(OUT)/range.o                                                             \
    $(OUT)/spec.o                                                              \

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/job_queue_unittests.o                                     \
         $(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/spec_unittests.o

all: $(OUT)/gen                                                                \
     $(OUT)/sim                                                                \

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -o $@ $< $(CFLAGS) -I$(GTEST_INCLUDE)

$(OUT)/gen: $(OBJ) $(OUT)/gen.o
	$(CC) -o $@ $^ $(GFLAGS_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/sim: $(OBJ) $(OUT)/sim.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o      \
    | $(OUTTESTS)/
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS) $(GTEST_LIB)

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -rf $(OUT)
