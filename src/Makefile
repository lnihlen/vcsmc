# VCSMC src/ Makefile.
# Dependencies: libpng, clang, OpenCL
# tools dependencies: python colormath, Wand, ImageMagick
CFLAGS+=-I. -I$(FARMHASH_INCLUDE) -I$(GFLAGS_INCLUDE) -I$(LIBZ26_INCLUDE)      \
    -I$(LIBYAML_INCLUDE) -I$(GPERF_INCLUDE) -I$(TBB_INCLUDE) -I$(DCMT_INCLUDE)
LIBS+=-lm -lpng $(LIBYAML_LIB) $(GFLAGS_LIB) $(FARMHASH_LIB) $(GPERF_LIB)      \
    -L$(OUT) -ltbb
OUTTESTS=$(OUT)/tests

# OS X-specific build flags
ifeq ($(UNAME), Darwin)
CFLAGS+=-I/usr/local/include
LDFLAGS+=-L/usr/local/lib -Wl,-no_pie
endif

ifeq ($(UNAME), Linux)
LDFLAGS+=-lpthread -lunwind
endif

DEPS=constants.h image_file.h types.h

COLOR_OBJ=$(OUT)/color.o                                                       \
          $(OUT)/color_table.o

IMAGE_OBJ=$(OUT)/bit_map.o                                                     \
          $(OUT)/gray_map.o                                                    \
          $(OUT)/image.o                                                       \
          $(OUT)/image_file.o                                                  \
          $(OUT)/value_map.o

OBJ=$(OUT)/assembler.o                                                         \
    $(OUT)/kernel.o                                                            \
    $(OUT)/range.o                                                             \
    $(OUT)/serialization.o                                                     \
    $(OUT)/sound_file.o                                                        \
    $(OUT)/spec.o                                                              \
    $(OUT)/tls_prng.o

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/bit_map_unittests.o                                       \
         $(OUTTESTS)/color_unittests.o                                         \
         $(OUTTESTS)/kernel_unittests.o                                        \
         $(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/serialization_unittests.o

all: $(OUT)/fit                                                                \
     $(OUT)/gen                                                                \
     $(OUT)/pcm                                                                \
     $(OUT)/respec                                                             \
     $(OUT)/sim                                                                \

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OUT)/color_distance_table.o: $(OUT)/build_error_distances
	$(OUT)/build_error_distances --output_directory=../out
	$(CC) -c -o $@ $(OUT)/color_distance_table.cc $(CFLAGS)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -o $@ $< $(CFLAGS) -I$(GTEST_INCLUDE)

$(OUT)/build_error_distances: $(IMAGE_OBJ) $(COLOR_OBJ)                        \
                              $(OUT)/build_error_distances.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/fit: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/fit.o                      \
            $(OUT)/color_distance_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/gen: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/gen.o                      \
            $(OUT)/color_distance_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/pcm: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/color_distance_table.o     \
            $(OUT)/pcm.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/respec: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/color_distance_table.o  \
               $(OUT)/respec.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/sim: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/sim.o                      \
            $(OUT)/color_distance_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o      \
    $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/color_distance_table.o | $(OUTTESTS)/
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS) $(GTEST_LIB)

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -rf $(OUT)
