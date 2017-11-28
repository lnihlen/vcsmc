# VCSMC src/ Makefile.
# Dependencies: libpng, clang, OpenCL
# tools dependencies: python colormath, Wand, ImageMagick
INCLUDES=-I. -I$(FARMHASH_INCLUDE) -I$(GFLAGS_INCLUDE) -I$(LIBZ26_INCLUDE)     \
    -I$(LIBYAML_INCLUDE) -I$(GPERF_INCLUDE) -I$(TBB_INCLUDE)
LIBS+=-lm -lpng $(LIBYAML_LIB) $(GFLAGS_LIB) $(FARMHASH_LIB) $(GPERF_LIB)      \
    -L$(OUT) -ltbb -lcudart
OUTTESTS=$(OUT)/tests

# OS X-specific build flags
ifeq ($(UNAME), Darwin)
INCLUDES+=-I/usr/local/include -I/Developer/NVIDIA/CUDA-8.0/include
LDFLAGS+=-L/usr/local/lib -Wl,-no_pie -L/Developer/NVIDIA/CUDA-8.0/lib
LDPATH=/Developer/NVIDIA/CUDA-8.0/lib
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
    $(OUT)/mssim.o                                                             \
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

all: $(OUT)/find_ciede_max_distance                                            \
     $(OUT)/fit                                                                \
     $(OUT)/gen                                                                \
     $(OUT)/pcm                                                                \
     $(OUT)/respec                                                             \
     $(OUT)/sim                                                                \

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDES)

$(OUT)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< --std=c++11 $(INCLUDES) -Xcompiler "-O2"

$(OUT)/color_distance_table.o: export LD_LIBRARY_PATH=$(OUT):$(LDPATH)
$(OUT)/color_distance_table.o: $(OUT)/build_error_distances
	$(OUT)/build_error_distances --output_directory=../out
	$(CC) -c -o $@ $(OUT)/color_distance_table.cc $(CFLAGS) $(INCLUDES)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -o $@ $< $(CFLAGS) -I$(GTEST_INCLUDE)

$(OUT)/build_error_distances: $(IMAGE_OBJ) $(COLOR_OBJ)                        \
                              $(OUT)/build_error_distances.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/find_ciede_max_distance: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ)               \
                                $(OUT)/color_distance_table.o                  \
                                $(OUT)/find_ciede_max_distance.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

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
