# VCSMC src/ Makefile.
# Dependencies: libpng, clang,CUDA, Intel TBB
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
CFLAGS+=-I/opt/cuda/include
LDFLAGS+=-lpthread -lunwind -L/opt/cuda/lib64
endif

DEPS=constants.h image_file.h types.h

COLOR_OBJ=$(OUT)/atari_ntsc_abgr_color_table.o                                 \
          $(OUT)/color.o

IMAGE_OBJ=$(OUT)/bit_map.o                                                     \
          $(OUT)/gray_map.o                                                    \
          $(OUT)/image.o                                                       \
          $(OUT)/image_file.o                                                  \
          $(OUT)/value_map.o

OBJ=$(OUT)/assembler.o                                                         \
    $(OUT)/ciede_2k.o                                                          \
    $(OUT)/cuda_utils.o                                                        \
    $(OUT)/kernel.o                                                            \
    $(OUT)/mssim.o                                                             \
    $(OUT)/range.o                                                             \
    $(OUT)/serialization.o                                                     \
    $(OUT)/sound_file.o                                                        \
    $(OUT)/spec.o                                                              \
    $(OUT)/tls_prng.o

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/bit_map_unittests.o                                       \
         $(OUTTESTS)/ciede_2k_unittests.o                                      \
         $(OUTTESTS)/kernel_unittests.o                                        \
         $(OUTTESTS)/mssim_unittests.o                                         \
         $(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/serialization_unittests.o

vcsmc: $(OUT)/find_ciede_max_distance                                          \
       $(OUT)/gen                                                              \
       $(OUT)/pcm                                                              \
       $(OUT)/respec

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

all: vcsmc tests

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(OPT) $(INCLUDES)

$(OUT)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< --std=c++11 $(INCLUDES) -Xcompiler "$(OPT) $(CFLAGS)"

$(OUT)/atari_ntsc_laba_color_table.o: export LD_LIBRARY_PATH=$(OUT):$(LDPATH)
$(OUT)/atari_ntsc_laba_color_table.o: $(OUT)/build_laba_table
	$(OUT)/build_laba_table --output_directory=../out
	$(CC) -c -o $@ $(OUT)/atari_ntsc_laba_color_table.cc $(CFLAGS) $(OPT)        \
    $(INCLUDES)

$(OUT)/atari_ntsc_nyuv_color_table.o: export LD_LIBRARY_PATH=$(OUT):$(LDPATH)
$(OUT)/atari_ntsc_nyuv_color_table.o: $(OUT)/build_nyuv_table
	$(OUT)/build_nyuv_table --output_directory=../out
	$(CC) -c -o $@ $(OUT)/atari_ntsc_nyuv_color_table.cc $(CFLAGS) $(OPT)        \
    $(INCLUDES)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -o $@ $< $(CFLAGS) $(OPT) $(INCLUDES) -I$(GTEST_INCLUDE)

$(OUTTESTS)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< --std=c++11 $(INCLUDES) -I$(GTEST_INCLUDE)               \
    -Xcompiler "$(OPT)"

$(OUT)/build_laba_table: $(IMAGE_OBJ) $(COLOR_OBJ)                             \
                         $(OUT)/build_laba_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/build_nyuv_table: $(IMAGE_OBJ) $(COLOR_OBJ)                             \
                         $(OUT)/build_nyuv_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/gen: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/gen.o                      \
            $(OUT)/atari_ntsc_nyuv_color_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/pcm: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ)                                   \
            $(OUT)/atari_ntsc_nyuv_color_table.o $(OUT)/pcm.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/respec: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ)                                \
               $(OUT)/atari_ntsc_nyuv_color_table.o $(OUT)/respec.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o      \
    $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/atari_ntsc_nyuv_color_table.o |           \
    $(OUTTESTS)/
	$(NVCC) -o $@ $^ $(LIBZ26_LIB) $(INCLUDES) $(LDFLAGS) $(LIBS)                \
    $(GTEST_LIB) -Xcompiler "$(OPT) $(CFLAGS)"

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -rf $(COLOR_OBJ) $(IMAGE_OBJ) $(OBJ) $(TEST_OBJ)                          \
    $(OUT)/build_nyuv_table{,.o}                                               \
    $(OUT)/atari_ntsc_nyuv_color_table.{cc,o}                                  \
    $(OUT)/gen{,.o}                                                            \
    $(OUT)/pcm{,.o}                                                            \
    $(OUT)/respec{,.o}                                                         \
    $(OUTTESTS)/run_unittests{,.o}

