# VCSMC src/ Makefile.
# Dependencies: libpng, clang, CUDA, Intel TBB
# tools dependencies: python colormath, Wand, ImageMagick
INCLUDES=-I. -I$(XXHASH_INCLUDE) -I$(GFLAGS_INCLUDE) -I$(LIBZ26_INCLUDE)       \
    -I$(LIBYAML_INCLUDE) -I$(GPERF_INCLUDE) -I$(TBB_INCLUDE) -I$(OUT)
LIBS+=-lm -lpng $(LIBYAML_LIB) $(GFLAGS_LIB) $(XXHASH_LIB) $(GPERF_LIB)        \
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
LDFLAGS+=-lpthread -lunwind -L/opt/cuda/lib64 -L/usr/lib/cuda/lib64
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
    $(OUT)/codon_table.o                                                       \
    $(OUT)/cuda_utils.o                                                        \
    $(OUT)/genome.o                                                            \
    $(OUT)/mssim.o                                                             \
    $(OUT)/range.o                                                             \
    $(OUT)/serialization.o                                                     \
    $(OUT)/sound_file.o                                                        \
    $(OUT)/state.o                                                             \
    $(OUT)/tls_prng.o

GEN_OBJ=$(OUT)/atari_ntsc_laba_color_table.o                                   \
        $(OUT)/codon_table.o

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/bit_map_unittests.o                                       \
         $(OUTTESTS)/ciede_2k_unittests.o                                      \
         $(OUTTESTS)/kernel_unittests.o                                        \
         $(OUTTESTS)/mssim_unittests.o                                         \
         $(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/serialization_unittests.o                                 \
         $(OUTTESTS)/state_unittests.o

vcsmc: $(OUT)/ciede                                                            \
       $(OUT)/pcm                                                              \
       $(OUT)/ssim

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

$(OUT)/codon_table.o: export LD_LIBRARY_PATH=$(OUT):$(LDPATH)
$(OUT)/codon_table.o: $(OUT)/build_codon_table
	$(OUT)/build_codon_table --output_directory=../out
	$(CC) -c -o $@ $(OUT)/codon_table.cc $(CFLAGS) $(OPT) $(INCLUDES)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -o $@ $< $(CFLAGS) $(OPT) $(INCLUDES) -I$(GTEST_INCLUDE)

$(OUTTESTS)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< --std=c++11 $(INCLUDES) -I$(GTEST_INCLUDE)               \
    -Xcompiler "$(OPT)"

$(OUT)/build_codon_table: $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/build_codon_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/build_laba_table: $(IMAGE_OBJ) $(COLOR_OBJ)                             \
                         $(OUT)/build_laba_table.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/ciede: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(OUT)/ciede.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/pcm: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(OUT)/pcm.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/ssim: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(OUT)/ssim.o
	$(CC) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o      \
    $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) | $(OUTTESTS)/
	$(NVCC) -o $@ $^ $(LIBZ26_LIB) $(INCLUDES) $(LDFLAGS) $(LIBS)                \
    $(GTEST_LIB) -Xcompiler "$(OPT) $(CFLAGS)"

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -rf $(COLOR_OBJ) $(IMAGE_OBJ) $(OBJ) $(TEST_OBJ)                          \
    $(OUT)/atari_ntsc_laba_color_table.{cc,o}                                  \
    $(OUT)/build_codon_table{,.o}                                              \
    $(OUT)/build_laba_table{,.o}                                               \
    $(OUT)/ciede{,.o}                                                          \
    $(OUT)/codon_table.{cc,h,o}                                                \
    $(OUT)/pcm{,.o}                                                            \
    $(OUT)/ssim{,.o}                                                           \
    $(OUTTESTS)/run_unittests{,.o}

