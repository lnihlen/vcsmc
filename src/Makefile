# VCSMC src/ Makefile.
# Dependencies: libpng, clang, CUDA, Intel TBB
# tools dependencies: python colormath, Wand, ImageMagick
OUTGEN=$(OUT)/generators
INCLUDES=-I. -I$(XXHASH_INCLUDE) -I$(GFLAGS_INCLUDE) -I$(LIBZ26_INCLUDE)       \
    -I$(LIBYAML_INCLUDE) -I$(GPERF_INCLUDE) -I$(TBB_INCLUDE) -I$(OUT)          \
    -I$(HALIDE_INCLUDE) -I$(OUTGEN)
LIBS+=-lm -lpng $(LIBYAML_LIB) $(GFLAGS_LIB) $(XXHASH_LIB) $(GPERF_LIB)        \
    -L$(OUT) -ltbb -lcudart -lHalide -L$(HALIDE_LIB)
OUTTESTS=$(OUT)/tests

ifeq ($(UNAME), Linux)
CFLAGS+=-I/opt/cuda/include -I/usr/lib/cuda/include
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

GENERATORS=$(OUTGEN)/ciede_2k_generator                                        \
           $(OUTGEN)/covariance_generator                                      \
           $(OUTGEN)/mean_generator                                            \
           $(OUTGEN)/ssim_generator                                            \
           $(OUTGEN)/rgb_to_lab_generator                                      \
           $(OUTGEN)/variance_generator

HALIDE_AOT=$(OUTGEN)/ciede_2k.o                                                \
           $(OUTGEN)/covariance.o                                              \
           $(OUTGEN)/mean.o                                                    \
           $(OUTGEN)/rgb_to_lab.o                                              \
           $(OUTGEN)/ssim.o                                                    \
           $(OUTGEN)/variance.o

OBJ=$(OUT)/assembler.o                                                         \
    $(OUT)/codon_table.o                                                       \
    $(OUT)/cuda_utils.o                                                        \
    $(OUT)/genome.o                                                            \
    $(OUT)/serialization.o                                                     \
    $(OUT)/sound_file.o                                                        \
    $(OUT)/state.o                                                             \
    $(OUT)/tls_prng.o

GEN_OBJ=$(OUT)/atari_ntsc_laba_color_table.o                                   \
        $(OUT)/codon_table.o

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/bit_map_unittests.o                                       \
         $(OUTTESTS)/ciede_2k_unittests.o                                      \
         $(OUTTESTS)/mssim_unittests.o                                         \
         $(OUTTESTS)/rgb_to_lab_unittests.o                                    \
         $(OUTTESTS)/serialization_unittests.o                                 \
         $(OUTTESTS)/state_unittests.o

vcsmc: $(OUT)/pcm                                                              \
       $(OUT)/ssim

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

all: vcsmc tests

$(OUT)/%.o: %.cc $(DEPS) $(HALIDE_AOT)
	$(CXX) -c -o $@ $< $(CFLAGS) $(OPT) $(INCLUDES)

$(OUT)/%.o: %.cu $(DEPS) $(HALIDE_AOT)
	$(NVCC) -c -o $@ $< --std=c++11 $(INCLUDES) -Xcompiler "$(OPT) $(CFLAGS)"

$(OUT)/atari_ntsc_laba_color_table.o: $(OUT)/build_laba_table
	$(OUT)/build_laba_table --output_directory=../out
	$(CXX) -c -o $@ $(OUT)/atari_ntsc_laba_color_table.cc $(CFLAGS) $(OPT)       \
    $(INCLUDES)

$(OUT)/codon_table.o: $(OUT)/build_codon_table
	$(OUT)/build_codon_table --output_directory=../out
	$(CXX) -c -o $@ $(OUT)/codon_table.cc $(CFLAGS) $(OPT) $(INCLUDES)

$(OUTGEN)/%: %.cc | $(OUTGEN)/
	$(CXX) -o $@ $^ $(HALIDE_DIR)/tools/GenGen.cpp -I$(HALIDE_INCLUDE) -lpthread \
  -lHalide -ldl -std=c++11 -march=native -O3 -L$(HALIDE_LIB)

$(OUTGEN)/ciede_2k.o: $(OUTGEN)/ciede_2k_generator | $(OUTGEN)/
	$(OUTGEN)/ciede_2k_generator -g ciede_2k -o $(OUTGEN) -e o,h target=host

$(OUTGEN)/covariance.o: $(OUTGEN)/covariance_generator | $(OUTGEN)/
	$(OUTGEN)/covariance_generator -g covariance -o $(OUTGEN) -e o,h target=host

$(OUTGEN)/mean.o: $(OUTGEN)/mean_generator | $(OUTGEN)/
	$(OUTGEN)/mean_generator -g mean -o $(OUTGEN) -e o,h target=host

$(OUTGEN)/rgb_to_lab.o: $(OUTGEN)/rgb_to_lab_generator | $(OUTGEN)/
	$(OUTGEN)/rgb_to_lab_generator -g rgb_to_lab -o $(OUTGEN) -e o,h target=host

$(OUTGEN)/ssim.o: $(OUTGEN)/ssim_generator | $(OUTGEN)/
	$(OUTGEN)/ssim_generator -g ssim -o $(OUTGEN) -e o,h target=host

$(OUTGEN)/variance.o: $(OUTGEN)/variance_generator | $(OUTGEN)/
	$(OUTGEN)/variance_generator -g variance -o $(OUTGEN) -e o,h target=host

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CXX) -c -o $@ $< $(CFLAGS) $(OPT) $(INCLUDES) -I$(GTEST_INCLUDE)

$(OUTTESTS)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< --std=c++11 $(INCLUDES) -I$(GTEST_INCLUDE)               \
    -Xcompiler "$(OPT)"

$(OUT)/build_codon_table: $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/build_codon_table.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/build_laba_table: $(IMAGE_OBJ) $(COLOR_OBJ)                             \
                         $(OUT)/build_laba_table.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/pcm: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(OUT)/pcm.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/ssim: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(OUT)/ssim.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o      \
    $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(HALIDE_AOT) | $(OUTTESTS)/
	$(NVCC) -o $@ $^ $(LIBZ26_LIB) $(INCLUDES) $(LDFLAGS) $(LIBS)                \
    $(GTEST_LIB) -Xcompiler "$(OPT) $(CFLAGS)"

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

$(OUTGEN)/:
	mkdir -p $(OUTGEN)

.PHONY: clean
clean:
	rm -rf $(COLOR_OBJ) $(IMAGE_OBJ) $(OBJ) $(TEST_OBJ) $(GEN_OBJ)               \
    $(HALIDE_AOT) $(GENERATORS)                                                \
    $(OUT)/atari_ntsc_laba_color_table.{cc,o}                                  \
    $(OUT)/build_codon_table{,.o}                                              \
    $(OUT)/build_laba_table{,.o}                                               \
    $(OUT)/codon_table.{cc,h,o}                                                \
    $(OUT)/pcm{,.o}                                                            \
    $(OUT)/ssim{,.o}                                                           \
    $(OUTTESTS)/run_unittests{,.o}

