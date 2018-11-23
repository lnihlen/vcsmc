# VCSMC src/ Makefile.
# Dependencies: libpng, clang, CUDA, Intel TBB
# tools dependencies: python colormath, Wand, ImageMagick
OUTGEN=$(OUT)/generators
INCLUDES=-I. -I$(XXHASH_INCLUDE) -I$(GFLAGS_INCLUDE) -I$(LIBZ26_INCLUDE)       \
    -I$(LIBYAML_INCLUDE) -I$(GPERF_INCLUDE) -I$(TBB_INCLUDE) -I$(OUT)          \
    -I$(HALIDE_INCLUDE) -I$(OUTGEN)
LIBS+=-lm -lpng $(LIBYAML_LIB) $(GFLAGS_LIB) $(XXHASH_LIB) $(GPERF_LIB)        \
    -L$(OUT) -ltbb -lHalide -L$(HALIDE_LIB) -lavcodec -lavformat -lavutil      \
    -lswscale
OUTTESTS=$(OUT)/tests

ifeq ($(UNAME), Linux)
LDFLAGS+=-lpthread -lunwind
endif

DEPS=constants.h image_file.h types.h

COLOR_OBJ=$(OUT)/atari_ntsc_rgb_color_table.o                                  \

IMAGE_OBJ=$(OUT)/bit_map.o                                                     \
          $(OUT)/gray_map.o                                                    \
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
    $(OUT)/gaussian_kernel.o                                                   \
    $(OUT)/genome.o                                                            \
    $(OUT)/serialization.o                                                     \
    $(OUT)/sound_file.o                                                        \
    $(OUT)/state.o                                                             \
    $(OUT)/tls_prng.o                                                          \
    $(OUT)/video_decoder.o

GEN_OBJ=$(OUT)/atari_ntsc_lab_color_table.o                                    \
        $(OUT)/codon_table.o

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/bit_map_unittests.o                                       \
         $(OUTTESTS)/ciede_2k_unittests.o                                      \
         $(OUTTESTS)/rgb_to_lab_unittests.o                                    \
         $(OUTTESTS)/serialization_unittests.o                                 \
         $(OUTTESTS)/ssim_unittests.o                                          \
         $(OUTTESTS)/state_unittests.o

vcsmc: $(OUT)/epfg                                                             \
       $(OUT)/split

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

all: vcsmc tests

$(OUT)/%.o: %.cc $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS) $(OPT) $(INCLUDES)

$(OUT)/atari_ntsc_lab_color_table.o: $(OUT)/build_lab_table
	$(OUT)/build_lab_table --output_directory=../out
	$(CXX) -c -o $@ $(OUT)/atari_ntsc_lab_color_table.cc $(CFLAGS) $(OPT)        \
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

$(OUT)/build_codon_table: $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/build_codon_table.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/build_lab_table: $(IMAGE_OBJ) $(COLOR_OBJ) $(OUT)/build_lab_table.o $(HALIDE_AOT)
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(GFLAGS_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/epfg: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(HALIDE_AOT)         \
    $(OUT)/epfg.o $(OUT)/epfg_options.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUT)/split: $(OBJ) $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(HALIDE_AOT)        \
    $(OUT)/split.o
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o      \
    $(IMAGE_OBJ) $(COLOR_OBJ) $(GEN_OBJ) $(HALIDE_AOT) | $(OUTTESTS)/
	$(CXX) -o $@ $^ $(LIBZ26_LIB) $(CFLAGS) $(OPT) $(LDFLAGS) $(LIBS) $(GTEST_LIB)

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

$(OUTGEN)/:
	mkdir -p $(OUTGEN)

.PHONY: clean
clean:
	rm -rf $(COLOR_OBJ) $(IMAGE_OBJ) $(OBJ) $(TEST_OBJ) $(GEN_OBJ)               \
    $(HALIDE_AOT) $(GENERATORS)                                                \
    $(OUT)/atari_ntsc_lab_color_table.{cc,o}                                   \
    $(OUT)/build_codon_table{,.o}                                              \
    $(OUT)/build_lab_table{,.o}                                                \
    $(OUT)/codon_table.{cc,h,o}                                                \
    $(OUT)/epfg{,.o}                                                           \
    $(OUT)/epfg_options.o                                                      \
    $(OUTTESTS)/run_unittests{,.o}

