# VCSMC src/ Makefile.
# Dependencies: libtiff, libpng, clang, OpenCL, gflags
# tools dependencies: python colormath, Wand, ImageMagick
CFLAGS+=-I. -O2
LIBS+=-lm -ltiff -lpng
OUTTESTS=$(OUT)/tests

UNAME:=$(shell uname)

# Linux-specific build flags
ifeq ($(UNAME), Linux)
LIBS+=-lOpenCL -lpthread
CFLAGS+=-DNVIDIA_OPENCL_LAMENESS
endif

# OS X-specific build flags
ifeq ($(UNAME), Darwin)
CFLAGS+=-I/opt/local/include
LDFLAGS+=-L/opt/local/lib
LIBS+=-framework OpenCL
endif

DEPS=constants.h image_file.h types.h
OBJ=$(OUT)/adjacency_map.o                                                     \
    $(OUT)/assembler.o                                                         \
    $(OUT)/block.o                                                             \
    $(OUT)/bit_map.o                                                           \
    $(OUT)/cl_buffer_impl.o                                                    \
    $(OUT)/cl_command_queue_impl.o                                             \
    $(OUT)/cl_device_context.o                                                 \
    $(OUT)/cl_image_impl.o                                                     \
    $(OUT)/cl_kernel_impl.o                                                    \
    $(OUT)/cl_program.o                                                        \
    $(OUT)/color.o                                                             \
    $(OUT)/color_distances_table.o                                             \
    $(OUT)/color_table.o                                                       \
    $(OUT)/frame_fitter.o                                                      \
    $(OUT)/gray_map.o                                                          \
    $(OUT)/image.o                                                             \
    $(OUT)/image_file.o                                                        \
    $(OUT)/line_kernel.o                                                       \
    $(OUT)/opcode.o                                                            \
    $(OUT)/player_fitter.o                                                     \
    $(OUT)/random.o                                                            \
    $(OUT)/range.o                                                             \
    $(OUT)/schedule.o                                                          \
    $(OUT)/spec.o                                                              \
    $(OUT)/state.o                                                             \
    $(OUT)/value_map.o                                                         \
    $(OUT)/video_frame_data.o                                                  \
    $(OUT)/video_frame_data_parser.o                                           \

TEST_OBJ=$(OUTTESTS)/adjacency_map_unittests.o                                 \
         $(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/bit_map_unittests.o                                       \
         $(OUTTESTS)/block_unittests.o                                         \
         $(OUTTESTS)/ciede2k_unittests.o                                       \
         $(OUTTESTS)/downsample_errors_unittests.o                             \
         $(OUTTESTS)/fft_radix_2_unittests.o                                   \
         $(OUTTESTS)/inverse_fft_norm_unittests.o                              \
         $(OUTTESTS)/k_means_unittests.o                                       \
         $(OUTTESTS)/lab_to_rgb_unittests.o                                    \
         $(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/rgb_to_lab_unittests.o                                    \
         $(OUTTESTS)/state_unittests.o                                         \

all: tests                                                                     \
     $(OUT)/asm                                                                \
     $(OUT)/bed                                                                \
     $(OUT)/check                                                              \
     $(OUT)/exty                                                               \
     $(OUT)/picc                                                               \
     $(OUT)/sim                                                                \
     $(OUT)/smap

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -o $@ $< $(CFLAGS) -I$(GTEST_INCLUDE)

$(OUT)/picc: $(OBJ) $(OUT)/picc.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/asm: $(OBJ) $(OUT)/asm.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/exty: $(OBJ) $(OUT)/exty.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/smap: $(OBJ) $(OUT)/smap.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/sim: $(OBJ) $(OUT)/sim.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/bed: $(OBJ) $(OUT)/bed.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/check: $(OBJ) $(OUT)/check.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o | $(OUTTESTS)/
	$(CC) -o $@ $^ $(GTEST_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -rf $(OUT)
