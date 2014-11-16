# VCSMC src/ Makefile.
# Dependencies: libtiff, libpng, clang, OpenCL
# tools dependencies: python colormath, Wand, ImageMagick
CFLAGS+=-I.
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
OBJ=$(OUT)/assembler.o                                                         \
    $(OUT)/block.o                                                             \
    $(OUT)/bit_map.o                                                           \
    $(OUT)/cl_buffer_impl.o                                                    \
    $(OUT)/cl_command_queue_impl.o                                             \
    $(OUT)/cl_device_context.o                                                 \
    $(OUT)/cl_image_impl.o                                                     \
    $(OUT)/cl_kernel_impl.o                                                    \
    $(OUT)/cl_program.o                                                        \
    $(OUT)/color.o                                                             \
    $(OUT)/color_table.o                                                       \
    $(OUT)/colu_strip.o                                                        \
    $(OUT)/gray_map.o                                                          \
    $(OUT)/image.o                                                             \
    $(OUT)/image_fitter.o                                                      \
    $(OUT)/opcode.o                                                            \
    $(OUT)/palette.o                                                           \
    $(OUT)/pixel_strip.o                                                       \
    $(OUT)/random.o                                                            \
    $(OUT)/range.o                                                             \
    $(OUT)/schedule.o                                                          \
    $(OUT)/shape_fit.o                                                         \
    $(OUT)/spec.o                                                              \
    $(OUT)/state.o                                                             \
    $(OUT)/tiff_image_file.o                                                   \
    $(OUT)/value_map.o

TEST_OBJ=$(OUTTESTS)/assembler_unittests.o                                     \
         $(OUTTESTS)/block_unittests.o                                         \
         $(OUTTESTS)/ciede2k_unittests.o                                       \
         $(OUTTESTS)/fft_radix_2_unittests.o                                   \
         $(OUTTESTS)/inverse_fft_norm_unittests.o                              \
         $(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/rgb_to_lab_unittests.o                                    \
         $(OUTTESTS)/shape_fit_unittests.o                                     \
         $(OUTTESTS)/state_unittests.o                                         \

all: $(OUT)/picc $(OUT)/asm $(OUT)/exty $(OUT)/smap tests

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -g -o $@ $< $(CFLAGS)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -g -o $@ $< $(CFLAGS) -I$(GTEST_INCLUDE)

$(OUT)/picc: $(OBJ) $(OUT)/picc.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/asm: $(OBJ) $(OUT)/asm.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/exty: $(OBJ) $(OUT)/exty.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUT)/smap: $(OBJ) $(OUT)/smap.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

tests: $(OUTTESTS)/run_unittests | $(OUTTESTS)/

$(OUTTESTS)/run_unittests: $(OBJ) $(TEST_OBJ) $(OUTTESTS)/run_unittests.o | $(OUTTESTS)/
	$(CC) -g -o $@ $^ $(GTEST_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -rf $(OUT)
