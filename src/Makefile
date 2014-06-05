# VCSMC src/ Makefile.
# Dependencies: libtiff, clang, OpenCL
# tools dependencies: python colormath, Wand, ImageMagick
# TODO: move mac-specific stuff to mac-specific vars
CFLAGS += -I. -I/opt/local/include
LDFLAGS += -L/opt/local/lib
LIBS += -lm -ltiff -framework OpenCL
OUTTESTS=$(OUT)/tests

DEPS=constants.h image_file.h strategy.h types.h
OBJ=$(OUT)/background_color_strategy.o                                         \
    $(OUT)/block.o                                                             \
    $(OUT)/cl_buffer_impl.o                                                    \
    $(OUT)/cl_command_queue_impl.o                                             \
    $(OUT)/cl_device_context.o                                                 \
    $(OUT)/cl_image_impl.o                                                     \
    $(OUT)/cl_kernel_impl.o                                                    \
    $(OUT)/color.o                                                             \
    $(OUT)/colu_strip.o                                                        \
    $(OUT)/do_nothing_strategy.o                                               \
    $(OUT)/image.o                                                             \
    $(OUT)/kernel.o                                                            \
    $(OUT)/log.o                                                               \
    $(OUT)/opcode.o                                                            \
    $(OUT)/pallette.o                                                          \
    $(OUT)/pixel_strip.o                                                       \
    $(OUT)/playfield_strategy.o                                                \
    $(OUT)/random.o                                                            \
    $(OUT)/range.o                                                             \
    $(OUT)/schedule.o                                                          \
    $(OUT)/spec.o                                                              \
    $(OUT)/state.o                                                             \
    $(OUT)/tiff_image_file.o

TEST_OBJ=$(OUTTESTS)/range_unittests.o                                         \
         $(OUTTESTS)/state_unittests.o                                         \

$(OUT)/%.o: %.cc $(DEPS)
	$(CC) -c -g -o $@ $< $(CFLAGS)

$(OUTTESTS)/%.o: %.cc $(DEPS) | $(OUTTESTS)/
	$(CC) -c -g -o $@ $< $(CFLAGS) -I$(GTEST_INCLUDE)

$(OUT)/picc: $(OBJ) $(OUT)/picc.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

tests: $(OUT)/run_unittests | $(OUTTESTS)/

$(OUT)/run_unittests: $(OBJ) $(TEST_OBJ) | $(OUTTESTS)/
	$(CC) -g -o $@ $^ $(GTEST_LIB) $(CFLAGS) $(LDFLAGS) $(LIBS)

$(OUTTESTS)/:
	mkdir -p $(OUTTESTS)

.PHONY: clean
clean:
	rm -f $(OBJ) $(OUT)/picc $(TEST_OBJ) $(OUT)/run_unittests

