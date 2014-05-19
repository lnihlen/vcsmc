# VCSMC Makefile.
# Dependencies: libtiff, clang
CC=clang
CFLAGS=-I. -DNTSC -std=c++11 -I/opt/local/include
# TODO: move mac-specific stuff to mac-specific vars
LDFLAGS=-L/opt/local/lib
LIBS=-lm -lstdc++ -ltiff -framework OpenCL

DEPS=constants.h image_file.h strategy.h types.h
OBJ=background_color_strategy.o                                                \
    cl_buffer_impl.o                                                           \
    cl_command_queue_impl.o                                                    \
    cl_device_context.o                                                        \
    cl_image_impl.o                                                            \
    cl_kernel_impl.o                                                           \
    color.o                                                                    \
    do_nothing_strategy.o                                                      \
    image.o                                                                    \
    kernel.o                                                                   \
    log.o                                                                      \
    opcode.o                                                                   \
    pixel_strip.o                                                              \
    playfield_strategy.o                                                       \
    scan_line.o state.o                                                        \
    tiff_image_file.o

%.o: %.cc $(DEPS)
	$(CC) -c -g -o $@ $< $(CFLAGS)

picc: $(OBJ) picc.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

ciede2k_test: ciede2k_test.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

rgb_to_lab_test: rgb_to_lab_test.o
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o picc ciede2k_test rgb_to_lab_test

