# VCSMC Makefile.
# Dependencies: libtiff, clang
CC=clang
CFLAGS=-I. -DNTSC -std=c++11 -I/opt/local/include
LDFLAGS=-L/opt/local/lib
LIBS=-lm -lstdc++ -ltiff

DEPS=constants.h image_file.h strategy.h types.h
OBJ=background_color_strategy.o color.o colu_strip.o do_nothing_strategy.o     \
    frame.o histogram.o image.o kernel.o log.o opcode.o picc.o scan_line.o     \
    state.o tiff_image_file.o

%.o: %.cc $(DEPS)
	$(CC) -c -g -o $@ $< $(CFLAGS)

picc: $(OBJ)
	$(CC) -g -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o picc

