# VCSMC Makefile.
# Dependencies: libtiff, clang
CC=clang
CFLAGS=-I. -DNTSC -std=c++11 -I/opt/local/include
LDFLAGS=-L/opt/local/lib
LIBS=-lm -lstdc++ -ltiff

DEPS=constants.h types.h
OBJ=background_color_strategy.o color.o frame.o histogram.o image.o kernel.o   \
    opcode.o picc.o scan_line.o state.o switches.o tiff_image_file.o

%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

picc: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o picasm

