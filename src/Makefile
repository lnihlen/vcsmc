# VCSMC Makefile.
# Dependencies: libtiff, clang
CC=clang
CFLAGS=-I. -DNTSC -std=c++11 -I/opt/local/include
LDFLAGS=-L/opt/local/lib
LIBS=-lm -lstdc++ -ltiff

DEPS=image.h image_file.h types.h
OBJ=color.o frame.o image.o picasm.o player.o scan_line.cc tiff_image_file.o

%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

picasm: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o picasm

