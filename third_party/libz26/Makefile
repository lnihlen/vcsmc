all: libz26

libz26:	src/libz26.c $(LIBZ26_OUT)/libz26.o

$(LIBZ26_OUT)/libz26.o: m4c | $(LIBZ26_OUT)/m4
	$(CC) $(CFLAGS) -I$(LIBZ26_OUT) -I$(LIBZ26_INCLUDE) -c src/libz26.c -o $(LIBZ26_OUT)/libz26.o

m4c: $(LIBZ26_OUT)/m4/catchuppixels.c $(LIBZ26_OUT)/m4/cpu.c

$(LIBZ26_OUT)/m4/catchuppixels.c:	src/z26/catchuppixels.m4
	m4 src/z26/catchuppixels.m4 > $(LIBZ26_OUT)/m4/catchuppixels.c

$(LIBZ26_OUT)/m4/cpu.c:	src/z26/cpu.m4
	m4 src/z26/cpu.m4 > $(LIBZ26_OUT)/m4/cpu.c

$(LIBZ26_OUT)/m4:
	mkdir -p $(LIBZ26_OUT)/m4
