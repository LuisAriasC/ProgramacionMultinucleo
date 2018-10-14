CC = nvcc
CFLAGS = -std=c++11
INCLUDES =
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
SOURCES = Assignment3.cu
OUTF = Assignment3.exe
OBJS = Assignment3.o

$(OUTF): $(OBJS)
        $(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
        $(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
        rm *.o $(OUTF)
