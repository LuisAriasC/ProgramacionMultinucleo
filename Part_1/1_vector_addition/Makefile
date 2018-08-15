CC = g++
CFLAGS = -std=c++11 -Wall -Ofast
INCLUDES =
LDFLAGS = -lpthread
SOURCES = 1_vector_add.cpp
OUTF = 1_vector_add

$(OUTF): 1_vector_add.o
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

1_vector_add.o: $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o 1_vector_add
