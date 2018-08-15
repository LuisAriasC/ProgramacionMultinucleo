CC = g++
CFLAGS = -std=c++11 -Wall -Ofast -fopenmp
INCLUDES =
LDFLAGS = -lpthread
SOURCES = 2_vector_add_omp.cpp
OUTF = 2_vector_add_omp

$(OUTF): 2_vector_add_omp.o
				$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

2_vector_add_omp.o: $(SOURCES)
				$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
				rm *.o 2_vector_add_omp
