#include "custom.h" /* Print matrix, fill matrix and check results */
#include "multMatrixOnHost.h" /* To call matrix multiplication on CPU*/
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <string.h>

//Default size for N if there are no arguments
#define defaultN 1000
#define defaultIterations 100

using namespace std;

int main(int argc, char **argv){

  int N, iterations;

  if (argc == 2) {
    N = atoi(argv[1]);
    iterations = defaultIterations;
  } else if (argc == 3){
    N = atoi(argv[1]);
    iterations = atoi(argv[2]);
  } else {
    N = defaultN;
    iterations = defaultIterations;
  }

  printf("%s starting...\n", argv[0]);

  // set up data size of matrix
  int nx = N;
  int ny = N;

  int nxy = nx * ny;
  int intBytes = nxy * sizeof(int);
  long longBytes = nxy * sizeof(long);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // Malloc host memory
  int *m_A, *m_B;
  long *m_R;
  m_A = (int *)malloc(intBytes);
  m_B = (int *)malloc(intBytes);
  m_R = (long *)malloc(longBytes);

  // initialize data at host side
  initialData(m_A, nxy);
  initialData(m_B, nxy);

  /*Variables to get the average times (avTime) and to set the iteration of multiplications (arSize)*/
  float avTime = 0.0;

  for (int i = 0; i < iterations; i++){
    memset(m_R, 0, longBytes);
    // Matrix multiplication
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(m_A, m_B, m_R, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    avTime += duration_ms.count();
  }

  //Get Average time on CPU
  avTime = avTime / iterations;
  printf("Average time for %d iterations is %f ms for a multiplication in a %dx%d matrix on Host \n", iterations, avTime, nx, ny );

  // Free host memory
  free(m_A);
  free(m_B);
  free(m_R);

    return (0);
}
