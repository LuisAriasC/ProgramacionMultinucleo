#include "custom.h" /* print matix, set matrix and check results */
#include "multMatrixOMP.h" /* Call the matrix multiplication on CPU with OPENMP*/
#include "multMatrixOnHost.h" /* Call the matrix multiplication on CPU */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <string.h>
#include <omp.h>

//Default size for N if there are no arguments
#define defaultN 1000
#define defaultIterations 100

using namespace std;

int main(int argc, char const *argv[]){

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

  printf("%s starting...\n\n", argv[0]);

  // Set up data size of matrix
  int nx = N;
  int ny = N;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(int);
  int lBytes = nxy * sizeof(long);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // Malloc host memory
  int *m_A, *m_B;
  long *m_R, *m_OMP;
  m_A = (int *)malloc(nBytes);
  m_B = (int *)malloc(nBytes);
  m_R = (long *)malloc(lBytes);
  m_OMP = (long *)malloc(lBytes);

  // Initialize data at host side
  initialData(m_A, nxy);
  initialData(m_B, nxy);

  printf("Calculating in CPU\n");
  float avTime = 0.0;

/**********************************************MULT IN HOST START****************************************************************************/
  for (int i = 0; i < iterations; i++){
    memset(m_R, 0, lBytes);
    // Matrix multiplication
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(m_A, m_B, m_R, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    avTime += duration_ms.count();
  }
  avTime = avTime / iterations;
/**********************************************MULT IN HOST END******************************************************************************/

  printf("Calculating in OpenMP\n");
  float avTime_omp = 0.0;
/**********************************************MULT ON OMP START*****************************************************************************/
  for (int i = 0; i < iterations; i++){
    memset(m_OMP, 0, lBytes);
    // Matrix multiplication
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOMP(m_A, m_B, m_OMP, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    avTime_omp += duration_ms.count();
  }
  avTime_omp = avTime_omp / iterations;
/**********************************************MULT ON OMP END*******************************************************************************/

  printf("Average time in CPU %dx%d matrix: %f\n", nx, ny, avTime);
  printf("Average time in OpenMO %dx%d matrix: %f\n", nx, ny, avTime_omp);
  printf("Checking result between cpu and OpenMP\n");
  checkResult(m_R, m_OMP, nxy);
  printf("Speedup: %f\n", avTime / avTime_omp);

  // Free host memory
  free(m_A);
  free(m_B);
  free(m_R);
  free(m_OMP);

  return (0);
}
