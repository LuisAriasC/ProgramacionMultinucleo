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

/*These three numbers are going to be the size N for the matrix in NxN */
#define N0  300 /*CHANGE FOR 1000*/
#define N1  400 /*CHANGE FOR 2000*/
#define N2  500 /*CHANGE FOR 4000*/

using namespace std;

int main(int argc, char const *argv[]){

  // Make an array with the three NxN sizes to test three different scenarios
  int test_n[3];
  test_n[0] = N0;
  test_n[1] = N1;
  test_n[2] = N2;

  printf("%s starting...\n\n", argv[0]);

  // Main loop to test the 3 diferet scenarios with diferent NxN sizes
  for (int i = 0; i < 3; i++) {
    // Set up data size of matrix
    int nx = test_n[i];
    int ny = test_n[i];

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Malloc host memory
    int *m_A, *m_B, *m_R, *m_OMP;
    m_A = (int *)malloc(nBytes);
    m_B = (int *)malloc(nBytes);
    m_R = (int *)malloc(nBytes);
    m_OMP = (int *)malloc(nBytes);

    // Initialize data at host side
    initialData(m_A, nxy);
    initialData(m_B, nxy);

    int iterations = 100;
    printf("Calculating in CPU\n");
    float avTime = 0.0;

/**********************************************MULT IN HOST START****************************************************************************/
    for (int i = 0; i < iterations; i++){
      memset(m_R, 0, nBytes);

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
      memset(m_OMP, 0, nBytes);

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

    printf("\n\n" );
  }

  return (0);
}
