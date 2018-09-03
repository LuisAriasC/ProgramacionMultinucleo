#include "custom.h"
#include "multMatrixOMP.h"
#include "multMatrixOnHost.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <string.h>
#include <omp.h>

#define N0  350
#define N1  500
#define N2  650

using namespace std;

int main(int argc, char const *argv[]){

  int test_n[3];
  test_n[0] = N0;
  test_n[1] = N1;
  test_n[2] = N2;

  printf("%s starting...\n\n", argv[0]);


  for (int i = 0; i < 3; i++) {
    // set up data size of matrix
    int nx = test_n[i];
    int ny = test_n[i];

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    int *m_A, *m_B, *m_R, *m_OMP;
    m_A = (int *)malloc(nBytes);
    m_B = (int *)malloc(nBytes);
    m_R = (int *)malloc(nBytes);
    m_OMP = (int *)malloc(nBytes);

    // initialize data at host side
    initialData(m_A, nxy);
    initialData(m_B, nxy);

    int iterations = 100;
    printf("Calculating in CPU\n");
    float avTime = 0.0;
    for (int i = 0; i < iterations; i++){
      memset(m_R, 0, nBytes);

      // Matrix multiplication
      auto start_cpu =  chrono::high_resolution_clock::now();
      //multMatrixOnHost(m_A, m_B, m_R, nx, ny);
      auto end_cpu =  chrono::high_resolution_clock::now();
      chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

      //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
      avTime += duration_ms.count();
    }

    avTime = avTime / iterations;
    printf("Average time for %d iterations is %f ms for a multiplication in a %dx%d matrix on Host \n", iterations, avTime, nx, ny );


    printf("Calculating in OpenMP\n");
    float avTime_omp = 0.0;
    for (int i = 0; i < iterations; i++){
      memset(m_OMP, 0, nBytes);

      // Matrix multiplication
      auto start_cpu =  chrono::high_resolution_clock::now();
      multMatrixOMP(m_A, m_B, m_OMP, nx, ny);
      auto end_cpu =  chrono::high_resolution_clock::now();
      chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
      //printMatrix(m_R, nx, ny);
      //cout << endl;
      //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
      avTime_omp += duration_ms.count();
    }

    avTime_omp = avTime_omp / iterations;
    printf("Average time for %d iterations is %f ms for a multiplication in a %dx%d matrix with OpenMP \n", iterations, avTime, nx, ny );


    printf("Checking result between cpu and OpenMP\n");
    checkResult(m_R, m_OMP, nxy);


    printf("Average time in CPU %dx%d matrix: %f\n", nx, ny, avTime);
    printf("Average time in OpenMO %dx%d matrix: %f\n", nx, ny, avTime_omp);
    printf("Speedup: %f\n", avTime / avTime_omp);

    // free host memory
    free(m_A);
    free(m_B);
    free(m_R);
    free(m_OMP);

    printf("\n\n" );
  }

  return (0);
}
