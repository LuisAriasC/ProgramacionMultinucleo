#include "custom.h"
#include "multMatrixOnHost.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <string.h>


#define N0  350
#define N1  500
#define N2  650

using namespace std;

int main(int argc, char **argv){

    int test_n[3];
    test_n[0] = N0;
    test_n[1] = N1;
    test_n[2] = N2;

    printf("%s starting...\n", argv[0]);

    for (int i = 0; i < 3; i++) {
      // set up data size of matrix
      int nx = test_n[i];
      int ny = test_n[i];

      int nxy = nx * ny;
      int nBytes = nxy * sizeof(float);
      printf("Matrix size: nx %d ny %d\n", nx, ny);

      // malloc host memory
      int *m_A, *m_B, *m_R;
      m_A = (int *)malloc(nBytes);
      m_B = (int *)malloc(nBytes);
      m_R = (int *)malloc(nBytes);

      // initialize data at host side
      initialData(m_A, nxy);
      initialData(m_B, nxy);

      /*Variables to get the average times (avTime) and to set the iteration of multiplications (arSize)*/
      float avTime = 0.0;
      int arSize = 100;

      for (int i = 0; i < arSize; i++){
        memset(m_R, 0, nBytes);

        // Matrix multiplication
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOnHost(m_A, m_B, m_R, nx, ny);
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

        //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
        avTime += duration_ms.count();
      }

      avTime = avTime / arSize;
      printf("Average time for %d iterations is %f ms for a multiplication in a %dx%d matrix on Host \n", arSize, avTime, nx, ny );

      // free host memory
      free(m_A);
      free(m_B);
      free(m_R);

      printf("\n\n" );
    }

    return (0);
}
