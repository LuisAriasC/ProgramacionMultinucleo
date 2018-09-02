#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <string.h>
#include <omp.h>

using namespace std;

void printMatrix(float *mat, const int nx, const int ny){
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++)
      cout << mat[ix] << " ";
    cout << endl;
    mat += nx;
  }

  return;
}

void initialData(float *ip, const int size){
    int i;
    for(i = 0; i < size; i++)
      ip[i] = i * 2;
      //ip[i] = (float)(rand() & 0xFF) / 10.0f;
    return;
}

void multMatrixOMP(float *A, float *B, float *C, const int nx, const int ny){
  float *ia = A;
  float *ib = B;
  float *ic = C;

  int i;
  #pragma omp parallel for private(i,j) shared(ia, ib, ic)
  for (i = 0; i < ny; i++)
  {
    for (int j = 0; j < nx; j++)
    {
        float sum = 0.0;
        for (int k = 0; k < ny ; k++)
          sum = sum + ia[i * nx + k] * ib[k * nx + j];
        ic[i * nx + j] = sum;
    }
  }

  return;
}

int main(int argc, char const *argv[]){

  printf("%s starting...\n", argv[0]);

  // set up data size of matrix
  int nx = 1 << 9;
  int ny = 1 << 9;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // malloc host memory
  float *m_A, *m_B, *m_R;
  m_A = (float *)malloc(nBytes);
  m_B = (float *)malloc(nBytes);
  m_R = (float *)malloc(nBytes);

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
    multMatrixOMP(m_A, m_B, m_R, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    //printMatrix(m_R, nx, ny);
    //cout << endl;
    //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
    avTime += duration_ms.count();
  }

  avTime = avTime / arSize;
  printf("Average time for %d iterations is %f ms for a multiplication in a %dx%d matrix with OpenMP \n", arSize, avTime, nx, ny );

  // free host memory
  free(m_A);
  free(m_B);
  free(m_R);

  return (0);
}
