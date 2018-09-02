#include "common.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cuda_fp16.h>
#include <chrono>
#include <string.h>
#include <omp.h>

using namespace std;

void multMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
        float sum = 0.0;
        for (int k = 0; k < ny ; k++)
          sum = sum + ia[i * nx + k] * ib[k * nx + j];
        ic[i * nx + j] = sum;
    }
  }

  return;
}


void multMatrixOMP(float *A, float *B, float *C, const int nx, const int ny){
  float *ia = A;
  float *ib = B;
  float *ic = C;

  int i;
  #pragma omp parallel for private(i) shared(ia, ib, ic)
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

// grid 2D block 1D
__global__ void multMatrixOnGPU2d1d(float *MatA, float *MatB, float *MatC, int nx, int ny) {

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    unsigned int idx;
    if (ix < nx && iy < ny){
        idx = iy * nx + ix;
        unsigned int col_position = idx % nx;
        unsigned int h_A_col_init = idx - col_position;
        //printf("Index en h_R es %d con fil y col %d %d\nEn h_A comienza a multiplicar desde index %d \nEn h_B comienza a multiplicar desde index %d\n\n", idx, iy, col_position, h_A_col_init, col_position);
        float sum = 0.0;
        for (int i = 0; i < nx; i++)
          sum = sum + MatA[h_A_col_init + i] * MatB[i * nx + col_position];
        MatC[idx] = sum;
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = 1 << 6;
    int ny = 1 << 6;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *h_R, *omp_R , *gpu_R;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_R = (float *)malloc(nBytes);
    omp_R = (float *)malloc(nBytes);
    gpu_R = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    int iterations = 100;

/**********************************************MULT IN HOST START****************************************************************************/
    float avTime_host = 0.0;
    for (int i = 0; i < iterations; i++){
      memset(h_R, 0, nBytes);

      // Matrix multiplication
      auto start_cpu =  chrono::high_resolution_clock::now();
      multMatrixOnHost(h_A, h_B, h_R, nx, ny);
      auto end_cpu =  chrono::high_resolution_clock::now();
      chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
      //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
      avTime_host += duration_ms.count();
    }

    avTime_host = avTime_host / iterations;
    printf("Average time for %d multiplications in host(no threads) with a matrix of %d x %d is %f ms\n", iterations, nx, ny, avTime_host );
/**********************************************MULT IN HOST END******************************************************************************/

/**********************************************MULT ON OMP START*****************************************************************************/
    float avTime_omp = 0.0;
    for (int i = 0; i < iterations; i++){
      memset(omp_R, 0, nBytes);

      // Matrix multiplication with OpenMP
      auto start_cpu =  chrono::high_resolution_clock::now();
      multMatrixOMP(h_A, h_B, omp_R, nx, ny);
      auto end_cpu =  chrono::high_resolution_clock::now();
      chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
      //printMatrix(m_R, nx, ny);
      //cout << endl;
      //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
      avTime_omp += duration_ms.count();
    }

    avTime_omp = avTime_omp / iterations;
    printf("Average time for %d multiplications in host(using OpenMP) with a matrix of %d x %d is %f ms\n", iterations, nx, ny, avTime_omp );
/**********************************************MULT ON OMP END*******************************************************************************/


    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = 128;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);


/**********************************************MULT ON GPU START*****************************************************************************/
    float avTime_gpu = 0.0;
    for (int i = 0; i < iterations; i++) {
      SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error setting d_MatC to 0");
      auto start_cpu =  chrono::high_resolution_clock::now();
      multMatrixOnGPU2d1d<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
      SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
      auto end_cpu =  chrono::high_resolution_clock::now();
      chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

      avTime_gpu += duration_ms.count();
    }

    avTime_gpu = avTime_gpu / iterations;
    printf("Average time for %d multiplications in GPU with a matrix of %d x %d is %f ms\n", iterations, nx, ny, avTime_gpu);
/**********************************************MULT ON GPU END*******************************************************************************/

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpu_R, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");


    // Check cpu and omp results
    printf("Checking result between cpu and omp\n");
    checkResult(h_R, omp_R, nxy);
    // Check cpu and gpu results
    printf("Checking result between cpu and gpu\n");
    checkResult(h_R, gpu_R, nxy);
    // Check omp and gpu results
    printf("Checking result between omp and gpu\n");
    checkResult(omp_R, gpu_R, nxy);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(h_R);
    free(omp_R);
    free(gpu_R);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
