#include "common.h"  /* Set cuda calls, print matix, set matrix and check results */
#include "multMatrixOnGPU2d1d.h" /*Call the matrix multiplication on cuda*/
#include "multMatrixOnHost.h" /* Call the matrix multiplication on CPU */
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <chrono>

/*These three numbers are going to be the size N for the matrix in NxN */
#define N0  300 /*CHANGE FOR 1000*/
#define N1  400 /*CHANGE FOR 2000*/
#define N2  500 /*CHANGE FOR 4000*/

using namespace std;

int main(int argc, char **argv){


  // Make an array with the three NxN sizes to test three different scenarios
    int test_n[3];
    test_n[0] = N0;
    test_n[1] = N1;
    test_n[2] = N2;

    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");
    printf("\n\n");

    // Main loop to test the 3 diferet scenarios with diferent NxN sizes
    for (int i = 0; i < 3; i++) {
      // set up data size of matrix
      int nx = test_n[i];
      int ny = test_n[i];

      int nxy = nx * ny;
      int nBytes = nxy * sizeof(float);
      printf("Matrix size: nx %d ny %d\n", nx, ny);

      // malloc host memory
      int *h_A, *h_B, *h_R, *gpu_R;
      h_A = (int *)malloc(nBytes);
      h_B = (int *)malloc(nBytes);
      h_R = (int *)malloc(nBytes);
      gpu_R = (int *)malloc(nBytes);

      // initialize data at host side

      initialData(h_A, nxy);
      initialData(h_B, nxy);

      int iterations = 100;
      printf("Calculating in CPU\n");
      float avTime = 0.0;

/**********************************************MULT IN HOST START****************************************************************************/
      for (int i = 0; i < iterations; i++){
        memset(h_R, 0, nBytes);

        // Matrix multiplication
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOnHost(h_A, h_B, h_R, nx, ny);
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

        //printf("multMatrixOnHost elapsed %f ms on iteration %d\n", duration_ms.count(), i);
        avTime += duration_ms.count();
      }
        avTime = avTime / iterations;
/**********************************************MULT IN HOST END******************************************************************************/


      // malloc device global memory
      int *d_MatA, *d_MatB, *d_MatC;
      SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
      SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
      SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

      // transfer data from host to device
      SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
      SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

      // invoke kernel at host side
      int dimx = 128 * ((nx + 128 -1) / 128);
      dim3 block(dimx, 1);
      dim3 grid((nx + block.x - 1) / block.x, ny);


/**********************************************MULT ON GPU START*****************************************************************************/
      printf("Calculating in GPU\n");
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
/**********************************************MULT ON GPU END*******************************************************************************/

      // SAFE_CALL kernel error
      SAFE_CALL(cudaGetLastError(), "Error with last error");

      // copy kernel result back to host side
      SAFE_CALL(cudaMemcpy(gpu_R, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

      printf("Average time in CPU for %dx%d matrix: %f\n", nx, ny, avTime);
      printf("Average time in GPU for %dx%d matrix: %f\n", nx, ny, avTime_gpu);
      printf("Checking result between cpu and gpu\n");
      checkResult(h_R, gpu_R, nxy);
      printf("Speedup: %f\n", avTime / avTime_gpu);

      // free device global memory
      SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
      SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
      SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

      // free host memory
      free(h_A);
      free(h_B);
      free(h_R);
      free(gpu_R);

      printf("\n\n" );
    }

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
