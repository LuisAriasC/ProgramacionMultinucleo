#include "common.h"  /* Set cuda calls, print matix, set matrix and check results */
#include "multMatrixOnHost.h" /* Call the matrix multiplication on CPU */
#include "multMatrixOMP.h" /* Call the matrix multiplication on CPU with OPENMP*/
#include "multMatrixOnGPU2d1d.h" /*Call the matrix multiplication on cuda*/
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cuda_fp16.h>
#include <chrono>
#include <string.h>
#include <omp.h>

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

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // Main loop to test the 3 diferet scenarios with diferent NxN sizes
    for (int i = 0; i < 3; i++) {

      // Set up data size of matrix
      int nx = test_n[i];
      int ny = test_n[i];

      int nxy = nx * ny;
      int nBytes = nxy * sizeof(float);
      printf("Matrix size: nx %d ny %d\n", nx, ny);

      // Malloc host memory
      int *h_A, *h_B, *h_R, *omp_R , *gpu_R;
      h_A = (int *)malloc(nBytes);
      h_B = (int *)malloc(nBytes);
      h_R = (int *)malloc(nBytes);
      omp_R = (int *)malloc(nBytes);
      gpu_R = (int *)malloc(nBytes);

      // Initialize data at host side with natural numbers
      initialData(h_A, nxy);
      initialData(h_B, nxy);

      // Set the number multiplications to calculate an average time
      int iterations = 100;

  /**********************************************MULT IN HOST START****************************************************************************/
      float avTime_host = 0.0;
      printf("Calculating on CPU with %dx%d\n", nx, ny);
      for (int i = 0; i < iterations; i++){
        // Set the host result matrix to 0
        memset(h_R, 0, nBytes);

        // Matrix multiplication in CPU
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOnHost(h_A, h_B, h_R, nx, ny);
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
        avTime_host += duration_ms.count();
      }
      // Get the average time for CPU
      avTime_host = avTime_host / iterations;
      printf("Average time for %d multiplications in host(no threads) with a matrix of %d x %d is %f ms\n", iterations, nx, ny, avTime_host );
  /**********************************************MULT IN HOST END******************************************************************************/

  /**********************************************MULT ON OMP START*****************************************************************************/
      float avTime_omp = 0.0;
      printf("Calculating on OpenMP with %dx%d\n", nx, ny);
      for (int i = 0; i < iterations; i++){
        // Set the OpenMP result matrix to 0
        memset(omp_R, 0, nBytes);

        // Matrix multiplication with OpenMP
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOMP(h_A, h_B, omp_R, nx, ny);
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
        avTime_omp += duration_ms.count();
      }

      // Get the average time for OpenMP
      avTime_omp = avTime_omp / iterations;
      printf("Average time for %d multiplications in host(using OpenMP) with a matrix of %d x %d is %f ms\n", iterations, nx, ny, avTime_omp );
  /**********************************************MULT ON OMP END*******************************************************************************/

      // Malloc device global memory
      int *d_MatA, *d_MatB, *d_MatC;
      SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
      SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
      SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

      // Transfer data from host to device
      SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
      SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

      // Invoke kernel at host side

      //Uncoment this to set the dinamicall threads calculation
      int dimx;
      if (nx > 1024) {
        //If nx > 1024 set the threads number in a block to 128
        dimx = 128;
      }
      else{
        //Dinamically set the number of threads per block
        dimx = 128 * ((nx + 128 -1) / 128);
      }
      // Comment the line bellow and uncoment the if and else above to set the dinamically calculation for threads
      //int dimx = 128;
      dim3 block(dimx, 1);
      dim3 grid((nx + block.x - 1) / block.x, ny);


  /**********************************************MULT ON GPU START*****************************************************************************/
      float avTime_gpu = 0.0;
      printf("Calculating on GPU with %dx%d\n", nx, ny);
      for (int i = 0; i < iterations; i++) {
        // Set the result matrix on device to 0
        SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error setting d_MatC to 0");

        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOnGPU2d1d<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
        auto end_cpu =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

        avTime_gpu += duration_ms.count();
      }
      // Get the average time for GPU
      avTime_gpu = avTime_gpu / iterations;
      printf("Average time for %d multiplications in GPU with a matrix of %d x %d is %f ms\n", iterations, nx, ny, avTime_gpu);
  /**********************************************MULT ON GPU END*******************************************************************************/

      // SAFE_CALL kernel error
      SAFE_CALL(cudaGetLastError(), "Error with last error");

      // Copy kernel result back to host side
      SAFE_CALL(cudaMemcpy(gpu_R, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

      // Check cpu and omp results
      printf("Checking result between CPU and OpenMP\n");
      checkResult(h_R, omp_R, nxy);
      printf("Speedup between CPU and OpenMP: %f\n", avTime_host / avTime_omp);

      // Check cpu and gpu results
      printf("Checking result between CPU and GPU\n");
      checkResult(h_R, gpu_R, nxy);
      printf("Speedup between CPU and GPU: %f\n", avTime_host / avTime_gpu);

      // Check omp and gpu results
      printf("Checking result between OpenMP and GPU\n");
      checkResult(omp_R, gpu_R, nxy);
      printf("Speedup between OpenMP and GPU: %f\n", avTime_omp / avTime_gpu);

      // Free device global memory
      SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
      SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
      SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

      // Free host memory
      free(h_A);
      free(h_B);
      free(h_R);
      free(omp_R);
      free(gpu_R);

      printf("\n\n\n");
    }

    // Reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
