#include "common.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cuda_fp16.h>
#include <chrono>

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

// grid 2D block 1D
__global__ void multMatrixOnGPU2d1d(float *MatA, float *MatB, float *MatC, int nx, int ny) {

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    //unsigned int col_position = idx % nx;
    //unsigned int row_position = (int)floorf ( (float)(idx / ny ));
    //unsigned int initial_col_mult = idx - col_position;

    unsigned int idx;
    if (ix < nx && iy < ny){
        idx = iy * nx + ix;
        unsigned int col_position = idx % nx;
        printf("Index en h_R es %d con fil y col %d %d\n", idx, iy, col_position);
        printf("En h_A comienza a multiplicar desde col %d fil %d\n", idx - col_position );
    }

    //float sum = 0.0;

    // if (ix < nx && iy < ny)
    //   for (int i = 0; i < nx; i++)
    //     sum = sum + MatA[initial_col_mult + i] * MatB[i * nx + row_position];
    //   MatC[idx] = sum;
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
    int nx = 1 << 2;
    int ny = 1 << 2;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(gpuRef, 0, nBytes);

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

    //printMatrix(h_A, nx, ny);
    //printMatrix(h_B, nx, ny);

    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU2d1d<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    auto end_cpu =  chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnGPU2d1d <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    printMatrix(gpuRef, nx, ny);
    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
