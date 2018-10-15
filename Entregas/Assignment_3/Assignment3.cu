#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>


using namespace std;
#define matrixSize 2000
//#define tileSize 8
#define tileSize 16
//#define tileSize 32

//Matrix Multiplication on CPU
void mulMatrix(long * MatA, long * MatB, long * MatR, const int size){
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        MatR[i * size + j] += MatA[k + i * size] * MatB[k * size + j];
}

//Matrix Multiplication on GPU with a 2D2D implementation
__global__ void multMatrixOnGPU2d2d(long *MatA, long *MatB, long *MatC, const int size){

  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  long sum = 0;

  if (ix < size && iy < size){
    for(int i = 0;i < size; i++)
        sum += MatA[ix * size + i] * MatB[i * size +iy];
    MatC[ix * size + iy] = sum;
  }
}

//Matrix Multiplication on GPU with Tiles
__global__ void multMatrixOnGPUWithTiles(long* MatA, long* MatB, long* MatC, const int size){

  unsigned int ix = threadIdx.x + tileSize * blockIdx.x;
  unsigned int iy = threadIdx.y + tileSize * blockIdx.y;
  unsigned int x = threadIdx.x;
  unsigned int y = threadIdx.y;

  __shared__ long tileA[tileSize][tileSize];
  __shared__ long tileB[tileSize][tileSize];


  //Init tile to 0
  for(int i = 0; i < tileSize; i++){
    for(int j = 0; j < tileSize; j++){
      tileA[i][j] = 0;
      tileB[i][j] = 0;
    }
  }

  long sum = 0;

  //Run over Tile in decreasive manner
  for (int i = (tileSize + size - 1) / tileSize; i >= 0; i--){
      //Just write the values for tileA[][]
      if (i * tileSize + x < size && iy < size)
        tileA[y][x] = MatA[(iy * size) + (i * tileSize) + x];

      //Just write the values for tileB[][]
      if (i * tileSize + y < size && ix < size)
        tileB[y][x] = MatB[(i * tileSize + y) * size + ix];

      __syncthreads();

      for (int j = 0; j < tileSize; j++)
          sum += tileA[y][j] * tileB[j][x];

      __syncthreads();
    }

    if (ix < size && iy < size){
      MatC[iy * size +ix] = sum;
    }
}

int main(int argc, char **argv){

    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = matrixSize;
    int ny = matrixSize;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(long);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    long *h_a, *h_b, *h_R, *gpu_R, *gpu_RT;
    h_a = (long *)malloc(nBytes);
    h_b = (long *)malloc(nBytes);
    h_R = (long *)malloc(nBytes);
    gpu_R = (long *)malloc(nBytes);
    gpu_RT = (long *)malloc(nBytes);

    // initialize data at host side
    initialData(h_a, nxy);
    initialData(h_b, nxy);
    memset(h_R, 0, nBytes);
    memset(gpu_RT, 0, nBytes);
    memset(gpu_R, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    mulMatrix(h_a, h_b, h_R, matrixSize);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnHost elapsed %f ms\n\n", duration_ms.count());

    // malloc device global memory
    long *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_a, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_b, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = tileSize;
    int dimy = tileSize;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    /* MATRIX MULT ON GPU */
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU2d2d<<<grid, block>>>(d_MatA, d_MatB, d_MatC, matrixSize);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpu_R, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    checkResult(h_R, gpu_R, nxy);

    /* MATRIX MULT WITH TILES */
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPUWithTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC, matrixSize);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    printf("sumMatrixOnGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,grid.y,block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpu_RT, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    checkResult(h_R, gpu_RT, nxy);


    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");


    // free host memory
    free(h_a);
    free(h_b);
    free(h_R);
    free(gpu_R);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
