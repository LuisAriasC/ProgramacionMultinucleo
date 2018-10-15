#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>


using namespace std;
#define N 2000
#define TILE 16
#define tileSize 16
#define TX 16
#define TY 16


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
    int nx = N;
    int ny = N;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(long);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    long *h_m1, *h_m2, *hostRef, *gpuRef, *gpuRefTiles;
    h_m1 = (long *)malloc(nBytes);
    h_m2 = (long *)malloc(nBytes);
    hostRef = (long *)malloc(nBytes);
    gpuRef = (long *)malloc(nBytes);
    gpuRefTiles = (long *)malloc(nBytes);

    // initialize data at host side

    initialData(h_m1, nxy);
    initialData(h_m2, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRefTiles, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    mulMatrix(h_m1, h_m2, hostRef, N);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnHost elapsed %f ms\n\n", duration_ms.count());

    // malloc device global memory
    long *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_m1, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_m2, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = TX;
    int dimy = TY;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    /*******************Normal********************************/
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU2d2d<<<grid, block>>>(d_MatA, d_MatB, d_MatC, N);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;


    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    checkResult(hostRef, gpuRef, nxy);

    /*******************Tiles********************************/
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPUWithTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC, N);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,grid.y,block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRefTiles, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    checkResult(hostRef, gpuRefTiles, nxy);




    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");


    // free host memory
    free(h_m1);
    free(h_m2);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
