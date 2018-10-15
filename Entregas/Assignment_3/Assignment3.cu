#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>


using namespace std;
#define N 2000
#define TILE 16
#define TX 16
#define TY 16


//multiplication of matrices in cpu
void mulMatrix(long * m_r, long * m1, long * m2){
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        m_r[i * N + j] += m1[k + i * N] * m2[k * N + j];
}

__global__ void mulMatrixGPU2D(long *MatA, long *MatB, long *MatC)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  long sum = 0;

  if (ix < N && iy < N)
  {
    for(int in =0;in<N;in++)
    {
        sum += MatA[ix*N+in] * MatB[in*N+iy];
    }
    MatC[ix*N+iy]=sum;
  }
}


__global__ void mulMatrixGPUTiles(long* A, long* B, long* C)
{
  long sum = 0;

  unsigned int ix = threadIdx.x + TILE * blockIdx.x;
  unsigned int iy = threadIdx.y + TILE * blockIdx.y;

  unsigned int x = threadIdx.x;
  unsigned int y = threadIdx.y;


  __shared__ long SharedA[TILE][TILE];
  __shared__ long SharedB[TILE][TILE];

  for(int a = 0; a < TILE;a++)// Se inician en 0 los arreglos
  {
    for(int b = 0; b < TILE; b++)
    {
      SharedA[a][b] = 0.0;
      SharedB[a][b] = 0.0;
    }
  }

  for (int a = (TILE + N - 1)/TILE; a >=0; a--) //Recorrer todas las tiles se hace Ceil para asegurar de tener todos los datos, se recorre de forma invertida para conservar los 0s.
    {
      if (a*TILE + x < N && iy < N) //Para que no intente acceder a espacios que no existen de la matriz A
        SharedA[y][x] = A[iy*N + a*TILE + x];

      if (a*TILE + y < N && ix < N)
        SharedB[y][x] = B[(a*TILE + y)*N + ix];

      __syncthreads();

      for (int b = 0; b < TILE; b++)
          sum += SharedA[y][b] * SharedB[b][x];

      __syncthreads();
    }

    if (ix < N && iy < N)
    {
      C[iy*N+ix] = sum;
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
    mulMatrix(hostRef, h_m1, h_m2);
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
    mulMatrixGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;


    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    if(checkResult(hostRef, gpuRef))
      printf("They are equal\n\n");
    else
      printf("They are different\n\n");

    /*******************Tiles********************************/
    start_cpu =  chrono::high_resolution_clock::now();
    mulMatrixGPUTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    printf("sumMatrixOnGPUTiles <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,grid.y,block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRefTiles, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    if(checkResult(hostRef, gpuRefTiles))
      printf("They are equal\n\n");
    else
      printf("They are different\n\n");




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
