/*
  Student Name: Luis Carlos Arias Camacho
  Sudent id: A01364808
*/

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

//Mult Matrix Size
#define mSize 2000
//Tilling matrix size
#define tSize 8


//Funcion de llenado de la matriz en tre 0 y 10 obtenida de la primera tarea
void fillMat(float * ip, const int size) {
  int i;
  for(i = 0; i < size; i++) {
    ip[i] = (rand() / (float)RAND_MAX * 10.0f);
  }
}

__global__ void multMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix < nx && iy < ny) {
    for(int i = 0; i < ny; i++) {
      MatC[ix*ny+iy] += MatA[ix*ny+i] * MatB[i*ny+iy];
    }
  }
}

//Funcion de matrix mult con tiles
__global__ void multMatrixTilled(float *A, float *B, float *C, int nx, int ny) {
  float sum = 0;
  //Algunas partes del codigo fueron obtenidas de los demos vistos en clase
  unsigned int ix = threadIdx.x + blockIdx.x * tSize;
  unsigned int iy = threadIdx.y + blockIdx.y * tSize;

  __shared__ float matTempA[tSize][tSize];
  __shared__ float matTempB[tSize][tSize];

  //Llenamos las matrices shared y iniciado de 0
  for(int i = 0; i < tSize; i ++) {
    for(int j = 0; j < tSize; j++) {
      matTempA[i][j] = 0;
      matTempB[i][j] = 0;
    }
  }

  //vamos a traves de todos los tiles
  for(int i = (tSize + nx - 1)/tSize; i >= 0; i--) {
    if((i * tSize + threadIdx.x) < nx && (iy < ny)) {
      matTempA[threadIdx.y][threadIdx.x] = A[(iy*ny) + (i*tSize+threadIdx.x)];
    }

    if((i * tSize + threadIdx.y) < ny && (ix < nx)) {
      matTempB[threadIdx.y][threadIdx.x] = B[(i*tSize+threadIdx.y) * nx + ix];
    }
    /*//__syncthreads(); command is a block level synchronization barrier. That means it is safe to be used when all threads in a
    //block reach the barrier. It is also possible to use __syncthreads() in conditional code but only when all
    //threads evaluate identically such code otherwise
    //the execution is likely to hang or produce unintended side effects*/

    __syncthreads(); //Tenemos que utilizar syncthreads despues de modificar las matrices en threadIdx
    for(int j = 0; j < tSize; j++) {
      sum += matTempA[threadIdx.y][j] * matTempB[j][threadIdx.x];
    }
    __syncthreads();
  }
  if(ix < nx && iy < ny) {
    C[iy*ny+ix] = sum;
  }
}

//Funcion obtenida de la primera matriz
void multMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  for(int i = 0; i < ny; i++) {
    for(int j = 0; j < nx; j++) {
      for(int k = 0; k < ny; k++) {
        //Operacion para hacer la regla del karatzo fila por culumna
        C[i * nx + j] += (A[i * nx + k] * B[k + nx * j]);
      }
    }
  }
}

//Funcion que checa el resultado el cual ya teniamos de la primera tarea
void checkResult(float *h_R, float *gpu_R, const int N)
{
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N*N; i++){
    if (fabs(h_R[i] - gpu_R[i]) > epsilon){
      match = 0;
      printf("host %f gpu %f\n", h_R[i], gpu_R[i]);
      break;
    }
  }
  if (match)
    printf("YES\n\n");
  else
    printf("No\n\n");
}







int main(int argc, char **argv){

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;

    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = mSize;
    int ny = mSize;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *h_R, *gpu_R;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_R = (float *)malloc(nBytes);
    gpu_R = (float *)malloc(nBytes);

    // initialize data at host side
    fillMat(h_A, nxy);
    fillMat(h_B, nxy);

    memset(h_R, 0, nBytes);
    memset(gpu_R, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(h_A, h_B, h_R, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("Matrix multiplication on Host elapsed %f ms\n\n", duration_ms.count());


    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = tSize;
    int dimy = tSize;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    printf("Matrix multiplication on GPU elapsed: %f ms\n", duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpu_R, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");
    // check device results
    checkResult(h_R, gpu_R, nxy);

    /*MATRIX MULT WITH TILLING*/
    // add matrix at host side for result SAFE_CALLs
    SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error setting d_MatC to 0");
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixTilled<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    printf("Matrix multiplication on GPU with a Tiling Matrix of %dx%d elapsed: %f ms\n", tSize, tSize, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpu_R, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(h_R, gpu_R, nxy);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(h_R);
    free(gpu_R);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
