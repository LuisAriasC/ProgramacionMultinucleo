#include <cstdio>
#include <cmath>

__global__ void vectorAdd(flat *a, float *b, float *c, int size){

  int tid = blockIdx.x * blockDim.x + theadIdx.x;

}

int main(){

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  int size = 100;
  size_t bytes = size * sizeof(float);

  h_a = new float[bytes]();
  h_b = new float[bytes]();
  h_c = new float[bytes]();

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  for (int i = 0; i < size; i++) {
    h_a[i] = 1;
    h_b[i] = 2;
  }

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  //Host es en cpu y device es en GPU, se pasan los arreglos a GPU, lo procesa con los devices (d_a, d_b) y despues de d_c lo regresamos a h_c
  //Lo ideal es configurar los bloques e hilos en potencias de dos

  //blockSize = número de hilos
  //gridSize = número de bloques
  int blockSize, gridSize;
  blockSize = 1024;
  gridSize = (int)ceil((float)size / blockSize)

  vectorAdd <<<gridSize, blockSize>>>(d_a, d_b, d_c, size);

}
