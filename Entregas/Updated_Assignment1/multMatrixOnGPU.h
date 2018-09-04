
// Matrix mult in a grid 2D block 1D
__global__ void multMatrixOnGPU2d1d(int *MatA, int *MatB, long *MatC, int nx, int ny) {

  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  if(ix < nx && iy < ny){
    long sum = 0;
    for (int i = 0; i < ny; i++) {
      sum += MatA[iy * nx + i] * MatB[i * ny + ix];
    }
    MatC[iy * nx + ix] = sum;
  }
}
