
// Matrix mult in a grid 2D block 1D
__global__ void multMatrixOnGPU2d1d(int *MatA, int *MatB, long *MatC, int nx, int ny) {

  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  if(ix < nx && iy < ny){
    int sum = 0;
    for (int i = 0; i < nx; i++) {
      sum += MatA[iy * ny + i] * MatB[i * nx + ix];
    }
    MatC[iy * ny + ix] = sum;
  }
}
