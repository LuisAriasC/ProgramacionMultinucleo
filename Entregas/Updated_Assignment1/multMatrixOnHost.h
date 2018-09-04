
//Matrix multiplication on host with an array
void multMatrixOnHost(int *A, int *B, long *C, const int nx, const int ny){
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
        float sum = 0.0;
        for (int k = 0; k < ny ; k++)
          sum = sum + A[i * nx + k] * B[k * nx + j];
        C[i * nx + j] = sum;
    }
  }
  return;
}
