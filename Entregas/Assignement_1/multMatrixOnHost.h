
//Matrix multiplication on host with an array
void multMatrixOnHost(int *A, int *B, int *C, const int nx, const int ny){
  int *ia = A;
  int *ib = B;
  int *ic = C;

  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
        float sum = 0.0;
        for (int k = 0; k < ny ; k++)
          sum = sum + ia[i * nx + k] * ib[k * nx + j];
        ic[i * nx + j] = sum;
    }
  }

  return;
}
