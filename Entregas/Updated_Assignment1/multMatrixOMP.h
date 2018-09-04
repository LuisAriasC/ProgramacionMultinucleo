
/*Make multiplication on OpenMP*/
void multMatrixOMP(int *A, int *B, long *C, const int nx, const int ny){
  /*Share i,j and k*/
  int i,j,k;
  #pragma omp parallel for private(i,j,k) shared(A, B, C)
  for (i = 0; i < ny; i++)
  {
    for (j = 0; j < nx; j++)
    {
        float sum = 0.0;
        for (k = 0; k < ny ; k++)
          sum = sum + A[i * nx + k] * B[k * nx + j];
        C[i * nx + j] = sum;
    }
  }
  return;
}
