void multMatrixOMP(float *A, float *B, float *C, const int nx, const int ny){

  float *ia = A;
  float *ib = B;
  float *ic = C;

  int i,j,k;
  #pragma omp parallel for private(i,j,k) shared(ia, ib, ic)
  for (i = 0; i < ny; i++)
  {
    for (j = 0; j < nx; j++)
    {
        float sum = 0.0;
        for (k = 0; k < ny ; k++)
          sum = sum + ia[i * nx + k] * ib[k * nx + j];
        ic[i * nx + j] = sum;
    }
  }

  return;
}
