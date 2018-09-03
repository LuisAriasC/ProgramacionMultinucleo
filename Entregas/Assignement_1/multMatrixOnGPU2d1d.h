// Matrix mult in a grid 2D block 1D
__global__ void multMatrixOnGPU2d1d(int *MatA, int *MatB, int *MatC, int nx, int ny) {

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    unsigned int idx;
    if (ix < nx && iy < ny){
        idx = iy * nx + ix;

        //Get the start position of the column to make the multiplication and run through the "rows" in the matrix B
        unsigned int col_position = idx % nx;
        //Get the initial position of the colums in the matrix A to make the multiplication and run through the colums in matrix A
        unsigned int h_A_col_init = idx - col_position;
        //printf("Index en h_R es %d con fil y col %d %d\nEn h_A comienza a multiplicar desde index %d \nEn h_B comienza a multiplicar desde index %d\n\n", idx, iy, col_position, h_A_col_init, col_position);
        float sum = 0.0;
        for (int i = 0; i < nx; i++)
          sum = sum + MatA[h_A_col_init + i] * MatB[i * nx + col_position];
        MatC[idx] = sum;
    }
}
