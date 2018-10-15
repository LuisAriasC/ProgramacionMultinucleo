#include <cstdlib>
#include <cstdio>

/*Print the matrix*/
void printMatrix(float *mat, const int nx, const int ny){
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++)
			printf("%f ", mat[ix] );
    printf("\n");
    mat += nx;
  }

  return;
}

/*Initialize data in a matrix*/
void initialData(long * ip, const int size){
    for(int i = 0; i < size; i++)
        ip[i] = (long)rand()/(RAND_MAX/ 10.0f);
    retur;
}

void fillMatrix(long * matrix)
{
    int i;
    int size = N*N;
    for(i = 0; i < size; i++)
    {
        matrix[i] = (long)rand()/(RAND_MAX/ 10.0f);
    }

    return;
}

/*Chech if two matrix are equal*/
void checkResult(float *hostRef, float *gpuRef, const int N){

    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Matrix 1 %f Matrix 2 %f in %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}
