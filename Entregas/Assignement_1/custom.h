#include <cstdlib>
#include <cstdio>

/*Print the matrix*/
void printMatrix(int *mat, const int nx, const int ny){
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++)
			printf("%d ", mat[ix] );
    printf("\n");
    mat += nx;
  }

  return;
}

/*Initialize data in a matrix*/
void initialData(int *ip, const int size){
    int i;
    for(i = 0; i < size; i++)
        //ip[i] = i * 2;
        /*Set a random integer between 1 and 10*/
        ip[i] = (int)(rand() % 10) + 1;
    return;
}

/*Chech if two matrix are equal*/
void checkResult(int *hostRef, int *gpuRef, const int N){

    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Matrix 1 %d Matrix 2 %d in %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}
