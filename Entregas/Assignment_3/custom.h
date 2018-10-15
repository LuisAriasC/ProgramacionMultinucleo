/*
  STUDENT: LUIS CARLOS ARIAS CAMACHO
  STUDENT ID: A01364808
*/

#include <cstdlib>
#include <cstdio>

/*Print the matrix*/
void printMatrix(long * ip, const int nx, const int ny){
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++)
			printf("%li ", ip[ix] );
    printf("\n");
    ip += nx;
  }
}

/*Initialize data in a matrix*/
void initialData(long * ip, const int size){
    for(int i = 0; i < size; i++)
        ip[i] = (long)rand()/(RAND_MAX/ 10.0f);
    return;
}

void checkResult(long *hostRef, long *gpuRef, const int N){

    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++){

        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Matrix 1 %li Matrix 2 %li in %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("MATCH!!!!\n\n");
    else
        printf("DO NOT MATCH!!!!\n\n");
}
