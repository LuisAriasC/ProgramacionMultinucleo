#include <iostream>
#include <pthread.h>

typedef struct thread_data_struct {
  int start;
  int end;
  int ny;
  int nx;
  int *m_A;
  int *m_B;
  long *m_R;
} thread_data_t;

void *multMatrixThreads(void *args){
  thread_data_t *data = (thread_data_t *) args;
  long sum;
  for (int i = data->start; i < data->end; i++) {
    for (int j = 0; j < data->nx; j++) {
      sum = 0.f;
      for (int k = 0; k < data->nx; k++) {
        sum += data->m_A[i * data->ny + k] * data->m_B[k * data->nx + j];
      }
      data->m_R[i * data->ny + j] = sum;
    }
  }
  pthread_exit(NULL);
}
