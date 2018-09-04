#include "custom.h"
#include "multMatrixOnHost.h"
#include "multMatrixThreads.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <string.h>

//Default size for N if there are no arguments
#define defaultN 1000
#define defaultIterations 100
#define num_threads 8

using namespace std;

int main(int argc, char const *argv[]){

  int N, iterations;

  if (argc == 2) {
    N = atoi(argv[1]);
    iterations = defaultIterations;
  } else if (argc == 3){
    N = atoi(argv[1]);
    iterations = atoi(argv[2]);
  } else {
    N = defaultN;
    iterations = defaultIterations;
  }

  printf("%s starting...\n\n", argv[0]);

  // Set up data size of matrix
  int nx = N;
  int ny = N;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(int);
  int lBytes = nxy * sizeof(long);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // Malloc host memory
  int *m_A, *m_B;
  long *m_R, *m_Threads;
  m_A = (int *)malloc(nBytes);
  m_B = (int *)malloc(nBytes);
  m_R = (long *)malloc(lBytes);
  m_Threads = (long *)malloc(lBytes);

  // Initialize data at host side
  initialData(m_A, nxy);
  initialData(m_B, nxy);

  //Initialize threads data
  pthread_t threads[num_threads];
  thread_data_t data[num_threads];

  int step = (int)(ny / num_threads);
  for (int i = 0; i < num_threads; i++) {
    data[i].start = step * i;
    data[i].end = step * (i + 1);
    data[i].nx = nx;
    data[i].ny = ny;
    data[i].m_A = m_A;
    data[i].m_B = m_B;
    data[i].m_R = m_Threads;
  }

  printf("Calculating in CPU\n");
  float avTime = 0.0;

/**********************************************MULT IN HOST START****************************************************************************/
  for (int i = 0; i < iterations; i++){
    memset(m_R, 0, lBytes);
    // Matrix multiplication
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnHost(m_A, m_B, m_R, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    avTime += duration_ms.count();
  }
  avTime = avTime / iterations;
/**********************************************MULT IN HOST END******************************************************************************/

  printf("Calculating with threads\n");
  float avTime_threads = 0.0;
/**********************************************MULT ON OMP START*****************************************************************************/
  for (int i = 0; i < iterations; i++){
    memset(m_Threads, 0, lBytes);
    // Matrix multiplication

    auto start_cpu =  chrono::high_resolution_clock::now();
    for (int o = 0; o < num_threads; o++) {
      pthread_create(&threads[o], NULL, multMatrixThreads, (void *)&data[o]);
    }
    for (int j = 0; j < num_threads; j++) {
      pthread_join(threads[i], NULL);
    }
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    avTime_threads += duration_ms.count();
  }
  avTime_threads = avTime_threads / iterations;
/**********************************************MULT ON OMP END*******************************************************************************/

  printf("Average time in CPU %dx%d matrix: %f\n", nx, ny, avTime);
  printf("Average time with Threads %dx%d matrix: %f\n", nx, ny, avTime_threads);
  printf("Checking result between cpu and OpenMP\n");
  checkResult(m_R, m_Threads, nxy);
  printf("Speedup: %f\n", avTime / avTime_threads);

  // Free host memory
  free(m_A);
  free(m_B);
  free(m_R);
  free(m_Threads);

  pthread_exit(NULL);
}
