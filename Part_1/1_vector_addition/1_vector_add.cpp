#include <cstdio>
#include <chrono> //Para medir tiempos de ejecucion

using namespace std;

void vectorAdd(float *a, float *b, float *c, int size){
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


int main(){

  float *a, *b, *c;
  int size = 1000000;

  a = new float[size]();
  b = new float[size]();
  c = new float[size]();

  for (int i = 0; i < size; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  auto start = chrono::high_resolution_clock::now();
  vectorAdd(a,b,c,size);
  auto end = chrono::high_resolution_clock::now();

  chrono::duration<float, std::milli> duration_ms = end - start;

  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += c[i];
  }

  printf("Resultado: %f\n", sum/size);
  printf("Tiempo(ms): %f\n", duration_ms.count());

  return 0;
}
