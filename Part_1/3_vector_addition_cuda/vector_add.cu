#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

using namespace std;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// inline double seconds()
// {
//     struct timeval tp;
//     struct timezone tzp;
//     int i = gettimeofday(&tp, &tzp);
//     return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
// }

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(float *a, float *b, float *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

int main( int argc, char* argv[] )
{
    cudaEvent_t start, stop;

    SAFE_CALL(cudaEventCreate(&start), "Error creating start event");
    SAFE_CALL(cudaEventCreate(&stop), "Error creating stop event");

    // Size of vectors
    int n = 1<<21;

    // Host input vectors
    float *h_a;
    float *h_b;

    //Host output vector
    float *h_c;

    // Device input vectors
    float *d_a;
    float *d_b;
    //Device output vector
    float *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(float);

    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Allocate memory for each vector on GPU
    SAFE_CALL(cudaMalloc(&d_a, bytes), "Error allocating da");
    SAFE_CALL(cudaMalloc(&d_b, bytes), "Error allocating db");
    SAFE_CALL(cudaMalloc(&d_c, bytes), "Error allocating dc");

    // Initialize vectors on host
    for(int i = 0; i < n; i++ ) {
        h_a[i] = 1 ;
        h_b[i] = 1 ;
    }

    // Copy host vectors to device
    SAFE_CALL(cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice), "Error copying ha -> da");
    SAFE_CALL(cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice), "Error copying hb -> db");

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 32;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);

    printf("Gridsize: %d Blocksize: %d\n", gridSize, blockSize);

    SAFE_CALL(cudaEventRecord(start, 0), "Error recording event");

    auto start_cpu =  chrono::high_resolution_clock::now();
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    auto end_cpu =  chrono::high_resolution_clock::now();

    SAFE_CALL(cudaEventRecord(stop, 0), "Error recording event stop");
    SAFE_CALL(cudaEventSynchronize(stop), "Error synchronizing events");

    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    float elapsedTime;

    SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop), "Error calculating elapsed time");

    printf("Time spent for %d elements: %.5f ms; %f\n",n, elapsedTime, duration_ms.count());

    // Copy array back to host
    SAFE_CALL(cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost ), "Error copying dc -> hc");

    // Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;

    for(int i=0; i<n; i++)
        sum += h_c[i];

    printf("final result: %f \n", sum/n);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
