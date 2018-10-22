/*
  Author: Luis Carlos Arias Camacho
  Student ID: A01364808
*/
#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "common.h"
#include <cuda_runtime.h>

#define img_dest "Images/"
#define default_image "dog3.jpeg"
#define C_SIZE 256

using namespace std;

/***

//This function is used to equalize an histogram

  //src_histogram - imput histogram as a one dimentional array of ints

***/
void equalize(int * src_histogram, int * eq_histogram, int size){
    int step = size / C_SIZE;
    int sum = 0;

    for(int i=0; i < C_SIZE; i++){
        sum += src_histogram[i];
        if (sum / step <= C_SIZE)
          eq_histogram[i] = sum / step;
        else
          eq_histogram[i] = C_SIZE - 1;
    }
}

// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
// grayWidthStep - number of gray bytes
__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep){
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)){
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		const int gray_tid = yIndex * grayWidthStep + xIndex;
		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];
		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void equalizer_cpu(const cv::Mat &input, cv::Mat &output, string imageName){

  int width = input.cols;
  int height = input.rows;
  int size_ = width * height;

  //Histogram
  int histo[C_SIZE]{};

  //Fill histogram
  for (int i = 0; i < size_; i++)
    histo[input.ptr()[i]]++;

  //Normalize
  int step = size_ / C_SIZE;
  int sum = 0;
  int n_histo[C_SIZE]{};
  for(int i=0; i < C_SIZE; i++){
      sum += histo[i];
      n_histo[i] = sum / step;
  }

  for (int i = 0; i < size_; i++)
    output.ptr()[i] = n_histo[input.ptr()[i]];

  cv::imwrite("Images/eq_cpu_" + imageName , output);
}



__global__ void get_histogram_kernel(unsigned char* output, int* histo,int width, int height, int grayWidthStep){

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)){
    const int tid = yIndex * grayWidthStep + xIndex;
    atomicAdd(&histo[(int)output[tid]], 1);
    __syncthreads();
	}
}



__global__ void equalizer_kernel(unsigned char* input, unsigned char* output, int * hist, int width, int height, int grayWidthStep, long totalSize){

    //2D Index of current thread
	unsigned int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int nxy = threadIdx.y * blockDim.x + threadIdx.x;
  //Location of gray pixel in output
  const int tid  = yIndex * grayWidthStep + xIndex;

  __shared__ int hist_s[256];
  hist_s[nxy] = 0;
  __syncthreads();

  if (nxy < 256)
    hist_s[nxy] = hist[nxy];
  __syncthreads();

  if((xIndex < width) && (yIndex < height))
      output[tid] = hist_s[input[tid]];
}



void convert_to_gray(const cv::Mat& input, cv::Mat& output, cv::Mat& eq_output, string imageName){

	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;
  int imSize = input.cols * input.rows;

	unsigned char *d_input, *d_output, *de_output;
  int * d_histogram, * df_histogram;
  int * histogram = (int *)malloc(C_SIZE * sizeof(int));
  int * f_histogram = (int *)malloc(C_SIZE * sizeof(int));
  for (int i = 0; i < C_SIZE; i++){
    histogram[i] = 0;
  }

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<unsigned char>(&de_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&d_histogram, C_SIZE * sizeof(int)), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&df_histogram, C_SIZE * sizeof(int)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
  SAFE_CALL(cudaMemset(d_histogram, 0, C_SIZE * sizeof(int)), "Error setting d_MatC to 0");

  const dim3 block(16, 16);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

	// Launch the color conversion kernel
	bgr_to_gray_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));
  // Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
  //Write the black & white image
  cv::imwrite("Images/bw_" + imageName , output);

  //printf("In CPU\n");
  equalizer_cpu(output, eq_output, imageName);
  memset(eq_output.ptr(), 0, colorBytes);
  //printf("END CPU\n");

  get_histogram_kernel<<<grid, block >>>(d_output, d_histogram, input.cols, input.rows, static_cast<int>(output.step));
  // Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  SAFE_CALL(cudaMemcpy(histogram, d_histogram, C_SIZE * sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

  //Equalize histogram
  equalize(histogram, f_histogram, imSize);
  //Copy normalized histogram to gpu
  SAFE_CALL(cudaMemcpy(df_histogram, f_histogram, C_SIZE * sizeof(int), cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
  equalizer_kernel<<<grid, block >>>(d_output, de_output, df_histogram, input.cols, input.rows, static_cast<int>(output.step), imSize);
  SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  SAFE_CALL(cudaMemcpy(eq_output.ptr(), de_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
  //Write the black & white image
  cv::imwrite("Images/eq_gpu_" + imageName , eq_output);

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]){

	string inputImage;

	if(argc < 2)
		inputImage = default_image;
  else
  	inputImage = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(img_dest + inputImage, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC1);
  //Create equalized output image
  cv::Mat eq_output(input.rows, input.cols, CV_8UC1);

	//Convert image to gray
	convert_to_gray(input, output, eq_output, inputImage);

	//Allow the windows to resize
  /*
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();
  */
	return 0;
}
