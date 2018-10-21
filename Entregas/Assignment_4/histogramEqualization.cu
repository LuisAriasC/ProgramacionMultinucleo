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
#define default_image "dog1.jpeg"
#define C_SIZE 256

using namespace std;

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


__global__ void equalize_image_kernel(unsigned char* output, int* histo,int width, int height, int grayWidthStep){

  __shared__ int n_histo[C_SIZE];

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int step_x = blockDim.x;

  int sizeImage = width * height;

	if ((xIndex < width) && (yIndex < height)){
    int h_index = (y * step_x) + x;
    int o_index = (yIndex * grayWidthStep) + xIndex;

    n_histo[h_index] = 0;
    __syncthreads();

    const int tid = yIndex * grayWidthStep + xIndex;
    atomicAdd(&histo[(int)output[tid]], 1);
    __syncthreads();

    //Normalized histogram
    int i;
    for (i = 0; i <= h_index; i++)
      n_histo[h_index] += histo[i];
    __syncthreads();

    /*
    unsigned int aux = (n_histo[h_index] * C_SIZE) / sizeImage;
    n_histo[h_index] = aux;
    __syncthreads();

    output[o_index] = n_histo[h_index];
    */
    if (o_index == 0) {
      for (int i = 0; i < C_SIZE; i++) {
        cout << n_histo[i]  << " "; 
      }
    }
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output, string imageName){


	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;
  //int imSize = input.cols * input.rows;

	unsigned char *d_input, *d_output;
  int * d_histogram;
  int * histogram = (int *)malloc(C_SIZE * sizeof(int));

  for (int i = 0; i < C_SIZE; i++)
    histogram[i] = 0;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc<int>(&d_histogram, C_SIZE * sizeof(int)), "CUDA Malloc Failed");

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

  equalize_image_kernel<<<grid, block >>>(d_output, d_histogram, input.cols, input.rows, static_cast<int>(output.step));
  // Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
  SAFE_CALL(cudaMemcpy(histogram, d_histogram, C_SIZE * sizeof(int), cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
  //Write the black & white image
  cv::imwrite("Images/eq_gpu_" + imageName , output);

  /*
  int sum = 0;
  for (int i = 0; i < C_SIZE; i++)
    sum += histogram[i];
    //printf("%d : %d\n", i, histogram[i]);
  printf("%d : %d\n", imSize, sum);
*/
	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
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

  //Normalized histogram
  long n_histo[C_SIZE]{};
  for (int i = 0; i < C_SIZE; i++){
      for(int j = 0; j <= i; j++)
          n_histo[i] += histo[j];
      unsigned int aux  = (n_histo[i]*C_SIZE) / size_;
      n_histo[i] = aux;
  }

  for (int i = 0; i < size_; i++)
    output.ptr()[i] = n_histo[input.ptr()[i]];

  cv::imwrite("Images/eq_" + imageName , output);
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
	convert_to_gray(input, output, inputImage);
  //equalizer_cpu(output, eq_output, inputImage);

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
