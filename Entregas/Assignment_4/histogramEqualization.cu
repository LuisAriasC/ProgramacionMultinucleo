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

#define default_image "Images/dog1.jpeg"

__shared__ int * histogram[256];

using namespace std;

// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
// grayWidthStep - number of gray bytes
__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep)
{
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

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  for (int i = 0; i < 256; i++)
    histogram[i] = 0;
  __syncthreads();
  
	if ((xIndex < width) && (yIndex < height)){
    const int tid = yIndex * grayWidthStep + xIndex;
    atomicAdd(histogram[(int)output[tid] % 256], 1);
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output){


	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

  int nBytes = 256 * sizeof(int);
  int *histo;
  histo = (int *)malloc(nBytes);
  for (int i = 0; i < 256; i++)
    histo[i] = 0;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	const dim3 block(16, 16);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

	// Launch the color conversion kernel
	bgr_to_gray_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");



	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

  //Write the black & white image
  cv::imwrite("Images/bw_outputImage.jpg" , output);

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
	cv::Mat input = cv::imread(inputImage, CV_LOAD_IMAGE_COLOR);

	if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	//Call the wrapper function
	convert_to_gray(input, output);

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