/*
  Assignment 2 - Image Blurring
  Author: Luis Carlos Arias Camacho
  Student ID: A01364808
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "common.h"
#include <cuda_runtime.h>

#define default_input_image "image.jpg"
#define size1 5
#define size2 11

using namespace std;

__global__ void blur_kernel(unsigned char* input_Image, unsigned char* output_Image, int width, int height, int colorWidthStep, int bM_size){

  //pixel margin for blur matrix
  const int margin = floor(bM_size / 2.0);
  const float multConstant = bM_size * bM_size;

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only pixels out of margin
	if ((xIndex < width) && (yIndex < height)) {

		int output_index = yIndex * colorWidthStep + (3 * xIndex);

		//Output pixels
		float blue = 0;
		float green = 0;
		float red = 0;

		//Pixels inside blur section
		if ((xIndex >= margin) && (xIndex < width - margin) && (yIndex >= margin) && (yIndex < height - margin)) {

			int index = 0;

			//Calculate blur average
			for (int i = xIndex - margin; i < xIndex + margin + 1; i++) {
				for (int j = yIndex - margin; j < yIndex + margin + 1; j++) {
					index = j * colorWidthStep + (3 * i);
					blue += input_Image[index];
					green += input_Image[index + 1];
					red += input_Image[index + 2];
				}
			}
			blue = blue / multConstant;
			green = green / multConstant;
			red = red / multConstant;
		} else {
			//If pixels are in margin range
			int input_index = yIndex * colorWidthStep + (3 * xIndex);
			blue = input_Image[input_index];
			green = input_Image[input_index + 1];
			red = input_Image[input_index + 2];
		}
		output_Image[output_index] = static_cast<unsigned char>(blue);
		output_Image[output_index + 1] = static_cast<unsigned char>(green);
		output_Image[output_index + 2] = static_cast<unsigned char>(red);
	}

}

void blur_GPU(const cv::Mat& input, cv::Mat& output, int blurMatrix_size){

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors
	size_t inputBytes = input.step * input.rows;
	size_t outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	chrono::duration<float, std::milli> duration_ms = chrono::high_resolution_clock::duration::zero();
	auto start_gpu =  chrono::high_resolution_clock::now();
	// Launch the color conversion kernel
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), blurMatrix_size);

	auto end_gpu =  chrono::high_resolution_clock::now();
	duration_ms = end_gpu - start_gpu;
	printf("Image blur elapsed %f ms in GPU with a blur matrix of %dx%d\n", duration_ms.count(), blurMatrix_size, blurMatrix_size);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Save image
	cv::imwrite("output" + to_string(blurMatrix_size) + "_gpu.jpg", output);

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]){

	string inputImage;
  int blurMatrix_size;
	int blurMatrix_size2;

	if(argc < 2){
		inputImage = default_input_image;
    blurMatrix_size = size1;
    blurMatrix_size2 = size2;
  } else if (argc == 2 ){
    inputImage = argv[1];
    blurMatrix_size = size1;
    blurMatrix_size2 = size2;
  } else {
    inputImage = argv[1];
    if (atoi(argv[2]) % 2 == 0) {
      blurMatrix_size = atoi(argv[2]);
      blurMatrix_size2 = size2;
    } else {
      blurMatrix_size = atoi(argv[2]) + 1;
      blurMatrix_size2 = size2;
    }
  }

	// Read input image from the disk
	cv::Mat input = cv::imread(inputImage, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	cv::Mat output(input.rows, input.cols, input.type());


	printf("Test on GPU\n");
	//Call the wrapper function
	blur_GPU(input, output, blurMatrix_size);
	blur_GPU(input, output, blurMatrix_size2);


  //Open windows to display images
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
