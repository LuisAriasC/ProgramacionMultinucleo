/*
  Assignment 2 - Image Blurring
  Author: Luis Carlos Arias Camacho
  Student ID: A01364808
  Compile line: g++ -o image_Blur_CPU imageBlur_CPU.cpp `pkg-config --cflags --libs opencv` -std=c++11 -fopenmp
 */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

#define default_input_image "image.jpg"
#define size1 5
#define size2 11

using namespace std;

void blur_OMP(const cv::Mat& input_Image, cv::Mat& output_Image, int blur_size){

  int colorWidthStep = static_cast<int>(input_Image.step);
  int margin = floor(blur_size / 2.0);
  float multConstant =  (blur_size * blur_size);

  printf("Margin %d -- Total pixels for blur matrix %f\n", margin, multConstant );

	size_t inputBytes = input_Image.step * input_Image.rows;
	unsigned char *input, *output;
	output = (unsigned char *) malloc(inputBytes * sizeof(unsigned char));
  input = (unsigned char *) malloc(inputBytes * sizeof(unsigned char));
	memcpy(input, input_Image.ptr(), inputBytes * sizeof(unsigned char));

  //Pixeld for the output
	float blue, green, red;

	int input_index, output_index;
  int i, j, cols, rows;


  #pragma omp parallel for private(i, j, cols, rows, blue, green, red, input_index, output_index) shared(input, output)
  for (i = 0; i < input_Image.cols; i++){
		blue = 0;
		green = 0;
		red = 0;

		for (j = 0; j < input_Image.rows; j++){

			if ((i >= margin) && (j >= margin) && (i < input_Image.cols - margin) && (j < input_Image.rows - margin)){

				input_index = 0;
        int cols, rows;
				//Average pixel color calculation
				for (int cols = i - margin; cols <= i + margin; cols++){
					for (int rows = j - margin; rows <= j + margin; rows++){
						input_index = rows * colorWidthStep + (3 * cols);
						blue = blue + input[input_index];
						green = green + input[input_index + 1];
						red = red + input[input_index + 2];
					}
				}
				blue = blue / multConstant;
				green = green / multConstant;
				red = red / multConstant;
			} else {
				input_index = j * colorWidthStep + (3 * i);
				blue = input[input_index];
				green = input[input_index + 1];
				red = input[input_index + 2];
			}
			output_index = j * colorWidthStep + (3 * i);
			output[output_index] = static_cast<unsigned char>(blue);
			output[output_index + 1] = static_cast<unsigned char>(green);
			output[output_index + 2] = static_cast<unsigned char>(red);
		}
	}

	memcpy(output_Image.ptr(), output, inputBytes * sizeof(unsigned char));

	//Save resultant image
	cv::imwrite("output_" + to_string(blur_size) + "pixelsOMP.jpg", output_Image);
}


/*blur in cpu*/
void blur_CPU(const cv::Mat& input_Image, cv::Mat& output_Image, int blur_size){

	int colorWidthStep = static_cast<int>(input_Image.step);
  int margin = floor(blur_size / 2.0);
  float multConstant =  (blur_size * blur_size);

  printf("Margin %d -- Total pixels for blur matrix %f\n", margin, multConstant );

	size_t inputBytes = input_Image.step * input_Image.rows;
	unsigned char *input, *output;
	output = (unsigned char *) malloc(inputBytes * sizeof(unsigned char));
  input = (unsigned char *) malloc(inputBytes * sizeof(unsigned char));
	memcpy(input, input_Image.ptr(), inputBytes * sizeof(unsigned char));

	//Pixeld for the output
	float blue, green, red;

	int input_index, output_index;

  for (int i = 0; i < input_Image.cols; i++){
		blue = 0;
		green = 0;
		red = 0;

		for (int j = 0; j < input_Image.rows; j++){

			if ((i >= margin) && (j >= margin) && (i < input_Image.cols - margin) && (j < input_Image.rows - margin)){

				input_index = 0;

				//Average pixel color calculation
				for (int cols = i - margin; cols < i + margin + 1; cols++){
					for (int rows = j - margin; rows < j + margin + 1; rows++){

						input_index = rows * colorWidthStep + (3 * cols);
						blue += input[input_index];
						green += input[input_index + 1];
						red += input[input_index + 2];
					}
				}
				blue = blue / multConstant;
				green = green / multConstant;
				red = red /multConstant;
			}
			else{
				input_index = j * colorWidthStep + (3 * i);
				blue = input[input_index];
				green = input[input_index + 1];
				red = input[input_index + 2];
			}
			output_index = j * colorWidthStep + (3 * i);
			output[output_index] = static_cast<unsigned char>(blue);
			output[output_index+1] = static_cast<unsigned char>(green);
			output[output_index+2] = static_cast<unsigned char>(red);
		}
	}

	memcpy(output_Image.ptr(), output, inputBytes * sizeof(unsigned char));

  //Write_image
  cv::imwrite("output_" + to_string(blur_size) + "pixelsCPU.jpg", output_Image);
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

	if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	cv::Mat output(input.rows, input.cols, input.type());

  printf("Test on CPU\n");

	chrono::duration<float, std::milli> duration_ms = chrono::high_resolution_clock::duration::zero();
	auto start =  chrono::high_resolution_clock::now();
	blur_CPU(input, output, blurMatrix_size);
	auto end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("Image blur elapsed %f ms in CPU with a blur matrix of %dx%d\n", duration_ms.count(), blurMatrix_size, blurMatrix_size);

  start =  chrono::high_resolution_clock::now();
	blur_CPU(input, output, blurMatrix_size2);
	end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("Image blur elapsed %f ms in CPU with a blur matrix of %dx%d\n", duration_ms.count(), blurMatrix_size2, blurMatrix_size2);

  printf("\n\n");

  printf("Test on OpenMP\n");

  start =  chrono::high_resolution_clock::now();
  blur_OMP(input, output, blurMatrix_size);
  end =  chrono::high_resolution_clock::now();
  duration_ms = end - start;
  printf("Image blur elapsed %f ms in OpenMP with a blur matrix of %dx%d\n", duration_ms.count(), blurMatrix_size2, blurMatrix_size2);

  start =  chrono::high_resolution_clock::now();
  blur_OMP(input, output, blurMatrix_size2);
  end =  chrono::high_resolution_clock::now();
  duration_ms = end - start;
  printf("Image blur elapsed %f ms in OpenMP with a blur matrix of %dx%d\n", duration_ms.count(), blurMatrix_size2, blurMatrix_size2);

  printf("\n\n");

  //Open windows to display images
	//Allow the windows to resize
	namedWindow("CPU INPUT", cv::WINDOW_NORMAL);
	namedWindow("CPU OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("CPU INPUT", input);
	imshow("CPU OUTPUT", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
