/*
  Assignment 2 - Image Blurring
  Author: Luis Carlos Arias Camacho
  Student ID: A01364808
 */

//g++ imageBlur_CPU.cpp `pkg-config --cflags --libs opencv`

#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define default_input_image "input_image.jpg"
#define blurM_size 5

 using namespace std;

void OMP_blur_image(const cv::Mat& M_input, cv::Mat& M_output)
{
	int colorWidthStep = static_cast<int>(M_input.step);
	size_t inputBytes = M_input.step*M_input.rows;
	unsigned char *input, *output;
	output = input = (unsigned char *) malloc(inputBytes*sizeof(unsigned char));

	memcpy(input, M_input.ptr(), inputBytes*sizeof(unsigned char));

	//pixel margin for blur matrix
	const unsigned int marginSize = 2;

	//Output pixels
	float out_blue = 0;
	float out_green = 0;
	float out_red = 0;

	int index, out_index;
	for (int i = 0; i < M_input.cols; i++)
	{
		out_blue = 0;
		out_green = 0;
		out_red = 0;
		for (int j = 0; j < M_input.rows; j++)
		{

			if ((i >= marginSize) && (j >= marginSize) && (i < M_input.cols - marginSize) && (j < M_input.rows - marginSize))
			{
				index = 0;
				#pragma omp parallel for collapse(2) default(shared) reduction (+:out_blue, out_green, out_red)
				//Average pixel color calculation
				for (int m_i = i - marginSize; m_i <= i + marginSize; m_i++)
				{
					for (int m_j = j - marginSize; m_j <= j + marginSize; m_j++)
					{
						index = m_j * colorWidthStep + (3 * m_i);
						out_blue = out_blue + input[index];
						out_green = out_green + input[index + 1];
						out_red = out_red + input[index + 2];
					}
				}
				out_blue /= 25;
				out_green /= 25;
				out_red /= 25;
			}
			else
			{
				index = j * colorWidthStep + (3 * i);
				out_blue = input[index];
				out_green = input[index + 1];
				out_red = input[index + 2];
			}
			out_index = j * colorWidthStep + (3 * i);
			output[out_index] = static_cast<unsigned char>(out_blue);
			output[out_index+1] = static_cast<unsigned char>(out_green);
			output[out_index+2] = static_cast<unsigned char>(out_red);
		}
	}

	memcpy(M_output.ptr(), output, inputBytes*sizeof(unsigned char));

	//Save resultant image
	cv::imwrite("OMP_Altered_Image.jpg", M_output);
}

void blur_CPU(const cv::Mat& input_Image, cv::Mat& output_Image, int blur_size){

	int colorWidthStep = static_cast<int>(input_Image.step);
  int margin = floor(blur_size / 2.0);
  float multConstant =  (blur_size * blur_size);

  printf("Margin %d Mult constant %f\n", margin, multConstant );

	size_t inputBytes = input_Image.step * input_Image.rows;
	unsigned char *input, *output;
	output = (unsigned char *) malloc(inputBytes * sizeof(unsigned char));
  input = (unsigned char *) malloc(inputBytes * sizeof(unsigned char));

	memcpy(input, input_Image.ptr(), inputBytes * sizeof(unsigned char));

	//pixel margin for blur matrix
	//const unsigned int marginSize = 2;

	//Output pixels
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
				for (int m_i = i - margin; m_i <= i + margin; m_i++){
					for (int m_j = j - margin; m_j <= j + margin; m_j++){

						input_index = m_j * colorWidthStep + (3 * m_i);
						blue += input[input_index];
						green += input[input_index + 1];
						red += input[input_index + 2];
            if (i%100 == 0) {
              printf("%f %f %f\n", blue, green, red);
            }
					}
				}
				blue /= multConstant;
				green /= multConstant;
				red /= multConstant;
        if (i%100 == 0) {
          printf("Dividido %f %f %f\n", blue, green, red);
        }
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

  cv::imwrite("output1.jpg", output_Image);
}

int main(int argc, char *argv[]){

  string inputImage;
  int blurMatrix_size;

	if(argc < 2){
		inputImage = default_input_image;
    blurMatrix_size = blurM_size;
  } else if (argc == 2 ){
    inputImage = argv[1];
    blurMatrix_size = blurM_size;
  } else {
    inputImage = argv[1];
    if (atoi(argv[2]) % 2 == 0) {
      blurMatrix_size = atoi(argv[2]);
    } else {
      blurMatrix_size = atoi(argv[2]) + 1;
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

  /* Maybe eliminate
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
  */
	// NO THREADS CPU TEST

	chrono::duration<float, std::milli> duration_ms = chrono::high_resolution_clock::duration::zero();
	auto start =  chrono::high_resolution_clock::now();

	blur_CPU(input, output, blurMatrix_size);

	auto end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("Image blur elapsed %f ms in CPU\n", duration_ms.count());

	/* ********* DISPLAY IMAGES **********/
	//Allow the windows to resize
	//namedWindow("CPU INPUT", cv::WINDOW_NORMAL);
	//namedWindow("CPU OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	//imshow("CPU INPUT", input);
	//imshow("CPU OUTPUT", output);

	//Wait for key press
	//cv::waitKey();


	// OMP CPU TEST
  /*
	duration_ms = chrono::high_resolution_clock::duration::zero();
	start =  chrono::high_resolution_clock::now();

	OMP_blur_image(input, output);

	end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("OMP image blurring elapsed %f ms\n\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("OMP INPUT", cv::WINDOW_NORMAL);
	namedWindow("OMP OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("OMP INPUT", input);
	imshow("OMP OUTPUT", output);

	//Wait for key press
	cv::waitKey();
  */

	return 0;
}
