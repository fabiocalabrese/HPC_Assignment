#include <opencv2/opencv.hpp>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/* as a reference for Mat "struct"
 *struct Mat_like {
  uchar *data; // Pointer to the pixel data (row-major)
  int rows; // Number of rows (height)
  int cols; // Number of columns (width)
  size_t step; // Number of bytes per row (can include padding)
  int channels; // Number of channels (1=gray, 3=BGR, etc.)
  int type; // Data type (e.g., CV_8UC1, CV_8UC3, etc.)
  int refcount; // Reference counting (shared memory)
}; */




__global__ void filter(const uchar* input, uchar* output, int width, int height, size_t step) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // we don't include the border
    if (col >= 1 && col < width - 1 && row >= 1 && row < height - 1) {
        int sum = 0;
        int w[3][3] = {
            {1, 2, 1},
            {3, 4, 3},
            {1, 2, 1}
        };

        // each thread must select the right row,
        // if we multiply by the number of bytes we get the right row (thread_y)
        size_t row_offset = row * step;

        for (int i = -1; i <= 1; ++i) {
            size_t r_offset = row_offset + (i * step); // now just move the pointer along the 3 possible rows in the block
            for (int j = -1; j <= 1; ++j) {
                int c = col + j;  // now just move the pointer along 3 possible columns
                sum += w[i + 1][j + 1] * input[r_offset + c]; // apply the filter  to each pixel
            }
        }
		
		
		if((sum / 16) > 255){
		   
		   output[row * step + col] = 255;
		   
		   }
        
		else{output[row * step + col] = sum / 16;} // once the output is obtained, we save the new pixel
    }
}

int main(int argc, char **argv) {
    // Check if image path is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return -1;
    }

    // Load the image from file
    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check if image is loaded successfully
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }


     // just to check some part of the matrix
    /*for (int row = 47; row < 50; ++row) {
        for (int col = 97; col < 100; ++col) {
        // outputImage is CV_8UC3, eahc pixel has 3 channels;
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(row, col);
             std::cout << "Pixel (" << row << "," << col << "): "
                  << "B=" << (int)pixel[0] << " "
                  << "G=" << (int)pixel[1] << " "
                  << "R=" << (int)pixel[2] << std::endl;
            }
        }*/

    int COLS = inputImage.cols;
    int ROWS = inputImage.rows;
    
    // Split the image into 3 channels
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);

    cv::Mat blue = bgrChannels[0];
    cv::Mat green = bgrChannels[1];
    cv::Mat red = bgrChannels[2];

    // Get the step size (important for proper memory allocation)
    size_t step = blue.step;  // All channels have the same step size

    // declaration of variables with malloc allocation in host
    uchar* output_blue = (uchar*)malloc(step * ROWS);
    uchar* output_green = (uchar*)malloc(step * ROWS);
    uchar* output_red = (uchar*)malloc(step * ROWS);

    // check the correctness of the allocation
    if (output_blue == NULL || output_green == NULL || output_red == NULL) {
        std::cerr << "Error: Could not allocate memory for output image!\n";
        exit(1);
    }

    // declaration of variables CUDA
    uchar* d_data_blue, *d_data_green, *d_data_red;
    uchar* d_output_blue, *d_output_green, *d_output_red;

    // creation of the events
    cudaEvent_t start_blue, stop_blue;
    float elapsedTime_blue;
    cudaEventCreate(&start_blue);
    cudaEventCreate(&stop_blue);

    cudaEvent_t start_green, stop_green;
    float elapsedTime_green;
    cudaEventCreate(&start_green);
    cudaEventCreate(&stop_green);

    cudaEvent_t start_red, stop_red;
    float elapsedTime_red;
    cudaEventCreate(&start_red);
    cudaEventCreate(&stop_red);

    // allocation of memory in cuda
    cudaMalloc(&d_data_blue, step * ROWS);
    cudaMalloc(&d_data_green, step * ROWS);
    cudaMalloc(&d_data_red, step * ROWS);
    cudaMalloc(&d_output_blue, step * ROWS);
    cudaMalloc(&d_output_green, step * ROWS);
    cudaMalloc(&d_output_red, step * ROWS);

    // copy of the data in cuda
    cudaMemcpy(d_data_blue, blue.data, step * ROWS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_green, green.data, step * ROWS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_red, red.data, step * ROWS, cudaMemcpyHostToDevice);

    // definition of grid, blocks, num of threads
    dim3 block(16, 16);
    int threadSize = 16;
    dim3 grid((COLS + threadSize - 1)/threadSize, (ROWS + threadSize - 1)/threadSize);

    // kernel for blue, green and red with computation of elapsed time
    cudaEventRecord(start_blue, 0);
    filter<<<grid, block>>>(d_data_blue, d_output_blue, COLS, ROWS, step);
	cudaEventRecord(stop_blue, 0);
    cudaEventSynchronize(stop_blue);
    cudaEventElapsedTime(&elapsedTime_blue, start_blue, stop_blue);

    cudaEventRecord(start_green, 0);
    filter<<<grid, block>>>(d_data_green, d_output_green, COLS, ROWS, step);
    cudaEventRecord(stop_green, 0);
    cudaEventSynchronize(stop_green);
    cudaEventElapsedTime(&elapsedTime_green, start_green, stop_green);

    cudaEventRecord(start_red, 0);
    filter<<<grid, block>>>(d_data_red, d_output_red, COLS, ROWS, step);
    cudaEventRecord(stop_red, 0);
    cudaEventSynchronize(stop_red);
    cudaEventElapsedTime(&elapsedTime_red, start_red, stop_red);

    // to synchronize the GPU with CPU
    cudaDeviceSynchronize();

    // copy the result from GPU to host
    cudaMemcpy(output_blue, d_output_blue, step * ROWS, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_green, d_output_green, step * ROWS, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_red, d_output_red, step * ROWS, cudaMemcpyDeviceToHost);

    // put together the result
    cv::Mat filtered_blue(ROWS, COLS, CV_8UC1, output_blue, step);
    cv::Mat filtered_green(ROWS, COLS, CV_8UC1, output_green, step);
    cv::Mat filtered_red(ROWS, COLS, CV_8UC1, output_red, step);

    std::vector<cv::Mat> channels = {filtered_blue, filtered_green, filtered_red};

   
    // define the output image
    cv::Mat outputImage;

    // merge the new three matrices
    cv::merge(channels, outputImage);

    // save the image
    cv::imwrite("output_filtered.jpg", outputImage);
  

     // Print timing results
    std::cout << "Blue channel processing time: " << elapsedTime_blue << " ms" << std::endl;
    std::cout << "Green channel processing time: " << elapsedTime_green << " ms" << std::endl;
    std::cout << "Red channel processing time: " << elapsedTime_red << " ms" << std::endl;
    std::cout << "Total processing time: " << (elapsedTime_blue + elapsedTime_green + elapsedTime_red) << " ms" << std::endl;

    // Destroy the events
    cudaEventDestroy(start_blue);
    cudaEventDestroy(stop_blue);
    cudaEventDestroy(start_green);
    cudaEventDestroy(stop_green);
    cudaEventDestroy(start_red);
    cudaEventDestroy(stop_red);

    // de-allocation of cuda memory
    cudaFree(d_data_blue);
    cudaFree(d_data_green);
    cudaFree(d_data_red);
    cudaFree(d_output_blue);
    cudaFree(d_output_green);
    cudaFree(d_output_red);

    // de-allocation of host memory
    free(output_blue);
    free(output_green);
    free(output_red);
    
    return 0;
}











#include <opencv2/opencv.hpp>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/* as a reference for Mat "struct"
 *struct Mat_like {
  uchar *data; // Pointer to the pixel data (row-major)
  int rows; // Number of rows (height)
  int cols; // Number of columns (width)
  size_t step; // Number of bytes per row (can include padding)
  int channels; // Number of channels (1=gray, 3=BGR, etc.)
  int type; // Data type (e.g., CV_8UC1, CV_8UC3, etc.)
  int refcount; // Reference counting (shared memory)
}; */




