#include <opencv2/opencv.hpp>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to apply a 3x3 weighted filter
__global__ void filter(const uchar* input, uchar* output, int width, int height, size_t step) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoid edges to prevent out-of-bounds memory access
    if (col >= 1 && col < width - 1 && row >= 1 && row < height - 1) {
        int sum = 0;
        int w[3][3] = {
            {1, 2, 1},
            {3, 4, 3},
            {1, 2, 1}
        };

        size_t row_offset = row * step;

        // Perform convolution with the 3x3 filter kernel
        for (int i = -1; i <= 1; ++i) {
            size_t r_offset = row_offset + (i * step);
            for (int j = -1; j <= 1; ++j) {
                int c = col + j;
                sum += w[i + 1][j + 1] * input[r_offset + c];
            }
        }

        int result = sum / 16; // Normalize sum by kernel weight total (16)
        // Clamp result to max 255 to avoid overflow
        output[row * step + col] = result > 255 ? 255 : result;
    }
}

int main(int argc, char **argv) {
    // Check if an image path is provided as argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return -1;
    }

    // Load color image using OpenCV
    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    int COLS = inputImage.cols;
    int ROWS = inputImage.rows;
    size_t step = inputImage.step; // Number of bytes in a row

    // Split image into BGR channels
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);

    cv::Mat blue = bgrChannels[0];
    cv::Mat green = bgrChannels[1];
    cv::Mat red = bgrChannels[2];

    // Define three different block size configurations to test
    dim3 blockSizes[] = {dim3(16, 16), dim3(32, 32), dim3(32, 8)};
    const char* logFiles[] = {"log_16x16_ref.txt", "log_32x32_ref.txt", "log_32x8_ref.txt"};

    // Loop through each block configuration
    for (int config = 0; config < 3; ++config) {
        dim3 block = blockSizes[config];
        // Calculate grid size to cover entire image
        dim3 grid((COLS + block.x - 1) / block.x, (ROWS + block.y - 1) / block.y);

        int totalThreads = grid.x * grid.y * block.x * block.y;

        // Allocate host memory for output channels
        uchar* output_blue = (uchar*)malloc(step * ROWS);
        uchar* output_green = (uchar*)malloc(step * ROWS);
        uchar* output_red = (uchar*)malloc(step * ROWS);

        // Device pointers for input and output image channels
        uchar* d_data_blue, *d_data_green, *d_data_red;
        uchar* d_output_blue, *d_output_green, *d_output_red;

        // Allocate device memory for input and output images
        cudaMalloc(&d_data_blue, step * ROWS);
        cudaMalloc(&d_data_green, step * ROWS);
        cudaMalloc(&d_data_red, step * ROWS);
        cudaMalloc(&d_output_blue, step * ROWS);
        cudaMalloc(&d_output_green, step * ROWS);
        cudaMalloc(&d_output_red, step * ROWS);

        // Copy input channels from host to device
        cudaMemcpy(d_data_blue, blue.data, step * ROWS, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data_green, green.data, step * ROWS, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data_red, red.data, step * ROWS, cudaMemcpyHostToDevice);

        // Create CUDA events to measure kernel execution time
        cudaEvent_t start, stop;
        float elapsedTime_blue, elapsedTime_green, elapsedTime_red;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Process blue channel and measure time
        cudaEventRecord(start);
        filter<<<grid, block>>>(d_data_blue, d_output_blue, COLS, ROWS, step);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_blue, start, stop);

        // Process green channel and measure time
        cudaEventRecord(start);
        filter<<<grid, block>>>(d_data_green, d_output_green, COLS, ROWS, step);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_green, start, stop);

        // Process red channel and measure time
        cudaEventRecord(start);
        filter<<<grid, block>>>(d_data_red, d_output_red, COLS, ROWS, step);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime_red, start, stop);

        cudaDeviceSynchronize();

        // Extract image base filename (strip path)
        const char* baseName = strrchr(argv[1], '/');
        baseName = baseName ? baseName + 1 : argv[1];

        // Append processing results to appropriate log file
        FILE *logfile = fopen(logFiles[config], "a");
        if (logfile != NULL) {
            fprintf(logfile,
                "Image: %s, Size:%dx%d, Block: %dx%d, Threads: %d, Blue Time: %.2f ms, Green Time: %.2f ms, Red Time: %.2f ms, Total Time: %.2f ms\n",
                baseName,
                COLS,
                ROWS,
                block.x, block.y,
                totalThreads,
                elapsedTime_blue,
                elapsedTime_green,
                elapsedTime_red,
                elapsedTime_blue + elapsedTime_green + elapsedTime_red);
            fclose(logfile);
        }

        // Free device memory
        cudaFree(d_data_blue);
        cudaFree(d_data_green);
        cudaFree(d_data_red);
        cudaFree(d_output_blue);
        cudaFree(d_output_green);
        cudaFree(d_output_red);

        // Free host memory
        free(output_blue);
        free(output_green);
        free(output_red);

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
