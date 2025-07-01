#include <opencv2/opencv.hpp>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to apply a 3x3 convolution filter
__global__ void filter(const uchar* input, uchar* output, int width, int height, size_t step) {
    // Calculate current pixel coordinates based on block and thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoid image borders to prevent out-of-bound accesses
    if (col >= 1 && col < width - 1 && row >= 1 && row < height - 1) {
        int sum = 0;

        // 3x3 filter kernel weights (weighted filter)
        int w[3][3] = {
            {1, 2, 1},
            {3, 4, 3},
            {1, 2, 1}
        };

        size_t row_offset = row * step;

        // Compute weighted sum convolution over 3x3 neighborhood around pixel (row, col)
        for (int i = -1; i <= 1; ++i) {
            size_t r_offset = row_offset + (i * step);
            for (int j = -1; j <= 1; ++j) {
                int c = col + j;
                sum += w[i + 1][j + 1] * input[r_offset + c];
            }
        }

        // Normalize the sum dividing by 16 (sum of kernel weights)
        // Clamp maximum value to 255 to avoid overflow in color channel
        if ((sum / 16) > 255) {
            output[row * step + col] = 255;
        } else {
            output[row * step + col] = sum / 16;
        }
    }
}

int main(int argc, char **argv) {
    // Check for correct program arguments (image path)
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return -1;
    }

    // Load image in color mode using OpenCV
    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    // Store image width and height
    int COLS = inputImage.cols;
    int ROWS = inputImage.rows;
    std::cout << "Image size: " << COLS << " x " << ROWS << std::endl;

    // Split the image into BGR color channels
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);

    cv::Mat blue = bgrChannels[0];
    cv::Mat green = bgrChannels[1];
    cv::Mat red = bgrChannels[2];

    // Get step size (number of bytes per row) for the blue channel (same for all)
    size_t step = blue.step;

    // Allocate host memory buffers for filtered output channels
    uchar* output_blue = (uchar*)malloc(step * ROWS);
    uchar* output_green = (uchar*)malloc(step * ROWS);
    uchar* output_red = (uchar*)malloc(step * ROWS);

    // Check memory allocation success
    if (!output_blue || !output_green || !output_red) {
        std::cerr << "Error: Could not allocate memory for output image!\n";
        exit(1);
    }

    // Declare pointers for device memory for input and output channels
    uchar* d_data_blue, *d_data_green, *d_data_red;
    uchar* d_output_blue, *d_output_green, *d_output_red;

    // Create CUDA events to time processing on blue channel
    cudaEvent_t start_blue, stop_blue;
    float elapsedTime_blue;
    cudaEventCreate(&start_blue);
    cudaEventCreate(&stop_blue);

    // Create CUDA events to time processing on green channel
    cudaEvent_t start_green, stop_green;
    float elapsedTime_green;
    cudaEventCreate(&start_green);
    cudaEventCreate(&stop_green);

    // Create CUDA events to time processing on red channel
    cudaEvent_t start_red, stop_red;
    float elapsedTime_red;
    cudaEventCreate(&start_red);
    cudaEventCreate(&stop_red);

    // Allocate device memory for input and output images (all channels)
    cudaMalloc(&d_data_blue, step * ROWS);
    cudaMalloc(&d_data_green, step * ROWS);
    cudaMalloc(&d_data_red, step * ROWS);
    cudaMalloc(&d_output_blue, step * ROWS);
    cudaMalloc(&d_output_green, step * ROWS);
    cudaMalloc(&d_output_red, step * ROWS);

    // Copy input image data from host to device for each channel
    cudaMemcpy(d_data_blue, blue.data, step * ROWS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_green, green.data, step * ROWS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_red, red.data, step * ROWS, cudaMemcpyHostToDevice);

    // Define block and grid dimensions for kernel launch (16x16 threads per block)
    dim3 block(16, 16);
    dim3 grid((COLS + 15) / 16, (ROWS + 15) / 16);
    int totalThreads = grid.x * grid.y * block.x * block.y;
    std::cout << "Total threads used: " << totalThreads << std::endl;

    // Launch kernel and time filtering for blue channel
    cudaEventRecord(start_blue, 0);
    filter<<<grid, block>>>(d_data_blue, d_output_blue, COLS, ROWS, step);
    cudaEventRecord(stop_blue, 0);
    cudaEventSynchronize(stop_blue);
    cudaEventElapsedTime(&elapsedTime_blue, start_blue, stop_blue);

    // Launch kernel and time filtering for green channel
    cudaEventRecord(start_green, 0);
    filter<<<grid, block>>>(d_data_green, d_output_green, COLS, ROWS, step);
    cudaEventRecord(stop_green, 0);
    cudaEventSynchronize(stop_green);
    cudaEventElapsedTime(&elapsedTime_green, start_green, stop_green);

    // Launch kernel and time filtering for red channel
    cudaEventRecord(start_red, 0);
    filter<<<grid, block>>>(d_data_red, d_output_red, COLS, ROWS, step);
    cudaEventRecord(stop_red, 0);
    cudaEventSynchronize(stop_red);
    cudaEventElapsedTime(&elapsedTime_red, start_red, stop_red);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    // Copy processed data from device back to host
    cudaMemcpy(output_blue, d_output_blue, step * ROWS, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_green, d_output_green, step * ROWS, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_red, d_output_red, step * ROWS, cudaMemcpyDeviceToHost);

    // Create OpenCV matrices for filtered channels (using allocated output buffers)
    cv::Mat filtered_blue(ROWS, COLS, CV_8UC1, output_blue, step);
    cv::Mat filtered_green(ROWS, COLS, CV_8UC1, output_green, step);
    cv::Mat filtered_red(ROWS, COLS, CV_8UC1, output_red, step);

    // Merge the filtered channels back into one color image
    std::vector<cv::Mat> channels = {filtered_blue, filtered_green, filtered_red};
    cv::Mat outputImage;
    cv::merge(channels, outputImage);

    // Extract original file name from input path to build output file name
    const char* baseName = strrchr(argv[1], '/');

    if (baseName != NULL) {
        baseName = baseName + 1; // Skip '/' character to get only the file name
    } else {
        baseName = argv[1]; // No '/' found, input is already a simple filename
    }

    // Create output file name with prefix "filtered_"
    char outputName[512];
    snprintf(outputName, sizeof(outputName), "filtered_%s", baseName);
    // Save filtered image to disk
    cv::imwrite(outputName, outputImage);

    // Print timing info for each color channel
    std::cout << "Blue channel processing time: " << elapsedTime_blue << " ms" << std::endl;
    std::cout << "Green channel processing time: " << elapsedTime_green << " ms" << std::endl;
    std::cout << "Red channel processing time: " << elapsedTime_red << " ms" << std::endl;
    std::cout << "Total processing time: " << (elapsedTime_blue + elapsedTime_green + elapsedTime_red) << " ms" << std::endl;

    // Retrieve GPU device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Append processing details to a log file
    FILE *logfile = fopen("log_gpu_processing.txt", "a");
    if (logfile != NULL) {
        fprintf(logfile,
                "Image: %s, Threads: %d, GPU: %s, Global Mem: %zu MB, Blue Time: %.2f ms, Green Time: %.2f ms, Red Time: %.2f ms, Total Time: %.2f ms\n",
                baseName,
                totalThreads,
                prop.name,
                prop.totalGlobalMem / (1024 * 1024),
                elapsedTime_blue,
                elapsedTime_green,
                elapsedTime_red,
                elapsedTime_blue + elapsedTime_green + elapsedTime_red
        );
        fclose(logfile);
    } else {
        fprintf(stderr, "Could not open log file for writing.\n");
    }

    // Clean up CUDA events
    cudaEventDestroy(start_blue);
    cudaEventDestroy(stop_blue);
    cudaEventDestroy(start_green);
    cudaEventDestroy(stop_green);
    cudaEventDestroy(start_red);
    cudaEventDestroy(stop_red);

    // Free CUDA device memory
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

    return 0;
}

