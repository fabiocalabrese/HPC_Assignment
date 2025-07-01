#include <opencv2/opencv.hpp>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


void filter(uchar* input,uchar* output, int height, int width,size_t step, int w[3][3]);



int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>\n";
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    int w[3][3] = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 1}
    };
    int COLS = inputImage.cols;
    int ROWS = inputImage.rows;
    std::cout << "Image size: " << COLS << " x " << ROWS << std::endl;

    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);

    cv::Mat blue = bgrChannels[0];
    cv::Mat green = bgrChannels[1];
    cv::Mat red = bgrChannels[2];

    size_t step = blue.step;

    uchar* output_blue = (uchar*)malloc(step * ROWS);
    uchar* output_green = (uchar*)malloc(step * ROWS);
    uchar* output_red = (uchar*)malloc(step * ROWS);

    uchar* input_blue = (uchar*)malloc(step * ROWS);
    uchar* input_green = (uchar*)malloc(step * ROWS);
    uchar* input_red = (uchar*)malloc(step * ROWS);




    if (!output_blue || !output_green || !output_red) {
        std::cerr << "Error: Could not allocate memory for output image!\n";
        exit(1);
    }

    if (!input_blue || !input_green || !input_red) {
        std::cerr << "Error: Could not allocate memory for input image!\n";
        exit(1);
    }


    memcpy(input_blue, blue.data, step * ROWS);
    memcpy(input_green, green.data, step * ROWS);
    memcpy(input_red, red.data, step * ROWS);

    clock_t start_blue;
    clock_t end_blue;
    clock_t start_green;
    clock_t end_green;
    clock_t start_red;
    clock_t end_red;



    start_blue = clock();
    filter(input_blue,output_blue,ROWS,COLS,step,w);
    end_blue = clock();

    start_green = clock();
    filter(input_green,output_green,ROWS,COLS,step,w);
    end_green = clock();


    start_red = clock();
    filter(input_red,output_red,ROWS,COLS,step,w);
    end_red = clock();


    double cpu_time_used_blue = ((double) (end_blue - start_blue)) / CLOCKS_PER_SEC;
    double cpu_time_used_green = ((double) (end_green - start_green)) / CLOCKS_PER_SEC;
    double cpu_time_useed_red = ((double) (end_red - start_red)) / CLOCKS_PER_SEC;

    std::cout<<"Elapsed time blue:"<< cpu_time_used_blue <<"s\n "
    "Elapsed time green:"<< cpu_time_used_green <<"s\n"
    "Elapsed time red"<< cpu_time_useed_red<<"s\n"
    "Total time:"<< (cpu_time_useed_red + cpu_time_used_green + cpu_time_used_blue ) <<"s\n"<< std::endl;



    cv::Mat filtered_blue(ROWS, COLS, CV_8UC1, output_blue, step);
    cv::Mat filtered_green(ROWS, COLS, CV_8UC1, output_green, step);
    cv::Mat filtered_red(ROWS, COLS, CV_8UC1, output_red, step);

    std::vector<cv::Mat> channels = {filtered_blue, filtered_green, filtered_red};
    cv::Mat outputImage;
    cv::merge(channels, outputImage);

    // Extraction of the name of the image.
    const char* baseName = strrchr(argv[1], '/');


    if (baseName != NULL) {

        baseName = baseName + 1; // Jump '/' to obtain the name of the file

    } else {

        baseName = argv[1]; // the name is already there, there is no '/'
    }

    char outputName[512];
    snprintf(outputName, sizeof(outputName), "filtered_%s", baseName);
    cv::imwrite(outputName, outputImage);

    FILE *logfile = fopen("log_cpu_processing.txt", "a");
    if (logfile != NULL) {
        fprintf(logfile,
                "Image: %s,  Blue Time: %.2f s, Green Time: %.2f s, Red Time: %.2f s, Total Time: %.2f s\n",
                baseName,
                cpu_time_used_blue,
                cpu_time_used_green,
                cpu_time_useed_red,
                cpu_time_useed_red + cpu_time_used_blue + cpu_time_used_green);

                fclose(logfile);
    } else {
        fprintf(stderr, "Could not open log file for writing.\n");
    }

    free(output_blue);
    free(output_green);
    free(output_red);

    free(input_blue);
    free(input_green);
    free(input_red);

    return 0;


}


void filter(uchar* input,uchar* output, int height, int width,size_t step, int w[3][3]) {
    for (int i = 1; i < height - 1; i++) {
        int row_ref = i*step;
        for (int j = 1; j < width - 1; j++) {
            int sum = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int col_ref = j + l;
                    sum += input[row_ref + k*step + col_ref] * w[k + 1][l + 1];
                }
            }
            
            if((sum/16) <= 255) {
                output[i*step + j] = sum / 16;
            } else {
                output[i*step + j] = 255;
            }
        }
    }
}
