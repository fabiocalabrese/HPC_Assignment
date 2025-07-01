#!/bin/bash

# Percorsi per OpenCV
OPENCV_INCLUDE=/usr/include/opencv4
OPENCV_LIB=/usr/lib/aarch64-linux-gnu

# Compilazione con g++
g++ filter_seq.cpp -o filter_seq \
  -I$OPENCV_INCLUDE \
  -L$OPENCV_LIB \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
  -std=c++11




for img in select/the/correct/path/of/the/images_folder/*.jpg; do
    ./filter_seq "$img"
done


