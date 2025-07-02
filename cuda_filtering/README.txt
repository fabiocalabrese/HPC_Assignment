CUDA FILTERING
==================================

This directory contains the codes of a GPU-accelerated implementation of a 3x3 filter using NVIDIA CUDA

Structure:
----------
- 'src/': contains 2 folders

	- 'cuda/',  with 4 files:
		- "filter_cuda.cu": compute the output images with the applied filter
		- "filter_cuda_time_computation.cu": compute the execution time for all block dimensions
		- "filter_debug.cu": code used to check arbitrary values (single pixel) of the image. It prints th		selected interval and the times related to each channel
		- "launcher.sh": used to compile and run the selected code given the correct path of the image

	- 'seq' contains the sequential implementation of the filter together the launcher

- 'result/': contains: 
	- the various output for reference images and the images with noise, named as "{noise value}_output".
	- 'log/' folder containing all the result in terms of time, block size, number of threads, images and names. 	Each log file is named as "log_{block size}_{noise value}.
		-'speedup_computation/': contains the code for the computation of the speedup and throughput


- 'images/': contains all the images used

-'data_analisys/': contains:
	- 'code/' containing all the python code for the plots
	- 'plot/' containing the bar and graph plot


COMPILE and RUN
---------------
On a machine embedded with a GPU (such as Mixto):

"./launcher.sh"

Make sure that the launcher is provided with the correct libraries such as OpenCV and the correct path of the images.

----------------------------------------------------------------------------------------------------------------
NOTE: the .sh file must be in the same folder of the file .cu you want to run or change the path in the .sh file
----------------------------------------------------------------------------------------------------------------


OUTPUT
-------------
Depending on the program executed different output files are generated:

- "cuda_filtering.cu" provides "{image name}_filtered.jpeg", the log file
- "filter_cuda_time_computation" provides the log file without the output images


