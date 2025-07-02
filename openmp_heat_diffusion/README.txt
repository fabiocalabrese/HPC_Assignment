HEAT DIFFUSION 2D - ISOTROPIC CASE
==================================

This directory contains the implementation of the 2D heat diffusion model with isotropic and anisotropic conductivity

Structure:
----------

- 'src/': contains the source files. It is divided into

	- 'isotropic/'

	- 'anisotropic/'

	For both the isotropic and anisotropic case there are 2 folders:

	- 'omp/': contains the .c and .sbatch file to compile and run the program in its openMP 		implementation

	- 'seq/': contains the .c sequential version of the program

- 'data_analysis/': it is divided into 'isotropic/' and 'anisotropic/'. Each folder contains:

	- 'code/': contains the code necessary to create the plots

	- 'heatmap/': contains the animation of the heat diffusion

	- 'images/': contains the plots

	- 'output/': contains the output logs

COMPILE AND RUN
----------------

For the OpenMP version:

	- In case you are using a cluster: "sbatch launcher.sbatch"

	-In case you are using a normal PC: "gcc -fopenmp heat_diffusion_a.c -o heat"



For the sequential version:

	Compile: "gcc heat_diffusion_seq.c -o heat"

	Run: "./heat"



