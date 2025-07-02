SYSTOLIC ARRAY USING MPI
==================================

This directory contains the implementation of the systolic array matrix multiplication using MPI

Structure:
----------

- 'src/' 

	- 'sequential/':  contains the traditional matrix multiplication


	- 'MPI_implemented/': 

		- contains three .c files, one for each matrix size. 
		The files are named in 	this way "systolic_array_{Matrix size}.c".

		- 'sbatch/': contains .sbatch files to compile and execute the MPI codes.
		The files are named as "run_N{Number of nodes}_{Number of processes}proc_{Matrix size}

- 'input/': contains the input matrices of different sizes and a python code to generate them

- 'output/': contains the output returned by the code


COMPILE and RUN
---------------

On a cluster: 

"sbatch {sbatch filename}"



OUTPUT
-------------
The program generates the following output files:
- The output matrix
- The time and number of processes contained in the file: 
   "time_N{Number of nodes}_{Number of processes}proc_{Matrix size}.csv"