SYSTOLIC ARRAY USING MPI
==================================

This directory contains the implementation of the systolic array matrix multiplication using MPI

Structure:
----------

- 'src/' 

	- 'sequential/':  contains the traditional matrix multiplication. If you want to run it, 		make sure that the parameter N referring to the matrix dimension coincides with 		the matrix in the input file and do the same for the input file names.


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

------------------------------------------------------------------------------------------------
NOTE: Make sure that the input matrices, the .c program and the .sbatch file are in the same folder. In this repository they are in different folders just to have a better organization of the files. Alternatively, you can just adjust the path in the .c and .sbatch files.
------------------------------------------------------------------------------------------------


OUTPUT
-------------
The program generates the following output files:
- The output matrix
- The time and number of processes contained in the file: 
   "time_N{Number of nodes}_{Number of processes}proc_{Matrix size}.csv"