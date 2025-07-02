HEAT DIFFUSION 2D - ISOTROPIC CASE
==================================

This directory contains the implementation of the 2D heat diffusion model with isotropic and anisotropic conductivity

Structure:
----------
For both the isotropic and anisotropic case there are 2 folders:
- omp (contains the .c and .sbatch file to compile and run the program in its openMP implementation)
- seq (contains the .c sequential version of the program)

OMP FOLDER
- heat_diffusion_omp.c
- launcher.sbatch

To compile and run the program:

If you are working with a cluster: "sbatch launcher.sbatch" 

If you are working with a normal PC: "gcc -fopenmp heat_diffusion_a.c -o heat


SEQ FOLDER 
-heat_diffusion_seq.c 

To compile the program and create the executable:

gcc heat_diffusion_seq.c -o heat

To run

./heat

