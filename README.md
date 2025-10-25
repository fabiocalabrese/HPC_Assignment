REPOSITORY STRUCTURE - HPC ASSIGNMENT
=======================================

This repository contains three independent projects developed for the High Performance Computing course at Politecnico di Torino. Each project resides in its own directory and is self-contained.

Each project folder includes a dedicated README file that explains the structure and provides instructions for compilation and execution.

Project Directories:
--------------------

1. openmp_heat_diffusion/
   - Simulation of 2D heat diffusion using sequential and OpenMP implementations.
   - Includes both isotropic and anisotropic models.
   - See openmp_heat_diffusion/README.txt for more details.

2. cuda_filter/
   - CUDA implementation of a 2D image filter (e.g., blur, sharpen, edge detection).
   - See cuda_filter/README.txt for compilation and usage instructions.

3. mpi_matrix_mult/
   - MPI-based matrix multiplication using a systolic array approach.
   - Designed to run with multiple MPI processes.
   - See mpi_matrix_mult/README.txt for build and execution instructions.

General Notes:
--------------
- Each folder is self-contained and can be compiled and executed independently.
- SLURM launcher scripts are included where needed for HPC environments.
- Source code is written in C or CUDA C, and can be compiled using gcc, nvcc, or mpicc as appropriate.

Authors:
-------
Carlo Mattioli - s349351@studenti.polito.it
Fabio Calabrese - s343467@studenti.polito.it
Michele Merla - s343500@studenti.polito.it

Politecnico di Torino – High Performance Computing Assignment
