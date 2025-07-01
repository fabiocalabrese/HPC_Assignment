#!/bin/bash
#SBATCH -J Ex1_assignment
#SBATCH --time=0:03:30
#SBATCH --partition=global
#SBATCH -n 16
#SBATCH --nodes=1
#SBATCH --mail-user=s349351@studenti.polito.it
#SBATCH --mail-type=end
mpicc systolic_array_2000.c -lm -o systolic_array_2000
mpirun ./systolic_array_2000  matrice_A_2000.csv matrice_B_2000.csv matrice_c_2000.csv time_N1_16proc_2000.csv
