#!/bin/bash
#SBATCH -J Ex1_assignment
#SBATCH --time=0:03:30
#SBATCH --partition=global
#SBATCH -n 25
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=13
#SBATCH --mail-user=s349351@studenti.polito.it
#SBATCH --mail-type=end
mpicc systolic_array_500.c -lm -o systolic_array_500
mpirun ./systolic_array_500  matrice_A_500.csv matrice_B_500.csv matrice_C_500.csv time_N2_25proc_500.csv
