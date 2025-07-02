#!/bin/bash
#SBATCH --job-name=OpenMP_exercise
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s343467@studenti.polito.it
#SBATCH --partition=global    # o la partizione CPU che usi
#SBATCH --time=00:01:30         # tempo massimo di esecuzione (modifica a piacere)
#SBATCH --nodes=1               # un nodo
#SBATCH --ntasks=1              # un processo (OpenMP è multithread, non multiprocessi)
#SBATCH --cpus-per-task=24       # numero di thread OpenMP
#SBATCH --output=OpenMP_exercise24.log
#SBATCH --mem=1G                # memoria (modifica se serve)

# ************************************************
# command to launch the job in the scheduler:
# sbatch openmp_launcher.sbatch

# Command to check the job status:
# squeue -u your_user
# ************************************************

# Carica eventuali moduli necessari (gcc con supporto openmp)
module load gcc

# Setta il numero di thread OpenMP (opzionale, puoi gestirlo anche da programma)
export OMP_NUM_THREADS=24
# Compila il codice OpenMP (adatta il nome del file sorgente)
gcc -fopenmp heat_diffusion_b_omp.c -o heat_a

# Esegui il programma
./heat_b

