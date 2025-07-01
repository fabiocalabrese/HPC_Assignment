import numpy as np

# Genera due matrici 16x16 di tipo float64 (double precision)
matrix1 = np.random.uniform(0.0, 100.0, size=(2000, 2000)).astype(np.float64)
matrix2 = np.random.uniform(0.0, 100.0, size=(2000, 2000)).astype(np.float64)

# Salva le matrici in file CSV con 3 decimali
np.savetxt("matrice_A_2000.csv", matrix1, delimiter=",", fmt="%.3f")
np.savetxt("matrice_B_2000.csv", matrix2, delimiter=",", fmt="%.3f")
