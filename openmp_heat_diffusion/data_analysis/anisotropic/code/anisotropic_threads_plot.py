import numpy as np
import matplotlib.pyplot as plt

filepath =  "OpenMP_iteration_with24cpu_b_part2.log" #"heat_b_12_CPU.txt"
matrix = []
threads_id = []
time = []

with open(filepath, 'r') as file:
    for line in file:
        line.strip()
        matrix.append(float(line))

for i in range(0, len(matrix), 2):
    threads_id.append(matrix[i])
    time.append(matrix[i+1])

fig, ax = plt.subplots()

plt.plot(threads_id, time, 'r')

plt.title("Performance execution of threads with 24 CPUs")
plt.ylabel('Time (s)')
plt.xlabel('Threads')
plt.savefig("heat_b_thread_24_CPU")
plt.show()
