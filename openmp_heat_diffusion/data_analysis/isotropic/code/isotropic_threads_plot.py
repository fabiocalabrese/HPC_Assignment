import numpy as np
import matplotlib.pyplot as plt

filepath =  "OpenMP_iteration_with4cpu_a_part2.log"#"OpenMP_iteration_with12cpu_a_part2.log" #"OpenMP_iteration_with24cpu_a_part2.log"
matrix = []
threads_id = []
time = []
#Taking datas from the file .txt in a matrix
with open(filepath, 'r') as file:
    for line in file:
        line.strip()
        matrix.append(float(line))

for i in range(0, len(matrix), 2):
    threads_id.append(matrix[i])
    time.append(matrix[i+1])

wfig, ax = plt.subplots()
#Plotting
plt.plot(threads_id, time, 'r')

plt.title("Performance execution of threads with 4 cores")
plt.ylabel('Time (s)')
plt.xlabel('Threads')
plt.savefig("heat_a_thread_4_cores")
plt.show()
