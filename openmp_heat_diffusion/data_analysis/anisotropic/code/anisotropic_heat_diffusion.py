import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from matplotlib.colors import LinearSegmentedColormap

#Matrix parameters
matrix_size = 1024
matrices = []
filepath = "output_heat_diffusion_b.txt"

#Open output file
with open(filepath, 'r') as file:
    matrix_now = [] #First initialization of a row vector
    for line in file: #Line for line
        if line.strip(): #Pass over empty lines
            row_list = list(map(float, line.strip().split())) #Conversion of rows in lists
            matrix_now.append(row_list) #adding the list (row) to the matrix
            if len(matrix_now) == matrix_size: #checks if the rows are 1024, so a matrix
                matrices.append(np.array(matrix_now)) #Append every matrix to final matrix
                matrix_now = [] #Initialization of another matrix

matrices = np.array(matrices)
#fig, ax = plt.subplots()
#start_row, end_row = 500, 550
#start_column, end_column = 500, 550


fig, ax = plt.subplots()#Function to call the plot
#colors = [
#    (0.678, 0.847, 0.902),  # light blue (RGB for light sky blue)
#    (1.0, 0.0, 0.0)         # red
#    ]

#cmap = LinearSegmentedColormap.from_list("lightblue_red", colors)
#Definition of the parameters of the plot
heatmap = ax.imshow(matrices[0], cmap='hot', interpolation='nearest', vmin=25, vmax=550,aspect='auto')
#print the colormap
plt.colorbar(heatmap)

#Function to update the frames
def update(frame):
    heatmap.set_data(matrices[frame])
    return [heatmap]

#Function for animation
ani = animation.FuncAnimation(
    fig, update, frames=matrices.shape[0], interval=100, blit=True
)

plt.show()

#Functions to save the animation
ani.save('heatmap_animation1.mp4', writer='ffmpeg')
ani.save('heatmap_animation_prova.gif', writer='pillow')
