import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

file_names = [
    "log_16x16_ref.txt",
    "log_32x8_ref.txt",
    "log_32x32_ref.txt"
    ]
times_for_image_16x16 = defaultdict(list)
times_for_image_32x8 = defaultdict(list)
times_for_image_32x32 = defaultdict(list)
threads_for_image = defaultdict(list)
dic_times = [times_for_image_16x16,times_for_image_32x8,times_for_image_32x32]


def name_normalization(name):
    if "N" in name:
        base = name.split("N")[0].strip()
        name = base + ".jpg"
    return name

def process_files(filepath):
    with open(filepath) as file:
        for row in file:
            parts = row.strip().split(',')

            figure_name = None
            times = None

            for data in parts:
                data.strip()

                if data.startswith("Image:"):
                    figure_name = data.split("Image:")[1].strip()
                    figure_name= name_normalization(figure_name)
                elif data.startswith(" Total Time:"):
                    string_value = data.split("Total Time:")[1].strip().replace("ms", "")
                    times = float(string_value)
                if figure_name and times is not None:
                    if filepath == "log_16x16_ref.txt":
                        times_for_image_16x16[figure_name].append(times)
                    elif filepath == "log_32x8_ref.txt":
                        times_for_image_32x8[figure_name].append(times)
                    elif filepath == "log_32x32_ref.txt":
                        times_for_image_32x32[figure_name].append(times)

def threads_evaluation(filepath):
    with open(filepath, 'r')as file:
        for row in file:
            parts = row.strip().split(',')

            figure_name = None
            thread = None
            t = 0;
            for data in parts:
                data.strip()
                if data.startswith("Image:"):
                    figure_name = data.split("Image:")[1].strip()
                    figure_name= name_normalization(figure_name)
                elif data.startswith(" Threads:"):
                    string_value = data.split("Threads:")[1].strip()
                    thread = int(string_value)
                if figure_name and thread is not None:
                    if t!=1:
                        threads_for_image[figure_name].append(thread)
                        t=1

def plot_graphs(x,y):
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.xlabel("Threads")
    plt.ylabel("Time")
    if y == times_for_image_16x16:
        plt.title("Number of threads with respect to time 16x16")
    elif y == times_for_image_32x8:
        plt.title("Number of threads with respect to time 32x8")
    elif y == times_for_image_32x32:
        plt.title("Number of threads with respect to time 32x32")
    plt.grid(True)
    plt.show()

for j in file_names:
    process_files(j)

threads_evaluation("log_32x8_ref.txt")

for i, times in enumerate(dic_times):
    x= []
    y=[]
    labels=[]

    for img,time in times.items():
        if img in threads_for_image:
            x.append(threads_for_image[img])
            y.append(time)
            labels.append(img)

    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    labels = np.array(labels)

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    labels_sorted = labels[sorted_indices]

    name = file_names[i].removesuffix(".txt")

    plt.figure()
    plt.xscale('log')
    plt.plot(x_sorted, y_sorted, marker='o', linestyle='-', color='blue', label=file_names[i])
    plt.title(f'{name}: Thread vs time')
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (ms)')
    plt.xticks(x_sorted, [str(int(val)) for val in x_sorted], rotation=45, fontsize=6)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{name}3.jpeg")

plt.show()

