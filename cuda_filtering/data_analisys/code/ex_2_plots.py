import matplotlib.pyplot as plt
from collections import defaultdict

file_names= [
    "log_16x16_ref.txt",
    "log_32x8_ref.txt",
    "log_32x32_ref.txt",
    "log_16x16_noise_50.txt",
    "log_32x8_noise_50.txt",
    "log_32x32_noise_50.txt",
    "log_16x16_noise_75.txt",
    "log_32x8_noise_75.txt",
    "log_32x32_noise_75.txt",
    "log_16x16_noise_90.txt",
    "log_32x8_noise_90.txt",
    "log_32x32_noise_90.txt"
]
file_labels = [f.replace("log_", "").replace(".txt", "") for f in file_names]
times_for_image = defaultdict(list)
width_image = defaultdict(list)
threads_image = defaultdict(list)

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
                    times_for_image[figure_name].append(times)

def name_normalization(name):
    if "N" in name:
        base = name.split("N")[0].strip()
        name = base + ".jpg"
    return name

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
                        threads_image[figure_name].append(thread)
                        t=1

def size_of_image(filepath):
    with open(filepath, 'r')as file:
        for row in file:
            parts = row.strip().split(',')

            figure_name = None
            size = None
            t = 0;
            for data in parts:
                data.strip()
                if data.startswith("Image:"):
                    figure_name = data.split("Image:")[1].strip()
                    figure_name= name_normalization(figure_name)
                elif data.startswith(" Size:"):
                    size = data.split("Size:")[1].strip()
                if figure_name and size is not None:
                    if t!=1:
                        width_image[figure_name].append(size)
                        t=1

def plot_histograms_per_image(times_for_image, file_labels):
    for image_name, times in times_for_image.items():
        plt.figure(figsize=(8, 5))

        # Crea etichette per ogni barra
        if file_labels and len(file_labels) == len(times):
            labels = file_labels
        else:
            labels = ["16x16_ref","32x8_ref","32x32_ref"]
            #labels = [f"Run {i+1}" for i in range(len(times))]
        thread = threads_image[image_name]
        size = width_image[image_name]
        # Istogramma per questa immagine
        plt.bar(labels, times, color='steelblue')
        plt.title(f"Times for image: {image_name}")
        plt.ylabel("Time (ms)")
        plt.xlabel(f"Measure of the image: {size} Number of threads: {thread}",)
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.savefig(f"{image_name}")
        plt.show()

for j in file_names:
    process_files(j)

threads_evaluation("log_32x8_ref.txt")
size_of_image("size_images.txt")
plot_histograms_per_image(times_for_image,file_labels)
