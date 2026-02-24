import matplotlib.pyplot as plt
import csv

def load_csv(filename):
    sizes, times = [], []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            sizes.append(row[0])
            times.append(float(row[1]))
    return sizes, times

cpu_sizes, cpu_times = load_csv("cpu_timings.csv")
gpu_sizes, gpu_times = load_csv("gpu_timings.csv")

plt.figure(figsize=(10, 6))
plt.plot(cpu_sizes, cpu_times, marker='o', label='CPU')
plt.plot(gpu_sizes, gpu_times, marker='s', label='GPU Naive')
plt.yscale('log')
plt.xlabel('Matrix Size (NxKxM)')
plt.ylabel('Time (microseconds, log scale)')
plt.title('CPU vs GPU Naive — Matrix Multiplication Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
print("Plot saved to comparison.png")
