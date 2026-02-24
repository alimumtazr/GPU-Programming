import matplotlib.pyplot as plt
import csv

def load_csv(filename, size_col, time_col):
    sizes, times = [], []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('#') or row[0] == size_col:
                continue
            sizes.append(row[0])
            times.append(float(row[1]))
    return sizes, times

cpu_sizes,    cpu_times    = load_csv("cpu_timings.csv",    "size", "cpu_us")
gpu_sizes,    gpu_times    = load_csv("gpu_timings.csv",    "size", "gpu_us")
tiled_sizes,  tiled_times  = load_csv("tiled_timings.csv",  "size", "tiled_us")

plt.figure(figsize=(11, 6))
plt.plot(cpu_sizes,   cpu_times,   marker='o', label='CPU')
plt.plot(gpu_sizes,   gpu_times,   marker='s', label='GPU Naive')
plt.plot(tiled_sizes, tiled_times, marker='^', label=f'GPU Tiled (32x32)')
plt.yscale('log')
plt.xlabel('Matrix Size (NxKxM)')
plt.ylabel('Time (microseconds, log scale)')
plt.title('CPU vs GPU Naive vs GPU Tiled — Matrix Multiplication')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
print("Plot saved to comparison.png")
