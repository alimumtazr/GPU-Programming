import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("task4_times.csv")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['size'], df['cpu_time_us'], marker='o', label='CPU Time (μs)')
plt.plot(df['size'], df['gpu_time_us'], marker='s', label='GPU Time (μs)')

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (μs)")
plt.title("CPU vs GPU Matrix Addition Times")
plt.grid(True)
plt.legend()
plt.xticks(df['size'])
plt.tight_layout()

# Save figure
plt.savefig("task4_times_plot.png")
plt.show()
