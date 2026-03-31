import pandas as pd

# Read the CPU-only CSV (from task2)
cpu_df = pd.read_csv("task2_cpu_times.csv")  # columns: size, cpu_time_us

# Read the GPU-only CSV (from task3)
gpu_df = pd.read_csv("task3_gpu_times.csv")  # columns: size, gpu_time_us

# Merge both on 'size'
combined_df = pd.merge(cpu_df, gpu_df, on="size")

# Save as task4_times.csv
combined_df.to_csv("task4_times.csv", index=False)

print("Combined CSV saved as task4_times.csv:")
print(combined_df)
