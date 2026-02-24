import matplotlib.pyplot as plt
import csv

tile_sizes, times = [], []

with open("tile_timings.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tile_sizes.append(int(row["tile_size"]))
        times.append(float(row["time_us"]))

plt.figure(figsize=(7, 5))
plt.plot(tile_sizes, times, marker='o', color='steelblue')
plt.xlabel('Tile Size')
plt.ylabel('Time (microseconds)')
plt.title('Tile Size vs Execution Time (1024x1024)')
plt.xticks(tile_sizes, [f'{t}x{t}' for t in tile_sizes])
plt.tight_layout()
plt.savefig("best_tile_size.png", dpi=150)
print("Plot saved to best_tile_size.png")
