import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
df = pd.read_csv("at_least_counts.csv")

# Pivot the data
heatmap_data = df.pivot(index="c", columns="b", values="count")

# Ensure both axes are sorted ascending
heatmap_data = heatmap_data.sort_index(ascending=True)  # y-axis (c)
heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)  # x-axis (b)

# Plot the heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="viridis")

# Fix axis orientation so origin is bottom-left
ax.invert_yaxis()

plt.title("Heatmap of count by b (x-axis) and c (y-axis)")
plt.xlabel("b")
plt.ylabel("c")
plt.tight_layout()
# plt.show()
plt.savefig("at_least_counts_heatmap.png")
