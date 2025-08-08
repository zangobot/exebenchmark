import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Patch
import scienceplots  # noqa: F401

plt.style.use("science")


def plot_spie_chart(ax, title, values, colors=None):
    num_slices = len(values)
    angle_per_slice = 360 / num_slices

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw concentric grid
    for r in [0.25, 0.5, 0.75, 1.0]:
        circle = Circle(
            (0, 0),
            r,
            facecolor="none",
            edgecolor="lightgray",
            linestyle="dotted",
            linewidth=0.6,
        )
        ax.add_patch(circle)

    # Use provided colors or fallback to tab10
    if colors is None:
        cmap = plt.cm.get_cmap("tab10", num_slices)
        colors = [cmap(i) for i in range(num_slices)]

    for i, (value, color) in enumerate(zip(values, colors)):
        # Start from North (0° up), rotate clockwise
        center_angle = -i * angle_per_slice + 90
        theta1 = center_angle - angle_per_slice / 2
        theta2 = center_angle + angle_per_slice / 2

        wedge = Wedge(
            center=(0, 0),
            r=value,
            theta1=theta1,
            theta2=theta2,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.add_patch(wedge)

    ax.set_title(title, fontsize=12, pad=4)


csv_file = pd.read_csv("ranking.csv")
labels = ["Performance", "Inference", "Temporal", "Robustness"]
colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]  # Color-blind friendly

num_rows, num_cols = 3, 10
fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 1.6 * num_rows))
axes = axes.flatten()

for i, (idx, row) in enumerate(csv_file.iterrows()):
    if i >= len(axes):
        break
    title = f"{row[1]} \n Rank = {row[-1]:.2f}"
    values = row[2 : 2 + len(labels)].tolist()
    plot_spie_chart(axes[i], title, values, colors=colors)


legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in zip(labels, colors)
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=len(labels),
    fontsize=14,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),  # Slightly below the figure
)

plt.tight_layout()
plt.savefig("spie_chart.pdf", dpi=300, bbox_inches="tight")  # Save the figure
