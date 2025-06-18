# %%

import pandas as pd
import matplotlib.pyplot as plt

metadata = pd.read_csv(
    "/Users/bridge/PhD/Code/exebenchmark/configurations/SPEAKEASY/all_metadata.csv"
)

# Check if 'timestamp' column exists
if "timestamp" not in metadata.columns:
    raise ValueError("The 'timestamp' column is missing from the CSV file.")

# Extract the year from the timestamp column
metadata["year_month"] = metadata["timestamp"].str[:7]
# Divide data by label
metadata_label_0 = metadata[metadata["label"] == 0].copy()
metadata_label_1 = metadata[metadata["label"] == 1].copy()

# Define the period range
start_year = 2019
end_year = 2022


def assign_period(dt):
    if pd.isnull(dt):
        return "unreal"
    year = dt.year
    if start_year <= year <= end_year:
        quarter = (dt.month - 1) // 4 + 1  # 3 quarters: months 1-4, 5-8, 9-12
        if quarter > 3:
            quarter = 3
        return f"{year}-Q{quarter}"
    else:
        return "unreal"


# Group by quarters for label 0
metadata_label_0["year_month"] = pd.to_datetime(
    metadata_label_0["year_month"], errors="coerce"
)
metadata_label_0["period"] = metadata_label_0["year_month"].apply(assign_period)
period_counts_label_0 = metadata_label_0["period"].value_counts().sort_index()

# Group by quarters for label 1
metadata_label_1["year_month"] = pd.to_datetime(
    metadata_label_1["year_month"], errors="coerce"
)
metadata_label_1["period"] = metadata_label_1["year_month"].apply(assign_period)
period_counts_label_1 = metadata_label_1["period"].value_counts().sort_index()

# # Extract year counts for label 0
# year_counts_label_0 = metadata_label_0['year'].value_counts().sort_index()
#
# # Extract year counts for label 1
# year_counts_label_1 = metadata_label_1['year'].value_counts().sort_index()

# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 30,  # General font size
        "axes.titlesize": 30,  # Title font size
        "axes.labelsize": 25,  # Axis label font size
        "xtick.labelsize": 30,  # X-axis tick label font size
        "ytick.labelsize": 30,  # Y-axis tick label font size
        "legend.fontsize": 20,  # Legend font size
    }
)

plt.figure(figsize=(50, 25))
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=1)
ax0 = period_counts_label_0.plot(
    kind="bar",
    title="Samples per 4-Month Period GOODWARE",
    xlabel="Year-4Month",
    ylabel="Count",
)
for i, count in enumerate(period_counts_label_0):
    ax0.text(i, count * 1.05, str(count), ha="center", va="bottom", fontsize=30)
plt.show()

plt.figure(figsize=(50, 25))
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=1)
ax1 = period_counts_label_1.plot(
    kind="bar",
    title="Samples per 4-Month Period MALWARE",
    xlabel="Year-4Month",
    ylabel="Count",
)
for i, count in enumerate(period_counts_label_1):
    ax1.text(i, count * 1.05, str(count), ha="center", va="bottom", fontsize=30)
plt.show()
