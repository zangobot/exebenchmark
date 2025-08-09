import pandas as pd
from pathlib import Path

INPUT_FILE = "ranking.csv"
OUTPUT_FILE = "tables/leaderboard_table.tex"
METRICS = ["P", "T", "R", "I"]

# === Load CSV ===
df = pd.read_csv(INPUT_FILE)

# Sort by Rank (highest first)
df_sorted = df.sort_values(by="rank", ascending=False)

# Rename columns
df_sorted = df_sorted.rename(
    columns={
        "model": "Model",
        "performance_metric": "P",
        "temporal_metric": "T",
        "robustness_metric": "R",
        "inference_metric": "I",
        "rank": "Rank",
    }
)

# Round numeric values first
for col in METRICS + ["Rank"]:
    df_sorted[col] = df_sorted[col].round(2)

# Highlight all best (ties included) for metrics
for col in METRICS:
    best_val = df_sorted[col].max()
    df_sorted[col] = df_sorted[col].apply(
        lambda x: f"\\textbf{{{x:.2f}}}" if x == best_val else f"{x:.2f}"
    )

# Highlight best Rank as well
best_rank = df_sorted["Rank"].max()
df_sorted["Rank"] = df_sorted["Rank"].apply(
    lambda x: f"\\textbf{{{x:.2f}}}" if x == best_rank else f"{x:.2f}"
)

# Final columns
final_df = df_sorted[["Model", "Rank", "P", "T", "R", "I"]]

# Export LaTeX table
latex_table = final_df.to_latex(
    index=False,
    escape=False,  # keep LaTeX bold
    column_format="lcccccc",
    caption=(
        "\\exebench leaderboard, detailing the value of each metric "
        "(performance as P, temporal as T, robustness as R, and inference as I)."
    ),
    label="tab:leaderboard",
    bold_rows=False,
)

# Add gray background in alternating rows
lines = latex_table.split("\n")
new_lines = []
in_table_body = False
data_row_count = 0

for i, line in enumerate(lines):
    if "\\midrule" in line:
        in_table_body = True
        new_lines.append(line)
        continue
    elif "\\bottomrule" in line:
        in_table_body = False

    if in_table_body and "&" in line and "\\\\" in line:
        data_row_count += 1
        if data_row_count % 2 == 1:
            new_lines.append("\\rowcolor{gray!10}")

    new_lines.append(line)

enhanced_latex = "\n".join(new_lines)

Path(OUTPUT_FILE).write_text(enhanced_latex)
print(f"LaTeX table with alternating gray blocks saved to {OUTPUT_FILE}")
