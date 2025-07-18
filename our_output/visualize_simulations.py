import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
years_to_n = {5: 60, 10: 120, 20: 240, 30: 360}
input_dir = "our_output/model_results_500"
file_pattern = os.path.join(input_dir, "d*year_theta_*_*_metrics.csv")

# LOAD & CONCAT
all_dfs = []
for path in glob(file_pattern):
    fname = os.path.basename(path).replace(".csv", "")
    parts = fname.split("_")
    years = int(parts[0][1:-4])
    theta = float(parts[2])
    model = parts[3].upper()
    
    df = pd.read_csv(path)
    df["n"] = years_to_n[years]
    df["theta"] = theta
    df["model"] = model
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

# ─────────────────────────────────────────
# SUMMARY TABLE: Group → Mean → Pivot
# ─────────────────────────────────────────
grp = df.groupby(["theta", "n", "model"], as_index=False).agg({
    "bias_test": "mean",
    "mase_test": "mean",
    "rmse_train": "mean",
    "rmse_test": "mean"
})

wide = grp.pivot(index=["theta", "n"], columns="model")
wide.columns = [f"{model}_{metric}" for metric, model in wide.columns]
wide = wide.reset_index()

# Rename columns
wide.columns = [
    "θ", "n",
    "BNB_bias", "NB_bias",
    "BNB_mase", "NB_mase",
    "BNB_rmse_train", "NB_rmse_train",
    "BNB_rmse_test", "NB_rmse_test"
]

# Drop bias columns (optional)
wide.drop(columns=["BNB_bias", "NB_bias"], inplace=True)

# ROUND numeric columns for display
rounded_wide = wide.copy()
for col in rounded_wide.columns[2:]:
    rounded_wide[col] = rounded_wide[col].round(3)

# DISPLAY the summary table
print(rounded_wide.to_string(index=False))
print(type(rounded_wide))

# ─────────────────────────────────────────
# PLOTTING: Nested 4×4 Panel (Grouped by Metric → Timespan)
# ─────────────────────────────────────────
sns.set(style="whitegrid", font_scale=0.9)
metrics = [
    ("bias_test", "Bias on test data"),
    ("mase_test", "MASE on test data"),
    ("rmse_train", "RMSE on training data"),
    ("rmse_test", "RMSE on test data"),
]
n_values = [60, 120, 240, 360]
thetas = sorted(df["theta"].unique())

fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharey=False)
axes = axes.reshape(4, 4)

for metric_col, (metric, label) in enumerate(metrics):
    valid_vals = df[metric].dropna()
    if not valid_vals.empty and np.isfinite(valid_vals).all():
        metric_min = valid_vals.min()
        metric_max = valid_vals.max()
        y_margin = 0.05 * (metric_max - metric_min)
        y_limits = (metric_min - y_margin, metric_max + y_margin)
    else:
        y_limits = None

    for i, n in enumerate(n_values):
        row = i // 2 + (2 * (metric_col // 2))  # 0–1 or 2–3
        col = i % 2 + (2 * (metric_col % 2))    # 0–1 or 2–3

        ax = axes[row, col]
        subset = df[df["n"] == n]

        sns.violinplot(
            data=subset,
            x="theta",
            y=metric,
            hue="model",
            order=thetas,
            palette={"BNB": "#FF3643", "NB": "#2DC4F6"},
            split=False,
            ax=ax
        )

        ax.set_title(f"{label} — n={n}", fontsize=9)
        ax.set_xlabel("θ", fontsize=8)
        if col % 2 == 0:
            ax.set_ylabel(label, fontsize=8)
        else:
            ax.set_ylabel("")

        if y_limits:
            ax.set_ylim(y_limits)

        if row != 0 or col != 0:
            ax.get_legend().remove()

        ax.set_xticklabels([str(t) for t in thetas], fontsize=7)
        ax.tick_params(axis='y', labelsize=7)

# Add legend to top-left plot
axes[0, 0].legend(title="Model", loc="lower left", fontsize=8, title_fontsize=9)

plt.tight_layout()
plt.savefig("our_output/metrics_simulations_panelplot.png", dpi=300, bbox_inches="tight")
plt.show()
