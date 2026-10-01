import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from collections import Counter

# ── تنظیمات سبک یکسان ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
PALETTE = "viridis"

df = pd.read_csv("classification_results.csv")

models      = ["CLIP", "ConvNeXt", "DeiT", "EfficientNet", "MobileViT", "ResNet", "SigLIP", "Swin", "ViT"]
label_cols  = models
latency_cols = {m: f"{m}_time" for m in models}

# ── 01: Class Distribution ───────────────────────────────────────────────────
all_labels = []
for col in label_cols:
    all_labels.extend(df[col].dropna().tolist())

label_counts = Counter(all_labels)
top_n = 20
top_labels = dict(label_counts.most_common(top_n))

fig, ax = plt.subplots(figsize=(14, 6))
colors = sns.color_palette(PALETTE, len(top_labels))
ax.bar(top_labels.keys(), top_labels.values(), color=colors)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Frequency")
ax.set_title(f"Top {top_n} Predicted Classes Across All Models", fontsize=14, weight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("01_Class_Distribution.png", dpi=300)
plt.close()

# ── 04: Category Consensus ───────────────────────────────────────────────────
def consensus_score(row):
    preds = [row[m] for m in models if pd.notna(row[m])]
    if not preds:
        return 0
    most_common_count = Counter(preds).most_common(1)[0][1]
    return most_common_count / len(preds)

df["consensus"] = df.apply(consensus_score, axis=1)

bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
df["consensus_bin"] = pd.cut(df["consensus"], bins=bins, labels=labels, right=False)
bin_counts = df["consensus_bin"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 5))
colors = sns.color_palette(PALETTE, len(bin_counts))
ax.bar(bin_counts.index.astype(str), bin_counts.values, color=colors)
ax.set_xlabel("Consensus Level")
ax.set_ylabel("Number of Images")
ax.set_title("Category Consensus Distribution Across Models", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("04_Category_Consensus.png", dpi=300)
plt.close()

# ── 05: Latency Comparison (mean) ────────────────────────────────────────────
mean_latency = {m: df[latency_cols[m]].mean() for m in models}

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette(PALETTE, len(mean_latency))
ax.bar(mean_latency.keys(), mean_latency.values(), color=colors)
ax.set_xlabel("Model")
ax.set_ylabel("Mean Inference Time (s)")
ax.set_title("Mean Inference Latency per Model", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("05_Latency_Comparison.png", dpi=300)
plt.close()

# ── 06: Throughput Comparison ────────────────────────────────────────────────
throughput = {m: 1.0 / mean_latency[m] for m in models}

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette(PALETTE, len(throughput))
ax.bar(throughput.keys(), throughput.values(), color=colors)
ax.set_xlabel("Model")
ax.set_ylabel("Throughput (images/s)")
ax.set_title("Estimated Throughput per Model", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("06_Throughput_Comparison.png", dpi=300)
plt.close()

# ── 07: Latency Boxplot ──────────────────────────────────────────────────────
latency_data = {m: df[latency_cols[m]].dropna() for m in models}

data_list  = [v.tolist() for v in latency_data.values()]
label_list = list(latency_data.keys())

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette(PALETTE, len(label_list))

bp = ax.boxplot(
    data_list,
    tick_labels=label_list,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="orange", linewidth=2),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

ax.set_xlabel("Model")
ax.set_ylabel("Inference Time (s)")
ax.set_title("Inference Time Distribution per Model (Boxplot, no outliers)", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("07_Latency_Boxplot.png", dpi=300)
plt.close()

print("✅ All 5 charts saved successfully.")
