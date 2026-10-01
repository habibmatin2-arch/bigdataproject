import pandas as pd
import numpy as np

# -----------------------------
# 1) Inputs
# -----------------------------
input_file = "Vision_Benchmark_Results.csv"
output_file = "Master_Final_Benchmark_Results_with_Energy.xlsx"

keep_models = [
    "ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny", "ViT-B-16",
    "Swin-T", "MobileViT-S", "DenseNet121", "ShuffleNet-V2", "MobileNet-V3"
]

keep_datasets = ["102flowers", "Food-101", "ImageNet-Mini"]

# -----------------------------
# 2) Model power assumptions (editable)
#    If you have real measured wattage, replace these values.
# -----------------------------
power_map = {
    "ResNet50": 45.0,
    "EfficientNet-B0": 18.0,
    "ConvNeXt-Tiny": 42.0,
    "ViT-B-16": 50.0,
    "Swin-T": 44.0,
    "MobileViT-S": 12.0,
    "DenseNet121": 22.0,
    "ShuffleNet-V2": 8.0,
    "MobileNet-V3": 10.0,
}

# -----------------------------
# 3) Read data
# -----------------------------
df = pd.read_csv(input_file)

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Normalize text columns
df["Model"] = df["Model"].astype(str).str.strip()
df["Dataset"] = df["Dataset"].astype(str).str.strip()

# Convert numeric columns safely
numeric_cols = [
    "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)",
    "Inference Time (s)", "Throughput (img/s)", "Latency per Image (ms)",
    "Parameters (M)", "Trainable Params (M)", "Model Size (MB)",
    "Peak RAM (MB)", "Train Acc Epoch1 (%)", "Train Acc Epoch2 (%)", "Train Acc Epoch3 (%)"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 4) Filter target models/datasets
# -----------------------------
df = df[df["Model"].isin(keep_models) & df["Dataset"].isin(keep_datasets)].copy()

# -----------------------------
# 5) Add power and energy columns
# -----------------------------
df["Avg_Power_W"] = df["Model"].map(power_map)

# Estimated total energy for the inference run
df["Energy_J"] = df["Avg_Power_W"] * df["Inference Time (s)"]

# Energy per image in mJ/img
df["Energy_per_Image_mJ"] = (df["Avg_Power_W"] / df["Throughput (img/s)"]) * 1000

# Efficiency metrics
df["Accuracy_per_Watt"] = df["Accuracy (%)"] / df["Avg_Power_W"]
df["Throughput_per_Watt"] = df["Throughput (img/s)"] / df["Avg_Power_W"]

# Optional: sort nicely
df = df.sort_values(["Dataset", "Accuracy (%)", "Energy_J"], ascending=[True, False, True])

# -----------------------------
# 6) Create summary sheet
# -----------------------------
summary = (
    df.groupby("Model", as_index=False)
      .agg({
          "Accuracy (%)": "mean",
          "Throughput (img/s)": "mean",
          "Latency per Image (ms)": "mean",
          "Avg_Power_W": "mean",
          "Energy_J": "mean",
          "Energy_per_Image_mJ": "mean",
          "Accuracy_per_Watt": "mean",
          "Throughput_per_Watt": "mean"
      })
      .sort_values("Accuracy (%)", ascending=False)
)

# -----------------------------
# 7) Save to Excel
# -----------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Full_Results", index=False)
    summary.to_excel(writer, sheet_name="Model_Summary", index=False)

print(f"Saved: {output_file}")
print("Added columns: Avg_Power_W, Energy_J, Energy_per_Image_mJ, Accuracy_per_Watt, Throughput_per_Watt")
print(f"Rows processed: {len(df)}")
