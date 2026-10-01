import pandas as pd
from itertools import combinations

df = pd.read_csv("classification_results.csv")

models = ["CLIP", "ConvNeXt", "DeiT", "EfficientNet",
          "MobileViT", "ResNet", "SigLIP", "Swin", "ViT"]

# ۱. Full agreement
full_agree = (df[models].nunique(axis=1) == 1).mean() * 100
print(f"Full Agreement (all 9): {full_agree:.2f}%")

# ۲. Majority agreement (≥5 models agree)
consensus = df[models].apply(lambda row: row.value_counts().iloc[0], axis=1)
majority = (consensus >= 5).mean() * 100
print(f"Majority Agreement (≥5): {majority:.2f}%")

# ۳. Strong majority (≥7)
strong = (consensus >= 7).mean() * 100
print(f"Strong Agreement (≥7): {strong:.2f}%")

# ۴. Mean consensus count
print(f"Mean consensus count: {consensus.mean():.2f}")

# ۵. Pairwise agreement matrix
print("\n=== Pairwise Agreement Matrix (%) ===")
matrix = pd.DataFrame(index=models, columns=models, dtype=float)
for m in models:
    matrix.loc[m, m] = 100.0
for m1, m2 in combinations(models, 2):
    val = (df[m1] == df[m2]).mean() * 100
    matrix.loc[m1, m2] = val
    matrix.loc[m2, m1] = val
print(matrix.round(1).to_string())

# ۶. Per-model top class
print("\n=== Per-Model Top Class ===")
for m in models:
    vc = df[m].value_counts()
    print(f"{m}: {df[m].nunique()} unique | top='{vc.index[0]}' ({vc.iloc[0]/len(df)*100:.1f}%)")

# ۷. Top 15 classes (by majority vote)
print("\n=== Top 15 Classes (majority vote) ===")
majority_class = df[models].apply(lambda row: row.value_counts().index[0], axis=1)
print(majority_class.value_counts().nlargest(15).to_string())
