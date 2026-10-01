import pandas as pd
from itertools import combinations

df = pd.read_csv("classification_results.csv")
models = ["CLIP","ConvNeXt","DeiT","EfficientNet","MobileViT","ResNet","SigLIP","Swin","ViT"]
preds = df[models]

# Consensus
agree_counts = preds.apply(lambda r: r.value_counts().iloc[0], axis=1)
print("Full (9/9):", (agree_counts==9).mean()*100)
print("Strong (>=7):", (agree_counts>=7).mean()*100)
print("Majority (>=5):", (agree_counts>=5).mean()*100)
print("Mean consensus:", agree_counts.mean())

# Pairwise agreement
for m1, m2 in combinations(models, 2):
    rate = (df[m1]==df[m2]).mean()*100
    print(f"{m1} vs {m2}: {rate:.1f}%")

# Unique classes per model
for m in models:
    print(f"{m}: {df[m].nunique()} classes")

# Top 5 majority-vote classes
from collections import Counter
majority = preds.apply(lambda r: r.value_counts().index[0], axis=1)
print(Counter(majority).most_common(5))

# Throughput per model
for m in models:
    t_col = f"{m}_time"
    thr = 1 / df[t_col].mean()
    print(f"{m}: {thr:.2f} img/s")
