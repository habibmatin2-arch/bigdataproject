import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os

# Create output directory
os.makedirs('paper_assets', exist_ok=True)

# 1. Define Architecture Families based on keywords
FAMILIES = {
    'CNNs': ['resnet', 'efficientnet', 'mobilenet', 'vgg', 'densenet', 'inception', 'convnext'],
    'Transformers': ['vit', 'swin', 'deit', 'cait', 'eva'],
    'Vision-Language': ['clip', 'siglip', 'blip', 'align']
}

def assign_family(model_name):
    name_lower = model_name.lower()
    for family, keywords in FAMILIES.items():
        if any(kw in name_lower for kw in keywords):
            return family
    return 'Other'

def main():
    print("Loading classification_results.csv...")
    try:
        df = pd.read_csv('classification_results.csv')
    except FileNotFoundError:
        print("Error: 'classification_results.csv' not found.")
        return

    # Clean data: drop filepath
    if 'filepath' in df.columns:
        df = df.drop(columns=['filepath'])
    
    # Filter to only use Top-1 predictions if columns have "Top-1" in name
    top1_cols = [c for c in df.columns if 'Top-1' in c]
    if top1_cols:
        df_models = df[top1_cols]
    else:
        df_models = df # Fallback if no Top-1 suffix exists

    # Normalize column names for assignment
    original_cols = df_models.columns

    print("Calculating metrics...")
    
    # --- Metric 1: Consensus Alignment ---
    # Find the majority vote (mode) for each row
    consensus_labels = df_models.mode(axis=1)[0]
    
    model_consensus_acc = {}
    for col in df_models.columns:
        acc = (df_models[col] == consensus_labels).mean() * 100
        model_consensus_acc[col] = acc
        
    family_consensus = {'CNNs': [], 'Transformers': [], 'Vision-Language': []}
    for col, acc in model_consensus_acc.items():
        fam = assign_family(col)
        if fam in family_consensus:
            family_consensus[fam].append(acc)
            
    avg_consensus = {fam: (np.mean(vals) if vals else 0.0) for fam, vals in family_consensus.items()}

    # --- Metric 2: Intra-family Agreement ---
    avg_intra = {}
    for fam in FAMILIES.keys():
        fam_models = [c for c in df_models.columns if assign_family(c) == fam]
        if len(fam_models) >= 2:
            agreements = []
            for m1, m2 in combinations(fam_models, 2):
                agr = (df_models[m1] == df_models[m2]).mean() * 100
                agreements.append(agr)
            avg_intra[fam] = np.mean(agreements)
        elif len(fam_models) == 1:
            avg_intra[fam] = 100.0 # Only one model implies 100% agreement with itself
        else:
            avg_intra[fam] = 0.0

    # --- Plotting ---
    print("Generating Figure 8: Generations Comparison Chart...")
    labels = list(FAMILIES.keys())
    consensus_scores = [avg_consensus[l] for l in labels]
    intra_scores = [avg_intra[l] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, consensus_scores, width, label='Alignment with Consensus (Majority)', color='#4c72b0', edgecolor='black')
    rects2 = ax.bar(x + width/2, intra_scores, width, label='Intra-Family Agreement', color='#dd8452', edgecolor='black')

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Architecture Generations in Image Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)

    # Add labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.1f%%', fontsize=10)
    ax.bar_label(rects2, padding=3, fmt='%.1f%%', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join('paper_assets', 'Figure8_Generations_Comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
