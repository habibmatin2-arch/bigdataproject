import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
import os

# تنظیمات پایه و مسیر فایل‌ها
INPUT_CSV = "classification_results.csv"  # مسیر فایل خود را اینجا قرار دهید
OUTPUT_DIR_EN = os.path.join("plots", "EN")
OUTPUT_DIR_FA = os.path.join("plots", "FA")

os.makedirs(OUTPUT_DIR_EN, exist_ok=True)
os.makedirs(OUTPUT_DIR_FA, exist_ok=True)

# تنظیم استایل پیش‌فرض
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def fa(text):
    """رفع مشکل نمایش حروف جدا و برعکس در زبان فارسی"""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

df = pd.read_csv(INPUT_CSV)
models = ['CLIP', 'ConvNeXt', 'DeiT', 'EfficientNet', 'MobileViT', 'ResNet', 'SigLIP', 'Swin', 'ViT']

df_melted = df.melt(id_vars=['filepath'], 
                    value_vars=models,
                    var_name='Model', 
                    value_name='Prediction')

mode_predictions = df[models].mode(axis=1)[0]
df['Consensus_Prediction'] = mode_predictions

def plot_class_distribution(lang='EN'):
    plt.figure(figsize=(14, 7))
    top_15 = df_melted['Prediction'].value_counts().nlargest(15)
    
    sns.barplot(x=top_15.index, y=top_15.values, hue=top_15.index, palette="mako", legend=False)
    
    if lang == 'EN':
        plt.title("Top 15 Predicted Classes Across All Models", fontsize=14, weight='bold')
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_EN, "01_Class_Distribution.png")
    else:
        plt.title(fa('توزیع ۱۵ کلاس پرتکرار در تمام مدل‌ها'), fontsize=14, weight='bold')
        plt.xlabel(fa("کلاس پیش‌بینی‌شده"), fontsize=12)
        plt.ylabel(fa("فراوانی"), fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_FA, "01_Class_Distribution.png")
        
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_agreement_heatmap(lang='EN'):
    plt.figure(figsize=(10, 8))
    agreement_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    
    for m1 in models:
        for m2 in models:
            agreement_matrix.loc[m1, m2] = (df[m1] == df[m2]).mean() * 100
            
    cbar_label = 'Agreement (%)' if lang == 'EN' else fa('میزان توافق (درصد)')
    sns.heatmap(agreement_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': cbar_label})
    
    if lang == 'EN':
        plt.title("Inter-Model Agreement (%)", fontsize=14, weight='bold')
        save_path = os.path.join(OUTPUT_DIR_EN, "02_Agreement_Heatmap.png")
    else:
        plt.title(fa('نقشه حرارتی میزان توافق مدل‌ها (درصد)'), fontsize=14, weight='bold')
        save_path = os.path.join(OUTPUT_DIR_FA, "02_Agreement_Heatmap.png")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_consensus_distribution(lang='EN'):
    plt.figure(figsize=(8, 8))
    
    def count_agreements(row):
        return (row[models] == row['Consensus_Prediction']).sum()
    
    df['Consensus_Count'] = df.apply(count_agreements, axis=1)
    consensus_counts = df['Consensus_Count'].value_counts().sort_index()
    
    colors = sns.color_palette("Set3", len(consensus_counts))
    plt.pie(consensus_counts.values, labels=[str(x) for x in consensus_counts.index], 
            autopct='%1.1f%%', startangle=140, colors=colors)
            
    if lang == 'EN':
        plt.title("Consensus Level (Number of Agreeing Models)", fontsize=14, weight='bold')
        save_path = os.path.join(OUTPUT_DIR_EN, "03_Consensus_Level.png")
    else:
        plt.title(fa('سطح خرد جمعی (تعداد مدل‌های هم‌نظر)'), fontsize=14, weight='bold')
        save_path = os.path.join(OUTPUT_DIR_FA, "03_Consensus_Level.png")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_performance_vs_scale(lang='EN'):
    plt.figure(figsize=(10, 6))
    
    paper_data = {
        'Model': ['MobileViT-S', 'ResNet-152', 'EfficientNet-B3', 'ViT-B/16'],
        'Params (M)': [5.6, 60.2, 12.0, 86.0],
        'Top-1 Acc (%)': [78.4, 80.6, 81.6, 77.9]
    }
    df_papers = pd.DataFrame(paper_data)
    
    sns.scatterplot(data=df_papers, x='Params (M)', y='Top-1 Acc (%)', s=200, color='crimson')
    
    for i, row in df_papers.iterrows():
        plt.annotate(row['Model'], (row['Params (M)'], row['Top-1 Acc (%)']), 
                     xytext=(5,5), textcoords='offset points', fontsize=11)
        
    if lang == 'EN':
        plt.title("Accuracy vs. Parameters (From Literature)", fontsize=14, weight='bold')
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("ImageNet Top-1 Accuracy (%)", fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_EN, "04_Accuracy_vs_Params.png")
    else:
        plt.title(fa('دقت بر حسب تعداد پارامترها (برگرفته از مقالات)'), fontsize=14, weight='bold')
        plt.xlabel(fa("تعداد پارامترها (میلیون)"), fontsize=12)
        plt.ylabel(fa("دقت ImageNet Top-1 (درصد)"), fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_FA, "04_Accuracy_vs_Params.png")
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_prediction_diversity(lang='EN'):
    plt.figure(figsize=(10, 6))
    
    unique_counts = {m: df[m].nunique() for m in models}
    x_vals = list(unique_counts.keys())
    y_vals = list(unique_counts.values())
    
    sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="crest", legend=False)
    
    if lang == 'EN':
        plt.title("Unique Classes Predicted by Model", fontsize=14, weight='bold')
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Number of Unique Classes", fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_EN, "05_Prediction_Diversity.png")
    else:
        plt.title(fa('تنوع کلاس‌های پیش‌بینی‌شده توسط هر مدل'), fontsize=14, weight='bold')
        plt.xlabel(fa("مدل"), fontsize=12)
        plt.ylabel(fa("تعداد کلاس‌های منحصربه‌فرد"), fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_FA, "05_Prediction_Diversity.png")
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_architecture_family(lang='EN'):
    plt.figure(figsize=(8, 6))
    
    cnn_models = ['ConvNeXt', 'EfficientNet', 'ResNet']
    vit_models = ['DeiT', 'Swin', 'ViT', 'SigLIP', 'CLIP']
    hybrid_models = ['MobileViT']
    
    accuracies = []
    for group, name, name_fa in zip([cnn_models, vit_models, hybrid_models], 
                                    ['CNNs', 'Transformers', 'Hybrid'],
                                    [fa('شبکه‌های عصبی کانولوشنی'), fa('ترانسفورمرها'), fa('هیبرید (ترکیبی)')]):
        group_acc = df[group].apply(lambda col: (col == df['Consensus_Prediction']).mean()).mean() * 100
        arch_name = name if lang == 'EN' else name_fa
        accuracies.append({'Architecture': arch_name, 'Agreement with Consensus (%)': group_acc})
        
    df_arch = pd.DataFrame(accuracies)
    sns.barplot(data=df_arch, x='Architecture', y='Agreement with Consensus (%)', hue='Architecture', palette="flare", legend=False)
    
    if lang == 'EN':
        plt.title("Architecture Family vs. Consensus", fontsize=14, weight='bold')
        plt.xlabel("Architecture Family", fontsize=12)
        plt.ylabel("Agreement with Consensus (%)", fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_EN, "06_Architecture_Family.png")
    else:
        plt.title(fa('مقایسه خانواده معماری‌ها با خرد جمعی'), fontsize=14, weight='bold')
        plt.xlabel(fa("خانواده معماری"), fontsize=12)
        plt.ylabel(fa("توافق با خرد جمعی (درصد)"), fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_FA, "06_Architecture_Family.png")
        
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_divergence(lang='EN'):
    plt.figure(figsize=(10, 6))
    
    divergence = {}
    for m in models:
        is_unique_vote = df.apply(lambda row: sum(row[models] == row[m]) == 1, axis=1)
        divergence[m] = is_unique_vote.mean() * 100
        
    x_vals = list(divergence.keys())
    y_vals = list(divergence.values())
    
    sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="magma", legend=False)
    
    if lang == 'EN':
        plt.title("Model Novelty / Divergence Rate (%)", fontsize=14, weight='bold')
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Divergence Rate (%)", fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_EN, "07_Divergence_Rate.png")
    else:
        plt.title(fa('نرخ پیش‌بینی‌های منحصربه‌فرد (تفاوت با سایرین)'), fontsize=14, weight='bold')
        plt.xlabel(fa("مدل"), fontsize=12)
        plt.ylabel(fa("نرخ واگرایی (درصد)"), fontsize=12)
        save_path = os.path.join(OUTPUT_DIR_FA, "07_Divergence_Rate.png")
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("در حال تولید نمودارها در دو زبان (FA و EN)...")
    
    for language in ['EN', 'FA']:
        plot_class_distribution(lang=language)
        plot_agreement_heatmap(lang=language)
        plot_consensus_distribution(lang=language)
        plot_performance_vs_scale(lang=language)
        plot_prediction_diversity(lang=language)
        plot_architecture_family(lang=language)
        plot_divergence(lang=language)
        
    print(f"تمامی ۱۴ نمودار با موفقیت در پوشه‌های '{OUTPUT_DIR_EN}' و '{OUTPUT_DIR_FA}' ذخیره شدند.")
