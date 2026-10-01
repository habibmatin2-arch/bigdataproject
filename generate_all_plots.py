import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==================== تنظیمات اولیه ====================
RESULTS_CSV = os.path.join('paper_results', 'Vision_Benchmark_Results.csv')
CLASSIFICATION_CSV = 'classification_results.csv'
OUTPUT_DIR = 'paper_figures'

# فونت و استایل کلی برای سازگاری با استانداردهای ژورنال‌های IEEE/Springer
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})
# =======================================================

def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Figures will be saved to: {os.path.abspath(OUTPUT_DIR)}")

def load_data():
    df_results = None
    df_class = None
    
    if os.path.exists(RESULTS_CSV):
        df_results = pd.read_csv(RESULTS_CSV)
        # پاکسازی داده‌ها و میانگین‌گیری برای حذف ردیف‌های تکراری اجراها
        numeric_cols = df_results.select_dtypes(include=['number']).columns
        df_results = df_results.groupby(['Dataset', 'Model'])[numeric_cols].mean().reset_index()
        print("[SUCCESS] Loaded and cleaned Vision_Benchmark_Results.csv")
    else:
        print(f"[WARNING] {RESULTS_CSV} not found.")

    if os.path.exists(CLASSIFICATION_CSV):
        df_class = pd.read_csv(CLASSIFICATION_CSV)
        print("[SUCCESS] Loaded classification_results.csv")
    else:
        print(f"[WARNING] {CLASSIFICATION_CSV} not found. Confusion matrices and class-wise plots will be skipped.")
        
    return df_results, df_class

# ----------------- 1. Figure 7: Speed vs Accuracy -----------------
def generate_speed_vs_accuracy(df_results, dataset_name):
    df_filtered = df_results[df_results['Dataset'] == dataset_name]
    if df_filtered.empty:
        return
        
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = sns.scatterplot(
        data=df_filtered,
        x='Throughput (img/s)',
        y='Accuracy (%)',
        hue='Model',
        style='Model',
        s=200,
        ax=ax,
        edgecolor='black',
        alpha=0.9
    )
    
    # افزودن لیبل‌ها بدون هم‌پوشانی شدید
    for _, row in df_filtered.iterrows():
        ax.text(
            row['Throughput (img/s)'] + 1.5,
            row['Accuracy (%)'] + 0.1,
            row['Model'],
            fontsize=9,
            fontweight='semibold'
        )
        
    # خطوط میانگین برای بخش‌بندی نمودار
    ax.axhline(y=df_filtered['Accuracy (%)'].mean(), color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=df_filtered['Throughput (img/s)'].mean(), color='gray', linestyle='--', linewidth=0.8)
    
    ax.set_title(f'Figure 7: Speed vs. Accuracy on {dataset_name}', fontweight='bold', pad=15)
    ax.set_xlabel('Throughput (images/second)  →  Faster')
    ax.set_ylabel('Accuracy (%)  →  Better')
    ax.legend(title='Architectures', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'Figure7_Speed_vs_Accuracy_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Generated Figure 7 (Speed vs Accuracy) for {dataset_name}")

# ----------------- 2. Figure 9: Parameter Efficiency -----------------
def generate_parameter_efficiency(df_results, dataset_name):
    df_filtered = df_results[df_results['Dataset'] == dataset_name].copy()
    if df_filtered.empty or 'Total Parameters' not in df_filtered.columns:
        return
        
    # تبدیل تعداد پارامتر به میلیون برای خوانایی بهتر
    df_filtered['Params (M)'] = df_filtered['Total Parameters'] / 1e6
    
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=df_filtered,
        x='Params (M)',
        y='Accuracy (%)',
        hue='Model',
        style='Model',
        s=200,
        ax=ax,
        edgecolor='black'
    )
    
    for _, row in df_filtered.iterrows():
        ax.text(
            row['Params (M)'] + 0.5,
            row['Accuracy (%)'] + 0.1,
            row['Model'],
            fontsize=9,
            fontweight='semibold'
        )
        
    ax.set_title(f'Figure 9: Parameter Efficiency on {dataset_name}', fontweight='bold', pad=15)
    ax.set_xlabel('Model Size (Million Parameters)  →  Lighter')
    ax.set_ylabel('Accuracy (%)  →  Better')
    ax.legend(title='Architectures', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'Figure9_Parameter_Efficiency_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Generated Figure 9 (Parameter Efficiency) for {dataset_name}")

# ----------------- 3. Figure 10: Multi-Dataset Comparison -----------------
def generate_dataset_comparison(df_results):
    if df_results is None or df_results.empty:
        return
        
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=df_results,
        x='Dataset',
        y='Accuracy (%)',
        hue='Model',
        ax=ax,
        edgecolor='black',
        errorbar=None
    )
    
    ax.set_title('Figure 10: Accuracy Comparison Across Different Datasets', fontweight='bold', pad=15)
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(title='Architectures', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # اضافه کردن مقادیر بالای هر ستون به صورت خلاصه
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=7, rotation=90, xytext=(0, 5),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure10_Multi_Dataset_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[PLOT] Generated Figure 10 (Multi-Dataset Comparison)")

# Helper function to extract ground truth from path if not explicitly present
def get_gt_label(df_class):
    if 'ground_truth' in df_class.columns:
        return 'ground_truth'
    elif 'label' in df_class.columns:
        return 'label'
    elif 'filepath' in df_class.columns:
        df_class['ground_truth'] = df_class['filepath'].apply(
            lambda x: x.replace('\\', '/').split('/')[-2] if len(x.replace('\\', '/').split('/')) >= 2 else "Unknown"
        )
        return 'ground_truth'
    return None

# ----------------- 4. Figure 6: Confusion Matrix -----------------
def generate_confusion_matrices(df_class, target_models):
    y_true_col = get_gt_label(df_class)
    if not y_true_col:
        print("[WARNING] Could not parse labels for Confusion Matrix.")
        return
        
    for model in target_models:
        if model not in df_class.columns:
            continue
            
        df_clean = df_class[[y_true_col, model]].dropna()
        df_clean = df_clean[df_clean[model] != 'Error']
        
        y_true = df_clean[y_true_col]
        y_pred = df_clean[model]
        
        labels = sorted(list(set(y_true).union(set(y_pred))))
        
        # اگر تعداد کلاس‌ها خیلی زیاد نباشد (مثلا زیر 20 کلاس) نمایش ماتریس عالی خواهد شد
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True if len(labels) <= 15 else False,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=True
        )
        
        ax.set_title(f'Figure 6: Confusion Matrix for {model}', fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Figure6_Confusion_Matrix_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Generated Figure 6 (Confusion Matrix) for {model}")

# ----------------- 5. Figure 8: Class-wise Precision vs Recall -----------------
def generate_classwise_performance(df_class, best_model):
    y_true_col = get_gt_label(df_class)
    if not y_true_col or best_model not in df_class.columns:
        return
        
    df_clean = df_class[[y_true_col, best_model]].dropna()
    df_clean = df_clean[df_clean[best_model] != 'Error']
    
    # تولید گزارش متنی کلاس‌ها برای استخراج مقادیر precision و recall
    report = classification_report(df_clean[y_true_col], df_clean[best_model], output_dict=True)
    
    # تبدیل به دیتافریم و حذف مقادیر میانگین کل (accuracy, macro avg, weighted avg)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_report.index))
    width = 0.35
    
    ax.bar(x - width/2, df_report['precision'], width, label='Precision', color='#1f77b4', edgecolor='black')
    ax.bar(x + width/2, df_report['recall'], width, label='Recall', color='#ff7f0e', edgecolor='black')
    
    ax.set_title(f'Figure 8: Class-wise Precision and Recall for {best_model}', fontweight='bold', pad=15)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_report.index, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'Figure8_Classwise_Performance_{best_model}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Generated Figure 8 (Class-wise Metrics) for {best_model}")

# ==================== اجرای اصلی ====================
def main():
    create_output_dir()
    df_results, df_class = load_data()
    
    # ۱. پردازش داده‌های توصیفی مدل‌ها (Vision_Benchmark_Results.csv)
    if df_results is not None:
        datasets = df_results['Dataset'].unique()
        print(f"[INFO] Detected datasets: {datasets}")
        
        for ds in datasets:
            generate_speed_vs_accuracy(df_results, ds)
            generate_parameter_efficiency(df_results, ds)
            
        generate_dataset_comparison(df_results)
        
    # ۲. پردازش تحلیل‌های پیش‌بینی (classification_results.csv)
    if df_class is not None:
        # انتخاب چند مدل برتر بر اساس بیشترین تکرار پیش‌بینی درست برای ماتریس درهم‌ریختگی
        non_meta_cols = [col for col in df_class.columns if col not in ['filepath', 'ground_truth', 'label']]
        print(f"[INFO] Models found in predictions: {non_meta_cols}")
        
        # رسم ماتریس درهم‌ریختگی برای ۳ مدل مطرح (مثلاً ConvNeXt-Tiny, ViT-B-16 و ResNet50)
        target_models_for_cm = [m for m in ['ConvNeXt-Tiny', 'ViT-B-16', 'ResNet50'] if m in non_meta_cols]
        if not target_models_for_cm and non_meta_cols:
            target_models_for_cm = non_meta_cols[:2]
            
        generate_confusion_matrices(df_class, target_models_for_cm)
        
        # رسم کارایی به تفکیک کلاس برای بهترین مدل (مثلاً ConvNeXt-Tiny یا برترین مدل موجود)
        best_model = 'ConvNeXt-Tiny' if 'ConvNeXt-Tiny' in non_meta_cols else (non_meta_cols[0] if non_meta_cols else None)
        if best_model:
            generate_classwise_performance(df_class, best_model)

    print("\n[FINISH] All figures have been generated and saved inside 'paper_figures' directory.")

if __name__ == "__main__":
    main()
