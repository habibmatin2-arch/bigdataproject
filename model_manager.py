import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import arabic_reshaper
from bidi.algorithm import get_display
import warnings

# غیرفعال کردن هشدارهای مزاحم
warnings.filterwarnings('ignore')

# تنظیم استایل کلی نمودارها
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------
# ۱. توابع کمکی برای زبان و مسیرها
# ---------------------------------------------------------
def fa(text):
    """تابع تبدیل متن برای نمایش صحیح فارسی"""
    return get_display(arabic_reshaper.reshape(text))

# ایجاد پوشه‌های ذخیره‌سازی
os.makedirs("plots/EN", exist_ok=True)
os.makedirs("plots/FA", exist_ok=True)

# ---------------------------------------------------------
# ۲. آماده‌سازی و پاکسازی داده‌ها
# ---------------------------------------------------------
# خواندن فایل CSV
try:
    df = pd.read_csv("classification_results.csv")
except FileNotFoundError:
    print("Error: classification_results.csv not found!")
    exit()

# فرض بر این است که ستون اول نام تصویر و بقیه ستون‌ها نام مدل‌ها هستند
id_col = df.columns[0]
model_cols = df.columns[1:]

# پاکسازی: تبدیل Error_Model_Load به مقدار خالی (NaN)
df[model_cols] = df[model_cols].replace('Error_Model_Load', np.nan)

# حذف کامل مدل‌هایی که کلا خطا داشته‌اند (مثل SigLIP در داده‌های شما)
df = df.dropna(axis=1, how='all')
valid_models = [col for col in df.columns if col != id_col]

# تبدیل داده‌ها از حالت عریض (Wide) به طولانی (Long) برای رسم راحت‌تر
df_long = df.melt(id_vars=[id_col], value_vars=valid_models, 
                  var_name='Model', value_name='Prediction')
df_long = df_long.dropna(subset=['Prediction'])

# ---------------------------------------------------------
# ۳. توابع رسم نمودار
# ---------------------------------------------------------
def plot_01_class_distribution(lang='EN'):
    plt.figure(figsize=(10, 6))
    top_n = 10
    top_classes = df_long['Prediction'].value_counts().nlargest(top_n)
    
    sns.barplot(x=top_classes.values, y=top_classes.index, palette="viridis", hue=top_classes.index, legend=False)
    
    title = f"Top {top_n} Most Predicted Classes" if lang == 'EN' else fa(f"۱۰ کلاس پر تکرار در پیش‌بینی‌ها")
    xlabel = "Frequency" if lang == 'EN' else fa("فراوانی")
    ylabel = "Class Name" if lang == 'EN' else fa("نام کلاس")
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/{lang}/01_Class_Distribution.png", dpi=300)
    plt.close()

def plot_02_agreement_heatmap(lang='EN'):
    plt.figure(figsize=(8, 6))
    
    # محاسبه ماتریس توافق
    agreement_matrix = pd.DataFrame(index=valid_models, columns=valid_models, dtype=float)
    for m1 in valid_models:
        for m2 in valid_models:
            # مقایسه ردیف به ردیف
            mask = df[m1].notna() & df[m2].notna()
            if mask.sum() > 0:
                agreement = (df.loc[mask, m1] == df.loc[mask, m2]).mean()
                agreement_matrix.loc[m1, m2] = agreement
            else:
                agreement_matrix.loc[m1, m2] = 0.0

    sns.heatmap(agreement_matrix, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
    
    title = "Model Agreement Heatmap" if lang == 'EN' else fa("نقشه حرارتی میزان توافق مدل‌ها")
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"plots/{lang}/02_Agreement_Heatmap.png", dpi=300)
    plt.close()

def plot_03_consensus_level(lang='EN'):
    plt.figure(figsize=(8, 8))
    
    # محاسبه سطح توافق برای هر تصویر
    consensus_counts = {'Unanimous': 0, 'Majority': 0, 'Minority': 0}
    
    for _, row in df[valid_models].iterrows():
        valid_preds = row.dropna().tolist()
        if not valid_preds:
            continue
            
        pred_counts = pd.Series(valid_preds).value_counts()
        max_votes = pred_counts.max()
        total_votes = len(valid_preds)
        
        if max_votes == total_votes and total_votes > 1:
            consensus_counts['Unanimous'] += 1
        elif max_votes > total_votes / 2:
            consensus_counts['Majority'] += 1
        else:
            consensus_counts['Minority'] += 1
            
    labels_en = list(consensus_counts.keys())
    labels_fa = [fa("توافق کامل"), fa("اکثریت"), fa("اقلیت")]
    sizes = list(consensus_counts.values())
    
    labels = labels_en if lang == 'EN' else labels_fa
    colors = sns.color_palette("pastel")[0:3]
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    title = "Prediction Consensus Levels" if lang == 'EN' else fa("سطوح توافق در پیش‌بینی‌ها")
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"plots/{lang}/03_Consensus_Level.png", dpi=300)
    plt.close()

def plot_04_prediction_diversity(lang='EN'):
    plt.figure(figsize=(10, 6))
    
    # تعداد کلاس‌های یکتا که هر مدل پیش‌بینی کرده است
    diversity = df_long.groupby('Model')['Prediction'].nunique().sort_values(ascending=False)
    
    sns.barplot(x=diversity.index, y=diversity.values, palette="magma", hue=diversity.index, legend=False)
    
    title = "Prediction Diversity by Model" if lang == 'EN' else fa("تنوع پیش‌بینی به تفکیک مدل")
    xlabel = "Model" if lang == 'EN' else fa("مدل")
    ylabel = "Unique Classes Predicted" if lang == 'EN' else fa("تعداد کلاس‌های منحصربه‌فرد")
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{lang}/04_Prediction_Diversity.png", dpi=300)
    plt.close()

def plot_05_divergence_rate(lang='EN'):
    plt.figure(figsize=(10, 6))
    
    divergence_rates = {}
    for model in valid_models:
        divergent_count = 0
        total_valid = 0
        for _, row in df[valid_models].iterrows():
            valid_preds = row.dropna()
            if model in valid_preds and len(valid_preds) > 2:
                # یافتن پیش‌بینی اکثریت
                majority_pred = valid_preds.value_counts().index[0]
                if valid_preds[model] != majority_pred:
                    divergent_count += 1
                total_valid += 1
        
        rate = (divergent_count / total_valid) * 100 if total_valid > 0 else 0
        divergence_rates[model] = rate
        
    div_series = pd.Series(divergence_rates).sort_values()
    
    sns.barplot(x=div_series.values, y=div_series.index, palette="rocket", hue=div_series.index, legend=False)
    
    title = "Model Divergence Rate from Majority" if lang == 'EN' else fa("نرخ انحراف هر مدل از نظر اکثریت")
    xlabel = "Divergence Rate (%)" if lang == 'EN' else fa("درصد انحراف (%)")
    ylabel = "Model" if lang == 'EN' else fa("مدل")
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/{lang}/05_Divergence_Rate.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# ۴. اجرای توابع
# ---------------------------------------------------------
print("Starting plot generation...")
for language in ['EN', 'FA']:
    print(f"Generating {language} plots...")
    plot_01_class_distribution(language)
    plot_02_agreement_heatmap(language)
    plot_03_consensus_level(language)
    plot_04_prediction_diversity(language)
    plot_05_divergence_rate(language)
    
print("All plots generated successfully! Check the 'plots/EN' and 'plots/FA' folders.")
