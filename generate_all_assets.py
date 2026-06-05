import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# پشتیبانی از زبان فارسی
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False
    print("Warning: 'arabic-reshaper' or 'python-bidi' not found. Persian text may not render correctly.")

# ==========================================
# تنظیمات مسیرها
# ==========================================
CSV_PATH = r"E:\vscode\big data\classification_results.csv"
PLOTS_BASE_DIR = r"E:\vscode\big data\plots"

# ایجاد پوشه‌های زبان‌ها
DIR_EN = os.path.join(PLOTS_BASE_DIR, 'en')
DIR_FA = os.path.join(PLOTS_BASE_DIR, 'fa')
os.makedirs(DIR_EN, exist_ok=True)
os.makedirs(DIR_FA, exist_ok=True)

# تنظیم استایل پیش‌فرض
sns.set_theme(style="whitegrid")

# ==========================================
# توابع کمکی
# ==========================================
def prep_text(text, lang='en'):
    """آماده‌سازی متن برای نمایش صحیح فارسی"""
    if lang == 'fa' and HAS_BIDI:
        return get_display(arabic_reshaper.reshape(text))
    return text

def plot_and_save(fig, filename_base, titles, xlabels, ylabels, lang_dirs):
    """ذخیره یک نمودار در دو زبان مختلف"""
    for lang in ['en', 'fa']:
        # تنظیم متون بر اساس زبان
        plt.title(prep_text(titles[lang], lang), fontsize=14, pad=15)
        if xlabels[lang]:
            plt.xlabel(prep_text(xlabels[lang], lang), fontsize=12)
        if ylabels[lang]:
            plt.ylabel(prep_text(ylabels[lang], lang), fontsize=12)
            
        plt.tight_layout()
        filepath = os.path.join(lang_dirs[lang], f"{filename_base}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# تابع اصلی تولید نمودارها
# ==========================================
def generate_all_plots(csv_path):
    print("Loading data...")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # تفکیک ستون‌ها
    pred_cols = [c for c in df.columns if c != 'filepath' and not c.endswith('_time')]
    time_cols = [c for c in df.columns if c.endswith('_time')]
    model_names = pred_cols
    
    # دیتافریم تمیز شده (حذف خطاها برای تحلیل‌های آماری پیش‌بینی)
    clean_df = df.copy()
    error_keywords = ["Error", "Error_Model_Load", "Error_Processing"]
    for col in pred_cols:
        clean_df[col] = clean_df[col].apply(lambda x: np.nan if pd.isna(x) or any(ek in str(x) for ek in error_keywords) else x)

    lang_dirs = {'en': DIR_EN, 'fa': DIR_FA}
    print(f"Analyzing {len(model_names)} models over {len(df)} images...")

    # ---------------------------------------------------------
    # 1. Prediction Diversity Heatmap
    # ---------------------------------------------------------
    print("1/10 Generating Prediction Diversity...")
    fig = plt.figure(figsize=(12, 8))
    pred_counts = clean_df[model_names].apply(pd.Series.value_counts).fillna(0)
    sns.heatmap(pred_counts, cmap="YlGnBu", annot=False, linewidths=.5)
    plot_and_save(fig, "01_Prediction_Diversity", 
                  titles={'en': 'Prediction Diversity Across Models', 'fa': 'تنوع پیش‌بینی‌ها در مدل‌های مختلف'},
                  xlabels={'en': 'Models', 'fa': 'مدل‌ها'},
                  ylabels={'en': 'Categories', 'fa': 'دسته‌بندی‌ها'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 2. Overall Class Distribution
    # ---------------------------------------------------------
    print("2/10 Generating Overall Distribution...")
    fig = plt.figure(figsize=(12, 6))
    all_preds = clean_df[model_names].melt(value_name='prediction').dropna()['prediction']
    top_classes = all_preds.value_counts().head(15)
    sns.barplot(x=top_classes.index, y=top_classes.values, palette="rocket")
    plt.xticks(rotation=45, ha='right')
    plot_and_save(fig, "02_Overall_Distribution", 
                  titles={'en': 'Top 15 Predicted Classes (Overall)', 'fa': '۱۵ کلاس با بیشترین پیش‌بینی (کلی)'},
                  xlabels={'en': 'Category', 'fa': 'دسته‌بندی'},
                  ylabels={'en': 'Total Count', 'fa': 'تعداد کل'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 3. Model Agreement (Overlap) Matrix
    # ---------------------------------------------------------
    print("3/10 Generating Agreement Matrix...")
    if len(model_names) > 1:
        overlap_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
        for m1 in model_names:
            for m2 in model_names:
                if m1 == m2:
                    overlap_matrix.loc[m1, m2] = 100.0
                else:
                    valid = clean_df[m1].notna() & clean_df[m2].notna()
                    agreement = (clean_df.loc[valid, m1] == clean_df.loc[valid, m2]).mean() * 100 if valid.sum() > 0 else 0
                    overlap_matrix.loc[m1, m2] = agreement
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': '%'})
        plot_and_save(fig, "03_Agreement_Matrix", 
                      titles={'en': 'Model Agreement Matrix (%)', 'fa': 'ماتریس توافق مدل‌ها (درصد)'},
                      xlabels={'en': '', 'fa': ''}, ylabels={'en': '', 'fa': ''}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 4. Consensus Level per Category
    # ---------------------------------------------------------
    print("4/10 Generating Category Consensus...")
    if len(model_names) > 1:
        majority_vote = clean_df[model_names].mode(axis=1)
        if not majority_vote.empty:
            consensus_label = majority_vote.iloc[:, 0]
            agreement_ratios = clean_df[model_names].apply(lambda row: (row == consensus_label[row.name]).mean() * 100, axis=1)
            cons_df = pd.DataFrame({'Category': consensus_label, 'Agreement': agreement_ratios}).dropna()
            avg_cons = cons_df.groupby('Category')['Agreement'].mean().sort_values(ascending=False).head(15)
            
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(x=avg_cons.index, y=avg_cons.values, palette="mako")
            plt.xticks(rotation=45, ha='right')
            plot_and_save(fig, "04_Category_Consensus", 
                          titles={'en': 'Consensus Percentage per Category', 'fa': 'درصد اجماع مدل‌ها در هر دسته‌بندی'},
                          xlabels={'en': 'Category', 'fa': 'دسته‌بندی'},
                          ylabels={'en': 'Average Consensus (%)', 'fa': 'میانگین اجماع (%)'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 5. Average Latency Comparison
    # ---------------------------------------------------------
    print("5/10 Generating Latency Comparison...")
    if time_cols:
        avg_times = clean_df[time_cols].mean().sort_values()
        clean_time_names = [x.replace('_time', '') for x in avg_times.index]
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=clean_time_names, y=avg_times.values, palette="magma")
        plt.xticks(rotation=45, ha='right')
        plot_and_save(fig, "05_Latency_Comparison", 
                      titles={'en': 'Average Inference Latency', 'fa': 'متوسط زمان پردازش هر تصویر'},
                      xlabels={'en': 'Model', 'fa': 'مدل'},
                      ylabels={'en': 'Seconds per Image', 'fa': 'ثانیه بر تصویر'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 6. Throughput Comparison (Images / Sec)
    # ---------------------------------------------------------
    print("6/10 Generating Throughput...")
    if time_cols:
        # جلوگیری از تقسیم بر صفر
        throughput = 1.0 / (clean_df[time_cols].mean() + 1e-6)
        throughput = throughput.sort_values(ascending=False)
        clean_tp_names = [x.replace('_time', '') for x in throughput.index]
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=clean_tp_names, y=throughput.values, palette="crest")
        plt.xticks(rotation=45, ha='right')
        plot_and_save(fig, "06_Throughput_Comparison", 
                      titles={'en': 'Throughput Comparison', 'fa': 'مقایسه توان عملیاتی مدل‌ها'},
                      xlabels={'en': 'Model', 'fa': 'مدل'},
                      ylabels={'en': 'Images / Second', 'fa': 'تصویر بر ثانیه'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 7. Latency Boxplot (Outlier Detection)
    # ---------------------------------------------------------
    print("7/10 Generating Latency Boxplot...")
    if time_cols:
        time_data = clean_df[time_cols].copy()
        time_data.columns = [x.replace('_time', '') for x in time_data.columns]
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(data=time_data, orient="h", palette="Set2")
        plot_and_save(fig, "07_Latency_Boxplot", 
                      titles={'en': 'Latency Distribution & Outliers', 'fa': 'توزیع زمان پردازش و مقادیر پرت'},
                      xlabels={'en': 'Seconds', 'fa': 'ثانیه'},
                      ylabels={'en': 'Model', 'fa': 'مدل'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 8. Error Rates per Model
    # ---------------------------------------------------------
    print("8/10 Generating Error Rates...")
    error_rates = {}
    for col in model_names:
        total = len(df[col])
        errors = df[col].astype(str).apply(lambda x: any(ek in x for ek in error_keywords) or x == 'nan').sum()
        error_rates[col] = (errors / total) * 100 if total > 0 else 0
    
    err_series = pd.Series(error_rates).sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=err_series.index, y=err_series.values, palette="Reds_r")
    plt.xticks(rotation=45, ha='right')
    plot_and_save(fig, "08_Error_Rates", 
                  titles={'en': 'Error Rates per Model', 'fa': 'نرخ خطای مدل‌ها'},
                  xlabels={'en': 'Model', 'fa': 'مدل'},
                  ylabels={'en': 'Error Percentage (%)', 'fa': 'درصد خطا (%)'}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 9. Overall Agreement Distribution
    # ---------------------------------------------------------
    print("9/10 Generating Agreement Distribution...")
    if len(model_names) > 1:
        # درصد توافق روی هر عکس
        def get_max_agreement(row):
            vals = row.dropna().values
            if len(vals) == 0: return 0
            u, c = np.unique(vals, return_counts=True)
            return (c.max() / len(model_names)) * 100

        agreements = clean_df[model_names].apply(get_max_agreement, axis=1)
        bins = [0, 50, 99.9, 100]
        labels = ['Low (<50%)', 'Majority (50-99%)', 'Full (100%)']
        binned = pd.cut(agreements, bins=bins, labels=labels, include_lowest=True).value_counts()
        
        fig = plt.figure(figsize=(8, 8))
        plt.pie(binned.values, labels=binned.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plot_and_save(fig, "09_Agreement_Distribution", 
                      titles={'en': 'Images Agreement Levels', 'fa': 'سطح توافق مدل‌ها روی تصاویر'},
                      xlabels={'en': '', 'fa': ''}, ylabels={'en': '', 'fa': ''}, lang_dirs=lang_dirs)

    # ---------------------------------------------------------
    # 10. Valid vs Error Counts (Stacked)
    # ---------------------------------------------------------
    print("10/10 Generating Valid vs Error Stacked Bar...")
    valid_counts = len(df) - err_series * len(df) / 100
    error_counts = err_series * len(df) / 100
    
    stack_df = pd.DataFrame({
        'Valid': valid_counts,
        'Error': error_counts
    })
    
    fig = plt.figure(figsize=(12, 6))
    stack_df.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], ax=plt.gca())
    plt.xticks(rotation=45, ha='right')
    plot_and_save(fig, "10_Valid_vs_Error", 
                  titles={'en': 'Valid Predictions vs Errors', 'fa': 'تعداد پیش‌بینی‌های موفق در برابر خطا'},
                  xlabels={'en': 'Model', 'fa': 'مدل'},
                  ylabels={'en': 'Number of Images', 'fa': 'تعداد تصاویر'}, lang_dirs=lang_dirs)

    print("\n✅ All 10 plots generated successfully!")
    print(f"English plots saved to: {DIR_EN}")
    print(f"Persian plots saved to: {DIR_FA}")

if __name__ == "__main__":
    generate_all_plots(CSV_PATH)
