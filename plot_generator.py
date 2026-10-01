import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------- تنظیمات -------------------
# مسیر فایل CSV نتایج
# اسکریپت فرض می‌کند که پوشه paper_results در کنار آن قرار دارد
RESULTS_CSV_PATH = os.path.join('paper_results', 'Vision_Benchmark_Results.csv')

# نام پوشه‌ای که نمودارها در آن ذخیره می‌شوند
OUTPUT_DIR = 'paper_figures'

# انتخاب دیتاست برای تحلیل (دیتاست‌های چالش‌برانگیز بهترین انتخاب هستند)
# می‌توانید این مقدار را به 'Food-101' نیز تغییر دهید
TARGET_DATASET = 'ImageNet-Mini'
# ---------------------------------------------

def plot_speed_vs_accuracy(df: pd.DataFrame, dataset_name: str):
    print(f"\n[INFO] Generating Speed vs. Accuracy plot for '{dataset_name}' dataset...")

    df_filtered = df[df['Dataset'] == dataset_name].copy()

    if df_filtered.empty:
        print(f"[ERROR] No data found for dataset '{dataset_name}'.")
        return

    # میانگین‌گیری روی داده‌های تکراری هر مدل
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns
    df_filtered = df_filtered.groupby(['Dataset', 'Model'])[numeric_cols].mean().reset_index()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter_plot = sns.scatterplot(
        data=df_filtered,
        x='Throughput (img/s)',
        y='Accuracy (%)',
        hue='Model',
        style='Model',
        s=220,
        ax=ax,
        edgecolor='black',
        alpha=0.9
    )

    for i, row in df_filtered.iterrows():
        ax.text(
            row['Throughput (img/s)'] + 1.2,
            row['Accuracy (%)'] + 0.2,
            row['Model'],
            fontsize=9.5,
            fontweight='semibold'
        )

    ax.set_title(f'Model Performance on {dataset_name}: Speed vs. Accuracy (Averaged)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Speed / Throughput (images/sec)  →  Faster', fontsize=12)
    ax.set_ylabel('Accuracy (%)  →  Better', fontsize=12)
    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left')

    ax.axhline(y=df_filtered['Accuracy (%)'].mean(), color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=df_filtered['Throughput (img/s)'].mean(), color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(OUTPUT_DIR, f'Figure7_Speed_vs_Accuracy_{dataset_name}_Fixed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Fixed plot saved to: {output_path}")
    plt.show()



def main():
    # ساخت پوشه خروجی اگر وجود نداشته باشد
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # خواندن فایل CSV
    try:
        df_results = pd.read_csv(RESULTS_CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] The results file was not found at '{RESULTS_CSV_PATH}'")
        print("Please ensure this script is in the same directory as your project folder.")
        return

    # فراخوانی تابع برای رسم نمودار
    plot_speed_vs_accuracy(df_results, TARGET_DATASET)


if __name__ == "__main__":
    main()
