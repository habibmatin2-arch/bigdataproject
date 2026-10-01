import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# تنظیم فونت و استایل علمی
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

csv_file = "Vision_Benchmark_Full_9Models.csv"
df = pd.read_csv(csv_file)

# مرتب‌سازی بر اساس تاخیر
df = df.sort_values(by="Latency Mean (ms)", ascending=True)

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

x = np.arange(len(df))
width = 0.42

# محور اول: تاخیر با خطای انحراف معیار
bars1 = ax1.bar(x - width/2, df["Latency Mean (ms)"], width, 
                yerr=df["Latency Std (ms)"], capsize=4, 
                label='Latency (ms) [Lower is Better]', color='#2b5c8f', edgecolor='black', alpha=0.85)

ax1.set_ylabel('Inference Latency (ms)', color='#2b5c8f', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#2b5c8f')
ax1.set_xticks(x)
ax1.set_xticklabels(df["Model"], rotation=30, ha='right', fontweight='bold')

# محور دوم: توان عملیاتی (Throughput)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, df["Throughput (img/s)"], width, 
                label='Throughput (img/s) [Higher is Better]', color='#e67e22', edgecolor='black', alpha=0.85)

ax2.set_ylabel('Throughput (Images / sec)', color='#e67e22', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#e67e22')

# عنوان و شبکه پس‌زمینه
plt.title('CPU Inference Efficiency Benchmark Across 9 Vision Architectures (ICCKE 2026)', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# راهنمای نمودار
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True)

out_name = "Benchmark_Latency_Throughput_9Models.png"
plt.savefig(out_name, bbox_inches='tight')
print(f"[OK] نمودار با کیفیت چاپ ذخیره شد: {out_name}")
plt.show()
