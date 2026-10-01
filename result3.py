import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import arabic_reshaper
from bidi.algorithm import get_display

# داده‌های ۹ مدل
models_data = [
    # Model Name, Parameters (M), Top-1 Accuracy (%), Family
    ("ResNet-152", 60.2, 78.3, "CNN-based"),
    ("EfficientNet-B3", 12.0, 81.6, "CNN-based"),
    ("ConvNeXt-B", 88.6, 83.8, "CNN-based"),
    ("ViT-B/16", 86.6, 81.8, "Vision Transformer"),
    ("DeiT-B", 86.6, 81.8, "Vision Transformer"),
    ("Swin-B", 88.0, 83.5, "Vision Transformer"),
    ("MobileViT-S", 5.6, 78.4, "Hybrid"),
    ("CLIP ViT-B/32", 87.8, 68.3, "Multimodal"),
    ("SigLIP", 87.0, 83.0, "Multimodal"),
]

# تبدیل به آرایه‌های numpy
model_names = [d[0] for d in models_data]
params = np.array([d[1] for d in models_data])
accuracy = np.array([d[2] for d in models_data])
families = [d[3] for d in models_data]

# رنگ‌بندی برای هر خانواده
family_colors = {
    "CNN-based": "#2E86AB",
    "Vision Transformer": "#A23B72",
    "Hybrid": "#F18F01",
    "Multimodal": "#C73E1D"
}

family_markers = {
    "CNN-based": "o",
    "Vision Transformer": "s",
    "Hybrid": "^",
    "Multimodal": "D"
}

colors = [family_colors[f] for f in families]
markers = [family_markers[f] for f in families]

# =========================
# نسخه انگلیسی
# =========================
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

for i, (name, param, acc, fam) in enumerate(models_data):
    ax.scatter(param, acc, 
              c=family_colors[fam], 
              marker=family_markers[fam], 
              s=250, 
              alpha=0.8, 
              edgecolors='black', 
              linewidths=1.5,
              zorder=3)
    
    # Annotation
    ax.annotate(f'{name}\n{acc:.1f}%',
                xy=(param, acc),
                xytext=(15, 15),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor=family_colors[fam], 
                         alpha=0.3,
                         edgecolor=family_colors[fam],
                         linewidth=1.5),
                arrowprops=dict(arrowstyle='->', 
                               connectionstyle='arc3,rad=0.3',
                               color='gray',
                               lw=1.2),
                zorder=4)

ax.set_xscale('log')
ax.set_xlabel('Number of Parameters (Millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Deep Learning Models: Top-1 Accuracy vs Parameters\n(9-Model Benchmark on ImageNet-1K)', 
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_xlim(3, 150)
ax.set_ylim(65, 86)

# Legend
legend_elements = [plt.Line2D([0], [0], marker=family_markers[fam], color='w', 
                              markerfacecolor=family_colors[fam], markersize=10,
                              markeredgecolor='black', markeredgewidth=1.5,
                              label=fam) 
                  for fam in family_colors.keys()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
         frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('04_Accuracy_vs_Params_EN.png', dpi=300, bbox_inches='tight')
print("✓ English version saved: 04_Accuracy_vs_Params_EN.png")
plt.close()

# =========================
# نسخه فارسی
# =========================
import matplotlib.font_manager as fm

# تنظیم فونت فارسی (B Nazanin)
try:
    font_path = '/usr/share/fonts/truetype/BNazanin.ttf'  # مسیر سیستم لینوکس
    # یا برای ویندوز: 'C:\\Windows\\Fonts\\BNazanin.ttf'
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
except:
    print("Warning: B Nazanin font not found. Using default.")

def reshape_persian(text):
    """تبدیل متن فارسی به فرم قابل نمایش"""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

# ترجمه خانواده‌ها
family_names_fa = {
    "CNN-based": "مبتنی بر CNN",
    "Vision Transformer": "ترنسفورمر بینایی",
    "Hybrid": "ترکیبی",
    "Multimodal": "چندحالته"
}

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

for i, (name, param, acc, fam) in enumerate(models_data):
    ax.scatter(param, acc, 
              c=family_colors[fam], 
              marker=family_markers[fam], 
              s=250, 
              alpha=0.8, 
              edgecolors='black', 
              linewidths=1.5,
              zorder=3)
    
    label_text = f'{name}\n{acc:.1f}%'
    ax.annotate(label_text,
                xy=(param, acc),
                xytext=(15, 15),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor=family_colors[fam], 
                         alpha=0.3,
                         edgecolor=family_colors[fam],
                         linewidth=1.5),
                arrowprops=dict(arrowstyle='->', 
                               connectionstyle='arc3,rad=0.3',
                               color='gray',
                               lw=1.2),
                zorder=4)

ax.set_xscale('log')
ax.set_xlabel(reshape_persian('تعداد پارامترها (میلیون)'), fontsize=14, fontweight='bold')
ax.set_ylabel(reshape_persian('دقت Top-1 (درصد)'), fontsize=14, fontweight='bold')

title_fa = reshape_persian('مقایسه مدل‌های یادگیری عمیق: دقت Top-1 در برابر تعداد پارامترها\n(بنچمارک ۹ مدل بر روی ImageNet-1K)')
ax.set_title(title_fa, fontsize=16, fontweight='bold', pad=20)

ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_xlim(3, 150)
ax.set_ylim(65, 86)

# Legend فارسی
legend_elements = [plt.Line2D([0], [0], marker=family_markers[fam], color='w', 
                              markerfacecolor=family_colors[fam], markersize=10,
                              markeredgecolor='black', markeredgewidth=1.5,
                              label=reshape_persian(family_names_fa[fam])) 
                  for fam in family_colors.keys()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
         frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('04_Accuracy_vs_Params_FA.png', dpi=300, bbox_inches='tight')
print("✓ Persian version saved: 04_Accuracy_vs_Params_FA.png")
plt.close()

print("\n=== Benchmark Complete ===")
print("9 models analyzed:")
for name, param, acc, fam in models_data:
    print(f"  • {name:20s} | {param:6.1f}M params | {acc:5.1f}% | {fam}")
