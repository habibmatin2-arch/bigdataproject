import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── colour palette ──────────────────────────────────────────────
C = {
    'data':      '#4A90D9',
    'preproc':   '#7B68EE',
    'cnn':       '#E8A838',
    'trans':     '#50C878',
    'hybrid':    '#FF7F7F',
    'consensus': '#20B2AA',
    'output':    '#708090',
    'arrow':     '#444444',
    'text_light':'#FFFFFF',
    'text_dark': '#1A1A1A',
    'bg_group':  '#F0F4FF',
}

def box(ax, x, y, w, h, label, sub='', color='#4A90D9', fontsize=10, subsize=8, radius=0.3):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    y_text = y + (0.18 if sub else 0)
    ax.text(x, y_text, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4)
    if sub:
        ax.text(x, y - 0.28, sub, ha='center', va='center',
                fontsize=subsize, color='#E8E8E8', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#444444'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=16),
                zorder=2)

def group_rect(ax, x, y, w, h, label, color='#E8EFF8'):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1,rounding_size=0.4",
                          facecolor=color, edgecolor='#AABBCC',
                          linewidth=1.2, zorder=1, linestyle='--')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.05, label, ha='center', va='bottom',
            fontsize=8.5, color='#556677', fontweight='bold', zorder=2)

# ════════════════════════════════════════════════════════════════
# ROW 1 — Data Sources
# ════════════════════════════════════════════════════════════════
ax.text(9, 13.5, 'Multi-Model Image Classification Pipeline',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color=C['text_dark'])

box(ax,  4.0, 12.5, 3.2, 0.85, 'Image Dataset', 'IMAGE_DIR  |  *.jpg / *.png', C['data'], 10, 7.5)
box(ax, 14.0, 12.5, 3.2, 0.85, 'Model Directory', 'MODELS_DIR  |  9 checkpoints', C['data'], 10, 7.5)

arrow(ax,  4.0, 12.07,  4.0, 11.45)
arrow(ax, 14.0, 12.07, 14.0, 11.45)

# ════════════════════════════════════════════════════════════════
# ROW 2 — Loading / detection
# ════════════════════════════════════════════════════════════════
box(ax,  4.0, 11.1, 3.2, 0.78, 'Image Loading', 'PIL.Image.open → RGB', C['preproc'], 9.5, 7.5)
box(ax, 14.0, 11.1, 3.2, 0.78, 'Model Type Detection', 'detect_model_type(config.json)', C['preproc'], 9, 7.5)

arrow(ax,  4.0, 10.71,  4.0, 10.15)
arrow(ax, 14.0, 10.71, 14.0, 10.15)

# ════════════════════════════════════════════════════════════════
# ROW 3 — Preprocessing
# ════════════════════════════════════════════════════════════════
box(ax,  4.0, 9.8, 3.2, 0.78, 'Pre-processing', 'Processor (per-model defaults)', C['preproc'], 9.5, 7.5)
box(ax, 14.0, 9.8, 3.2, 0.78, 'Model Loading', 'AutoModel / timm.create_model', C['preproc'], 9.5, 7.5)

# converge arrows to inference block
arrow(ax,  5.6,  9.8,  7.3,  8.35)
arrow(ax, 12.4,  9.8, 10.7,  8.35)

# ════════════════════════════════════════════════════════════════
# ROW 4 — Model family group + inference boxes
# ════════════════════════════════════════════════════════════════
group_rect(ax, 0.5, 7.1, 17.0, 1.55, 'Inference  (torch.no_grad())', '#EEF6EE')

# CNN branch
box(ax,  3.0, 7.9, 3.6, 1.0, 'CNN Models',
    'ResNet · EfficientNet · ConvNeXt', C['cnn'], 9, 7.5)

# Transformer branch
box(ax,  9.0, 7.9, 4.4, 1.0, 'Transformer Models',
    'ViT · DeiT · Swin · CLIP · SigLIP', C['trans'], 9, 7.5)

# Hybrid branch
box(ax, 15.0, 7.9, 2.8, 1.0, 'Hybrid',
    'MobileViT', C['hybrid'], 9, 7.5)

# sub-note: zero-shot path
ax.text(9.0, 7.25, 'CLIP / SigLIP: text-guided zero-shot  →  logits_per_image → softmax',
        ha='center', va='center', fontsize=7.5, color='#336633', style='italic')

arrow(ax,  3.0, 7.4,  4.5, 6.55)
arrow(ax,  9.0, 7.4,  9.0, 6.55)
arrow(ax, 15.0, 7.4, 13.5, 6.55)

# ════════════════════════════════════════════════════════════════
# ROW 5 — Argmax / label mapping
# ════════════════════════════════════════════════════════════════
box(ax,  9.0, 6.2, 5.5, 0.78, 'Predicted Label',
    'logits.argmax(-1)  →  id2label / SHORT_LABELS', C['preproc'], 9, 7.5)

arrow(ax, 9.0, 5.81, 9.0, 5.25)

# ════════════════════════════════════════════════════════════════
# ROW 6 — CSV accumulation
# ════════════════════════════════════════════════════════════════
box(ax, 9.0, 4.9, 5.5, 0.78, 'Results Accumulation',
    'classification_results.csv   |   44 441 rows', C['data'], 9, 7.5)

arrow(ax, 9.0, 4.51, 9.0, 3.95)

# ════════════════════════════════════════════════════════════════
# ROW 7 — Consensus
# ════════════════════════════════════════════════════════════════
box(ax, 9.0, 3.6, 5.5, 0.78, 'Consensus Computation',
    'mode(axis=1)  →  Consensus_Prediction  +  Consensus_Count', C['consensus'], 9, 7.5)

arrow(ax, 9.0, 3.21, 9.0, 2.65)

# ════════════════════════════════════════════════════════════════
# ROW 8 — Outputs  (7 figures + CSV)
# ════════════════════════════════════════════════════════════════
group_rect(ax, 0.4, 1.15, 17.2, 1.3, 'Analysis & Outputs', '#FFF5EE')

plots = [
    ('Class\nDist.', 1.6),
    ('Agreement\nHeatmap', 3.6),
    ('Consensus\nLevel', 5.6),
    ('Acc vs\nParams', 7.6),
    ('Pred\nDiversity', 9.6),
    ('Arch\nFamily', 11.6),
    ('Divergence\nRate', 13.6),
    ('Output\nCSV', 15.8),
]
for label, xp in plots:
    col = C['output'] if 'CSV' in label else C['output']
    col = '#5B7FA6' if 'CSV' not in label else '#708090'
    box(ax, xp, 1.75, 1.75, 0.95, label, '', col, 7.5)

# arrows from consensus to each output box
for _, xp in plots:
    ax.annotate('', xy=(xp, 2.23), xytext=(9.0, 3.21),
                arrowprops=dict(arrowstyle='->', color='#999999',
                                lw=1.0, mutation_scale=10,
                                connectionstyle='arc3,rad=0.0'),
                zorder=2)

# ════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════
legend_items = [
    (C['data'],      'Data / Storage'),
    (C['preproc'],   'Processing / Loading'),
    (C['cnn'],       'CNN family'),
    (C['trans'],     'Transformer family'),
    (C['hybrid'],    'Hybrid family'),
    (C['consensus'], 'Consensus'),
    ('#5B7FA6',      'Visualisation'),
]
patches = [mpatches.Patch(facecolor=c, edgecolor='white', label=l) for c, l in legend_items]
ax.legend(handles=patches, loc='lower left', bbox_to_anchor=(0.0, 0.0),
          fontsize=8, framealpha=0.85, ncol=4,
          title='Component type', title_fontsize=8)

plt.tight_layout(pad=0.3)
plt.savefig('method_pipeline.pdf', dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
plt.savefig('method_pipeline.png', dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
plt.show()
print("Saved: method_pipeline.pdf  &  method_pipeline.png")
