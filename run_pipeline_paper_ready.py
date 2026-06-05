import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
import os

# ── فونت فارسی ──────────────────────────────────────────────
try:
    from matplotlib import font_manager
    # استفاده از فونت‌های موجود ویندوز
    fa_font = font_manager.FontProperties(family='Tahoma')
except:
    fa_font = None

# ── ترجمه لیبل‌ها ────────────────────────────────────────────
FA_LABELS = {
    "miniskirt, mini": "دامن کوتاه",
    "lampshade, lamp shade": "آباژور",
    "apron": "پیش‌بند",
    "diaper, nappy, napkin": "پوشک",
    "jean, blue jean, denim": "شلوار جین",
    "sweatshirt": "سویشرت",
    "swimming trunks, bathing trunks": "شورت شنا",
    "jersey, T-shirt, tee shirt": "تی‌شرت",
    "pajama, pyjama, pj's, jammies": "پیژامه",
    "suit, suit of clothes": "کت‌وشلوار",
    "bathing cap, swimming cap": "کلاه شنا",
    "hair slide": "گیره مو",
    "cardigan": "کاردیگان",
    "ski mask": "ماسک اسکی",
    "dumbbell": "دمبل",
    "hoopskirt, crinoline": "دامن حلقه‌ای",
    "wool, woolen, woollen": "پشمی",
    "bulletproof vest": "جلیقه ضدگلوله",
    "running shoe": "کفش ورزشی",
    "maillot": "مایو",
    "brassiere, bra, bandeau": "سوتین",
    "bikini, two-piece": "بیکینی",
    "bow tie, bow-tie, bowtie": "پاپیون",
    "stole": "شال",
    "fur coat": "پالتو خز",
    "lab coat, laboratory coat": "روپوش آزمایشگاه",
    "trench coat": "بارانی",
    "gown": "لباس بلند",
    "academic gown, academic robe, judge's robe": "ردای دانشگاهی",
    "abaya": "عبا",
    "cloak": "شنل",
    "poncho": "پانچو",
    "overskirt": "روی‌دامن",
    "sarong": "ساروگ",
    "miniskirt": "دامن کوتاه",
    "sock": "جوراب",
    "stocking": "جوراب ساق‌بلند",
    "sandal": "صندل",
    "loafer": "کفش لوفر",
    "clog": "کفش چوبی",
    "cowboy boot": "چکمه کابوی",
    "cowboy hat, ten-gallon hat": "کلاه کابوی",
    "sombrero": "سومبررو",
    "mortarboard": "کلاه فارغ‌التحصیلی",
    "helmet": "کلاه ایمنی",
    "crash helmet": "کلاه موتور",
    "swimming cap": "کلاه شنا",
    "bonnet, poke bonnet": "کلاه بنت",
    "beret": "کلاه برت",
    "mitten": "دستکش انگشت‌نما",
    "glove": "دستکش",
    "umbrella": "چتر",
    "purse": "کیف دستی",
    "backpack, back pack, knapsack, packsack, rucksack, haversack": "کوله‌پشتی",
    "wallet, billfold, notecase, pocketbook": "کیف پول",
    "sunglasses, dark glasses, shades": "عینک آفتابی",
    "mask": "ماسک",
    "neck brace": "گردن‌بند طبی",
    "Windsor tie": "کراوات ویندزور",
    "bolo tie, bolo, bola tie, bola": "کراوات بولو",
    "necktie, tie": "کراوات",
}

def fa(label):
    return FA_LABELS.get(label, label)

# ── خواندن CSV ───────────────────────────────────────────────
df = pd.read_csv("classification_results.csv")
MODEL_COLS = [c for c in df.columns if c != "filepath"]

# ═══════════════════════════════════════════════════════════════
# ۱. Overlap Matrix
# ═══════════════════════════════════════════════════════════════

all_labels = sorted(set(df[MODEL_COLS].values.ravel()))
top_n = 20
freq = Counter(df[MODEL_COLS].values.ravel())
top_labels = [l for l, _ in freq.most_common(top_n)]

overlap = pd.DataFrame(0, index=top_labels, columns=top_labels)
for _, row in df[MODEL_COLS].iterrows():
    preds = set(row.values)
    for a in preds:
        for b in preds:
            if a in top_labels and b in top_labels:
                overlap.loc[a, b] += 1

def make_overlap(lang):
    fig, ax = plt.subplots(figsize=(13, 11))
    if lang == "fa":
        tick_labels = [fa(l) for l in top_labels]
        title = "ماتریس هم‌پوشانی پیش‌بینی‌ها (۲۰ کلاس برتر)"
        xlabel, ylabel = "کلاس", "کلاس"
        matplotlib.rcParams['axes.unicode_minus'] = False
    else:
        tick_labels = [l[:25] for l in top_labels]
        title = "Prediction Overlap Matrix (Top 20 Classes)"
        xlabel, ylabel = "Class", "Class"

    im = ax.imshow(overlap.values, cmap="YlOrRd")
    ax.set_xticks(range(top_n)); ax.set_yticks(range(top_n))

    if lang == "fa" and fa_font:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontproperties=fa_font, fontsize=9)
        ax.set_yticklabels(tick_labels, fontproperties=fa_font, fontsize=9)
        ax.set_title(title, fontproperties=fa_font, fontsize=13, pad=12)
        ax.set_xlabel(xlabel, fontproperties=fa_font)
        ax.set_ylabel(ylabel, fontproperties=fa_font)
    else:
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    cbar_label = "تعداد هم‌رخداد" if lang == "fa" else "Co-occurrence count"
    cbar = plt.colorbar(im, ax=ax)
    if lang == "fa" and fa_font:
        cbar.set_label(cbar_label, fontproperties=fa_font)
    else:
        cbar.set_label(cbar_label)
    
    plt.tight_layout()
    fname = f"overlap_matrix_{lang}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

make_overlap("en")
make_overlap("fa")

# ═══════════════════════════════════════════════════════════════
# ۲. Per-category Consensus
# ═══════════════════════════════════════════════════════════════

n_models = len(MODEL_COLS)
df["consensus"] = df[MODEL_COLS].apply(
    lambda r: max(Counter(r.values).values()) / n_models, axis=1
)
df["majority_class"] = df[MODEL_COLS].apply(
    lambda r: Counter(r.values).most_common(1)[0][0], axis=1
)

consensus_by_class = (
    df.groupby("majority_class")["consensus"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "avg_consensus", "count": "n_images"})
    .query("n_images >= 50")
    .sort_values("avg_consensus", ascending=False)
    .head(25)
)

def make_consensus(lang):
    fig, ax = plt.subplots(figsize=(12, 7))
    labels_raw = consensus_by_class.index.tolist()
    values = consensus_by_class["avg_consensus"].values
    colors = plt.cm.RdYlGn(values)

    if lang == "fa":
        tick_labels = [fa(l) for l in labels_raw]
        title = "میانگین اجماع مدل‌ها به تفکیک کلاس (حداقل ۵۰ تصویر)"
        xlabel = "میانگین اجماع"
        ylabel = "کلاس"
    else:
        tick_labels = [l[:30] for l in labels_raw]
        title = "Average Model Consensus per Class (min 50 images)"
        xlabel = "Average Consensus"
        ylabel = "Class"

    bars = ax.barh(range(len(labels_raw)), values, color=colors)
    ax.set_yticks(range(len(labels_raw)))

    if lang == "fa" and fa_font:
        ax.set_yticklabels(tick_labels, fontproperties=fa_font, fontsize=9)
        ax.set_title(title, fontproperties=fa_font, fontsize=13)
        ax.set_xlabel(xlabel, fontproperties=fa_font)
        ax.set_ylabel(ylabel, fontproperties=fa_font)
    else:
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fname = f"consensus_per_class_{lang}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

make_consensus("en")
make_consensus("fa")
