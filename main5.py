import os
import sys
import glob
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import gc

# Imports for Persian text rendering
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False
    print("Warning: 'arabic-reshaper' or 'python-bidi' not found. Persian text may not render correctly.")
    print("Please run: pip install arabic-reshaper python-bidi")

# ==========================================
# 0. تنظیمات محیطی — باید قبل از import pyspark باشد
# ==========================================
_python_exe = sys.executable

os.environ['PYSPARK_PYTHON']        = _python_exe
os.environ['PYSPARK_DRIVER_PYTHON'] = _python_exe
os.environ['SPARK_LOCAL_IP']        = '127.0.0.1'
os.environ['PYTHONPATH']            = os.path.dirname(_python_exe)

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# ==========================================
# 1. تنظیمات مسیرها
# ==========================================
IMAGE_DIR  = r"E:\vscode\big data\maxData\data"
MODELS_DIR = r"E:\vscode\big data\models"
OUTPUT_CSV = r"E:\vscode\big data\classification_results.csv"
OUTPUT_PLOTS_BASE = r"E:\vscode\big data\plots"

# ایجاد پوشه‌های ذخیره نمودارها
os.makedirs(os.path.join(OUTPUT_PLOTS_BASE, 'en'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PLOTS_BASE, 'fa'), exist_ok=True)

# ==========================================
# 2. دسته‌بندی‌های پوشاک
# ==========================================
CANDIDATE_LABELS = [
    "a photo of a shirt or t-shirt",
    "a photo of pants or jeans",
    "a photo of a skirt",
    "a photo of a dress",
    "a photo of a jacket or coat",
    "a photo of shoes or sneakers",
    "a photo of a bag or backpack",
    "a photo of an accessory",
    "a photo of other clothing"
]

SHORT_LABELS = [
    "jersey, T-shirt, tee shirt", 
    "jean, blue jean, denim", 
    "miniskirt, mini", 
    "suit, suit of clothes",
    "cardigan", 
    "running shoe", 
    "backpack, back pack, knapsack, packsack, rucksack, haversack", 
    "hair slide", 
    "sweatshirt"
]

# ==========================================
# 3. تشخیص نوع مدل
# ==========================================
def detect_model_type(model_path: str) -> str:
    folder_name = os.path.basename(model_path).lower()

    if "siglip"       in folder_name: return "siglip"
    if "clip"         in folder_name: return "clip"
    if "mobilevit"    in folder_name: return "unknown"
    if "vit"          in folder_name: return "vit"
    if "resnet"       in folder_name: return "resnet"
    if "efficientnet" in folder_name: return "efficientnet"

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "").lower()
        arch       = cfg.get("architectures", [""])[0].lower()
        if "siglip"       in model_type or "siglip"       in arch: return "siglip"
        if "clip"         in model_type or "clip"         in arch: return "clip"
        if "mobilevit"    in model_type or "mobilevit"    in arch: return "unknown"
        if "vit"          in model_type or "vit"          in arch: return "vit"
        if "resnet"       in model_type or "resnet"       in arch: return "resnet"
        if "efficientnet" in model_type or "efficientnet" in arch: return "efficientnet"

    return "unknown"

# ==========================================
# 4. انتخاب مدل توسط کاربر 
# ==========================================
def select_model() -> list[tuple[str, str]]:
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"پوشه مدل‌ها پیدا نشد: {MODELS_DIR}")

    available = [
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
        and len(os.listdir(os.path.join(MODELS_DIR, d))) > 0
    ]

    if not available:
        raise RuntimeError("هیچ مدلی در پوشه models پیدا نشد.")

    print("\n" + "="*50)
    print("  ALL MODEL:")
    print("="*50)
    print("  [0] Process all Models")
    for i, name in enumerate(available, 1):
        model_path = os.path.join(MODELS_DIR, name)
        mtype      = detect_model_type(model_path)
        print(f"  [{i}] {name}  (نوع: {mtype})")
    print("="*50)

    while True:
        try:
            choice = int(input(f"\n Enter number a model (0-{len(available)}): "))
            if choice == 0:
                print("\n✓  Select All Model ")
                selected = []
                for name in available:
                    path = os.path.join(MODELS_DIR, name)
                    selected.append((path, detect_model_type(path)))
                return selected
                
            elif 1 <= choice <= len(available):
                selected_name = available[choice - 1]
                selected_path = os.path.join(MODELS_DIR, selected_name)
                selected_type = detect_model_type(selected_path)
                print(f"\n✓ مدل انتخاب شده: {selected_name}  (نوع: {selected_type})\n")
                return [(selected_path, selected_type)]
            else:
                print(f"عدد بین 0 تا {len(available)} وارد کنید.")
        except ValueError:
            print("لطفاً یک عدد صحیح وارد کنید.")

# ==========================================
# 5. نگاشت نام timm
# ==========================================
TIMM_NAME_MAP = {
    "resnet-50":       "resnet50.a1_in1k",
    "resnet_50":       "resnet50.a1_in1k",
    "resnet50":        "resnet50.a1_in1k",
    "efficientnet-b0": "efficientnet_b0.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
}

def get_timm_name(folder_name: str) -> str:
    key = folder_name.lower()
    if key in TIMM_NAME_MAP:
        return TIMM_NAME_MAP[key]
    normalized = key.replace("-", "").replace("_", "")
    for k, v in TIMM_NAME_MAP.items():
        if k.replace("-", "").replace("_", "") == normalized:
            return v
    return key

# ==========================================
# 6. تابع پردازش هر پارتیشن (Worker)
# ==========================================
def make_partition_fn(model_path: str, model_type: str):
    _candidate_labels = CANDIDATE_LABELS[:]
    _short_labels     = SHORT_LABELS[:]
    _model_path       = model_path
    _model_type       = model_type
    _timm_name        = get_timm_name(os.path.basename(model_path))
    _python_exe_w     = _python_exe  

    def classify_images_partition(iterator):
        import os, sys, torch, time
        from PIL import Image

        os.environ['PYSPARK_PYTHON']        = _python_exe_w
        os.environ['PYSPARK_DRIVER_PYTHON'] = _python_exe_w

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Worker] model={_model_path}  type={_model_type}  device={device}")

        processor = None
        model     = None
        use_timm  = False
        id2label  = {}

        try:
            if _model_type == "clip":
                from transformers import CLIPProcessor, CLIPModel
                model     = CLIPModel.from_pretrained(_model_path).to(device)
                processor = CLIPProcessor.from_pretrained(_model_path)
                
            elif _model_type == "siglip":
                from transformers import AutoProcessor, AutoModel
                model     = AutoModel.from_pretrained(_model_path).to(device)
                processor = AutoProcessor.from_pretrained(_model_path)

            elif _model_type == "vit":
                from transformers import ViTForImageClassification, ViTImageProcessor
                model     = ViTForImageClassification.from_pretrained(_model_path).to(device)
                processor = ViTImageProcessor.from_pretrained(_model_path)
                if hasattr(model.config, "id2label"):
                    id2label = model.config.id2label

            elif _model_type in ("resnet", "efficientnet"):
                timm_ok = False
                try:
                    import timm
                    from timm.data import resolve_data_config, create_transform

                    weight_file = os.path.join(_model_path, "pytorch_model.bin")
                    safetensor  = os.path.join(_model_path, "model.safetensors")
                    has_weights = os.path.exists(weight_file) or os.path.exists(safetensor)

                    if has_weights:
                        model = timm.create_model(_timm_name, pretrained=False, num_classes=1000)
                        if os.path.exists(safetensor):
                            from safetensors.torch import load_file
                            state = load_file(safetensor)
                        else:
                            state = torch.load(weight_file, map_location=device)
                        model.load_state_dict(state, strict=False)
                    else:
                        model = timm.create_model(_timm_name, pretrained=True, num_classes=1000)

                    model = model.to(device).eval()

                    try:
                        from timm.data import ImageNetInfo
                        _info = ImageNetInfo()
                        id2label = {i: _info.index_to_description(i) for i in range(1000)}
                    except Exception:
                        id2label = {i: f"class_{i}" for i in range(1000)}

                    cfg_timm  = resolve_data_config({}, model=model)
                    processor = create_transform(**cfg_timm)
                    use_timm  = True
                    timm_ok   = True
                except Exception as e:
                    pass

                if not timm_ok:
                    from transformers import AutoImageProcessor, AutoModelForImageClassification
                    processor = AutoImageProcessor.from_pretrained(_model_path)
                    model     = AutoModelForImageClassification.from_pretrained(_model_path).to(device)
                    if hasattr(model.config, "id2label"):
                        id2label = model.config.id2label

            else:  
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                processor = AutoImageProcessor.from_pretrained(_model_path)
                model     = AutoModelForImageClassification.from_pretrained(_model_path).to(device)
                if hasattr(model.config, "id2label"):
                    id2label = model.config.id2label

            model.eval()

            for batch in iterator:
                predictions = []
                latencies = []
                for img_path in batch['filepath']:
                    try:
                        image = Image.open(img_path).convert("RGB")
                        
                        start_time = time.time()
                        
                        if _model_type == "siglip":
                            inputs = processor(
                                text=_candidate_labels, images=image,
                                return_tensors="pt", padding="max_length", truncation=True
                            ).to(device)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            probs    = torch.sigmoid(outputs.logits_per_image)
                            best_idx = probs.argmax().item()
                            predictions.append(_short_labels[best_idx])
                            
                        elif _model_type == "clip":
                            inputs = processor(
                                text=_candidate_labels, images=image,
                                return_tensors="pt", padding=True
                            ).to(device)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            probs    = outputs.logits_per_image.softmax(dim=1)
                            best_idx = probs.argmax().item()
                            predictions.append(_short_labels[best_idx])

                        elif _model_type == "vit":
                            inputs = processor(images=image, return_tensors="pt").to(device)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            best_idx = outputs.logits.argmax(-1).item()
                            label = id2label.get(best_idx, str(best_idx))
                            predictions.append(label)

                        else:  
                            if use_timm:
                                tensor = processor(image).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    logits = model(tensor)
                            else:
                                inputs = processor(images=image, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    logits = model(pixel_values=inputs["pixel_values"]).logits

                            best_idx = logits.argmax(-1).item()
                            label = id2label.get(best_idx, str(best_idx))
                            predictions.append(label)

                        end_time = time.time()
                        latencies.append(end_time - start_time)

                    except Exception as e:
                        print(f"Image Error ({img_path}): {e}")
                        predictions.append("Error_Processing")
                        latencies.append(0.0)

                yield pd.DataFrame({
                    'filepath':   list(batch['filepath']),
                    'prediction': predictions,
                    'time': latencies
                })

        except Exception as err:
            print(f"[ERROR] Loading model {_model_type} failed: {err}")
            for batch in iterator:
                yield pd.DataFrame({
                    'filepath': list(batch['filepath']),
                    'prediction': ["Error_Model_Load"] * len(batch),
                    'time': [0.0] * len(batch)
                })
        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return classify_images_partition

# ==========================================
# Helper for Persian Text
# ==========================================
def prep_text(text, lang='en'):
    if lang == 'fa' and HAS_BIDI:
        return get_display(arabic_reshaper.reshape(text))
    return text

# ==========================================
# 7. توابع مربوط به مصورسازی داده (14-Chart Visualization)
# ==========================================
def generate_plots(csv_path):
    print("\n[Visualizations] Generating analysis plots...")
    if not os.path.exists(csv_path):
        print("CSV file not found for plotting.")
        return
        
    df = pd.read_csv(csv_path)
    
    # تفکیک ستون‌های پیش‌بینی از ستون‌های زمان
    pred_cols = [c for c in df.columns if c != 'filepath' and not c.endswith('_time')]
    time_cols = [c for c in df.columns if c.endswith('_time')]
    
    clean_df = df.copy()
    error_keywords = ["Error", "Error_Model_Load", "Error_Processing"]
    for col in pred_cols:
        clean_df[col] = clean_df[col].apply(lambda x: np.nan if pd.isna(x) or any(ek in str(x) for ek in error_keywords) else x)

    # Dictionary of configs for EN and FA
    langs = {
        'en': {
            'dir': os.path.join(OUTPUT_PLOTS_BASE, 'en'),
            'overlap_title': 'Model Agreement (Prediction Overlap)',
            'consensus_title': 'Consensus Percentage per Category',
            'consensus_ylabel': 'Consensus (%)',
            'consensus_xlabel': 'Category',
            'latency_title': 'Average Inference Latency per Model',
            'latency_ylabel': 'Latency (Seconds per Image)'
        },
        'fa': {
            'dir': os.path.join(OUTPUT_PLOTS_BASE, 'fa'),
            'overlap_title': 'میزان توافق مدل‌ها (هم‌پوشانی پیش‌بینی)',
            'consensus_title': 'درصد اجماع مدل‌ها در هر دسته‌بندی',
            'consensus_ylabel': 'درصد اجماع (%)',
            'consensus_xlabel': 'دسته‌بندی',
            'latency_title': 'متوسط زمان استنتاج هر مدل',
            'latency_ylabel': 'زمان (ثانیه بر تصویر)'
        }
    }

    # 1. Prediction Overlap Matrix (Agreement)
    for lang, cfg in langs.items():
        if len(pred_cols) > 1:
            overlap_matrix = pd.DataFrame(index=pred_cols, columns=pred_cols, dtype=float)
            for m1 in pred_cols:
                for m2 in pred_cols:
                    if m1 == m2:
                        overlap_matrix.loc[m1, m2] = 100.0
                    else:
                        valid_mask = clean_df[m1].notna() & clean_df[m2].notna()
                        if valid_mask.sum() > 0:
                            agreement = (clean_df.loc[valid_mask, m1] == clean_df.loc[valid_mask, m2]).mean() * 100
                        else:
                            agreement = 0.0
                        overlap_matrix.loc[m1, m2] = agreement
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(overlap_matrix, annot=True, fmt=".1f", cmap="Blues")
            plt.title(prep_text(cfg['overlap_title'], lang))
            plt.tight_layout()
            plt.savefig(os.path.join(cfg['dir'], 'Prediction_Overlap_Matrix.png'))
            plt.close()

    # 2. Consensus Level per Category
    if len(pred_cols) > 1:
        majority_vote = clean_df[pred_cols].mode(axis=1)
        if not majority_vote.empty:
            consensus_label = majority_vote.iloc[:, 0]
            agreement_ratios = clean_df[pred_cols].apply(lambda row: (row == consensus_label[row.name]).mean() * 100, axis=1)
            consensus_df = pd.DataFrame({'Category': consensus_label, 'Agreement': agreement_ratios}).dropna()
            
            if not consensus_df.empty:
                avg_consensus = consensus_df.groupby('Category')['Agreement'].mean().sort_values(ascending=False).head(15)
                
                for lang, cfg in langs.items():
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x=avg_consensus.index, y=avg_consensus.values, palette="viridis")
                    plt.title(prep_text(cfg['consensus_title'], lang))
                    plt.ylabel(prep_text(cfg['consensus_ylabel'], lang))
                    plt.xlabel(prep_text(cfg['consensus_xlabel'], lang))
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(cfg['dir'], 'Category_Consensus.png'))
                    plt.close()

    # 3. Latency Comparison
    if time_cols:
        avg_times = clean_df[time_cols].mean().sort_values()
        # Remove '_time' suffix for display
        avg_times.index = [idx.replace('_time', '') for idx in avg_times.index]
        
        for lang, cfg in langs.items():
            plt.figure(figsize=(10, 6))
            sns.barplot(x=avg_times.index, y=avg_times.values, palette="magma")
            plt.title(prep_text(cfg['latency_title'], lang))
            plt.ylabel(prep_text(cfg['latency_ylabel'], lang))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(cfg['dir'], 'Latency_Comparison.png'))
            plt.close()

    print("✅ Core visualization plots saved successfully in 'plots/en' and 'plots/fa'.")

# ==========================================
# 8. تابع اصلی
# ==========================================
def main():
    selected_models = select_model()

    print("Initializing Spark Session...")
    spark = (
        SparkSession.builder
        .appName("Clothing_Classification")
        .config("spark.pyspark.python",        _python_exe)
        .config("spark.pyspark.driver.python", _python_exe)
        .config("spark.driver.host",           "127.0.0.1")
        .config("spark.driver.bindAddress",    "127.0.0.1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "4")
        .config("spark.driver.memory",   "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.python.worker.reuse",         "false")
        .config("spark.network.timeout",             "600s")
        .config("spark.executor.heartbeatInterval",  "60s")
        .config("spark.hadoop.io.native.lib.available", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print(f"Scanning images in: {IMAGE_DIR}")
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    total = len(image_paths)
    print(f"Found {total} images.")
    if total == 0:
        print("No images found! Exiting...")
        spark.stop()
        return

    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing results at {OUTPUT_CSV}. Merging new predictions...")
        final_combined_df = pd.read_csv(OUTPUT_CSV)
        final_combined_df['filepath'] = final_combined_df['filepath'].astype(str)
        
        new_paths_df = pd.DataFrame({'filepath': image_paths})
        final_combined_df = pd.merge(new_paths_df, final_combined_df, on='filepath', how='left')
    else:
        final_combined_df = pd.DataFrame({'filepath': image_paths})

    schema = StructType([
        StructField("filepath",   StringType(), True),
        StructField("prediction", StringType(), True),
        StructField("time",       FloatType(), True)
    ])

    num_partitions = 8 
    spark_df_base = spark.createDataFrame(pd.DataFrame({'filepath': image_paths})).repartition(num_partitions)

    for model_path, model_type in selected_models:
        model_name = os.path.basename(model_path)
        time_col_name = f"{model_name}_time"
        print(f"\n==============================================")
        print(f"  Starting classification with: {model_name}  ")
        print(f"==============================================\n")
        
        partition_fn = make_partition_fn(model_path, model_type)
        
        result_df = spark_df_base.mapInPandas(partition_fn, schema=schema)
        
        print(f"Collecting results for {model_name}...")
        temp_pdf = result_df.toPandas()
        
        # تغییر نام ستون‌ها متناسب با مدل
        temp_pdf.rename(columns={'prediction': model_name, 'time': time_col_name}, inplace=True)
        
        if model_name in final_combined_df.columns:
            final_combined_df.drop(columns=[model_name], inplace=True)
        if time_col_name in final_combined_df.columns:
            final_combined_df.drop(columns=[time_col_name], inplace=True)
            
        final_combined_df = pd.merge(final_combined_df, temp_pdf, on='filepath', how='left')
        
        final_combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"--> Saved partial results to {OUTPUT_CSV}")

        gc.collect()

    print(f"\n✅ All processing complete! Final combined results saved → {OUTPUT_CSV}")
    spark.stop()
    
    # اجرای تولید نمودارهای نهایی
    generate_plots(OUTPUT_CSV)

if __name__ == "__main__":
    main()
