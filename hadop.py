import os
import sys

# ۱. تنظیم متغیرها در بالاترین خط ممکن (قبل از ایمپورت اسپارک)
python_path = r"E:\vscode\big data\venv\Scripts\python.exe"
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

import io
import pandas as pd
import torch
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

# تنظیم مسیر هادوپ به درایو E
hadoop_home = r'E:\hadoop'
os.environ['HADOOP_HOME'] = hadoop_home
os.environ['hadoop.home.dir'] = hadoop_home
os.environ['PATH'] = os.path.join(hadoop_home, 'bin') + os.pathsep + os.environ.get('PATH', '')

# ==========================================
# 1. تنظیمات و پیکربندی (Configuration)
# ==========================================

# مدل مورد نظر خود را اینجا انتخاب کنید: "clip" یا "resnet" یا "vit"
SELECTED_MODEL = "clip" 

# مسیرها (با استفاده از abspath برای جلوگیری از خطای ویندوز)
IMAGE_FOLDER_PATH = os.path.abspath(r"E:\vscode\big data\maxData\data")
OUTPUT_PARQUET = os.path.abspath(r"E:\vscode\big data\embeddings.parquet")

# دیکشنری پیکربندی مدل‌ها
MODEL_CONFIGS = {
    "clip": {
        "name": r"E:\vscode\big data\tools",
        "processor_class": "AutoProcessor",
        "model_class": "CLIPModel"
    },
    "resnet": {
        "name": "microsoft/resnet-50",
        "processor_class": "AutoImageProcessor",
        "model_class": "ResNetModel"
    },
    "vit": {
        "name": "google/vit-base-patch16-224-in21k",
        "processor_class": "AutoImageProcessor",
        "model_class": "ViTModel"
    }
}

# ==========================================
# 2. تابع سازنده پردازشگر اسپارک (Factory)
# ==========================================

def get_feature_extractor(model_type):
    """
    این تابع بر اساس مدل انتخاب شده، یک تابع پردازشی (Iterator) برای اسپارک برمی‌گرداند.
    """
    config = MODEL_CONFIGS[model_type]
    
    def extract_features_iterator(iterator):
        # 1. وارد کردن کتابخانه‌ها درون Worker اسپارک
        from transformers import AutoProcessor, AutoImageProcessor, CLIPModel, ResNetModel, ViTModel
        
        # انتخاب کلاس‌های مناسب بر اساس Config
        ProcessorClass = eval(config["processor_class"])
        ModelClass = eval(config["model_class"])
        
        # 2. بارگذاری مدل و پردازشگر
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = ProcessorClass.from_pretrained(config["name"])
        model = ModelClass.from_pretrained(config["name"]).to(device)
        model.eval()

        # 3. پردازش دسته‌ها (Batches)
        for pdf in iterator:
            paths = pdf['path'].tolist()
            contents = pdf['content'].tolist()
            
            valid_images = []
            valid_paths = []
            
            for path, content in zip(paths, contents):
                try:
                    img = Image.open(io.BytesIO(content)).convert("RGB")
                    valid_images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
            
            if not valid_images:
                continue
                
            # آماده‌سازی تصاویر برای مدل
            inputs = processor(images=valid_images, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # استفاده از متد مناسب بر اساس نوع مدل
                if model_type == "clip":
                    embeddings = model.get_image_features(**inputs)
                else:
                    outputs = model(**inputs)
                    
                    # 4. یکسان‌سازی خروجی مدل‌های مختلف
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        # مخصوص ResNet
                        embeddings = outputs.pooler_output
                        if embeddings.dim() > 2:
                            embeddings = torch.flatten(embeddings, start_dim=1)
                    elif hasattr(outputs, 'last_hidden_state'):
                        # مخصوص ViT (میانگین‌گیری از توکن‌ها)
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                    else:
                        raise ValueError("Unknown model output format")
                
                # نرمال‌سازی بردارها (L2 Normalization)
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
            
            # بازگرداندن نتایج این پارتیشن
            yield pd.DataFrame({
                "path": valid_paths,
                "embedding": embeddings.tolist()
            })
            
    return extract_features_iterator

# ==========================================
# 3. اجرای پایپ‌لاین اسپارک
# ==========================================

def main():
    if not os.path.exists(IMAGE_FOLDER_PATH):
        raise FileNotFoundError(f"Directory not found: {IMAGE_FOLDER_PATH}")

    print(f"Initializing Spark. Selected Model: {SELECTED_MODEL}...")
    spark = SparkSession.builder \
        .appName("MultiModel_Image_Embedding") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.pyspark.python", python_path) \
        .config("spark.pyspark.driver.python", python_path) \
        .getOrCreate()
    
    # خواندن فایل‌های تصویری به صورت باینری
    print("Reading images...")
    images_df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(IMAGE_FOLDER_PATH)

    # تعریف ساختار خروجی (Schema)
    output_schema = StructType([
        StructField("path", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), True)
    ])

    # دریافت تابع پردازشی مخصوص مدل انتخاب شده
    extraction_function = get_feature_extractor(SELECTED_MODEL)

    print("Extracting features (This may take a while)...")
    # اعمال تابع روی پارتیشن‌های اسپارک
    result_df = images_df.select("path", "content") \
                         .mapInPandas(extraction_function, schema=output_schema)

    print(f"Saving results to Parquet: {OUTPUT_PARQUET}...")
    result_df.write \
        .mode("overwrite") \
        .parquet(OUTPUT_PARQUET)
        
    print("Pipeline finished successfully!")
    
    # نمایش چند سطر برای اطمینان
    spark.read.parquet(OUTPUT_PARQUET).show(3, truncate=80)
    spark.stop()

if __name__ == "__main__":
    main()
