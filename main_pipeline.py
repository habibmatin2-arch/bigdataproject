import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
import faiss

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from typing import Iterator

# ---------------------------------------------------------
# 1. Spark Session Setuppy -m pip install .

# ---------------------------------------------------------
from pyspark.sql import SparkSession
print("Script is starting...")
spark = SparkSession.builder \
    .appName("CLIP_Distributed_Feature_Extraction") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
print("Script is finishing...")

# ---------------------------------------------------------
# 2. Distributed UDF for CLIP Embedding
# ---------------------------------------------------------
# این تابع روی هر Worker در Spark اجرا می‌شود
@pandas_udf(ArrayType(FloatType()))
def get_clip_embeddings(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # تنظیمات سخت‌افزار
    device = "cpu"
    
    # لود کردن مدل فقط یک بار در هر Worker
    model, preprocess = clip.load("ViT-B/32", device=device, download_root="./")
    
    for paths_batch in iterator:
        batch_embeddings = []
        for path in paths_batch:
            try:
                # خواندن و پیش‌پردازش تصویر
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)
                
                # استخراج ویژگی
                with torch.no_grad():
                    features = model.encode_image(image)
                    # تبدیل تنسور به لیست ساده پایتون برای ذخیره در دیتافریم اسپارک
                    features_np = features.cpu().numpy().flatten().tolist()
                    batch_embeddings.append(features_np)
            except Exception as e:
                # در صورت خرابی تصویر، مقدار None برگردانده می‌شود
                batch_embeddings.append(None)
                
        yield pd.Series(batch_embeddings)

# ---------------------------------------------------------
# 3. Main Data Processing Pipeline
# ---------------------------------------------------------
def main():
    # مسیر پوشه تصاویر خود را اینجا وارد کنید
    image_dir = "path/to/your/images" 
    
    # گرفتن لیست تمام تصاویر
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found!")
        return

    # ساخت DataFrame اولیه
    df = spark.createDataFrame([(p,) for p in image_files], ["image_path"])
    print(f"Total images to process: {df.count()}")

    # اعمال تابع توزیع‌شده استخراج ویژگی
    print("Extracting features using PySpark and CLIP...")
    features_df = df.withColumn("embedding", get_clip_embeddings(col("image_path")))
    
    # حذف تصاویری که پردازش نشدند (null)
    features_df = features_df.dropna(subset=["embedding"])
    
    # برای ساخت FAISS، داده‌ها را جمع‌آوری می‌کنیم
    # نکته: در ابعاد کلان‌داده، داده‌ها به جای collect در Parquet ذخیره می‌شوند.
    print("Collecting results for indexing...")
    results = features_df.collect()
    
    paths = [row['image_path'] for row in results]
    embeddings = np.array([row['embedding'] for row in results]).astype('float32')

    # ---------------------------------------------------------
    # 4. Building FAISS Index
    # ---------------------------------------------------------
    dimension = 512 # ابعاد خروجی مدل CLIP
    print(f"Building FAISS index with dimension: ${dimension}$ ...")
    
    # ساخت یک اندیس Flat بر اساس فاصله L2
    index = faiss.IndexFlatL2(dimension)
    
    # اضافه کردن بردارها به اندیس
    index.add(embeddings)
    
    # ذخیره اندیس در هارد دیسک
    faiss.write_index(index, "images_faiss.index")
    print("FAISS Index saved successfully!")

    # ---------------------------------------------------------
    # 5. Example Search (Querying)
    # ---------------------------------------------------------
    print("Testing Search System...")
    # فرض کنید می‌خواهیم شبیه‌ترین تصاویر به تصویر اول را پیدا کنیم
    k = 5 # پیدا کردن 5 تصویر شبیه
    query_vector = embeddings[0:1] # باید 2D array باشد
    
    distances, indices = index.search(query_vector, k)
    
    print("\nSearch Results:")
    print(f"Query Image: {paths[0]}")
    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}: {paths[idx]} (Distance: {distances[0][i]:.4f})")

    print("✅ Pipeline executed successfully!")
    print("🔍 Search Results:")
    print("Indices of top matching images:", indices)
    print("Distances:", distances)
if __name__ == "__main__":
    main()
