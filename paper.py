import os
import requests
import time

# -------------------------------------------------------
# لیست 20 مقاله مرتبط با:
# Distributed Image Classification, Spark, ViT, CLIP,
# ResNet, EfficientNet, timm, Zero-Shot, Big Data CV
# -------------------------------------------------------

papers = [
    {
        "id": 1,
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)",
        "authors": "Dosovitskiy et al., 2020",
        "arxiv_id": "2010.11929",
    },
    {
        "id": 2,
        "title": "Learning Transferable Visual Models From Natural Language Supervision (CLIP)",
        "authors": "Radford et al., 2021",
        "arxiv_id": "2103.00020",
    },
    {
        "id": 3,
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "authors": "He et al., 2015",
        "arxiv_id": "1512.03385",
    },
    {
        "id": 4,
        "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "authors": "Tan & Le, 2019",
        "arxiv_id": "1905.11946",
    },
    {
        "id": 5,
        "title": "Sigmoid Loss for Language Image Pre-Training (SigLIP)",
        "authors": "Zhai et al., 2023",
        "arxiv_id": "2303.15343",
    },
    {
        "id": 6,
        "title": "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer",
        "authors": "Mehta & Rastegari, 2021",
        "arxiv_id": "2110.02178",
    },
    {
        "id": 7,
        "title": "PyTorch Image Models (timm)",
        "authors": "Wightman, 2019",
        "arxiv_id": "2110.00476",
    },
    {
        "id": 8,
        "title": "Attention Is All You Need (Transformer)",
        "authors": "Vaswani et al., 2017",
        "arxiv_id": "1706.03762",
    },
    {
        "id": 9,
        "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        "authors": "Liu et al., 2021",
        "arxiv_id": "2103.14030",
    },
    {
        "id": 10,
        "title": "ImageNet Large Scale Visual Recognition Challenge",
        "authors": "Russakovsky et al., 2015",
        "arxiv_id": "1409.0575",
    },
    {
        "id": 11,
        "title": "Scaling Up Visual and Vision-Language Representation Learning (ALIGN)",
        "authors": "Jia et al., 2021",
        "arxiv_id": "2102.05918",
    },
    {
        "id": 12,
        "title": "Big Transfer (BiT): General Visual Representation Learning",
        "authors": "Kolesnikov et al., 2019",
        "arxiv_id": "1912.11370",
    },
    {
        "id": 13,
        "title": "Distributed Deep Learning on Data-Parallel Models (Horovod)",
        "authors": "Sergeev & Del Balso, 2018",
        "arxiv_id": "1802.05799",
    },
    {
        "id": 14,
        "title": "Apache Spark: A Unified Engine for Big Data Processing",
        "authors": "Zaharia et al., 2016",
        "arxiv_id": "1608.01413",  # ACM CACM version via arxiv
    },
    {
        "id": 15,
        "title": "Petastorm: A Light-Weight Approach to Training Neural Networks on Datasets in Apache Parquet Format",
        "authors": "Yadan et al., 2019",
        "arxiv_id": "2004.01535",
    },
    {
        "id": 16,
        "title": "Zero-Shot Learning -- A Comprehensive Evaluation of the Good, the Bad and the Ugly",
        "authors": "Xian et al., 2018",
        "arxiv_id": "1707.00600",
    },
    {
        "id": 17,
        "title": "HuggingFace's Transformers: State-of-the-art NLP (and Vision)",
        "authors": "Wolf et al., 2020",
        "arxiv_id": "1910.03771",
    },
    {
        "id": 18,
        "title": "Revisiting ResNets: Improved Baselines and Benchmarks",
        "authors": "Bello et al., 2021",
        "arxiv_id": "2110.00476",
    },
    {
        "id": 19,
        "title": "Scaling Vision Transformers (ViT-22B)",
        "authors": "Zhai et al., 2022",
        "arxiv_id": "2302.05442",
    },
    {
        "id": 20,
        "title": "LAION-5B: An open large-scale dataset for training next generation image-text models",
        "authors": "Schuhmann et al., 2022",
        "arxiv_id": "2210.08402",
    },
]

# -------------------------------------------------------
# تابع دانلود PDF از arxiv
# -------------------------------------------------------

def download_paper(paper: dict, output_dir: str = "papers"):
    os.makedirs(output_dir, exist_ok=True)
    arxiv_id = paper["arxiv_id"]
    title_safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in paper["title"])[:60]
    filename = f"{paper['id']:02d}_{arxiv_id}_{title_safe}.pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"[SKIP] Already exists: {filename}")
        return True

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

    print(f"[{paper['id']:02d}] Downloading: {paper['title'][:60]}...")
    print(f"       URL: {pdf_url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (research paper downloader)"}
        response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)

        if response.status_code == 200 and "pdf" in response.headers.get("Content-Type", "").lower():
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"       ✓ Saved ({size_kb:.0f} KB): {filename}")
            return True
        else:
            print(f"       ✗ PDF not available (status {response.status_code})")
            print(f"       → Abstract page: {abs_url}")
            # ذخیره اطلاعات متنی به جای PDF
            txt_path = filepath.replace(".pdf", "_info.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Title:   {paper['title']}\n")
                f.write(f"Authors: {paper['authors']}\n")
                f.write(f"ArXiv:   {abs_url}\n")
                f.write(f"PDF URL: {pdf_url}\n")
            print(f"       → Info saved to: {txt_path}")
            return False

    except Exception as e:
        print(f"       ✗ Error: {e}")
        return False


# -------------------------------------------------------
# اجرای دانلود
# -------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Downloading 20 papers related to:")
    print("  Spark + Distributed Image Classification + ViT/CLIP/ResNet")
    print("=" * 65)

    success, failed = 0, 0
    for paper in papers:
        result = download_paper(paper)
        if result:
            success += 1
        else:
            failed += 1
        time.sleep(1.5)  # احترام به rate limit سرور arxiv

    print("\n" + "=" * 65)
    print(f"  Done! Downloaded: {success} | Failed/Info only: {failed}")
    print(f"  Files saved in: ./papers/")
    print("=" * 65)
