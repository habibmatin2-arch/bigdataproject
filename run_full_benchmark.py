import os
import time
import psutil
import torch
import torchvision.models as models
import pandas as pd
import numpy as np

# تنظیم بهینه‌سازی CPU
torch.set_num_threads(os.cpu_count() or 4)
torch.set_grad_enabled(False)
device = torch.device('cpu')

# تعریف معماری ۹ مدل بر اساس مخزن torchvision
MODEL_DICT = {
    "MobileNet-V3": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    "ShuffleNet-V2": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    "MobileViT-S": None,  # مدیریت اختصاصی در لودر
    "EfficientNet-B0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "DenseNet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
    "ConvNeXt-Tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    "Swin-T": (models.swin_t, models.Swin_T_Weights.DEFAULT),
    "ResNet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "ViT-B-16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
}

def load_model(name):
    if name == "MobileViT-S":
        import timm
        model = timm.create_model("mobilevit_s", pretrained=True)
    else:
        builder, weights = MODEL_DICT[name]
        model = builder(weights=weights)
    model.eval()
    return model

def benchmark_model(name, num_warmup=15, num_runs=50, num_repeats=3):
    print(f"\n==========================================")
    print(f"[*] در حال ارزیابی مدل: {name}")
    print(f"==========================================")
    
    try:
        model = load_model(name)
    except Exception as e:
        print(f"[!] خطا در بارگذاری {name}: {e}")
        return None

    # محاسبه تعداد پارامترها و اندازه مدل
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    dummy_input = torch.randn(1, 3, 224, 224)

    # ۱. فاز گرم‌کردن (Warm-up)
    print(" -> فاز گرم‌کردن پردازنده (Warm-up)...")
    for _ in range(num_warmup):
        _ = model(dummy_input)

    # ۲. فاز بنچمارک چندمرحله‌ای
    latencies_all_runs = []
    print(f" -> اجرای {num_repeats} تکرار ارزیابی ({num_runs} تصویر در هر تکرار)...")
    
    for r in range(num_repeats):
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # میلی‌ثانیه
        latencies_all_runs.append(np.mean(times))
        print(f"    تکرار {r+1}: میانگین تأخیر = {np.mean(times):.2f} ms")

    mean_latency = float(np.mean(latencies_all_runs))
    std_latency = float(np.std(latencies_all_runs))
    throughput = 1000.0 / mean_latency if mean_latency > 0 else 0

    # بررسی مصرف رم
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    res = {
        "Model": name,
        "Parameters (M)": round(total_params, 2),
        "Model Size (MB)": round(model_size_mb, 2),
        "Latency Mean (ms)": round(mean_latency, 2),
        "Latency Std (ms)": round(std_latency, 2),
        "Throughput (img/s)": round(throughput, 2),
        "RAM Usage (MB)": round(peak_ram_mb, 2)
    }
    print(f"[+] نتیجه: {mean_latency:.2f} ± {std_latency:.2f} ms | Throughput: {throughput:.2f} img/s")
    return res

if __name__ == "__main__":
    results = []
    for model_name in MODEL_DICT.keys():
        row = benchmark_model(model_name)
        if row:
            results.append(row)

    out_df = pd.DataFrame(results)
    out_path = "Vision_Benchmark_Full_9Models.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[OK] تمام شد! نتایج کامل در فایل '{out_path}' ذخیره گردید.")
