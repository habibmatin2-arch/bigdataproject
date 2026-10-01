import os
import time
import psutil
import torch
import numpy as np
import pandas as pd

# غیرفعال‌سازی ماژول‌های مانیتورینگ متفرقه
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

torch.set_num_threads(os.cpu_count() or 4)
torch.set_grad_enabled(False)

import timm

print("[*] در حال ارزیابی اختصاصی MobileViT-S...")
model = timm.create_model("mobilevit_s", pretrained=True)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

# Warm-up
for _ in range(15):
    _ = model(dummy)

# Benchmark
latencies = []
for r in range(3):
    t_runs = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(dummy)
        t1 = time.perf_counter()
        t_runs.append((t1 - t0) * 1000.0)
    latencies.append(np.mean(t_runs))
    print(f"  تکرار {r+1}: {np.mean(t_runs):.2f} ms")

mean_lat = float(np.mean(latencies))
std_lat = float(np.std(latencies))
tp = 1000.0 / mean_lat

total_params = sum(p.numel() for p in model.parameters()) / 1e6
size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

print(f"\n[+] MobileViT-S: {mean_lat:.2f} ± {std_lat:.2f} ms | Throughput: {tp:.2f} img/s")

# الحاق به فایل نتایج
csv_path = "Vision_Benchmark_Full_9Models.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    new_row = {
        "Model": "MobileViT-S",
        "Parameters (M)": round(total_params, 2),
        "Model Size (MB)": round(size_mb, 2),
        "Latency Mean (ms)": round(mean_lat, 2),
        "Latency Std (ms)": round(std_lat, 2),
        "Throughput (img/s)": round(tp, 2),
        "RAM Usage (MB)": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] مدل MobileViT-S با موفقیت به {csv_path} افزوده شد.")
