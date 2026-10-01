import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# ─────────────────────────────────────────────
# Spark environment
# ─────────────────────────────────────────────
_python_exec = sys.executable.replace("\\", "/")
os.environ['PYSPARK_PYTHON']        = _python_exec
os.environ['PYSPARK_DRIVER_PYTHON'] = _python_exec

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BASE_DATASET_DIR = r"E:\vscode\big data\DataSet"

AVAILABLE_DATASETS = {
    "1": ("102flowers",    os.path.join(BASE_DATASET_DIR, "102flowers")),
    "2": ("Food-101",      os.path.join(BASE_DATASET_DIR, "food-101", "food-101", "images")),
    "3": ("MaxData",       os.path.join(BASE_DATASET_DIR, "maxData", "data")),
    "4": ("MiniDataset",   os.path.join(BASE_DATASET_DIR, "minidataset", "images")),
    "5": ("ImageNet-Mini", os.path.join(BASE_DATASET_DIR, "imagenet-mini", "train")),
}

AVAILABLE_MODELS = [
    "ResNet50",
    "EfficientNet-B0",
    "MobileNet-V3",
    "ViT-B-16",
    "Swin-T",
    "ConvNeXt-Tiny",
    "MobileViT-S",
    "DenseNet121",
    "ShuffleNet-V2",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "Vision_Benchmark_Results.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Menu helper
# ─────────────────────────────────────────────
def show_menu(title: str, options: list) -> list:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"  {len(options)+1}. All (همه موارد)")
    print(f"{'='*50}")

    raw = input("انتخاب (یک عدد یا چند عدد با کاما): ").strip()
    all_idx = str(len(options) + 1)

    try:
        chosen_ids = [s.strip() for s in raw.split(",")]
        if all_idx in chosen_ids:
            return options
        selected = [options[int(i) - 1] for i in chosen_ids if i.isdigit()]
        return selected if selected else options
    except Exception:
        return options

# ─────────────────────────────────────────────
# Spark worker (All PyTorch logic goes inside to prevent PicklingError)
# ─────────────────────────────────────────────
def spark_evaluate(args):
    model_name, dataset_name, dataset_path = args
    
    # ── IMPORTS INSIDE WORKER ──
    import time
    import os
    import psutil
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision import datasets
    from sklearn.metrics import precision_score, recall_score, f1_score

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    LINEAR_PROBE_EPOCHS = 3
    LINEAR_PROBE_LR     = 1e-3
    LINEAR_PROBE_BATCH  = 64

    def get_model(m_name: str):
        w = "DEFAULT"
        mapping = {
            "ResNet50":        lambda: models.resnet50(weights=w),
            "EfficientNet-B0": lambda: models.efficientnet_b0(weights=w),
            "MobileNet-V3":    lambda: models.mobilenet_v3_small(weights=w),
            "ViT-B-16":        lambda: models.vit_b_16(weights=w),
            "Swin-T":          lambda: models.swin_t(weights=w),
            "ConvNeXt-Tiny":   lambda: models.convnext_tiny(weights=w),
            "DenseNet121":     lambda: models.densenet121(weights=w),
            "ShuffleNet-V2":   lambda: models.shufflenet_v2_x1_0(weights=w),
        }
        if m_name in mapping:
            return mapping[m_name]()
        if m_name == "MobileViT-S":
            if hasattr(models, "mobilevit_s"):
                return models.mobilevit_s(weights=w)
            return models.mobilenet_v2(weights=w)
        raise ValueError(f"Unknown model: {m_name}")

    def replace_head(model, m_name: str, num_classes: int):
        name = m_name.lower()
        if "resnet" in name or "shufflenet" in name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif "efficientnet" in name or "mobilenet" in name or "mobilevit" in name:
            if hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
        elif "densenet" in name:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif "vit" in name or "swin" in name or "convnext" in name:
            if hasattr(model, "head"):
                if isinstance(model.head, nn.Sequential):
                    in_features = model.head[-1].in_features
                    model.head[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, num_classes)
            elif hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
        return model

    def freeze_backbone(model, m_name: str):
        name = m_name.lower()
        for param in model.parameters():
            param.requires_grad = False

        if "resnet" in name or "shufflenet" in name:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif "efficientnet" in name or "mobilenet" in name or "mobilevit" in name:
            if hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True
        elif "densenet" in name:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif "vit" in name or "swin" in name or "convnext" in name:
            if hasattr(model, "head"):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True

    def count_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters())

    def count_trainable_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_size_mb(model) -> float:
        total = sum(p.numel() * p.element_size() for p in model.parameters())
        total += sum(b.numel() * b.element_size() for b in model.buffers())
        return total / (1024 ** 2)

    def is_valid_image(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in VALID_EXTENSIONS

    def measure_latency(model, _device_val, runs: int = 30) -> float:
        dummy = torch.randn(1, 3, 224, 224).to(_device_val)
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                model(dummy)
                times.append((time.perf_counter() - t0) * 1000)
        return round(float(np.mean(times)), 3)

    def linear_probe_train(model, train_loader, num_classes: int, m_name: str, _device_val) -> list:
        model = replace_head(model, m_name, num_classes)
        freeze_backbone(model, m_name)
        model = model.to(_device_val)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LINEAR_PROBE_LR
        )

        epoch_accs = []
        model.train()
        for epoch in range(LINEAR_PROBE_EPOCHS):
            correct, total = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(_device_val), labels.to(_device_val)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                preds    = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
            epoch_acc = correct / total * 100 if total > 0 else 0.0
            epoch_accs.append(round(epoch_acc, 2))

        return epoch_accs

    def evaluate_model(m_name, d_name, d_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            full_dataset = datasets.ImageFolder(d_path, transform=transform, is_valid_file=is_valid_image)
        except Exception as e:
            return {"error": f"Cannot load dataset '{d_name}': {e}", "Model": m_name, "Dataset": d_name}

        n_total = len(full_dataset)
        if n_total == 0:
            return {"error": f"No images found in '{d_path}'", "Model": m_name, "Dataset": d_name}

        num_classes  = len(full_dataset.classes)
        # اجرای پیش‌فرض روی پردازنده (CPU) مانند کد اصلی خودتان:
        _device      = torch.device("cpu")

        n_train = int(0.8 * n_total)
        n_test  = n_total - n_train
        train_ds, test_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=LINEAR_PROBE_BATCH, shuffle=True,  num_workers=0)
        test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

        model    = get_model(m_name).to(_device)
        params   = count_parameters(model)
        size_mb  = get_model_size_mb(model)
        process  = psutil.Process(os.getpid())

        probe_start  = time.time()
        epoch_accs   = linear_probe_train(model, train_loader, num_classes, m_name, _device)
        probe_time   = round(time.time() - probe_start, 2)
        trainable_params = count_trainable_parameters(model)

        latency_ms = measure_latency(model, _device)

        model.eval()
        all_preds, all_labels = [], []
        peak_ram = process.memory_info().rss / (1024 ** 2)
        infer_start = time.time()

        with torch.no_grad():
            for images, labels in test_loader:
                images   = images.to(_device)
                outputs  = model(images)
                preds    = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                ram_now  = process.memory_info().rss / (1024 ** 2)
                peak_ram = max(peak_ram, ram_now)

        infer_time = time.time() - infer_start
        n_test_act = len(all_labels)

        accuracy   = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        precision  = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        recall     = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        f1         = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        throughput = n_test_act / infer_time if infer_time > 0 else 0.0
        total_time = probe_time + infer_time

        ep1 = epoch_accs[0] if len(epoch_accs) > 0 else 0.0
        ep2 = epoch_accs[1] if len(epoch_accs) > 1 else 0.0
        ep3 = epoch_accs[2] if len(epoch_accs) > 2 else 0.0

        del model

        return {
            "Dataset":                  d_name,
            "Model":                    m_name,
            "Num Classes":              num_classes,
            "Train Samples":            n_train,
            "Test Samples":             n_test_act,
            "Accuracy (%)":             round(accuracy,   2),
            "Precision (%)":            round(precision,  2),
            "Recall (%)":               round(recall,     2),
            "F1-Score (%)":             round(f1,         2),
            "Inference Time (s)":       round(infer_time, 2),
            "Probe Train Time (s)":     probe_time,
            "Total Time (s)":           round(total_time, 2),
            "Throughput (img/s)":       round(throughput, 2),
            "Latency per Image (ms)":   latency_ms,
            "Parameters (M)":           round(params / 1e6, 2),
            "Trainable Params (M)":     round(trainable_params / 1e6, 2),
            "Model Size (MB)":          round(size_mb, 2),
            "Peak RAM (MB)":            round(peak_ram, 2),
            "Train Acc Epoch1 (%)":     ep1,
            "Train Acc Epoch2 (%)":     ep2,
            "Train Acc Epoch3 (%)":     ep3,
        }

    try:
        result = evaluate_model(model_name, dataset_name, dataset_path)
        return result if result is not None else {}
    except Exception as e:
        import traceback
        return {"error": str(e) + "\n" + traceback.format_exc(),
                "Model": model_name, "Dataset": dataset_name}

# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main():
    try:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device_str = "cpu"

    print("\n" + "="*60)
    print("  Welcome to the Vision Models Benchmark Pipeline!")
    print(f"  Python : {_python_exec}")
    print(f"  Device : {device_str}")
    print("="*60)

    # ── انتخاب دیتاست ──
    dataset_labels = [name for name, _ in AVAILABLE_DATASETS.values()]
    chosen_dataset_names = show_menu("Select Dataset(s)", dataset_labels)
    chosen_datasets = [
        (name, path)
        for name, path in AVAILABLE_DATASETS.values()
        if name in chosen_dataset_names
    ]

    # ── انتخاب مدل ──
    chosen_models = show_menu("Select Model(s)", AVAILABLE_MODELS)

    # ── راه‌اندازی Spark ──
    print("\n[INFO] Starting Spark session...")
    spark = (
        SparkSession.builder
        .appName("VisionBenchmark")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.pyspark.python",            _python_exec)
        .config("spark.pyspark.driver.python",     _python_exec)
        .config("spark.network.timeout",              "600s")
        .config("spark.executor.heartbeatInterval",    "60s")
        .config("spark.python.worker.reuse",          "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    total_tasks = len(chosen_datasets) * len(chosen_models)
    print(f"[INFO] Total tasks: {total_tasks}")

    results = []

    for ds_name, ds_path in chosen_datasets:
        if not os.path.isdir(ds_path):
            print(f"\n[WARNING] Dataset path not found, skipping: {ds_path}")
            continue

        print(f"\n{'─'*50}")
        print(f"[DATASET] {ds_name}  →  {ds_path}")
        print(f"{'─'*50}")

        ds_tasks = [(m, ds_name, ds_path) for m in chosen_models]
        rdd      = spark.sparkContext.parallelize(ds_tasks, numSlices=len(ds_tasks))
        batch    = rdd.map(spark_evaluate).collect()

        for r in batch:
            if r and "error" in r:
                print(f"  [ERROR] {r.get('Model','?')}: {r['error'][:200]}...")
            elif r:
                print(f"  [OK] {r['Model']:20s} | "
                      f"Acc={r.get('Accuracy (%)', 0):5.1f}% | "
                      f"F1={r.get('F1-Score (%)', 0):5.1f}% | "
                      f"{r.get('Throughput (img/s)', 0):6.1f} img/s | "
                      f"Lat={r.get('Latency per Image (ms)', 0):.1f}ms | "
                      f"RAM={r.get('Peak RAM (MB)', 0):.0f}MB")
                results.append(r)

    spark.stop()
    print("\n[INFO] Spark session stopped.")

    # ── ذخیره نتایج ──
    if not results:
        print("\n[ERROR] No results generated.")
        return

    df = pd.DataFrame(results)

    # ستون‌های عددی برای summary
    numeric_cols = [
        "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)",
        "Inference Time (s)", "Probe Train Time (s)", "Total Time (s)",
        "Throughput (img/s)", "Latency per Image (ms)",
        "Parameters (M)", "Model Size (MB)", "Peak RAM (MB)",
        "Train Acc Epoch1 (%)", "Train Acc Epoch2 (%)", "Train Acc Epoch3 (%)",
    ]

    summary_df = (
        df.groupby(["Dataset", "Model"])[numeric_cols]
        .mean()
        .reset_index()
        .round(2)
    )

    best_acc   = summary_df.loc[summary_df["Accuracy (%)"].idxmax()]
    best_f1    = summary_df.loc[summary_df["F1-Score (%)"].idxmax()]
    best_speed = summary_df.loc[summary_df["Throughput (img/s)"].idxmax()]
    best_lat   = summary_df.loc[summary_df["Latency per Image (ms)"].idxmin()]

    print("\n" + "="*60)
    print("  BENCHMARK SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    print(f"  Best Accuracy  : {best_acc['Model']} on {best_acc['Dataset']} "
          f"→ {best_acc['Accuracy (%)']:.2f}%")
    print(f"  Best F1-Score  : {best_f1['Model']} on {best_f1['Dataset']} "
          f"→ {best_f1['F1-Score (%)']:.2f}%")
    print(f"  Fastest (img/s): {best_speed['Model']} on {best_speed['Dataset']} "
          f"→ {best_speed['Throughput (img/s)']:.1f} img/s")
    print(f"  Lowest Latency : {best_lat['Model']} on {best_lat['Dataset']} "
          f"→ {best_lat['Latency per Image (ms)']:.2f} ms")
    print("="*60)

    write_header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode='a', header=write_header, index=False)
    print(f"\n[SUCCESS] Results saved to: {RESULTS_CSV}")
    print(f"\n[INFO] CSV columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"         • {col}")


if __name__ == "__main__":
    main()
