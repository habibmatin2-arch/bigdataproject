import os
import sys
import time
import gc
import math
import logging
import threading
import argparse
import platform
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as torchvision_models

import timm
from transformers import AutoProcessor, AutoModel, CLIPVisionModel, SiglipVisionModel
import pynvml

# ==========================================
# 0. LOGGING & GLOBAL CONFIG
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VisionBenchmark")

# Fix OpenMP conflict on Windows if needed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. NVML POWER LOGGING ENGINE
# ==========================================
class NVMLPowerLogger:
    """
    Background thread sampler for GPU metrics using pynvml.
    Samples power (W), temperature (°C), and GPU utilization (%) at fixed intervals.
    """
    def __init__(self, device_index: int = 0, sampling_interval_ms: int = 20):
        self.device_index = device_index
        self.interval = sampling_interval_ms / 1000.0
        self.running = False
        self.thread = None
        self.samples: List[Dict[str, Any]] = []
        self.handle = None
        self._init_nvml()

    def _init_nvml(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            logger.info(f"NVML initialized for GPU {self.device_index}: {pynvml.nvmlDeviceGetName(self.handle)}")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.handle = None

    def start(self, model_name: str, dataset_name: str, batch_id: int = 0):
        if self.handle is None:
            return
        self.current_model = model_name
        self.current_dataset = dataset_name
        self.current_batch_id = batch_id
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def _sample_loop(self):
        while self.running:
            t_start = time.perf_counter()
            try:
                # NVML power is reported in milliwatts
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                
                self.samples.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'Time_Sec': time.time(),
                    'Model': self.current_model,
                    'Dataset': self.current_dataset,
                    'Batch_ID': self.current_batch_id,
                    'Power_W': p_mw / 1000.0,
                    'Temp_C': temp,
                    'GPU_Util (%)': util
                })
            except Exception as e:
                pass
            
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, self.interval - elapsed)
            time.sleep(sleep_time)

    def stop() -> List[Dict[str, Any]]:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        return self.samples

    def measure_idle_power(self, duration_sec: float = 5.0) -> float:
        """Measures average baseline power idle before runs."""
        if self.handle is None:
            return 0.0
        logger.info(f"Measuring GPU idle baseline power for {duration_sec}s...")
        idle_samples = []
        t_end = time.time() + duration_sec
        while time.time() < t_end:
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                idle_samples.append(p_mw / 1000.0)
            except Exception:
                pass
            time.sleep(0.05)
        mean_idle = float(np.mean(idle_samples)) if idle_samples else 0.0
        logger.info(f"Measured GPU Idle Base Power: {mean_idle:.2f} W")
        return mean_idle

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

# ==========================================
# 2. VLM ADAPTER WRAPPERS (CLIP & SigLIP)
# ==========================================
class VLMClassifierHead(nn.Module):
    """
    Standard linear classifier evaluation adapter for CLIP / SigLIP visual encoders.
    """
    def __init__(self, model_type: str, num_classes: int = 1000):
        super().__init__()
        self.model_type = model_type.lower()
        if 'clip' in self.model_type:
            self.vlm = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            hidden_dim = self.vlm.config.hidden_size
        elif 'siglip' in self.model_type:
            self.vlm = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
            hidden_dim = self.vlm.config.hidden_size
        else:
            raise ValueError(f"Unsupported VLM adapter: {model_type}")

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vlm(pixel_values=x)
        # Extract pooled sequence output or class token representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0, :]
        return self.classifier(feats)

# ==========================================
# 3. MODEL BUILDER FACTORY
# ==========================================
def build_model(model_name: str, num_classes: int, device: torch.device) -> Tuple[nn.Module, Tuple[int, int]]:
    """
    Instantiates target benchmark models and returns model with default image input resolution.
    """
    img_size = (224, 224)
    
    if model_name == "ResNet50":
        m = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "EfficientNet-B0":
        m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    elif model_name == "ConvNeXt-Tiny":
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
    elif model_name == "ViT-B-16":
        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "Swin-T":
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
    elif model_name == "MobileViT-S":
        m = timm.create_model("mobilevit_s", pretrained=True, num_classes=num_classes)
    elif model_name == "DeiT-Small":
        m = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "CLIP-ViT-B/32":
        m = VLMClassifierHead("clip", num_classes=num_classes)
    elif model_name == "SigLIP-ViT-B/16":
        m = VLMClassifierHead("siglip", num_classes=num_classes)
    else:
        raise ValueError(f"Unknown target model name: {model_name}")

    m = m.to(device)
    m.eval()
    return m, img_size

# ==========================================
# 4. DATASET FACTORY & SYNTHETIC FALLBACK
# ==========================================
class DummyDataset(Dataset):
    """Fallback synthetic dataset if custom path structure does not exist."""
    def __init__(self, img_size=(224, 224), length=200, num_classes=10):
        self.img_size = img_size
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(3, *self.img_size)
        y = idx % self.num_classes
        return x, y

def get_dataloader(ds_name: str, root_path: str, batch_size: int, img_size: Tuple[int, int]) -> DataLoader:
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if folder structure exists
    if os.path.exists(root_path) and len(os.listdir(root_path)) > 0:
        try:
            from torchvision.datasets import ImageFolder
            ds = ImageFolder(root=root_path, transform=transform)
            logger.info(f"Loaded real dataset '{ds_name}' with {len(ds)} images.")
        except Exception as e:
            logger.warning(f"Error loading ImageFolder at {root_path}: {e}. Falling back to DummyDataset.")
            ds = DummyDataset(img_size=img_size, length=128, num_classes=10)
    else:
        logger.warning(f"Path '{root_path}' not found for '{ds_name}'. Initializing Synthetic Dummy Dataset.")
        ds = DummyDataset(img_size=img_size, length=128, num_classes=10)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 5. METRICS & MEMORY PROFILING UTILS
# ==========================================
def count_parameters(model: nn.Module) -> Tuple[float, float]:
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return round(total, 3), round(trainable, 3)

def compute_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return round((param_size + buffer_size) / (1024 ** 2), 2)

# ==========================================
# 6. CORE INFERENCE & ENERGY EVALUATION PIPELINE
# ==========================================
def run_benchmark_instance(
    model_name: str,
    ds_name: str,
    ds_path: str,
    batch_size: int,
    device: torch.device,
    power_logger: NVMLPowerLogger,
    idle_power_w: float
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    # Clean GPU memory before benchmarking
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Temporary build dataset to resolve number of classes
    loader = get_dataloader(ds_name, ds_path, batch_size, (224, 224))
    num_classes = getattr(loader.dataset, 'classes', None)
    num_classes = len(num_classes) if num_classes else 10

    # Build model
    model, img_size = build_model(model_name, num_classes, device)
    total_params, trainable_params = count_parameters(model)
    model_size_mb = compute_model_size_mb(model)

    # Refresh loader with model target image resolution
    loader = get_dataloader(ds_name, ds_path, batch_size, img_size)
    num_samples = len(loader.dataset)

    # Warm-up phase (10 batches)
    logger.info(f"Warming up {model_name} on {ds_name} (BS={batch_size})...")
    dummy_input = torch.randn(batch_size, 3, *img_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize(device)

    # Start Inference & NVML Power Tracking
    all_preds, all_targets = [], []
    raw_logs = []
    
    power_logger.start(model_name=model_name, dataset_name=ds_name)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    batch_samples = power_logger.stop()
    raw_logs.extend(batch_samples)

    total_time = end_time - start_time
    peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

    # Metric computations
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)) * 100)
    prec = float(precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100)
    rec = float(recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100)
    f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100)

    throughput = round(num_samples / total_time, 2)
    latency_ms = round((total_time / num_samples) * 1000, 3)

    # Power & Integration Analysis (Trapezoidal rule)
    if len(raw_logs) > 1:
        powers = [s['Power_W'] for s in raw_logs]
        times = [s['Time_Sec'] for s in raw_logs]
        times_rel = [t - times[0] for t in times]
        
        avg_power = float(np.mean(powers))
        max_power = float(np.max(powers))
        
        # Total Energy E (Joules) = Integral P(t) dt
        total_energy_j = float(np.trapz(powers, x=times_rel))
        # Active Net Energy (subtract idle baseline)
        net_powers = [max(0.0, p - idle_power_w) for p in powers]
        net_energy_j = float(np.trapz(net_powers, x=times_rel))
    else:
        avg_power = idle_power_w
        max_power = idle_power_w
        total_energy_j = avg_power * total_time
        net_energy_j = 0.0

    energy_per_img_mj = round((total_energy_j * 1000.0) / num_samples, 3)
    acc_per_watt = round(acc / avg_power, 3) if avg_power > 0 else 0.0
    throughput_per_watt = round(throughput / avg_power, 3) if avg_power > 0 else 0.0

    summary_row = {
        'Model': model_name,
        'Dataset': ds_name,
        'Image_Resolution': f"{img_size[0]}x{img_size[1]}",
        'Batch_Size': batch_size,
        'Test_Samples': num_samples,
        'Accuracy (%)': round(acc, 2),
        'Precision (%)': round(prec, 2),
        'Recall (%)': round(rec, 2),
        'F1-Score (%)': round(f1, 2),
        'Inference_Time (s)': round(total_time, 3),
        'Throughput (img/s)': throughput,
        'Latency per Image (ms)': latency_ms,
        'Parameters (M)': total_params,
        'Trainable Params (M)': trainable_params,
        'Model Size (MB)': model_size_mb,
        'Peak VRAM (MB)': peak_vram_mb,
        'Avg_Power (W)': round(avg_power, 2),
        'Max_Power (W)': round(max_power, 2),
        'Total_Energy (J)': round(total_energy_j, 2),
        'Net_Energy (J)': round(net_energy_j, 2),
        'Energy_per_Image (mJ/img)': energy_per_img_mj,
        'Accuracy_per_Watt': acc_per_watt,
        'Throughput_per_Watt': throughput_per_watt
    }

    return summary_row, raw_logs

# ==========================================
# 7. MAIN ORCHESTRATOR & EXCEL EXPORT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Vision Hardware Power Benchmark Pipeline")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32], help="Batch sizes to benchmark")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    parser.add_argument("--output", type=str, default="Master_Final_Benchmark_Results_with_Energy.xlsx")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.error("CUDA is required for hardware power profiling!")
        sys.exit(1)

    # Define paths configuration
    DATASETS = {
        '102flowers': './data/102flowers',
        'Food-101': './data/food-101',
        'ImageNet-Mini': './data/imagenet-mini',
        'maxData': './data/maxData',
        'minidataset': './data/minidataset'
    }

    MODELS = [
        'ResNet50',
        'EfficientNet-B0',
        'ConvNeXt-Tiny',
        'ViT-B-16',
        'Swin-T',
        'MobileViT-S',
        'DeiT-Small',
        'CLIP-ViT-B/32',
        'SigLIP-ViT-B/16'
    ]

    # Initialize NVML Power Logger
    power_logger = NVMLPowerLogger(device_index=0, sampling_interval_ms=20)
    idle_power_w = power_logger.measure_idle_power(duration_sec=3.0)

    # Collect Hardware Metadata
    handle = power_logger.handle
    gpu_name = pynvml.nvmlDeviceGetName(handle) if handle else torch.cuda.get_device_name(0)
    driver_ver = pynvml.nvmlSystemGetDriverVersion() if handle else "N/A"
    
    metadata_df = pd.DataFrame([{
        'GPU Name': gpu_name,
        'Driver Version': driver_ver,
        'CUDA Version': torch.version.cuda,
        'PyTorch Version': torch.__version__,
        'Host CPU': platform.processor(),
        'Host OS': f"{platform.system()} {platform.release()}",
        'Idle Power Base (W)': round(idle_power_w, 2),
        'Benchmark Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    summary_results = []
    raw_logs_all = []

    # Run Benchmark Loops
    for bs in args.batch_sizes:
        for ds_name, ds_path in DATASETS.items():
            for m_name in MODELS:
                logger.info(f"--- Running: Model={m_name} | Dataset={ds_name} | BS={bs} ---")
                try:
                    summary_row, raw_logs = run_benchmark_instance(
                        model_name=m_name,
                        ds_name=ds_name,
                        ds_path=ds_path,
                        batch_size=bs,
                        device=device,
                        power_logger=power_logger,
                        idle_power_w=idle_power_w
                    )
                    summary_results.append(summary_row)
                    raw_logs_all.extend(raw_logs)
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM Error encountered for {m_name} on {ds_name} (BS={bs}). Skipping!")
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Execution Error for {m_name} on {ds_name}: {e}")
                    torch.cuda.empty_cache()

    power_logger.shutdown()

    # Aggregate Analysis Across Runs
    summary_df = pd.DataFrame(summary_results)
    raw_logs_df = pd.DataFrame(raw_logs_all)

    if not summary_df.empty:
        aggregated_df = summary_df.groupby(['Model', 'Dataset']).agg(
            Accuracy_Mean=('Accuracy (%)', 'mean'),
            Accuracy_Std=('Accuracy (%)', 'std'),
            Latency_Mean=('Latency per Image (ms)', 'mean'),
            Latency_Std=('Latency per Image (ms)', 'std'),
            Energy_per_Image_Mean=('Energy_per_Image (mJ/img)', 'mean'),
            Energy_per_Image_Std=('Energy_per_Image (mJ/img)', 'std'),
            Throughput_per_Watt_Mean=('Throughput_per_Watt', 'mean')
        ).reset_index().round(3)
    else:
        aggregated_df = pd.DataFrame()

    # Save Results to Multi-Sheet Excel File
    logger.info(f"Exporting final benchmark results to {args.output}...")
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        metadata_df.to_excel(writer, sheet_name='Hardware_Metadata', index=False)
        raw_logs_df.to_excel(writer, sheet_name='Raw_Inference_Logs', index=False)
        summary_df.to_excel(writer, sheet_name='Run_Summary', index=False)
        aggregated_df.to_excel(writer, sheet_name='Aggregated_Results', index=False)

    logger.info("✓ Master Benchmark Execution Completed Successfully!")

if __name__ == "__main__":
    main()
