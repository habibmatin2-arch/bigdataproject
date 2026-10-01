#!/usr/bin/env python3
"""
env_report.py — run this ON THE MACHINE WHERE THE EXPERIMENTS RAN
(the CPU-only Windows workstation used for the paper), then send the output
back so Table V can be finalized with exact specifications.

Usage:  python env_report.py
"""
import platform, sys, os

print("=== ICCKE 2026 Paper 1073 - Table V environment report ===")
print()

# --- CPU ---
proc = platform.processor() or "(empty - copy the CPU name from Task Manager > Performance > CPU)"
print("platform.processor() :", proc)
try:
    print("os.cpu_count()       :", os.cpu_count(), "logical cores")
except Exception as e:
    print("os.cpu_count()       : error", e)

# --- OS / Python ---
print("platform.platform()  :", platform.platform())
print("platform.machine()   :", platform.machine())
print("Python               :", sys.version.split()[0])

# --- ML stack ---
try:
    import torch
    print("torch                :", torch.__version__)
    try:
        print("torch threads        :", torch.get_num_threads())
    except Exception:
        pass
    print("torch.cuda.is_available():", torch.cuda.is_available())
except Exception as e:
    print("torch                : NOT IMPORTABLE ->", e)

try:
    import torchvision
    print("torchvision          :", torchvision.__version__)
except Exception as e:
    print("torchvision          : NOT IMPORTABLE ->", e)

try:
    import pyspark
    print("pyspark              :", pyspark.__version__)
except Exception as e:
    print("pyspark              : NOT IMPORTABLE ->", e)

# --- RAM (optional) ---
try:
    import psutil
    b = psutil.virtual_memory().total
    print("RAM (psutil)         : %.1f GB" % (b / 1024**3))
except Exception:
    print("RAM (psutil)         : psutil not installed - read total RAM from Task Manager")

print()
print("=== PLEASE ALSO SEND MANUALLY ===")
print("1. Exact CPU marketing name (e.g. 'Intel Core i7-9750H') - from Task Manager > Performance > CPU")
print("2. Total installed RAM (GB) - from Task Manager > Performance > Memory")
print("3. OS edition/version if platform.platform() above looks incomplete (winver)")
print()
print("Copy everything above and send it back - Table V will be finalized from it.")
