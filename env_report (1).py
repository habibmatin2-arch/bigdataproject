#!/usr/bin/env python3
"""
env_report.py (v2) -- run this ON THE MACHINE WHERE THE EXPERIMENTS RAN,
inside the SAME Python environment you used for the paper experiments
(same venv / conda env / interpreter as your main pipeline code).

It NEVER crashes on a missing library: it reports what is missing,
prints the exact fix, and (if pyspark + java work) runs a tiny
local-mode Spark smoke test.

Usage:  python env_report.py
"""

import os
import platform
import subprocess
import sys

BAR = "=" * 64


def try_import(name):
    """Return (version, None) or (None, error_text). Never raises."""
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed (no __version__)"), None
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def java_version():
    """First line of 'java -version', or a clear NOT FOUND message."""
    try:
        p = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=20
        )
        lines = ((p.stderr or "") + (p.stdout or "")).strip().splitlines()
        return lines[0] if lines else "exit=%d" % p.returncode
    except FileNotFoundError:
        return "NOT FOUND (java is not on PATH)"
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)


def ram_gb():
    """Total physical RAM in GB via WinAPI, then psutil, then sysconf."""
    try:
        import ctypes

        class MSE(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MSE()
        st.dwLength = ctypes.sizeof(MSE)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 1024.0 ** 3
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / 1024.0 ** 3
    except Exception:
        pass
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024.0 ** 3
    except Exception:
        return None


def cpu_name():
    """Windows CPU marketing name from the registry (no psutil needed)."""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        try:
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
        finally:
            key.Close()
    except Exception:
        return None


def main():
    pyspark_ver = None

    print(BAR)
    print("ICCKE 2026 Paper 1073 - Table V environment report (v2)")
    print(BAR)

    print()
    print("[1] Python interpreter -- must be the SAME env your")
    print("    experiments ran with (same venv / conda env)!")
    print("  sys.executable      :", sys.executable)
    print("  Python version      :", sys.version.split()[0])
    print("  platform.platform() :", platform.platform())
    print("  machine             :", platform.machine())

    print()
    print("[2] CPU / RAM / cores")
    print("  platform.processor():", platform.processor())
    cpu = cpu_name()
    print("  CPU marketing name  :",
          cpu or "(not detected - Task Manager > Performance > CPU)")
    print("  logical cores       :", os.cpu_count())
    ram = ram_gb()
    print("  total RAM           :",
          ("%.1f GB" % ram) if ram else "(not detected - Task Manager > Performance > Memory)")

    print()
    print("[3] Python packages")
    for lib in ("torch", "torchvision", "numpy", "pandas", "pyspark"):
        ver, err = try_import(lib)
        if ver is not None:
            print("  %-12s: %s" % (lib, ver))
            if lib == "pyspark":
                pyspark_ver = ver
        else:
            print("  %-12s: MISSING -> %s" % (lib, err))

    try:
        import torch

        print("  torch threads       :", torch.get_num_threads())
        print("  torch.cuda available:", torch.cuda.is_available())
    except Exception:
        pass

    print()
    print("[4] Java (required by PySpark)")
    print("  java -version       :", java_version())

    print()
    print("[5] PySpark local-mode smoke test")
    if pyspark_ver is None:
        print("  SKIPPED - pyspark is not importable in THIS interpreter.")
        print("  FIX (run in the same terminal / env as your experiments):")
        print("      python -m pip install pyspark")
        print("  (~300 MB download; if pip rejects your Python version, try:")
        print("       python -m pip install pyspark==3.5.1 )")
    else:
        try:
            from pyspark.sql import SparkSession

            spark = (
                SparkSession.builder.master("local[1]")
                .appName("env_probe")
                .config("spark.ui.enabled", "false")
                .config("spark.log.level", "ERROR")
                .getOrCreate()
            )
            print("  SparkSession OK     : spark %s" % spark.version)
            print("  defaultParallelism  :", spark.sparkContext.defaultParallelism)
            spark.stop()
            print("  => PySpark works on this machine.")
        except Exception as exc:
            print("  FAILED -> %s: %s" % (type(exc).__name__, exc))
            print("  Most common causes on Windows:")
            print("    - Java missing/wrong: install JDK 17 (Temurin/Adoptium)")
            print("      and set JAVA_HOME, or with Anaconda:")
            print("      conda install -c conda-forge openjdk=17")
            print("    - winutils.exe / NativeIO / hadoop.dll error:")
            print("      Windows needs winutils.exe + hadoop.dll in HADOOP_HOME/bin")

    print()
    print(BAR)
    print("Copy EVERYTHING above and send it back to the chat.")
    print("Additionally (only if something above says 'not detected'):")
    print("  1. CPU name  - Task Manager > Performance > CPU")
    print("  2. Total RAM - Task Manager > Performance > Memory")
    print("  3. OS edition - run 'winver'")
    print(BAR)


if __name__ == "__main__":
    main()
