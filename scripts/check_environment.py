"""
Environment checker for the CogNVS reproduction project.

Run this after activating the `cognvs` conda environment to confirm
your setup matches what the project expects. Works on machines
without a GPU too (CUDA fields will just report unavailable).

Usage:
    python scripts/check_environment.py
"""

import sys
import platform


def check_python():
    print("Python version:", sys.version.split()[0])
    print("Platform:", platform.platform())


def check_torch():
    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED")
        return

    print("torch version:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        print("CUDA version:", torch.version.cuda)
        gpu_count = torch.cuda.device_count()
        print("GPU count:", gpu_count)
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  GPU {i}: {name} ({total_mem_gb:.1f} GB)")
    else:
        print("No CUDA-capable GPU detected (or CPU-only torch build installed).")


def check_key_packages():
    packages = ["diffusers", "transformers", "accelerate", "numpy", "gradio"]
    print("\nKey package versions:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")


if __name__ == "__main__":
    print("=" * 50)
    print("CogNVS Reproduction — Environment Check")
    print("=" * 50)
    check_python()
    print()
    check_torch()
    check_key_packages()
    print("=" * 50)