#!/usr/bin/env python3
"""
PolypNet VLM Setup Script
=========================
Downloads and verifies the VLM model weights needed for clinical report generation.

Two options:
  1. LLaVA-1.5-7B (best quality, ~14 GB VRAM, ~28 GB download)
  2. BLIP-2 Flan-T5-XL (lighter, ~16 GB download, runs on 8 GB VRAM)

Usage:
  python3 setup_vlm.py              # auto-select based on GPU VRAM
  python3 setup_vlm.py --model blip2
  python3 setup_vlm.py --model llava
  python3 setup_vlm.py --check      # only check what's installed, no download
"""

import argparse
import subprocess
import sys


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU found: {torch.cuda.get_device_name(0)} — {vram:.1f} GB VRAM")
            return vram
        else:
            print("[INFO] No GPU found — will run on CPU (very slow for LLaVA)")
            return 0.0
    except ImportError:
        print("[WARN] torch not installed")
        return 0.0


def install_deps():
    print("\n[STEP] Installing VLM dependencies...")
    packages = [
        "transformers>=4.37.0",
        "accelerate>=0.26.0",
        "bitsandbytes>=0.41.0",   # for 4-bit quantization (saves VRAM)
        "sentencepiece",
        "pillow",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    print("[OK] Dependencies installed")


def download_blip2():
    print("\n[STEP] Downloading BLIP-2 (Salesforce/blip2-flan-t5-xl) ...")
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print("  Downloading processor...")
    Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    print("  Downloading model weights (~16 GB, this may take a while)...")
    Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        load_in_8bit=True,         # 8-bit quantization — halves memory usage
        device_map="auto",
    )
    print("[OK] BLIP-2 downloaded and ready")


def download_llava():
    print("\n[STEP] Downloading LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) ...")
    import torch
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from transformers import BitsAndBytesConfig
    print("  Downloading processor...")
    LlavaNextProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print("  Downloading model weights (~28 GB, this will take a while)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("[OK] LLaVA-1.5-7B downloaded and ready")


def check_installation():
    print("\n[CHECK] VLM Installation Status")
    print("=" * 45)

    # Check transformers
    try:
        import transformers
        print(f"[OK] transformers {transformers.__version__}")
    except ImportError:
        print("[FAIL] transformers not installed — run: pip install transformers")

    # Check BLIP-2
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache("Salesforce/blip2-flan-t5-xl", "config.json")
        if path:
            print("[OK] BLIP-2 model cached locally")
        else:
            print("[MISS] BLIP-2 not cached — run setup_vlm.py --model blip2")
    except Exception:
        print("[MISS] BLIP-2 not cached")

    # Check LLaVA
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache("llava-hf/llava-1.5-7b-hf", "config.json")
        if path:
            print("[OK] LLaVA-1.5-7B model cached locally")
        else:
            print("[MISS] LLaVA not cached — run setup_vlm.py --model llava")
    except Exception:
        print("[MISS] LLaVA not cached")

    # Check GPU
    check_gpu()


def main():
    parser = argparse.ArgumentParser(description="PolypNet VLM Setup")
    parser.add_argument("--model", choices=["blip2", "llava", "auto"], default="auto",
                        help="Which VLM to download (default: auto-select based on VRAM)")
    parser.add_argument("--check", action="store_true",
                        help="Only check installation status, no download")
    args = parser.parse_args()

    if args.check:
        check_installation()
        return

    install_deps()
    vram = check_gpu()

    model = args.model
    if model == "auto":
        model = "llava" if vram >= 10 else "blip2"
        print(f"\n[AUTO] Selected model: {model} (VRAM={vram:.1f} GB)")

    if model == "blip2":
        download_blip2()
    else:
        download_llava()

    print("\n[DONE] VLM setup complete!")
    print("       The pipeline will automatically use the downloaded model.")
    print("       Run: python3 test_terminal_pipeline.py")


if __name__ == "__main__":
    main()
