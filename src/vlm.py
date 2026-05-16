"""
PolypNet VLM Integration
========================
Priority order for clinical description generation:
  1. LLaVA-1.5-7B  (best, ~14 GB VRAM)
  2. BLIP-2 Flan-T5-XL (lighter, ~8 GB VRAM)
  3. Structured text fallback (no GPU required)

Run `python3 setup_vlm.py` once to download the model weights.
"""

import torch
from PIL import Image
import numpy as np
from typing import Optional, Dict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# BLIP-2 Generator (lighter alternative to LLaVA)
# ─────────────────────────────────────────────────────────────────────────────

class Blip2Generator:
    """
    BLIP-2 based clinical description generator.
    Smaller and faster than LLaVA while still providing VLM-quality output.
    Model: Salesforce/blip2-flan-t5-xl
    """

    MODEL_ID = "Salesforce/blip2-flan-t5-xl"

    def __init__(self, device: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            from huggingface_hub import try_to_load_from_cache

            # Only load if already cached (don't auto-download in pipeline)
            cached = try_to_load_from_cache(self.MODEL_ID, "config.json")
            if not cached:
                raise FileNotFoundError(
                    f"BLIP-2 not cached. Run: python3 setup_vlm.py --model blip2"
                )

            print(f"[INFO] Loading BLIP-2 from cache...")
            self.processor = Blip2Processor.from_pretrained(self.MODEL_ID)
            load_kw = {"device_map": "auto"}
            if self.device == "cuda":
                load_kw["load_in_8bit"] = True     # save VRAM
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.MODEL_ID, **load_kw
            )
            print(f"[OK] BLIP-2 loaded on {self.device}")

        except Exception as e:
            print(f"[WARN] BLIP-2 not available: {e}")

    @property
    def available(self) -> bool:
        return self.model is not None

    def generate_description(self, image_path: str, mask_path: str) -> str:
        """Generate clinical description from colonoscopy image + mask overlay."""
        overlay = _create_overlay(image_path, mask_path)

        prompts = [
            ("Question: Describe this colonoscopy image. What polyp characteristics "
             "can you see — size, shape, surface texture, and morphology? "
             "Answer:"),
            ("Question: Based on the highlighted green region (polyp), "
             "what clinical follow-up would you recommend? Answer:"),
        ]

        responses = []
        for prompt in prompts:
            inputs = self.processor(
                images=overlay,
                text=prompt,
                return_tensors="pt"
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=256)
            responses.append(self.processor.decode(out[0], skip_special_tokens=True))

        return _format_blip2_report(responses, image_path, mask_path)


# ─────────────────────────────────────────────────────────────────────────────
# LLaVA-1.5-7B Generator (highest quality)
# ─────────────────────────────────────────────────────────────────────────────

class LlavaGenerator:
    """
    LLaVA-1.5-7B based clinical description generator.
    Best quality output. Requires ~14 GB VRAM or ~28 GB RAM.
    Run `python3 setup_vlm.py --model llava` to download weights first.
    """

    MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    def __init__(self, model_name: str = None, device: str = "cuda"):
        self.model_name = model_name or self.MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

        print(f"\n[INFO] Loading LLaVA model: {self.model_name}")
        print(f"   Device: {self.device}")

        try:
            # llava-1.5-7b-hf uses LlavaProcessor/LlavaForConditionalGeneration.
            # Using LlavaNext* classes causes a KeyError on 'image_sizes' because
            # LlavaNextProcessor expects image_sizes from LlavaNextImageProcessor,
            # but this checkpoint ships a plain CLIPImageProcessor.
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            from huggingface_hub import try_to_load_from_cache

            # Only load if cached
            cached = try_to_load_from_cache(self.model_name, "config.json")
            if not cached:
                raise FileNotFoundError(
                    f"LLaVA not cached. Run: python3 setup_vlm.py --model llava"
                )

            self.processor = LlavaProcessor.from_pretrained(self.model_name)

            load_kw = {"device_map": "auto", "low_cpu_mem_usage": True}
            if self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                                    bnb_4bit_compute_dtype=torch.float16)
                    load_kw["quantization_config"] = bnb_config
                except Exception:
                    # bitsandbytes not available, use fp16 instead
                    load_kw["torch_dtype"] = torch.float16
            else:
                load_kw["torch_dtype"] = torch.float32

            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name, **load_kw
            )
            print("[OK] LLaVA model loaded successfully")

        except ImportError:
            print("[WARN] transformers not installed. Run: pip install transformers accelerate")
            self.model = None
            self.processor = None
        except Exception as e:
            print(f"[WARN] LLaVA not available: {e}")
            self.model = None
            self.processor = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def create_overlay(self, image_path: str, mask_path: str,
                       output_path: Optional[str] = None) -> Image.Image:
        return _create_overlay(image_path, mask_path, output_path)

    def generate_description(self, image_path: str, mask_path: str,
                              prompt_template: Optional[str] = None) -> str:
        if not self.available:
            return self._generate_fallback_description(image_path, mask_path)

        overlay = _create_overlay(image_path, mask_path)

        if prompt_template is None:
            # LLaVA-1.5 uses "USER: <image>\n...\nASSISTANT:" format.
            # The [INST]/[/INST] format is for LLaVA-Next (Mistral base).
            prompt_template = (
                "USER: <image>\n"
                "You are an expert gastroenterologist analyzing a colonoscopy image "
                "with a highlighted polyp region (shown in green overlay).\n\n"
                "Please provide a detailed clinical assessment including:\n"
                "1. Polyp size estimation (small <5mm, medium 5-10mm, large >10mm)\n"
                "2. Polyp shape and morphology (sessile, pedunculated, flat)\n"
                "3. Surface characteristics (smooth, irregular, ulcerated)\n"
                "4. Location and distribution\n"
                "5. Recommended follow-up actions based on findings\n\n"
                "Provide your assessment in a professional clinical format.\n"
                "ASSISTANT:"
            )

        # LlavaProcessor (v1.5) does not produce image_sizes, so no filtering needed.
        inputs = self.processor(
            text=prompt_template,
            images=overlay,
            return_tensors="pt"
        ).to(self.device)

        print("\n[INFO] Generating clinical description with LLaVA...")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        description = self.processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt prefix — LLaVA-1.5 echoes the full input in its output
        if "ASSISTANT:" in description:
            description = description.split("ASSISTANT:")[-1].strip()

        return description

    def _generate_fallback_description(self, image_path: str, mask_path: str) -> str:
        """Used when LLaVA model weights are not loaded."""
        return get_best_description(image_path, mask_path)

    def batch_generate(self, image_mask_pairs: list, output_dir: str) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        descriptions = {}
        print(f"\n[INFO] Generating descriptions for {len(image_mask_pairs)} images...")
        for i, (img_path, mask_path) in enumerate(image_mask_pairs, 1):
            print(f"\nProcessing {i}/{len(image_mask_pairs)}: {Path(img_path).name}")
            try:
                description = self.generate_description(img_path, mask_path)
                img_name = Path(img_path).stem
                desc_file = output_path / f"{img_name}_description.txt"
                with open(desc_file, 'w') as f:
                    f.write(f"Image: {img_path}\nMask: {mask_path}\n")
                    f.write(f"\n{'='*80}\n")
                    f.write(description)
                descriptions[img_name] = description
                print(f"[OK] Description saved to {desc_file}")
            except Exception as e:
                print(f"[WARN] Error processing {img_path}: {e}")
                descriptions[Path(img_path).stem] = f"Error: {str(e)}"

        print(f"\n[OK] Batch processing complete. {len(descriptions)} descriptions generated.")
        return descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point — tries VLMs in priority order
# ─────────────────────────────────────────────────────────────────────────────

def get_best_description(image_path: str, mask_path: str) -> str:
    """
    Try VLMs in priority order:
      1. LLaVA-1.5-7B (if cached + GPU available)
      2. BLIP-2       (if cached)
      3. Structured template fallback
    """
    # 1. Try LLaVA
    try:
        llava = LlavaGenerator()
        if llava.available:
            print("[VLM] Using LLaVA-1.5-7B for clinical report")
            return llava.generate_description(image_path, mask_path)
    except Exception as e:
        print(f"[WARN] LLaVA failed: {e}")

    # 2. Try BLIP-2
    try:
        blip = Blip2Generator()
        if blip.available:
            print("[VLM] Using BLIP-2 for clinical report")
            return blip.generate_description(image_path, mask_path)
    except Exception as e:
        print(f"[WARN] BLIP-2 failed: {e}")

    # 3. Structured fallback (always works)
    print("[VLM] Using structured text fallback")
    return _structured_fallback(image_path, mask_path)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_overlay(image_path: str, mask_path: str,
                    output_path: Optional[str] = None) -> Image.Image:
    """Overlay segmentation mask (green) on original image."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    if Path(mask_path).exists():
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)
        mask_np = np.array(mask)
        overlay = image_np.copy()
        overlay[mask_np > 127] = (
            overlay[mask_np > 127] * 0.6 +
            np.array([0, 255, 0]) * 0.4
        ).astype(np.uint8)
    else:
        overlay = image_np

    result = Image.fromarray(overlay.astype(np.uint8))
    if output_path:
        result.save(output_path)
    return result


def _format_blip2_report(responses: list, image_path: str, mask_path: str) -> str:
    """Format BLIP-2 multi-turn responses into a clinical report."""
    import cv2
    sep = "=" * 55
    dash = "-" * 53

    coverage = 0.0
    if Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            coverage = float(np.sum(mask > 127)) / mask.size * 100

    r1 = responses[0] if len(responses) > 0 else "N/A"
    r2 = responses[1] if len(responses) > 1 else "N/A"

    return (
        f"POLYPNET - VLM CLINICAL ASSESSMENT (BLIP-2)\n"
        f"{sep}\n\n"
        f"{dash}\n"
        f"1. MORPHOLOGY & CHARACTERISTICS\n"
        f"{dash}\n"
        f"   {r1}\n\n"
        f"{dash}\n"
        f"2. CLINICAL RECOMMENDATIONS\n"
        f"{dash}\n"
        f"   {r2}\n\n"
        f"{dash}\n"
        f"3. SEGMENTATION METRICS\n"
        f"{dash}\n"
        f"   Polyp area coverage: {coverage:.2f}% of image\n\n"
        f"{dash}\n"
        f"DISCLAIMER\n"
        f"{dash}\n"
        f"AI-generated preliminary report. Must be reviewed\n"
        f"by a qualified gastroenterologist.\n"
    )


def _structured_fallback(image_path: str, mask_path: str) -> str:
    """Structured clinical report without any VLM (always available)."""
    import cv2
    img = cv2.imread(image_path)
    h, w = img.shape[:2] if img is not None else (0, 0)

    coverage = 0.0
    size_cat = "undetermined"
    contour_count = 0

    if Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            coverage = float(np.sum(mask > 127)) / mask.size * 100
            contours, _ = cv2.findContours(
                (mask > 127).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_count = len(contours)
            if contours:
                lc = max(contours, key=cv2.contourArea)
                _, _, cw, ch = cv2.boundingRect(lc)
                avg = (cw + ch) / 2
                size_cat = ("small (<5 mm)" if avg < 50 else
                            "medium (5–10 mm)" if avg < 100 else
                            "large (>10 mm)")

    sep  = "=" * 55
    dash = "-" * 53
    return (
        f"POLYPNET - AUTOMATED CLINICAL ASSESSMENT\n"
        f"{sep}\n\n"
        f"SYSTEM:  PolypNet v1.0 (Att-UNet + YOLO + CNN Ensemble)\n\n"
        f"{dash}\n"
        f"1. SEGMENTATION ANALYSIS\n"
        f"{dash}\n"
        f"   Polyp area coverage : {coverage:.2f}% of image\n"
        f"   Estimated size      : {size_cat}\n"
        f"   Detected regions    : {contour_count}\n"
        f"   Image dimensions    : {w}x{h} px\n\n"
        f"{dash}\n"
        f"2. CLINICAL RECOMMENDATIONS\n"
        f"{dash}\n"
        f"   - Pathological examination recommended.\n"
        f"   - Follow-up colonoscopy as per standard guidelines.\n"
        f"   - Document lesion size, location in patient record.\n\n"
        f"{dash}\n"
        f"DISCLAIMER\n"
        f"{dash}\n"
        f"AI-generated preliminary report. Must be reviewed\n"
        f"by a qualified gastroenterologist before any\n"
        f"clinical decision is made.\n"
        f"\nNOTE: Install VLM for richer natural-language reports:\n"
        f"      python3 setup_vlm.py\n"
    )


def test_llava_installation() -> bool:
    """Test if LLaVA dependencies are properly installed."""
    try:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        print("[OK] LLaVA dependencies are installed")
        return True
    except ImportError as e:
        print(f"[FAIL] LLaVA dependencies missing: {e}")
        print("\nTo install, run:")
        print("  pip install transformers accelerate pillow torch")
        return False
