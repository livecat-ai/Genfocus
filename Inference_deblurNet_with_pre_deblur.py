import os
import argparse
import torch
from PIL import Image

from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything
from Genfocus.runtime import (
    format_device_label,
    get_pipeline_execution_device,
    resolve_torch_device,
    run_with_flux_memory_policy,
    will_use_tiled_denoise,
)

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEBLUR_LORA_PATH = "."
DEBLUR_WEIGHT_NAME = "deblurNet_with_pre_deblur.safetensors"

def resize_and_pad_image(img: Image.Image, target_long_side: int) -> Image.Image:
    w, h = img.size

    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)

        if w >= h:
            new_w = target_max
            scale = target_max / w
            new_h = int(h * scale)
        else:
            new_h = target_max
            scale = target_max / h
            new_w = int(w * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16

        final_w = max(final_w, 16)
        final_h = max(final_h, 16)

        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        right = left + final_w
        bottom = top + final_h

        return img.crop((left, top, right, bottom))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16

    if final_w == w and final_h == h:
        return img

    return img.resize((final_w, final_h), Image.LANCZOS)

def main():
    parser = argparse.ArgumentParser(description="Genfocus DeblurNet CLI Inference (with pre-deblur support)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input blurry image")
    parser.add_argument("--pre_deblur_input", "-p", type=str, default=None, help="Path to the pre-deblurred image")
    parser.add_argument("--output", "-o", type=str, default="output_deblur.png", help="Path to save the output image")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling tricks (NO_TILED_DENOISE=True)")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (default: 28)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DeblurNet generation.")
    parser.add_argument("--long_side", type=int, default=0, help="Resize long side (0 = keep original)")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device index to use when running on GPU")
    parser.add_argument(
        "--offload_mode",
        choices=["auto", "sequential", "model", "none"],
        default="auto",
        help="CUDA offload mode. 'auto' uses the most aggressive safe mode based on available VRAM.",
    )
    parser.add_argument(
        "--vram_safety_margin_gb",
        type=float,
        default=0.5,
        help="VRAM headroom reserved when --offload_mode=auto selects a mode.",
    )
    
    args = parser.parse_args()


    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Input image not found: {args.input}")

    if args.pre_deblur_input and not os.path.exists(args.pre_deblur_input):
        raise FileNotFoundError(f"❌ Pre-deblur image not found: {args.pre_deblur_input}")

    if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
        print(f"❌ Warning: {DEBLUR_WEIGHT_NAME} not found in {DEBLUR_LORA_PATH}")

    device, dtype = resolve_torch_device(args.gpu_id)
    print(f"🚀 Device detected: {format_device_label(device, args.gpu_id)}")

    print(f"🔄 Loading and preprocessing main input: {args.input}")
    raw_img = Image.open(args.input).convert("RGB")
    clean_input_processed = resize_and_pad_image(raw_img, args.long_side)
    w, h = clean_input_processed.size
    print(f"✅ Main image processed size: {w}x{h}")
    print(f"ℹ️ Inference steps: {args.steps}")
    print(f"ℹ️ Seed: {args.seed}")


    if args.pre_deblur_input:
        print(f"🔄 Loading pre-deblur input: {args.pre_deblur_input}")
        pre_img = Image.open(args.pre_deblur_input).convert("RGB")

        pre_img_processed = pre_img.resize((w, h), Image.LANCZOS)
    else:
        print("⚠️ No pre-deblur input provided. Using a black image as default.")
        pre_img_processed = Image.new("RGB", (w, h), (0, 0, 0))


    print("🔄 Loading FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    print("🔄 Loading Deblur LoRA...")
    try:
        pipe_flux.load_lora_weights(DEBLUR_LORA_PATH, weight_name=DEBLUR_WEIGHT_NAME, adapter_name="deblurring")
        pipe_flux.set_adapters(["deblurring"])
    except Exception as e:
        print(f"❌ Failed to load Deblur LoRA: {e}")
        return


    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(args.disable_tiling) or force_no_tile
    print(f"🚀 Starting Generation... (Tiled Denoise Disabled: {no_tiled_denoise})")
    tiled_expected = not no_tiled_denoise and will_use_tiled_denoise(w, h)
    

    cond0 = Condition(pre_img_processed, "deblurring", [0, 32], 1.0)
    cond1 = Condition(clean_input_processed, "deblurring", [0, 0], 1.0)

    def run_deblur(_active_mode: str):
        seed_everything(args.seed)
        execution_device = get_pipeline_execution_device(pipe_flux, device)
        gen = torch.Generator(device=execution_device).manual_seed(args.seed)
        with torch.no_grad():
            return generate(
                pipe_flux,
                height=h,
                width=w,
                prompt="a sharp photo with everything in focus",
                num_inference_steps=args.steps,
                conditions=[cond0, cond1],
                generator=gen,
                NO_TILED_DENOISE=no_tiled_denoise,
            ).images[0]

    deblurred_img, _ = run_with_flux_memory_policy(
        pipe=pipe_flux,
        requested_mode=args.offload_mode,
        device=device,
        gpu_id=args.gpu_id,
        width=w,
        height=h,
        condition_count=2,
        safety_margin_gb=args.vram_safety_margin_gb,
        tiled_expected=tiled_expected,
        run_fn=run_deblur,
    )

    deblurred_img.save(args.output)
    print(f"✅ Successfully saved deblurred image to: {args.output}")

if __name__ == "__main__":
    main()
