import os
import argparse
import torch
from PIL import Image

from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEBLUR_LORA_PATH = "."
DEBLUR_WEIGHT_NAME = "deblurNet.safetensors"

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
    parser = argparse.ArgumentParser(description="Genfocus DeblurNet CLI Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", "-o", type=str, default="output_deblur.png", help="Path to save the output image")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling tricks (NO_TILED_DENOISE=True)")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (default: 28)")
    parser.add_argument("--long_side", type=int, default=0, help="Resize long side (0 = keep original)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Input image not found: {args.input}")

    if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
        print(f"❌ Warning: {DEBLUR_WEIGHT_NAME} not found in {DEBLUR_LORA_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"🚀 Device detected: {device}")


    print(f"🔄 Loading and preprocessing image: {args.input}")
    raw_img = Image.open(args.input).convert("RGB")
    clean_input_processed = resize_and_pad_image(raw_img, args.long_side)
    w, h = clean_input_processed.size
    print(f"✅ Processed image size: {w}x{h}")


    print("🔄 Loading FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    
    if device == "cuda":
        pipe_flux.to("cuda")


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
    
    cond0 = Condition(clean_input_processed, "deblurring", [0, 0], 1.0)
    seed_everything(42)


    deblurred_img = generate(
        pipe_flux,
        height=h,
        width=w,
        prompt="a sharp photo with everything in focus",
        num_inference_steps=args.steps,
        conditions=[cond0],
        NO_TILED_DENOISE=no_tiled_denoise,
    ).images[0]

    deblurred_img.save(args.output)
    print(f"✅ Successfully saved deblurred image to: {args.output}")

if __name__ == "__main__":
    main()