import os
import argparse
import numpy as np
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
DEBLUR_WEIGHT_NAME = "deblurNet.safetensors"
DEBLUR_PROMPT = "a sharp photo with everything in focus"

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


def build_tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    max_stride = tile_size - overlap
    if max_stride <= 0:
        raise ValueError("❌ pixel tile overlap must be smaller than pixel tile size")

    final_start = length - tile_size
    tile_count = int(np.ceil(final_start / max_stride)) + 1
    starts = np.rint(np.linspace(0, final_start, tile_count)).astype(int).tolist()
    return sorted(dict.fromkeys(starts))


def build_blend_ramp(overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones(0, dtype=np.float32)

    steps = np.arange(1, overlap + 1, dtype=np.float32)
    return np.sin((steps / (overlap + 1)) * (np.pi / 2.0)) ** 2


def build_feather_mask(
    tile_width: int,
    tile_height: int,
    left_overlap: int,
    right_overlap: int,
    top_overlap: int,
    bottom_overlap: int,
) -> np.ndarray:
    weight_x = np.ones(tile_width, dtype=np.float32)
    weight_y = np.ones(tile_height, dtype=np.float32)

    left_overlap = min(left_overlap, tile_width - 1)
    right_overlap = min(right_overlap, tile_width - 1)
    top_overlap = min(top_overlap, tile_height - 1)
    bottom_overlap = min(bottom_overlap, tile_height - 1)

    if left_overlap > 0:
        weight_x[:left_overlap] = np.minimum(weight_x[:left_overlap], build_blend_ramp(left_overlap))
    if right_overlap > 0:
        weight_x[-right_overlap:] = np.minimum(weight_x[-right_overlap:], build_blend_ramp(right_overlap)[::-1])

    if top_overlap > 0:
        weight_y[:top_overlap] = np.minimum(weight_y[:top_overlap], build_blend_ramp(top_overlap))
    if bottom_overlap > 0:
        weight_y[-bottom_overlap:] = np.minimum(weight_y[-bottom_overlap:], build_blend_ramp(bottom_overlap)[::-1])

    return np.outer(weight_y, weight_x)


def run_deblur_image(
    pipe_flux: FluxPipeline,
    image: Image.Image,
    steps: int,
    no_tiled_denoise: bool,
    seed: int,
) -> Image.Image:
    w, h = image.size
    cond = Condition(image, "deblurring", [0, 0], 1.0)
    seed_everything(seed)
    execution_device = get_pipeline_execution_device(pipe_flux, torch.device("cpu"))
    generator = torch.Generator(device=execution_device).manual_seed(seed)
    return generate(
        pipe_flux,
        height=h,
        width=w,
        prompt=DEBLUR_PROMPT,
        num_inference_steps=steps,
        conditions=[cond],
        generator=generator,
        NO_TILED_DENOISE=no_tiled_denoise,
    ).images[0]


def run_pixel_tiled_deblur(
    pipe_flux: FluxPipeline,
    image: Image.Image,
    steps: int,
    tile_size: int,
    overlap: int,
    seed: int,
) -> Image.Image:
    full_w, full_h = image.size
    if full_w < tile_size or full_h < tile_size:
        print("🧩 Pixel tiling skipped because the processed image is smaller than the requested tile size.")
        return run_deblur_image(pipe_flux, image, steps, no_tiled_denoise=True, seed=seed)

    x_starts = build_tile_starts(full_w, tile_size, overlap)
    y_starts = build_tile_starts(full_h, tile_size, overlap)
    tile_total = len(x_starts) * len(y_starts)

    print(
        f"🧩 Pixel-space tiling enabled: tile size {tile_size}, overlap {overlap}, total tiles {tile_total}, steps per tile {steps}, seed per tile {seed}"
    )

    image_sum = np.zeros((full_h, full_w, 3), dtype=np.float32)
    image_weight = np.zeros((full_h, full_w, 1), dtype=np.float32)

    tile_idx = 0
    for y_idx, top in enumerate(y_starts):
        for x_idx, left in enumerate(x_starts):
            tile_idx += 1
            right = left + tile_size
            bottom = top + tile_size
            print(f"   ► Pixel tile {tile_idx}/{tile_total}: x={left}:{right}, y={top}:{bottom}")

            tile_input = image.crop((left, top, right, bottom))
            tile_output = run_deblur_image(
                pipe_flux,
                tile_input,
                steps=steps,
                no_tiled_denoise=True,
                seed=seed,
            )

            tile_arr = np.asarray(tile_output, dtype=np.float32)
            left_overlap = 0 if x_idx == 0 else max((x_starts[x_idx - 1] + tile_size) - left, 0)
            right_overlap = 0 if x_idx == len(x_starts) - 1 else max((left + tile_size) - x_starts[x_idx + 1], 0)
            top_overlap = 0 if y_idx == 0 else max((y_starts[y_idx - 1] + tile_size) - top, 0)
            bottom_overlap = 0 if y_idx == len(y_starts) - 1 else max((top + tile_size) - y_starts[y_idx + 1], 0)
            weight = build_feather_mask(
                tile_width=tile_size,
                tile_height=tile_size,
                left_overlap=left_overlap,
                right_overlap=right_overlap,
                top_overlap=top_overlap,
                bottom_overlap=bottom_overlap,
            )[..., None]

            image_sum[top:bottom, left:right] += tile_arr * weight
            image_weight[top:bottom, left:right] += weight

    merged = image_sum / np.clip(image_weight, 1e-6, None)
    return Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description="Genfocus DeblurNet CLI Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", "-o", type=str, default="output_deblur.png", help="Path to save the output image")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling tricks (NO_TILED_DENOISE=True)")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (default: 28)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Pixel-space tiling reuses this exact seed for every tile.")
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
    parser.add_argument(
        "--tile_mode",
        choices=["latent", "pixel"],
        default="latent",
        help="Tiling strategy. 'latent' uses the existing latent-space tiled denoise path. 'pixel' deblurs independent RGB tiles before stitching.",
    )
    parser.add_argument(
        "--pixel_tile_size",
        type=int,
        default=512,
        help="Tile size in pixels for --tile_mode pixel. Should be a multiple of 16.",
    )
    parser.add_argument(
        "--pixel_tile_overlap",
        type=int,
        default=0,
        help="Tile overlap in pixels for --tile_mode pixel. Use 0 to isolate per-tile generation quality without feather blending.",
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Input image not found: {args.input}")

    if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
        print(f"❌ Warning: {DEBLUR_WEIGHT_NAME} not found in {DEBLUR_LORA_PATH}")

    device, dtype = resolve_torch_device(args.gpu_id)
    print(f"🚀 Device detected: {format_device_label(device, args.gpu_id)}")


    print(f"🔄 Loading and preprocessing image: {args.input}")
    raw_img = Image.open(args.input).convert("RGB")
    clean_input_processed = resize_and_pad_image(raw_img, args.long_side)
    w, h = clean_input_processed.size
    print(f"✅ Processed image size: {w}x{h}")
    print(f"ℹ️ Inference steps: {args.steps}")
    print(f"ℹ️ Seed: {args.seed}")

    if args.pixel_tile_size < 16 or args.pixel_tile_size % 16 != 0:
        raise ValueError("❌ --pixel_tile_size must be a multiple of 16 and at least 16")
    if args.pixel_tile_overlap < 0 or args.pixel_tile_overlap >= args.pixel_tile_size:
        raise ValueError("❌ --pixel_tile_overlap must be non-negative and smaller than --pixel_tile_size")

    print("🔄 Loading FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    
    print("🔄 Loading Deblur LoRA...")
    try:
        pipe_flux.load_lora_weights(
            DEBLUR_LORA_PATH,
            weight_name=DEBLUR_WEIGHT_NAME,
            adapter_name="deblurring",
        )
        pipe_flux.set_adapters(["deblurring"])
    except Exception as e:
        print(f"❌ Failed to load Deblur LoRA: {e}")
        return
    
    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(args.disable_tiling) or force_no_tile
    print(
        f"🚀 Starting Generation... (Tile Mode: {args.tile_mode}, Tiled Denoise Disabled: {no_tiled_denoise})"
    )
    tiled_expected = args.tile_mode == "latent" and (not no_tiled_denoise) and will_use_tiled_denoise(w, h)

    def run_deblur(_active_mode: str):
        if args.tile_mode == "pixel":
            return run_pixel_tiled_deblur(
                pipe_flux,
                clean_input_processed,
                steps=args.steps,
                tile_size=args.pixel_tile_size,
                overlap=args.pixel_tile_overlap,
                seed=args.seed,
            )

        return run_deblur_image(
            pipe_flux,
            clean_input_processed,
            steps=args.steps,
            no_tiled_denoise=no_tiled_denoise,
            seed=args.seed,
        )

    deblurred_img, _ = run_with_flux_memory_policy(
        pipe=pipe_flux,
        requested_mode=args.offload_mode,
        device=device,
        gpu_id=args.gpu_id,
        width=min(w, args.pixel_tile_size) if args.tile_mode == "pixel" else w,
        height=min(h, args.pixel_tile_size) if args.tile_mode == "pixel" else h,
        condition_count=1,
        safety_margin_gb=args.vram_safety_margin_gb,
        tiled_expected=tiled_expected,
        run_fn=run_deblur,
    )

    deblurred_img.save(args.output)
    print(f"✅ Successfully saved deblurred image to: {args.output}")

if __name__ == "__main__":
    main()
