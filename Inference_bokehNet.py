import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

try:
    import depth_pro
    HAS_DEPTH_PRO = True
except ImportError:
    HAS_DEPTH_PRO = False

MODEL_ID = "black-forest-labs/FLUX.1-dev"
BOKEH_LORA_DIR = "."
BOKEH_WEIGHT_NAME = "bokehNet.safetensors"
MAX_COC = 100.0  # defocus normalization

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

        final_w = max((new_w // 16) * 16, 16)
        final_h = max((new_h // 16) * 16, 16)
        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        return img.crop((left, top, left + final_w, top + final_h))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16
    if final_w == w and final_h == h:
        return img
    return img.resize((final_w, final_h), Image.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description="Genfocus BokehNet CLI Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the All-In-Focus input image")
    parser.add_argument("--output", "-o", type=str, default="output_bokeh.png", help="Path to save the output image")
    parser.add_argument("--k_value", "-k", type=float, default=15.0, help="Blur strength K (default: 20.0)")
    
    parser.add_argument("--mask", "-m", type=str, default=None, help="Path to the in-focus mask image")
    parser.add_argument("--point", "-p", type=str, default=None, help="Focus point 'x,y' on the ORIGINAL image (e.g., '512,300')")
    
    
    parser.add_argument("--depth", "-d", type=str, default=None, help="Path to pre-computed depth (.npy). If None, Depth Pro is used.")
    
    
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling tricks (NO_TILED_DENOISE=True)")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (default: 28)")
    parser.add_argument("--long_side", type=int, default=0, help="Resize long side (0 = keep original)")
    
    args = parser.parse_args()

    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Input image not found: {args.input}")
    if args.mask and not os.path.exists(args.mask):
        raise FileNotFoundError(f"❌ Mask image not found: {args.mask}")
    if args.depth and not os.path.exists(args.depth):
        raise FileNotFoundError(f"❌ Depth array not found: {args.depth}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"🚀 Device detected: {device}")

    
    print(f"🔄 Loading input: {args.input}")
    raw_img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = raw_img.size
    clean_input = resize_and_pad_image(raw_img, args.long_side)
    w, h = clean_input.size
    print(f"✅ Input processed size: {w}x{h}")

    
    if args.depth:
        print(f"🔄 Loading provided depth map: {args.depth}")
        depth_arr = np.load(args.depth).astype(np.float32)
        depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)
        safe_depth = np.where(depth_arr > 0.0, depth_arr, np.finfo(np.float32).max)
        disp = 1.0 / safe_depth
    else:
        if not HAS_DEPTH_PRO:
            raise ImportError("❌ 'depth_pro' is not installed, but --depth was not provided. Please provide a depth map or install depth_pro.")
        print("🔄 Running Depth Pro for dynamic depth prediction...")
        depth_model, depth_transform = depth_pro.create_model_and_transforms()
        depth_model.eval().to(device)
        
        img_t = depth_transform(clean_input).to(device)
        with torch.no_grad():
            pred = depth_model.infer(img_t, f_px=None)
        depth_arr = pred["depth"].cpu().numpy().squeeze()
        depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)
        safe_depth = np.where(depth_arr > 0.0, depth_arr, np.finfo(np.float32).max)
        disp = 1.0 / safe_depth
        print("✅ Depth predicted successfully.")

    
    if args.mask:
        print(f"🎯 Using Mask to determine focus point: {args.mask}")
        mask_img = Image.open(args.mask).convert("L").resize((w, h), Image.NEAREST)
        mask_np = np.array(mask_img)
        valid_disp = disp[mask_np > 0]
        if len(valid_disp) > 0:
            disp_focus = float(np.median(valid_disp))
        else:
            print("⚠️ Mask is empty! Defaulting to median depth of the entire image.")
            disp_focus = float(np.median(disp))
    elif args.point:
        px, py = map(int, args.point.split(','))
        
        tx = min(max(int(px * (w / orig_w)), 0), w - 1)
        ty = min(max(int(py * (h / orig_h)), 0), h - 1)
        print(f"🎯 Using Point to determine focus point: Original({px}, {py}) -> Resized({tx}, {ty})")
        disp_focus = float(disp[ty, tx])
    else:
        print("🎯 No Mask or Point provided. Defaulting to Center focus.")
        tx, ty = w // 2, h // 2
        disp_focus = float(disp[ty, tx])

    print(f"ℹ️ Calculated disp_focus: {disp_focus:.4f}")

    
    disp_minus_focus = disp - np.float32(disp_focus)
    defocus_abs = np.abs(args.k_value * disp_minus_focus)
    defocus_t = torch.from_numpy(defocus_abs).unsqueeze(0).float()
    cond_map = (defocus_t / MAX_COC).clamp(0, 1).repeat(3, 1, 1).unsqueeze(0)

   
    print("🔄 Loading FLUX pipeline...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        pipe_flux.to("cuda")

    print("🔄 Loading Bokeh LoRA...")
    try:
        pipe_flux.load_lora_weights(BOKEH_LORA_DIR, weight_name=BOKEH_WEIGHT_NAME, adapter_name="bokeh")
        pipe_flux.set_adapters(["bokeh"])
    except Exception as e:
        print(f"❌ Failed to load Bokeh LoRA: {e}")
        return

   
    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(args.disable_tiling) or force_no_tile
    print(f"🚀 Starting Generation (K={args.k_value})... (Tiled Denoise Disabled: {no_tiled_denoise})")
    
    cond_img = Condition(clean_input, "bokeh")
    cond_dmf = Condition(cond_map, "bokeh", [0, 0], 1.0, No_preprocess=True)

    seed_everything(42)
    gen = torch.Generator(device=device).manual_seed(1234)

    with torch.no_grad():
        generated_bokeh = generate(
            pipe_flux,
            height=h,
            width=w,
            prompt="an excellent photo with a large aperture",
            num_inference_steps=args.steps,
            conditions=[cond_img, cond_dmf],
            guidance_scale=1.0,
            kv_cache=False,
            generator=gen,
            NO_TILED_DENOISE=no_tiled_denoise,
        ).images[0]

    
    generated_bokeh.save(args.output)
    print(f"✅ Successfully saved bokeh image to: {args.output}")

if __name__ == "__main__":
    main()