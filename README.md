# Generative Refocusing: Flexible Defocus Control from a Single Image

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=green)](https://generative-refocusing.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2512.16923)
[![YouTube Tutorial](https://img.shields.io/badge/YouTube-Tutorial-red?logo=youtube&logoColor=red)](https://youtu.be/CMh_jGDl-RE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/nycu-cplab/Genfocus-Demo)

<div align="center">
  <img src="./assets/demo_vid.gif" width="50%" alt="Demo Video">
</div>

</div>

---

## ⚡ Quick Start

Follow the steps below to set up the environment and run the inference demo.

### 1. Installation

Clone the repository:

```bash
git clone git@github.com:rayray9999/Genfocus.git
cd Genfocus
```

Environment setup:

```bash
conda create -n Genfocus python=3.12
conda activate Genfocus
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 2. Download Weights

You can download the pre-trained models using the following commands. Ensure you are in the `Genfocus` root directory.

```bash
# 1. Download main models to the root directory
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/bokehNet.safetensors
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/deblurNet.safetensors

# 2. Setup checkpoints directory and download auxiliary model
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/checkpoints/depth_pro.pt
cd ..
```

### 3. Run Gradio Demo

Launch the interactive web interface locally:  
> **Note:** This project uses [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). You must request access and authenticate locally before running the demo.  
> ⚠️ **VRAM warning:** GPU memory usage can be high. This fork adds automatic FLUX offloading and explicit GPU/VRAM controls to make inference more practical on lower-memory cards.  
> **Tip (Advanced Settings):** If you want lower memory use or faster inference, try lowering **num_inference_steps**, resizing the input with **Long Side**, or using the environment variables shown below.

```bash
python demo.py
```

The demo will be accessible at `http://127.0.0.1:7860` in your browser.

Optional demo runtime controls:

```bash
GENFOCUS_GPU_ID=0 \
GENFOCUS_OFFLOAD_MODE=auto \
GENFOCUS_VRAM_SAFETY_MARGIN_GB=0.5 \
python demo.py
```

`GENFOCUS_OFFLOAD_MODE` accepts `auto`, `sequential`, `model`, or `none`. Use `auto` unless you need to force a specific placement strategy.

---

## ⚡ Inference (Command Line)

You can also run inference directly via the command line for our different models.

### 1. DeblurNet

Restore sharp details from blurry images.
> ⚠️ **Update Notice:** If you downloaded the weights before our recent inference command updates, please re-download the new `deblurNet.safetensors` to ensure best performance.

```bash
python Inference_deblurNet.py \
  --input inference_example/Blurry_example.jpg \
  --output Deblurred_output.png
```

* **`--input`** / **`-i`**: Path to the input blurry image. 
* **`--output`** / **`-o`**: Path to save the output image. 

Low-memory example:

```bash
python Inference_deblurNet.py \
  --input inference_example/Blurry_example.jpg \
  --output Deblurred_output.png \
  --offload_mode auto \
  --vram_safety_margin_gb 0.5 \
  --steps 20
```

High-resolution pixel-tiled example:

```bash
python Inference_deblurNet.py \
  --input inference_example/Blurry_example.jpg \
  --output Deblurred_output_pixel_tiles.png \
  --long_side 1728 \
  --tile_mode pixel \
  --pixel_tile_size 512 \
  --pixel_tile_overlap 0 \
  --offload_mode auto
```

For higher-resolution DeblurNet inference, `--tile_mode pixel` runs independent RGB crops through the model before stitching. This can produce better detail than latent-space tiling because each tile is encoded independently instead of slicing one full-image latent after VAE encoding.

---

### 2. BokehNet 

Add realistic bokeh effects to an All-In-Focus (AIF) image using a depth/focus mask.

```bash
python Inference_bokehNet.py \
  --input inference_example/AIF_example.png \
  --mask inference_example/AIF_mask.png \
  --depth inference_example/AIF_example_pred.npy \
  --k_value 15 \
  --output Bokeh_output.png 
```

* **`--input`** / **`-i`**: Path to the All-In-Focus input image. 
* **`--mask`** / **`-m`**: Path to the in-focus mask image.
* **`--point`** / **`-p`**: Focus point `x,y` on the ORIGINAL image (e.g., `512,300`).
* **`--depth`** / **`-d`**: Path to a pre-computed depth map (`.npy` file). If not provided, Depth Pro is automatically used.
* **`--k_value`** / **`-k`**: Blur strength K. 
* **`--output`** / **`-o`**: Path to save the output image. 

Low-memory example:

```bash
python Inference_bokehNet.py \
  --input inference_example/AIF_example.png \
  --mask inference_example/AIF_mask.png \
  --depth inference_example/AIF_example_pred.npy \
  --k_value 15 \
  --output Bokeh_output.png \
  --offload_mode auto \
  --long_side 1536
```

---

### 3. DeblurNet variant (with pre-deblur module)

A variant of DeblurNet that utilizes a pre-deblurring module for heavily degraded images.

**Note:** Please download the specific weight for this variant before running the inference:
```bash
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/deblurNet_with_pre_deblur.safetensors
```

```bash
python Inference_deblurNet_with_pre_deblur.py \
  --input inference_example/Blurry_example.jpg \
  --pre_deblur_input inference_example/Blurry_example_pre_deblur.jpg \
  --output Deblurred_output_with_pre_deblur.png 
```

* **`--input`** / **`-i`**: Path to the input blurry image. 
* **`--pre_deblur_input`**: Path to the pre-processed/pre-deblurred image.
* **`--output`** / **`-o`**: Path to save the output image. 

### ⚙️ Runtime and Memory Arguments

These arguments are available on `Inference_deblurNet.py`, `Inference_bokehNet.py`, and `Inference_deblurNet_with_pre_deblur.py`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--disable_tiling` | Flag | `False` | Force disable tiling (`NO_TILED_DENOISE=True`). *Note: Tiling is auto-disabled if the shortest edge is < 512px.* |
| `--steps` | Integer | `28` | Number of inference steps. Higher steps usually yield better details but take longer. |
| `--seed` | Integer | `42` | Random seed. For DeblurNet pixel tiling, the same seed is reused for every tile. |
| `--long_side` | Integer | `0` | Resize the longest edge of the image (aspect ratio preserved, padded to multiple of 16). `0` keeps original size. |
| `--gpu_id` | Integer | `0` | CUDA device index to use when running on GPU. |
| `--offload_mode` | String | `auto` | FLUX memory policy. `auto` checks current free VRAM and tries the fastest viable mode, falling back if needed. `sequential` minimizes VRAM use, `model` uses more VRAM for better speed, and `none` keeps the full FLUX pipeline on GPU. |
| `--vram_safety_margin_gb` | Float | `0.5` | Headroom reserved by `auto` mode before it chooses a more aggressive GPU placement. Increase this if the GPU is shared with other workloads. |

### 🧩 DeblurNet Pixel Tiling

These arguments are available only on `Inference_deblurNet.py`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--tile_mode` | String | `latent` | Tiling strategy. `latent` uses the original latent-space tiled denoise path. `pixel` deblurs independent RGB tiles before stitching. |
| `--pixel_tile_size` | Integer | `512` | Tile size for `--tile_mode pixel`. Use a multiple of `16`. |
| `--pixel_tile_overlap` | Integer | `0` | Requested overlap for `--tile_mode pixel`. Must be smaller than `--pixel_tile_size`. Overlap regions are feather-blended during stitching. Use `0` to isolate per-tile generation quality without blending. |

DeblurNet tiling notes:
- `latent` tiling is lighter-weight, but it still encodes the full resized image once and then slices tiles in latent space. It is not equivalent to independent `512x512` crop inference.
- `pixel` tiling is slower, but it matched the model's preferred `512x512` operating scale more closely in local testing.
- In local testing on a `2304x1584` example, pixel tiling stayed strongest up to about `--long_side 1728`. Beyond that, some fine details started to clump inside tiles, which likely indicates a scale/distribution limit rather than a seam-blending issue.
- If seams are visible, try a small overlap such as `--pixel_tile_overlap 64`. If details look over-smoothed or inconsistent, try `--pixel_tile_overlap 0`.

### Suggested Settings

| Goal | Suggested flags |
| --- | --- |
| Lowest VRAM | `--offload_mode sequential --steps 8 --long_side 1024` |
| Balanced default | `--offload_mode auto --steps 20` |
| Faster on high-VRAM GPU | `--offload_mode none` or `--offload_mode model` |
| High-res DeblurNet detail | `--tile_mode pixel --pixel_tile_size 512 --long_side 1536` |
| Test maximum useful DeblurNet scale | Increase `--long_side` gradually and compare texture/detail quality. |

---

## 🧩 ComfyUI Integration

A ComfyUI implementation of Genfocus is available, thanks to [Eric Rollei](https://github.com/EricRollei)!  
Check it out here: 👉 **[comfyui-refocus](https://github.com/EricRollei/comfyui-refocus)**

-----

## 🗺️ Roadmap & TODO

We are actively working on improving this project. Current progress:

  - [x] **Upload Model Weights**
  - [x] **Release HF Demo & Gradio Code** (with tiling tricks for high-res images)
  - [x] **Release Inference Code** (Support for adjustable parameters/settings)
  - [ ] **Release Benchmark data**
  - [ ] **Release Training Code and Data**

-----

## 🔗 Citation

If you find this project useful for your research, please consider citing:

```bibtex
@article{Genfocus2025,
  title={Generative Refocusing: Flexible Defocus Control from a Single Image},
  author={Tuan Mu, Chun-Wei and Huang, Jia-Bin and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.16923},
  year={2025}
}
```

## 📧 Contact

For any questions or suggestions, please open an issue or contact me at [raytm9999.cs09@nycu.edu.tw](mailto:raytm9999.cs09@nycu.edu.tw).

<div align="center">
  <br>
  <p>Star 🌟 this repository if you like it!</p>
</div>
