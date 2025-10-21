# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project performs **RHM (Reference Height Map) resolution enhancement** using depth-guided super-resolution. It coregisters low-resolution ground-truth RHM raster chips with high-resolution aerial orthophotography chips, then increases the RHM resolution by fusing it with monocular depth estimates from the aerial imagery. The wavelet-based fusion preserves the accurate low-frequency structure of the RHM while adding high-frequency details from the depth estimation.

## Core Architecture

### Two-Stage Pipeline

1. **Depth Inference** (`infer_depth.py`): Generates high-resolution depth maps from aerial orthophotography using pre-trained models
2. **Coregistration & Resolution Enhancement**:
   - `explore.ipynb`: Interactive proof-of-concept notebook with detailed explanations
   - `enhance_rhm.py`: Batch processing script for production workflows

### Key Components

**Depth Inference (`infer_depth.py`)**
- Model loaders for MiDaS (DPT_Large, DPT_Hybrid, MiDaS_small) and ZoeDepth (ZoeD_N, ZoeD_K) via torch.hub
- Inference functions that handle model-specific preprocessing and postprocessing
- MiDaS: outputs inverse depth (1/d), requires bicubic interpolation to original resolution
- ZoeDepth: outputs metric depth directly
- Output: both `.npy` (raw float32) and `.png` (16-bit normalized) depth maps

**Coregistration & Resolution Enhancement Workflow**

`explore.ipynb` - Interactive proof-of-concept:
- Preprocessing: RHM is resampled to match the target high resolution (downsampled then bicubic upsampled to simulate low-res ground truth)
- Edge detection via Canny on Gaussian-blurred images for robust feature detection
- **Coregistration**: Aerial orthophotography (depth) defines ground truth spatial location; RHM is warped to match aerial coordinates using OpenCV's `findTransformECC` affine registration on edge maps
- **Distribution matching**: Both depth and aligned RHM are normalized to match the original RHM's mean and standard deviation using z-score normalization, ensuring consistent statistical properties before fusion
- Scale/offset matching: linear regression between depth and aligned RHM for intensity calibration
- **Resolution enhancement**: SWT2 wavelet fusion combines ground-truth low-frequency structure from aligned RHM with high-frequency details from depth estimation
  - `alpha` controls low-freq blend (0=all RHM ground truth, 1=all depth)
  - `gain` amplifies depth high-frequency details for sharper output
  - Result: high-resolution RHM preserving accurate coarse structure with added fine detail

`enhance_rhm.py` - Batch processing script:
- Implements the same workflow as the notebook for production use
- Processes all matching RHM chips automatically
- Preserves georeferencing from aerial images using rasterio
- Command-line interface with configurable parameters
- Progress tracking and error handling for robust batch processing

## Running the Code

### Depth Inference

```bash
# Using MiDaS Large (default)
python infer_depth.py img_in --model midas_large --out_dir depth_out

# Using ZoeDepth
python infer_depth.py img_in --model zoedepth_n --out_dir depth_out

# Other models: midas_hybrid, midas_small, zoedepth_k
```

**Inputs:** Directory with 512x512 aerial orthophotography raster chips (`.png`, `.jpg`, `.jpeg`)
**Outputs:** `{basename}.npy` (raw depth) and `{basename}.png` (16-bit visualization)

### Coregistration & Resolution Enhancement (Interactive)

Run cells in `explore.ipynb` sequentially. The notebook expects:
- `img_in/`: aerial orthophotography chips
- `depth_out/`: depth maps from `infer_depth.py`
- `rhm_reference/`: low-resolution ground-truth RHM chips (`.tif` files)

Key parameters in final fusion cell:
- `wavelet`, `level`: wavelet type and decomposition levels
- `alpha`: blend factor for low frequencies (0=all RHM, 1=all depth)
- `gain`: multiplier for high-frequency depth details

### Batch Processing (Production)

For processing multiple chips, use `enhance_rhm.py`:

```bash
# Basic usage
python enhance_rhm.py --img_dir img_in --depth_dir depth_out --rhm_dir rhm_reference --out_dir rhm_enhanced

# With custom parameters
python enhance_rhm.py --img_dir img_in --depth_dir depth_out --rhm_dir rhm_reference --out_dir rhm_enhanced \
  --wavelet db2 --level 7 --alpha 0.0 --gain 1.5 --verbose

# See all options
python enhance_rhm.py --help
```

**Key features**:
- Batch processes all matching RHM chips
- Preserves georeferencing from aerial images (requires `rasterio`)
- Progress tracking with `tqdm`
- Automatic file matching by basename
- Skips existing outputs (use `--overwrite` to replace)

**Inputs**: Same directory structure as notebook
**Outputs**: `{basename}_enhanced.tif` files with georeferencing preserved

**Parameters**:
- `--rhm_scale`: RHM scaling factor (default: 0.075)
- `--wavelet`: Wavelet type (default: db2)
- `--level`: Decomposition levels (default: 7)
- `--alpha`: Low-freq blend (default: 0.0 = all RHM)
- `--gain`: High-freq amplification (default: 1.0)
- `--verbose`: Detailed output per chip
- `--fallback_no_registration`: Continue with unaligned images if ECC registration fails

## Dependencies

PyTorch-based depth models loaded via `torch.hub`:
- MiDaS: `intel-isl/MiDaS`
- ZoeDepth: `isl-org/ZoeDepth`

Image processing:
- OpenCV (`cv2`): I/O, color conversion, affine transforms, ECC registration
- PIL/Pillow (`Image`): image loading and resizing
- PyWavelets (`pywt`): stationary wavelet transform for fusion
- NumPy, SciPy

Batch processing (`enhance_rhm.py`):
- rasterio: Georeferencing preservation (optional but recommended)
- tqdm: Progress tracking

## Data Flow

1. **Input**: Aerial orthophotography chips (512x512) + low-resolution RHM ground truth (.tif)
2. **Depth inference**: Orthophotography → `infer_depth.py` → high-res depth maps in `depth_out/`
3. **Load data**: Notebook loads orthophotography, depth estimates, and RHM chips; captures original RHM statistics (mean, std)
4. **Resample RHM**: Low-res RHM is resampled to target high resolution (simulating coarse ground truth)
5. **Coregister**: Aerial image (depth) defines ground truth location; RHM is warped to match via ECC affine alignment on blurred/edge-detected versions
6. **Normalize distributions**: Both depth and aligned RHM are z-score normalized and scaled to match original RHM distribution
7. **Calibrate**: Linear scale/offset matching between depth and aligned RHM for intensity consistency
8. **Super-resolve**: Wavelet fusion combines aligned RHM low-frequency (ground truth structure) with depth high-frequency (fine details)
9. **Output**: High-resolution RHM preserving accurate coarse structure with enhanced detail from depth estimation

## Important Notes

- **Primary goal**: RHM resolution enhancement (super-resolution) using depth-guided wavelet fusion
- All processing assumes 512x512 aerial orthophotography chips as input
- RHM inputs are lower resolution ground truth; depth estimation provides high-frequency detail source
- MiDaS outputs are inverted (1/depth) in `run_midas()` at infer_depth.py:51
- 16-bit PNG normalization uses 2nd-98th percentile clipping in `normalize_to_16bit()` at infer_depth.py:27
- Wavelet fusion uses SWT (not DWT) to avoid downsampling artifacts that would degrade resolution
- Reference height maps are scaled by `rhm_scale=0.075` (7.5cm/1m) in notebook before bicubic upsampling
- **Coregistration approach**: Aerial orthophotography defines ground truth spatial location; RHM is warped to aerial coordinates (not vice versa)
- **Distribution matching**: Depth and RHM are normalized to have the same mean and standard deviation as the original RHM before fusion, ensuring statistical consistency and keeping outputs in expected value ranges
- Spatial misalignment between depth and RHM will produce artifacts in fused output, making coregistration essential
- The notebook is fully documented with markdown cells explaining each step and the reasoning behind the approach
