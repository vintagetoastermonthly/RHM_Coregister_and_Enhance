#!/usr/bin/env python3
"""
RHM Resolution Enhancement - Batch Processing Script

This script performs depth-guided super-resolution on Reference Height Maps (RHM)
by fusing them with monocular depth estimates from aerial orthophotography.

Assumes that infer_depth.py has already been run to generate depth maps.

Workflow:
1. Load aerial orthophotography, depth estimates, and low-resolution RHM
2. Coregister RHM to aerial image coordinates (aerial defines ground truth)
3. Normalize distributions to match original RHM statistics
4. Calibrate intensity using linear regression
5. Fuse using stationary wavelet transform (SWT)
6. Output high-resolution RHM with preserved georeferencing

Usage:
    python enhance_rhm.py --img_dir img_in --depth_dir depth_out --rhm_dir rhm_reference --out_dir rhm_enhanced
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
import pywt
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Georeferencing will not be preserved.")
    print("Install with: pip install rasterio")


def load_and_resample_rhm(rhm_path: Path, rhm_scale: float, target_dim: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Load RHM and resample to target resolution.

    Returns:
        rhm_original: Original RHM data (for statistics)
        rhm_resampled: Resampled RHM array (0-255 normalized)
        mean_rhm: Original mean
        std_rhm: Original standard deviation
    """
    rhm = Image.open(rhm_path)
    rhm_original = np.array(rhm).astype(np.float32)

    # Compute statistics from original
    mean_rhm = np.mean(rhm_original)
    std_rhm = np.std(rhm_original)

    # Resample to simulate low-resolution then upscale
    rhm = rhm.resize((int(target_dim * rhm_scale), int(target_dim * rhm_scale)))
    rhm = rhm.resize((target_dim, target_dim), resample=Image.BICUBIC)

    # Normalize to 0-255
    rhm_arr = np.array(rhm).astype(np.float32)
    rhm_arr = rhm_arr / np.max(rhm_arr) * 255.0

    return rhm_original, rhm_arr, mean_rhm, std_rhm


def load_and_normalize_depth(depth_path: Path) -> np.ndarray:
    """
    Load depth map and normalize to 0-255 range (inverted).
    """
    depth = Image.open(depth_path)
    depth_arr = np.array(depth).astype(np.float32)

    # Normalize and invert
    depth_arr = depth_arr / np.max(depth_arr)
    depth_arr = (1.0 - depth_arr) * 255.0

    return depth_arr


def coregister_rhm_to_depth(rhm: np.ndarray, depth: np.ndarray,
                            blur_sigma: float = 10.0,
                            fallback_no_registration: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Coregister RHM to depth using ECC affine registration.

    Depth defines the ground truth spatial location (from aerial image).
    RHM is warped to match aerial coordinates.

    Parameters:
        rhm: RHM array to be aligned
        depth: Depth array (reference)
        blur_sigma: Gaussian blur sigma for preprocessing
        fallback_no_registration: If True, return identity transform on failure instead of raising

    Returns:
        rhm_aligned: Aligned RHM array (or original if fallback and failed)
        warp_matrix: Affine transformation matrix (2x3) (or identity if fallback and failed)
        cc: ECC correlation coefficient (or None if fallback and failed)
    """
    # Preprocessing: blur for robust alignment
    ref = cv2.GaussianBlur(depth, (0, 0), blur_sigma)
    mov = cv2.GaussianBlur(rhm, (0, 0), blur_sigma)

    # Normalize to 8-bit for Canny edge detection
    ref_8u = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov_8u = cv2.normalize(mov, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ECC affine registration
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_8u.astype(np.float32),  # Reference: depth
            mov_8u.astype(np.float32),  # Moving: RHM
            warp_matrix,
            warp_mode,
            criteria
        )

        # Apply transformation to align RHM
        rhm_aligned = cv2.warpAffine(
            rhm,
            warp_matrix,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        return rhm_aligned, warp_matrix, cc

    except cv2.error as e:
        if fallback_no_registration:
            # Return unregistered RHM with identity transform
            return rhm.copy(), np.eye(2, 3, dtype=np.float32), None
        else:
            # Re-raise the exception
            raise


def match_distributions(depth: np.ndarray, rhm: np.ndarray,
                       mean_rhm: float, std_rhm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize depth and RHM to match original RHM distribution.

    Uses z-score normalization followed by scaling to target statistics.
    """
    depth_matched = (depth - np.mean(depth)) / np.std(depth) * std_rhm + mean_rhm
    rhm_matched = (rhm - np.mean(rhm)) / np.std(rhm) * std_rhm + mean_rhm

    return depth_matched, rhm_matched


def calibrate_intensity(depth: np.ndarray, rhm: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Calibrate depth intensity using linear regression to match RHM.

    Returns:
        depth_calibrated: Calibrated depth
        a: Scale factor
        b: Offset
    """
    mask = np.isfinite(rhm) & np.isfinite(depth)

    # Linear regression: depth' = a * depth + b
    a = np.cov(depth[mask], rhm[mask])[0, 1] / (np.var(depth[mask]) + 1e-9)
    b = rhm[mask].mean() - a * depth[mask].mean()

    depth_calibrated = a * depth + b

    return depth_calibrated, a, b


def fuse_swt2(rhm_arr: np.ndarray, depth_arr: np.ndarray,
              wavelet: str, level: int, alpha: float, gain: float) -> np.ndarray:
    """
    Fuse RHM and depth using Stationary Wavelet Transform.

    Parameters:
        rhm_arr: Reference height map (ground truth low-freq)
        depth_arr: Depth estimate (high-freq details)
        wavelet: Wavelet type (e.g., 'db2', 'haar', 'sym4')
        level: Decomposition levels
        alpha: Low-freq blend (0=all RHM, 1=all depth)
        gain: High-freq amplification factor

    Returns:
        Fused image with RHM structure + depth details
    """
    # Decompose both images
    cr = pywt.swt2(rhm_arr, wavelet=wavelet, level=level)
    cd = pywt.swt2(depth_arr, wavelet=wavelet, level=level)

    # Handle different pywt output formats
    if isinstance(cr, list):
        # Format: [(cA_L, (cH_L, cV_L, cD_L)), ..., (cA_1, (cH_1, cV_1, cD_1))]
        fused = []
        for (cA_r, (cHr, cVr, cDr)), (cA_d, (cHd, cVd, cDd)) in zip(cr, cd):
            # Low-freq: blend RHM and depth based on alpha
            cA = (1 - alpha) * cA_r + alpha * cA_d
            # High-freq: use depth details with gain
            fused.append((cA, (gain * cHd, gain * cVd, gain * cDd)))
        return pywt.iswt2(fused, wavelet=wavelet)
    else:
        # Format: (cA_top, [(cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)])
        cA_rhm, details_rhm = cr
        cA_depth, details_depth = cd
        # Low-freq: blend
        cA = (1 - alpha) * cA_rhm + alpha * cA_depth
        # High-freq: use depth details with gain
        details = [(gain * cH, gain * cV, gain * cD) for (cH, cV, cD) in details_depth]
        return pywt.iswt2((cA, details), wavelet=wavelet)


def process_chip(img_path: Path, depth_path: Path, rhm_path: Path,
                output_path: Path, rhm_scale: float, target_dim: int,
                wavelet: str, level: int, alpha: float, gain: float,
                fallback_no_registration: bool = False,
                verbose: bool = False) -> str:
    """
    Process a single image chip.

    Parameters:
        fallback_no_registration: If True, continue without registration on ECC failure

    Returns:
        'success': Processing completed with registration
        'warning': Processing completed without registration (fallback used)
        'failed': Processing failed
    """
    try:
        # Load data
        if verbose:
            print(f"  Loading {rhm_path.name}...")

        rhm_original, rhm_arr, mean_rhm, std_rhm = load_and_resample_rhm(
            rhm_path, rhm_scale, target_dim
        )
        depth_arr = load_and_normalize_depth(depth_path)

        # Coregister RHM to depth
        if verbose:
            print(f"  Coregistering...")

        rhm_aligned, warp_matrix, cc = coregister_rhm_to_depth(
            rhm_arr, depth_arr, fallback_no_registration=fallback_no_registration
        )

        # Track whether registration succeeded
        registration_failed = (cc is None)

        if cc is not None:
            if verbose:
                print(f"    ECC correlation: {cc:.4f}")
        else:
            # Registration failed, using fallback
            print(f"    Warning: Registration failed for {rhm_path.name}, proceeding without alignment")

        # Prepare for fusion
        depth_for_fusion = depth_arr.copy()
        rhm_for_fusion = rhm_aligned.copy()

        # Distribution matching
        depth_for_fusion, rhm_for_fusion = match_distributions(
            depth_for_fusion, rhm_for_fusion, mean_rhm, std_rhm
        )

        # Intensity calibration
        depth_calibrated, a, b = calibrate_intensity(depth_for_fusion, rhm_for_fusion)

        if verbose:
            print(f"    Calibration: depth' = {a:.4f} * depth + {b:.4f}")

        # Create mask and fill NaNs
        mask = np.isfinite(rhm_for_fusion) & np.isfinite(depth_calibrated)
        rhm_filled = np.nan_to_num(rhm_for_fusion, nan=float(np.nanmean(rhm_for_fusion[mask])))
        depth_filled = np.nan_to_num(depth_calibrated, nan=float(np.nanmean(depth_calibrated[mask])))

        # Wavelet fusion
        if verbose:
            print(f"  Fusing with SWT (wavelet={wavelet}, level={level})...")

        fused = fuse_swt2(rhm_filled, depth_filled, wavelet, level, alpha, gain)

        # Restore NaN mask
        fused[~mask] = np.nan

        # Save output with georeferencing
        if verbose:
            print(f"  Saving to {output_path.name}...")

        save_geotiff(fused, output_path, img_path, rhm_path)

        # Return status based on whether registration succeeded
        if registration_failed:
            return 'warning'
        else:
            return 'success'

    except Exception as e:
        print(f"Error processing {rhm_path.name}: {e}")
        return 'failed'


def save_geotiff(data: np.ndarray, output_path: Path,
                 img_path: Path, rhm_path: Path):
    """
    Save data as GeoTIFF with spatial reference from aerial image.

    Falls back to simple TIFF if rasterio not available.
    """
    if HAS_RASTERIO:
        try:
            # Try to read georeferencing from aerial image
            # First try the image itself
            with rasterio.open(img_path) as src:
                transform = src.transform
                crs = src.crs

            # Write output with georeferencing
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(data, 1)

            return

        except Exception as e:
            print(f"    Warning: Could not preserve georeferencing: {e}")
            print(f"    Falling back to plain TIFF")

    # Fallback: save as plain TIFF
    from PIL import Image
    # Clip and convert to uint16
    data_clipped = np.nan_to_num(data, nan=0)
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    data_norm = (data_clipped - data_min) / (data_max - data_min + 1e-9)
    data_uint16 = (data_norm * 65535).astype(np.uint16)

    Image.fromarray(data_uint16).save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Batch RHM resolution enhancement using depth-guided super-resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory with aerial orthophotography chips')
    parser.add_argument('--depth_dir', type=str, required=True,
                       help='Directory with depth maps (from infer_depth.py)')
    parser.add_argument('--rhm_dir', type=str, required=True,
                       help='Directory with low-resolution RHM ground truth')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for enhanced RHM')

    # Processing parameters
    parser.add_argument('--rhm_scale', type=float, default=0.075,
                       help='RHM scaling factor (default: 0.075 = 7.5cm/1m)')
    parser.add_argument('--dim', type=int, default=512,
                       help='Target dimension (default: 512)')

    # Wavelet fusion parameters
    parser.add_argument('--wavelet', type=str, default='db2',
                       help='Wavelet type (default: db2)')
    parser.add_argument('--level', type=int, default=7,
                       help='Decomposition levels (default: 7)')
    parser.add_argument('--alpha', type=float, default=0.0,
                       help='Low-freq blend: 0=all RHM, 1=all depth (default: 0.0)')
    parser.add_argument('--gain', type=float, default=1.0,
                       help='High-freq amplification factor (default: 1.0)')

    # Options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing outputs')
    parser.add_argument('--fallback_no_registration', action='store_true',
                       help='Continue processing without registration if ECC fails (use unaligned images)')

    args = parser.parse_args()

    # Convert to Path objects
    img_dir = Path(args.img_dir)
    depth_dir = Path(args.depth_dir)
    rhm_dir = Path(args.rhm_dir)
    out_dir = Path(args.out_dir)

    # Validate inputs
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        sys.exit(1)
    if not depth_dir.exists():
        print(f"Error: Depth directory not found: {depth_dir}")
        sys.exit(1)
    if not rhm_dir.exists():
        print(f"Error: RHM directory not found: {rhm_dir}")
        sys.exit(1)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all RHM files
    rhm_files = sorted(rhm_dir.glob('*.tif'))

    if not rhm_files:
        print(f"Error: No .tif files found in {rhm_dir}")
        sys.exit(1)

    print(f"Found {len(rhm_files)} RHM chips to process")
    print(f"Processing parameters:")
    print(f"  RHM scale: {args.rhm_scale}")
    print(f"  Wavelet: {args.wavelet}, Level: {args.level}")
    print(f"  Alpha: {args.alpha}, Gain: {args.gain}")
    if args.fallback_no_registration:
        print(f"  Fallback mode: ENABLED (will process unaligned if registration fails)")
    print()

    # Process each chip
    success_count = 0
    warning_count = 0
    fail_count = 0

    for rhm_path in tqdm(rhm_files, desc="Processing chips"):
        base_name = rhm_path.stem

        # Find corresponding files
        img_path = img_dir / f"{base_name}.png"
        if not img_path.exists():
            img_path = img_dir / f"{base_name}.jpg"
        if not img_path.exists():
            print(f"Warning: No image found for {base_name}, skipping")
            fail_count += 1
            continue

        depth_path = depth_dir / f"{base_name}.png"
        if not depth_path.exists():
            depth_path = depth_dir / f"{base_name}.npy"
        if not depth_path.exists():
            print(f"Warning: No depth map found for {base_name}, skipping")
            fail_count += 1
            continue

        output_path = out_dir / f"{base_name}_enhanced.tif"

        # Skip if exists and not overwrite
        if output_path.exists() and not args.overwrite:
            if args.verbose:
                print(f"Skipping {base_name} (already exists)")
            continue

        # Process chip
        if args.verbose:
            print(f"\nProcessing {base_name}:")

        status = process_chip(
            img_path, depth_path, rhm_path, output_path,
            args.rhm_scale, args.dim,
            args.wavelet, args.level, args.alpha, args.gain,
            fallback_no_registration=args.fallback_no_registration,
            verbose=args.verbose
        )

        if status == 'success':
            success_count += 1
        elif status == 'warning':
            warning_count += 1
        else:  # 'failed'
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    if warning_count > 0:
        print(f"  Warnings: {warning_count} (processed without registration)")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {out_dir}")


if __name__ == '__main__':
    main()
