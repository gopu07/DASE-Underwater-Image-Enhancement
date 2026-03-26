"""
DASE: Depth-Aware Scene-Adaptive Enhancement for Underwater Images
==================================================================
Author: [Your Name]
Date: 2024

This implementation extends the hybrid mean-max color correction baseline with:
1. Depth estimation using pretrained MiDaS
2. Depth-stratified color correction (near/mid/far zones)
3. Scene-adaptive yellow tone preservation
4. Depth-weighted contrast fusion (HE + CLAHE + LA)

Usage:
    python underwater_enhancement.py --image underwater.jpg
    python underwater_enhancement.py --batch input_dir/ --output results/
    python underwater_enhancement.py --compare underwater.jpg
"""

# ============================================================================
# IMPORTS
# ============================================================================
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import argparse
import time
import csv
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union

try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Warning] scikit-image not found. SSIM/PSNR will use fallback implementations.")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Global configuration parameters for the DASE pipeline."""

    # Depth zone segmentation
    NUM_ZONES: int = 3
    NEAR_PERCENTILE: int = 33
    FAR_PERCENTILE: int = 66

    # Yellow content detection (HSV space)
    YELLOW_HUE_MIN: int = 20    # degrees (OpenCV: 10)
    YELLOW_HUE_MAX: int = 40    # degrees (OpenCV: 20)
    YELLOW_SAT_MIN: int = 50    # 0-255
    YELLOW_THRESHOLD: float = 0.15  # minimum ratio of yellow pixels

    # Zone-specific scaling factors
    NEAR_SCALE_FACTOR: float = 0.3       # blend toward neutral
    FAR_SCALE_MULTIPLIER: float = 1.5    # aggressive far boost
    YELLOW_GREEN_MULTIPLIER: float = 0.7  # reduce green suppression

    # CLAHE parameters
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)

    # MiDaS model selection: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
    MIDAS_MODEL: str = "MiDaS_small"

    # Supported image extensions
    IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and convert to RGB uint8.

    Args:
        path: Filepath to the image.

    Returns:
        RGB image as np.ndarray of shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    # Handle grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Handle BGRA (alpha channel)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # OpenCV loads BGR → convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save an RGB image to disk.

    Args:
        image: RGB image (H, W, 3), dtype uint8 or float [0,1].
        path:  Output filepath (extension determines format).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Normalize a depth map to the [0, 1] range.

    Args:
        depth: Raw depth array of any shape.

    Returns:
        Depth map normalized to [0, 1].
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def ensure_rgb_float(image: np.ndarray) -> np.ndarray:
    """Return float32 image in [0,1] from uint8 or float input."""
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip and convert float [0,1] image to uint8 [0,255]."""
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


# ============================================================================
# BASELINE METHOD (Professor's Work)
# ============================================================================

def hybrid_mean_max_correction(image: np.ndarray) -> np.ndarray:
    """
    Baseline color correction using the hybrid mean-max method.

    Algorithm:
        1. ravg  = mean(R channel)
        2. Rmax, Gmax, Bmax = per-channel maximum values
        3. gsf = mean(ravg, Bmax) / Gmax
           bsf = mean(ravg, Gmax) / Bmax
           rsf = mean(Gmax, Bmax) / Rmax
        4. R' = clip(R * rsf), G' = clip(G * gsf), B' = clip(B * bsf)

    Args:
        image: RGB image (H, W, 3), dtype uint8 (0-255).

    Returns:
        Color-corrected RGB image, dtype uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    img = image.astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    ravg = float(np.mean(R))
    Rmax = float(np.max(R))
    Gmax = float(np.max(G))
    Bmax = float(np.max(B))

    # Scaling factors (add epsilon to avoid division by zero)
    gsf = np.mean([ravg, Bmax]) / (Gmax + 1e-6)
    bsf = np.mean([ravg, Gmax]) / (Bmax + 1e-6)
    rsf = np.mean([Gmax, Bmax]) / (Rmax + 1e-6)

    R_corr = np.clip(R * rsf, 0, 255).astype(np.uint8)
    G_corr = np.clip(G * gsf, 0, 255).astype(np.uint8)
    B_corr = np.clip(B * bsf, 0, 255).astype(np.uint8)

    return np.stack([R_corr, G_corr, B_corr], axis=2)


def apply_contrast_LAB(image: np.ndarray, method: str = 'CLAHE') -> np.ndarray:
    """
    Apply contrast enhancement in CIE LAB color space on the L channel.

    Args:
        image:  RGB image (H, W, 3), dtype uint8.
        method: 'HE'    – Histogram Equalization
                'CLAHE' – Contrast Limited Adaptive HE
                'LA'    – Linear (min-max) Stretching

    Returns:
        Contrast-enhanced RGB image, dtype uint8.

    Raises:
        ValueError: If method is not recognized.
    """
    if method not in ('HE', 'CLAHE', 'LA'):
        raise ValueError(f"Unknown contrast method '{method}'. Choose HE, CLAHE, or LA.")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B_ch = cv2.split(lab)

    if method == 'HE':
        L_enhanced = cv2.equalizeHist(L)

    elif method == 'CLAHE':
        clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMIT,
            tileGridSize=Config.CLAHE_TILE_SIZE
        )
        L_enhanced = clahe.apply(L)

    else:  # 'LA' – linear stretching
        L_float = L.astype(np.float32)
        L_min, L_max = L_float.min(), L_float.max()
        if L_max - L_min > 1e-6:
            L_enhanced = ((L_float - L_min) / (L_max - L_min) * 255).astype(np.uint8)
        else:
            L_enhanced = L

    lab_enhanced = cv2.merge([L_enhanced, A, B_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def baseline_enhance(image: np.ndarray, contrast_method: str = 'CLAHE') -> np.ndarray:
    """
    Complete baseline enhancement pipeline.

    Steps:
        1. Hybrid mean-max color correction
        2. LAB-space contrast enhancement

    Args:
        image:           RGB uint8 input image.
        contrast_method: One of 'HE', 'CLAHE', 'LA'.

    Returns:
        Enhanced RGB uint8 image.
    """
    color_corrected = hybrid_mean_max_correction(image)
    enhanced = apply_contrast_LAB(color_corrected, method=contrast_method)
    return enhanced


# ============================================================================
# DEPTH ESTIMATION
# ============================================================================

_midas_model_cache: Dict = {}


def load_midas_model(model_type: str = Config.MIDAS_MODEL):
    """
    Load a pretrained MiDaS depth estimation model from torch.hub.

    Args:
        model_type: One of 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'.

    Returns:
        Tuple (model, transform) ready for inference.
    """
    global _midas_model_cache

    if model_type in _midas_model_cache:
        return _midas_model_cache[model_type]

    print(f"[Depth] Loading MiDaS model: {model_type} …")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    except Exception as e:
        print(f"[Depth] torch.hub failed ({e}). Attempting timm fallback …")
        try:
            import timm  # noqa: F401
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            model_type = "MiDaS_small"
        except Exception as e2:
            raise RuntimeError(
                f"Cannot load any MiDaS model. Install torch + timm and check internet connection.\n{e2}"
            )

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    _midas_model_cache[model_type] = (model, transform, device)
    print(f"[Depth] Model loaded on {device}.")
    return _midas_model_cache[model_type]


def estimate_depth(
    image: np.ndarray,
    model_bundle=None
) -> np.ndarray:
    """
    Estimate a per-pixel depth map using MiDaS.

    Args:
        image:        RGB image (H, W, 3), uint8.
        model_bundle: (model, transform, device) from load_midas_model(),
                      or None to auto-load.

    Returns:
        Normalized depth map (H, W) as float32 in [0, 1].
        Smaller values → nearer objects (MiDaS outputs inverse depth;
        we normalize so 0 = near, 1 = far).
    """
    if model_bundle is None:
        model_bundle = load_midas_model()

    model, transform, device = model_bundle
    h, w = image.shape[:2]

    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_raw = prediction.cpu().numpy()

    # MiDaS returns inverse depth (larger = closer).
    # Invert so that 0 = near, 1 = far, then normalize.
    depth_inverted = -depth_raw
    depth_norm = normalize_depth(depth_inverted)
    return depth_norm


def dummy_depth_map(image: np.ndarray) -> np.ndarray:
    """
    Generate a simple synthetic depth map (gradient top→bottom) as a fallback
    when MiDaS is unavailable.

    Args:
        image: RGB image (H, W, 3).

    Returns:
        Depth map (H, W) float32 in [0, 1].
    """
    h, w = image.shape[:2]
    depth = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
    return depth


# ============================================================================
# DEPTH ZONE SEGMENTATION
# ============================================================================

def segment_depth_zones(
    depth_map: np.ndarray,
    num_zones: int = Config.NUM_ZONES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition the depth map into three spatial zones using percentiles.

    Args:
        depth_map: Normalized depth (H, W), float32 in [0, 1].
        num_zones: Number of zones (currently fixed at 3).

    Returns:
        Tuple (near_mask, mid_mask, far_mask) – boolean arrays of shape (H, W).
    """
    near_thresh = float(np.percentile(depth_map, Config.NEAR_PERCENTILE))
    far_thresh  = float(np.percentile(depth_map, Config.FAR_PERCENTILE))

    near_mask = depth_map < near_thresh
    mid_mask  = (depth_map >= near_thresh) & (depth_map < far_thresh)
    far_mask  = depth_map >= far_thresh

    return near_mask, mid_mask, far_mask


# ============================================================================
# YELLOW CONTENT DETECTION
# ============================================================================

def detect_yellow_content(
    image: np.ndarray,
    threshold: float = Config.YELLOW_THRESHOLD
) -> bool:
    """
    Detect whether the image contains significant yellow-toned regions.

    Yellow objects (fish, coral, artificial lighting) need gentler green
    suppression to avoid unnatural hue shifts.

    Detection uses HSV color space:
        - Hue in [YELLOW_HUE_MIN, YELLOW_HUE_MAX] (OpenCV scale: /2)
        - Saturation > YELLOW_SAT_MIN

    Args:
        image:     RGB image (H, W, 3), uint8.
        threshold: Minimum fraction of yellow pixels to trigger detection.

    Returns:
        True if yellow content exceeds threshold.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, _ = cv2.split(hsv)

    # OpenCV hue is half the standard degree value (0-179 for 0-359°)
    h_min = Config.YELLOW_HUE_MIN // 2   # ≈10
    h_max = Config.YELLOW_HUE_MAX // 2   # ≈20

    yellow_mask = (H >= h_min) & (H <= h_max) & (S > Config.YELLOW_SAT_MIN)
    yellow_ratio = yellow_mask.sum() / yellow_mask.size

    return bool(yellow_ratio > threshold)


# ============================================================================
# DEPTH-AWARE COLOR CORRECTION  (Novel Contribution #1)
# ============================================================================

def compute_zone_scaling_factors(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    zone: str,
    yellow_content: bool
) -> Tuple[float, float, float]:
    """
    Compute per-channel scaling factors for a specific depth zone.

    Args:
        R, G, B:       Float32 channel arrays for pixels in the zone.
        zone:          'near', 'mid', or 'far'.
        yellow_content: Whether scene-wide yellow was detected.

    Returns:
        (rsf, gsf, bsf) – scaling factors for Red, Green, Blue.
    """
    if R.size == 0 or G.size == 0 or B.size == 0:
        return 1.0, 1.0, 1.0

    ravg = float(np.mean(R))
    Rmax = float(np.max(R)) + 1e-6
    Gmax = float(np.max(G)) + 1e-6
    Bmax = float(np.max(B)) + 1e-6

    # Standard (mid) scaling factors — professor's baseline
    rsf_base = np.mean([Gmax, Bmax]) / Rmax
    gsf_base = np.mean([ravg, Bmax]) / Gmax
    bsf_base = np.mean([ravg, Gmax]) / Bmax

    if zone == 'near':
        # Gentle blending toward neutral (factor 1.0)
        alpha = Config.NEAR_SCALE_FACTOR
        rsf = 1.0 + alpha * (rsf_base - 1.0)
        gsf = 1.0 + alpha * (gsf_base - 1.0)
        bsf = 1.0 + alpha * (bsf_base - 1.0)

    elif zone == 'mid':
        rsf, gsf, bsf = rsf_base, gsf_base, bsf_base

    else:  # 'far'
        rsf = Config.FAR_SCALE_MULTIPLIER * rsf_base
        gsf = gsf_base
        bsf = bsf_base

    # Scene-adaptive yellow preservation: reduce green suppression
    if yellow_content and gsf < 1.0:
        gsf = 1.0 + Config.YELLOW_GREEN_MULTIPLIER * (gsf - 1.0)

    return float(rsf), float(gsf), float(bsf)


def depth_aware_color_correction(
    image: np.ndarray,
    depth_map: np.ndarray
) -> np.ndarray:
    """
    Apply depth-stratified, spatially-varying color correction.

    NOVEL CONTRIBUTION: Rather than applying a single global white-balance,
    different scaling factors are computed for near, mid, and far depth zones
    to account for the non-linear light absorption in water.

    Pipeline:
        1. Segment depth map → near / mid / far masks
        2. Detect yellow content (scene-level)
        3. For each zone compute zone-specific scaling factors
        4. Build per-pixel scaling maps by blending zone factors
        5. Apply pixel-wise multiplication and clip

    Args:
        image:     RGB image (H, W, 3), uint8.
        depth_map: Normalized depth (H, W), float32 in [0, 1].

    Returns:
        Depth-aware color-corrected RGB image, uint8.
    """
    img_f = image.astype(np.float32)
    R_full = img_f[:, :, 0]
    G_full = img_f[:, :, 1]
    B_full = img_f[:, :, 2]

    near_mask, mid_mask, far_mask = segment_depth_zones(depth_map)
    yellow_content = detect_yellow_content(image)

    if yellow_content:
        print("[ColorCorr] Yellow content detected → adapting green suppression.")

    # Compute zone-specific scaling factors
    zones = {'near': near_mask, 'mid': mid_mask, 'far': far_mask}
    zone_factors: Dict[str, Tuple[float, float, float]] = {}

    for zone_name, mask in zones.items():
        if mask.any():
            r_zone = R_full[mask]
            g_zone = G_full[mask]
            b_zone = B_full[mask]
            rsf, gsf, bsf = compute_zone_scaling_factors(
                r_zone, g_zone, b_zone, zone_name, yellow_content
            )
        else:
            rsf, gsf, bsf = 1.0, 1.0, 1.0
        zone_factors[zone_name] = (rsf, gsf, bsf)

    # Build spatially-varying scaling maps (H, W)
    rsf_map = np.ones_like(R_full)
    gsf_map = np.ones_like(G_full)
    bsf_map = np.ones_like(B_full)

    for zone_name, mask in zones.items():
        rsf, gsf, bsf = zone_factors[zone_name]
        rsf_map[mask] = rsf
        gsf_map[mask] = gsf
        bsf_map[mask] = bsf

    # Smooth boundaries with a small Gaussian blur to avoid hard transitions
    blur_k = 51  # must be odd
    rsf_map = cv2.GaussianBlur(rsf_map, (blur_k, blur_k), 0)
    gsf_map = cv2.GaussianBlur(gsf_map, (blur_k, blur_k), 0)
    bsf_map = cv2.GaussianBlur(bsf_map, (blur_k, blur_k), 0)

    R_out = np.clip(R_full * rsf_map, 0, 255).astype(np.uint8)
    G_out = np.clip(G_full * gsf_map, 0, 255).astype(np.uint8)
    B_out = np.clip(B_full * bsf_map, 0, 255).astype(np.uint8)

    return np.stack([R_out, G_out, B_out], axis=2)


# ============================================================================
# CONTRAST ENHANCEMENT METHODS
# ============================================================================

def histogram_equalization_LAB(image: np.ndarray) -> np.ndarray:
    """
    Apply global Histogram Equalization to the L channel in LAB space.

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        HE-enhanced RGB uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B_ch = cv2.split(lab)
    L_eq = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L_eq, A, B_ch]), cv2.COLOR_LAB2RGB)


def clahe_LAB(
    image: np.ndarray,
    clip_limit: float = Config.CLAHE_CLIP_LIMIT,
    tile_size: Tuple[int, int] = Config.CLAHE_TILE_SIZE
) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB space.

    Args:
        image:      RGB uint8 image (H, W, 3).
        clip_limit: CLAHE contrast limit (default 2.0).
        tile_size:  Grid tile size (default (8, 8)).

    Returns:
        CLAHE-enhanced RGB uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L_clahe = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L_clahe, A, B_ch]), cv2.COLOR_LAB2RGB)


def linear_adjustment_LAB(image: np.ndarray) -> np.ndarray:
    """
    Apply linear (min-max) stretching to the L channel in LAB space.

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        Linearly stretched RGB uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B_ch = cv2.split(lab)
    L_f = L.astype(np.float32)
    L_min, L_max = L_f.min(), L_f.max()
    if L_max - L_min > 1e-6:
        L_stretched = ((L_f - L_min) / (L_max - L_min) * 255).astype(np.uint8)
    else:
        L_stretched = L
    return cv2.cvtColor(cv2.merge([L_stretched, A, B_ch]), cv2.COLOR_LAB2RGB)


# ============================================================================
# DEPTH-WEIGHTED CONTRAST FUSION  (Novel Contribution #3)
# ============================================================================

def compute_fusion_weights(
    depth_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel fusion weights for the three contrast methods.

    Strategy:
        - Near field (depth → 0): prefer LA (gentle, avoids blown-out surfaces)
        - Mid field  (depth ≈ 0.5): prefer CLAHE (balanced local enhancement)
        - Far field  (depth → 1): prefer HE  (aggressive, recovers distant detail)

    Formulae:
        w_la    = 1.0 - depth
        w_clahe = 1.0 - |depth - 0.5| * 2
        w_he    = depth
        (then L1-normalized at each pixel)

    Args:
        depth_map: Normalized depth (H, W), float32 in [0, 1].

    Returns:
        (w_la, w_clahe, w_he) – weight maps in [0,1] summing to 1 per pixel.
    """
    d = depth_map.astype(np.float32)

    w_la    = 1.0 - d
    w_clahe = 1.0 - np.abs(d - 0.5) * 2.0
    w_he    = d

    # Ensure non-negative
    w_la    = np.clip(w_la, 0, 1)
    w_clahe = np.clip(w_clahe, 0, 1)
    w_he    = np.clip(w_he, 0, 1)

    # Normalize to sum to 1 per pixel
    w_sum = w_la + w_clahe + w_he + 1e-8
    w_la    /= w_sum
    w_clahe /= w_sum
    w_he    /= w_sum

    return w_la, w_clahe, w_he


def depth_weighted_contrast_fusion(
    image: np.ndarray,
    depth_map: np.ndarray
) -> np.ndarray:
    """
    Fuse HE, CLAHE, and LA contrast-enhanced images weighted by depth.

    NOVEL CONTRIBUTION: Instead of applying one fixed contrast method
    globally, different methods are blended based on scene depth, providing
    locally adaptive enhancement throughout the image.

    Args:
        image:     Color-corrected RGB uint8 image (H, W, 3).
        depth_map: Normalized depth (H, W), float32 in [0, 1].

    Returns:
        Depth-fused contrast-enhanced RGB uint8 image.
    """
    # Generate all three contrast variants
    he_out    = histogram_equalization_LAB(image).astype(np.float32)
    clahe_out = clahe_LAB(image).astype(np.float32)
    la_out    = linear_adjustment_LAB(image).astype(np.float32)

    # Compute per-pixel weights and expand to (H, W, 1) for broadcasting
    w_la, w_clahe, w_he = compute_fusion_weights(depth_map)
    w_la    = w_la[:, :, np.newaxis]
    w_clahe = w_clahe[:, :, np.newaxis]
    w_he    = w_he[:, :, np.newaxis]

    # Weighted fusion
    fused = w_la * la_out + w_clahe * clahe_out + w_he * he_out
    return np.clip(fused, 0, 255).astype(np.uint8)


# ============================================================================
# MAIN DASE PIPELINE
# ============================================================================

def dase_enhance(
    image: np.ndarray,
    depth_model_bundle=None,
    return_intermediate: bool = False
) -> Union[np.ndarray, Dict]:
    """
    Complete DASE (Depth-Aware Scene-Adaptive Enhancement) pipeline.

    Steps:
        1. Estimate depth map via MiDaS (or fallback gradient map)
        2. Depth-aware spatially-varying color correction
        3. Depth-weighted multi-contrast fusion (HE + CLAHE + LA)

    Args:
        image:               Input RGB uint8 underwater image (H, W, 3).
        depth_model_bundle:  (model, transform, device) from load_midas_model(),
                             or None to auto-load (or use fallback if unavailable).
        return_intermediate: If True, return dict with all intermediate results.

    Returns:
        Enhanced RGB uint8 image, or dict with intermediate results.
    """
    print("[DASE] Starting depth-aware enhancement …")

    # Step 1 – Depth estimation
    try:
        if depth_model_bundle is None:
            depth_model_bundle = load_midas_model()
        depth_map = estimate_depth(image, depth_model_bundle)
        print("[DASE] Depth estimation complete.")
    except Exception as e:
        print(f"[DASE] Depth estimation failed ({e}). Using gradient fallback.")
        depth_map = dummy_depth_map(image)

    # Step 2 – Depth-aware color correction
    color_corrected = depth_aware_color_correction(image, depth_map)
    print("[DASE] Depth-aware color correction applied.")

    # Step 3 – Depth-weighted contrast fusion
    enhanced = depth_weighted_contrast_fusion(color_corrected, depth_map)
    print("[DASE] Depth-weighted contrast fusion complete.")

    if return_intermediate:
        return {
            'original': image,
            'depth_map': depth_map,
            'color_corrected': color_corrected,
            'enhanced': enhanced,
        }
    return enhanced


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1, img2: RGB images of the same shape, uint8 or float.

    Returns:
        PSNR value in dB. Returns inf if images are identical.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for PSNR.")

    if SKIMAGE_AVAILABLE:
        return float(psnr_func(img1, img2, data_range=255))

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.

    Args:
        img1, img2: RGB images of the same shape, uint8.

    Returns:
        SSIM value in [-1, 1].
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for SSIM.")

    if SKIMAGE_AVAILABLE:
        return float(ssim_func(img1, img2, channel_axis=2, data_range=255))

    # Fallback: compute per-channel mean SSIM
    def _ssim_channel(c1, c2):
        c1, c2 = c1.astype(np.float64), c2.astype(np.float64)
        mu1, mu2 = c1.mean(), c2.mean()
        sigma1 = c1.std()
        sigma2 = c2.std()
        sigma12 = np.mean((c1 - mu1) * (c2 - mu2))
        C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
        return ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))

    scores = [_ssim_channel(img1[:,:,i], img2[:,:,i]) for i in range(3)]
    return float(np.mean(scores))


def _uicm(image: np.ndarray) -> float:
    """Underwater Image Colorfulness Measure (UICM component)."""
    img_f = image.astype(np.float32) / 255.0
    R, G, B = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    RG = R - G
    YB = 0.5*(R + G) - B
    mu_rg, sigma_rg = RG.mean(), RG.std()
    mu_yb, sigma_yb = YB.mean(), YB.std()
    uicm = -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sigma_rg**2 + sigma_yb**2)
    return float(uicm)


def _uism(image: np.ndarray) -> float:
    """Underwater Image Sharpness Measure (UISM component)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.mean(edges))


def _uiconm(image: np.ndarray) -> float:
    """Underwater Image Contrast Measure (UIConM component)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(gray))


def compute_uiqm(image: np.ndarray) -> float:
    """
    Compute Underwater Image Quality Measure (UIQM).

    Formula:
        UIQM = c1*UICM + c2*UISM + c3*UIConM
        Typical weights: c1=0.0282, c2=0.2953, c3=3.5753

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        UIQM score (higher is better).
    """
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm   = _uicm(image)
    uism   = _uism(image)
    uiconm = _uiconm(image)
    uiqm   = c1 * uicm + c2 * uism + c3 * uiconm
    return float(uiqm)


def compute_uciqe(image: np.ndarray) -> float:
    """
    Compute Underwater Color Image Quality Evaluation (UCIQE).

    Based on chroma, saturation, and luminance contrast in CIELab.
    Formula:
        UCIQE = c1*sigma_c + c2*con_l + c3*mu_s
        Typical: c1=0.4680, c2=0.2745, c3=0.2576

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        UCIQE score (higher is better).
    """
    lab  = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L    = lab[:,:,0] / 255.0 * 100.0   # 0-100
    A    = lab[:,:,1] - 128.0           # −128…+127
    B_ch = lab[:,:,2] - 128.0

    # Chroma
    chroma  = np.sqrt(A**2 + B_ch**2)
    sigma_c = float(chroma.std())

    # Luminance contrast (top-1% minus bottom-1%)
    l_sorted = np.sort(L.ravel())
    n = len(l_sorted)
    con_l = float(l_sorted[int(0.99*n)] - l_sorted[int(0.01*n)])

    # Saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:,:,1] / 255.0
    mu_s = float(S.mean())

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1*sigma_c + c2*con_l + c3*mu_s)


def compute_entropy(image: np.ndarray) -> float:
    """
    Compute image information entropy (Shannon entropy of grayscale histogram).

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        Entropy value in bits.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    prob = hist / (hist.sum() + 1e-8)
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))
    return entropy


def evaluate_enhancement(
    original: np.ndarray,
    enhanced: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute a comprehensive set of image quality metrics.

    Args:
        original:     Original underwater image (uint8 RGB).
        enhanced:     Enhanced image (uint8 RGB).
        ground_truth: Optional clean reference image for full-reference metrics.

    Returns:
        Dictionary mapping metric name → value.
    """
    metrics: Dict[str, float] = {}

    # No-reference metrics (enhanced image only)
    metrics['UIQM']    = compute_uiqm(enhanced)
    metrics['UCIQE']   = compute_uciqe(enhanced)
    metrics['Entropy'] = compute_entropy(enhanced)

    # Full-reference metrics (require ground truth)
    if ground_truth is not None:
        gt = ground_truth
        # Resize if needed
        if enhanced.shape != gt.shape:
            gt = cv2.resize(gt, (enhanced.shape[1], enhanced.shape[0]))
        try:
            metrics['PSNR'] = compute_psnr(enhanced, gt)
            metrics['SSIM'] = compute_ssim(enhanced, gt)
        except Exception as e:
            print(f"[Eval] Full-reference metric error: {e}")

    # Improvement over original (no-reference)
    metrics['UIQM_orig']  = compute_uiqm(original)
    metrics['UCIQE_orig'] = compute_uciqe(original)
    metrics['Delta_UIQM'] = metrics['UIQM'] - metrics['UIQM_orig']

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert a depth map to a colormapped RGB visualization.

    Args:
        depth_map: Normalized depth (H, W), float32 in [0, 1].

    Returns:
        Colormap visualization as RGB uint8 (H, W, 3).
    """
    depth_norm = normalize_depth(depth_map)
    colored = (cm.plasma(depth_norm)[:, :, :3] * 255).astype(np.uint8)  # type: ignore[attr-defined]
    return colored


def visualize_depth_zones(
    image: np.ndarray,
    depth_map: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the input image alongside the depth map and zone segmentation.

    Args:
        image:     RGB uint8 image.
        depth_map: Normalized depth (H, W).
        save_path: If provided, save figure to this path.
    """
    near_mask, mid_mask, far_mask = segment_depth_zones(depth_map)

    # Build an RGB zone visualization
    zone_vis = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    zone_vis[near_mask] = [0, 200, 50]    # green  = near
    zone_vis[mid_mask]  = [200, 200, 0]   # yellow = mid
    zone_vis[far_mask]  = [200, 0, 50]    # red    = far

    # Blend with original image for context
    overlay = cv2.addWeighted(image, 0.6, zone_vis, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image);        axes[0].set_title("Original");      axes[0].axis('off')
    axes[1].imshow(depth_map, cmap='plasma'); axes[1].set_title("Depth Map"); axes[1].axis('off')
    axes[2].imshow(overlay);      axes[2].set_title("Depth Zones (G=Near, Y=Mid, R=Far)"); axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Zone visualization saved: {save_path}")
    plt.show()
    plt.close(fig)


def create_comparison_figure(
    images: Dict[str, np.ndarray],
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create a side-by-side comparison figure of multiple enhancement results.

    Args:
        images:    Ordered dict { 'Original': arr, 'Baseline': arr, 'DASE': arr, … }.
        metrics:   Optional per-method metrics for subtitle annotation.
        save_path: If provided, save the figure here.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(img)
        subtitle = title
        if metrics and title in metrics:
            m = metrics[title]
            parts = []
            for k in ('UIQM', 'UCIQE', 'PSNR', 'SSIM'):
                if k in m:
                    parts.append(f"{k}={m[k]:.3f}")
            if parts:
                subtitle += "\n" + " | ".join(parts)
        ax.set_title(subtitle, fontsize=11)
        ax.axis('off')

    plt.suptitle("Underwater Enhancement Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Comparison figure saved: {save_path}")
    plt.show()
    plt.close(fig)


# ============================================================================
# SINGLE IMAGE PIPELINE ENTRY POINT
# ============================================================================

def enhance_image(
    image_path: str,
    method: str = 'dase',
    contrast_method: str = 'CLAHE',
    save_path: Optional[str] = None,
    depth_model_bundle=None
) -> np.ndarray:
    """
    Load and enhance a single underwater image.

    Args:
        image_path:         Path to input image.
        method:             'baseline' or 'dase'.
        contrast_method:    For baseline: 'HE', 'CLAHE', or 'LA'.
        save_path:          If given, save the result here.
        depth_model_bundle: Pre-loaded MiDaS bundle (avoids reloading).

    Returns:
        Enhanced RGB uint8 image.
    """
    image = load_image(image_path)
    print(f"[Pipeline] Image loaded: {image_path}  shape={image.shape}")

    if method == 'baseline':
        result = baseline_enhance(image, contrast_method=contrast_method)
    elif method == 'dase':
        result = dase_enhance(image, depth_model_bundle=depth_model_bundle)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'baseline' or 'dase'.")

    if save_path:
        save_image(result, save_path)
        print(f"[Pipeline] Saved: {save_path}")

    return result


# ============================================================================
# COMPARISON
# ============================================================================

def compare_methods(
    image_path: str,
    output_dir: Optional[str] = None,
    depth_model_bundle=None
) -> Dict:
    """
    Run both baseline and DASE on a single image and compare results.

    Args:
        image_path:         Path to the input image.
        output_dir:         If provided, save enhanced images and figure.
        depth_model_bundle: Pre-loaded MiDaS bundle.

    Returns:
        Dictionary with keys:
            'original', 'baseline', 'dase',
            'baseline_metrics', 'dase_metrics'
    """
    print(f"\n{'='*60}")
    print(f" Comparing methods on: {Path(image_path).name}")
    print(f"{'='*60}")

    image = load_image(image_path)

    # Run baseline
    t0 = time.time()
    baseline_result = baseline_enhance(image)
    baseline_time = time.time() - t0
    print(f"[Compare] Baseline done in {baseline_time:.2f}s")

    # Run DASE
    t0 = time.time()
    dase_result = dase_enhance(image, depth_model_bundle=depth_model_bundle)
    dase_time = time.time() - t0
    print(f"[Compare] DASE done in {dase_time:.2f}s")

    # Evaluate
    baseline_metrics = evaluate_enhancement(image, baseline_result)
    dase_metrics     = evaluate_enhancement(image, dase_result)

    print("\n[Compare] Metrics:")
    print(f"  {'Metric':<14}  {'Baseline':>10}  {'DASE':>10}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}")
    for key in ('UIQM', 'UCIQE', 'Entropy', 'Delta_UIQM'):
        b_val = baseline_metrics.get(key, float('nan'))
        d_val = dase_metrics.get(key, float('nan'))
        print(f"  {key:<14}  {b_val:>10.4f}  {d_val:>10.4f}")

    result = {
        'original':         image,
        'baseline':         baseline_result,
        'dase':             dase_result,
        'baseline_metrics': baseline_metrics,
        'dase_metrics':     dase_metrics,
        'baseline_time':    baseline_time,
        'dase_time':        dase_time,
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        save_image(baseline_result, f"{output_dir}/{stem}_baseline.png")
        save_image(dase_result,     f"{output_dir}/{stem}_dase.png")
        create_comparison_figure(
            images={
                'Original': image,
                'Baseline': baseline_result,
                'DASE':     dase_result,
            },
            metrics={
                'Baseline': baseline_metrics,
                'DASE':     dase_metrics,
            },
            save_path=f"{output_dir}/{stem}_comparison.png"
        )

    return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_single_image(
    input_path: str,
    output_dir: str,
    methods: List[str] = ('baseline', 'dase'),
    depth_model_bundle=None
) -> Dict[str, Dict[str, float]]:
    """
    Process one image with all requested methods.

    Args:
        input_path:         Path to the image.
        output_dir:         Directory to save results.
        methods:            List of methods to apply.
        depth_model_bundle: Pre-loaded MiDaS bundle.

    Returns:
        Dict { method: metrics_dict }
    """
    image = load_image(input_path)
    stem  = Path(input_path).stem
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, float]] = {}
    result_images = {'Original': image}

    for method in methods:
        t0 = time.time()
        if method == 'baseline':
            result = baseline_enhance(image)
        elif method == 'dase':
            result = dase_enhance(image, depth_model_bundle=depth_model_bundle)
        else:
            print(f"[Batch] Unknown method '{method}', skipping.")
            continue

        elapsed = time.time() - t0
        out_path = f"{output_dir}/{stem}_{method}.png"
        save_image(result, out_path)

        metrics = evaluate_enhancement(image, result)
        metrics['time_s'] = elapsed
        all_metrics[method] = metrics
        result_images[method.upper()] = result
        print(f"[Batch] {stem} | {method} | UIQM={metrics['UIQM']:.4f} | {elapsed:.2f}s")

    # Save comparison figure
    create_comparison_figure(
        images=result_images,
        metrics={k.upper(): v for k, v in all_metrics.items()},
        save_path=f"{output_dir}/{stem}_comparison.png"
    )

    return all_metrics

# evaluate_with_gt.py
import pandas as pd
from pathlib import Path
from underwater_enhancement import load_image, evaluate_enhancement, dase_enhance, baseline_enhance

raw_dir = Path("UIEB/raw-890")
ref_dir = Path("UIEB/reference-890")
results = []

for raw_path in sorted(raw_dir.glob("*.png")):
    ref_path = ref_dir / raw_path.name
    if not ref_path.exists():
        continue

    image = load_image(str(raw_path))
    ground_truth = load_image(str(ref_path))

    baseline = baseline_enhance(image)
    dase = dase_enhance(image)

    b_metrics = evaluate_enhancement(image, baseline, ground_truth=ground_truth)
    d_metrics = evaluate_enhancement(image, dase, ground_truth=ground_truth)

    results.append({
        'file': raw_path.name,
        'baseline_PSNR': b_metrics.get('PSNR'),
        'baseline_SSIM': b_metrics.get('SSIM'),
        'baseline_UIQM': b_metrics['UIQM'],
        'baseline_UCIQE': b_metrics['UCIQE'],
        'dase_PSNR': d_metrics.get('PSNR'),
        'dase_SSIM': d_metrics.get('SSIM'),
        'dase_UIQM': d_metrics['UIQM'],
        'dase_UCIQE': d_metrics['UCIQE'],
    })

df = pd.DataFrame(results)
df.to_csv("full_evaluation.csv", index=False)

# Print summary
print(df[['baseline_PSNR','dase_PSNR','baseline_UIQM','dase_UIQM']].mean())


def batch_process(
    input_dir: str,
    output_dir: str,
    methods: List[str] = ('baseline', 'dase'),
    save_metrics: bool = True
) -> None:
    """
    Process all images in a directory with all specified methods.

    Args:
        input_dir:    Directory containing input images.
        output_dir:   Directory to write enhanced images and metrics.
        methods:      List of methods to apply to every image.
        save_metrics: Whether to save a CSV summary.
    """
    input_dir_p = Path(input_dir)
    if not input_dir_p.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    image_paths = [
        p for p in sorted(input_dir_p.iterdir())
        if p.suffix.lower() in Config.IMAGE_EXTENSIONS
    ]
    if not image_paths:
        print(f"[Batch] No images found in {input_dir}")
        return

    print(f"[Batch] Found {len(image_paths)} images. Methods: {methods}")

    # Pre-load MiDaS once if dase is requested
    depth_model_bundle = None
    if 'dase' in methods:
        try:
            depth_model_bundle = load_midas_model()
        except Exception as e:
            print(f"[Batch] MiDaS unavailable ({e}). Using depth fallback.")

    all_results: List[Dict] = []
    total_start = time.time()

    for img_path in image_paths:
        print(f"\n[Batch] Processing: {img_path.name}")
        try:
            metrics = process_single_image(
                str(img_path), output_dir, methods, depth_model_bundle
            )
            for method, m in metrics.items():
                row = {'file': img_path.name, 'method': method}
                row.update(m)
                all_results.append(row)
        except Exception as e:
            print(f"[Batch] Error on {img_path.name}: {e}")

    total_time = time.time() - total_start
    print(f"\n[Batch] Done. {len(image_paths)} images in {total_time:.1f}s")

    if save_metrics and all_results:
        csv_path = f"{output_dir}/metrics_summary.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        keys = list(all_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[Batch] Metrics saved: {csv_path}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DASE – Depth-Aware Scene-Adaptive Underwater Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python underwater_enhancement.py --image sample.jpg
  python underwater_enhancement.py --image sample.jpg --method dase --save results/out.png
  python underwater_enhancement.py --compare sample.jpg --output results/
  python underwater_enhancement.py --batch images/ --output results/
        """
    )
    parser.add_argument('--image',   type=str, help='Single image to enhance')
    parser.add_argument('--compare', type=str, help='Compare baseline vs DASE on image')
    parser.add_argument('--batch',   type=str, help='Directory of images for batch processing')
    parser.add_argument('--output',  type=str, default='results', help='Output directory')
    parser.add_argument('--save',    type=str, default=None,      help='Save path for single image output')
    parser.add_argument('--method',  type=str, default='dase',    choices=['baseline', 'dase'],
                        help='Enhancement method')
    parser.add_argument('--contrast', type=str, default='CLAHE',  choices=['HE', 'CLAHE', 'LA'],
                        help='Contrast method for baseline')
    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Entry point: CLI or example usage."""
    # If arguments are provided, use CLI mode
    if len(sys.argv) > 1:
        parser = _build_cli()
        args   = parser.parse_args()

        if args.compare:
            compare_methods(args.compare, output_dir=args.output)

        elif args.batch:
            batch_process(args.batch, args.output)

        elif args.image:
            result = enhance_image(
                args.image,
                method=args.method,
                contrast_method=args.contrast,
                save_path=args.save or f"{args.output}/{Path(args.image).stem}_{args.method}.png"
            )
            metrics = evaluate_enhancement(load_image(args.image), result)
            print("\n[Result] Quality Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            parser.print_help()
        return

    # -----------------------------------------------------------------------
    # Example / demo mode (no CLI arguments)
    # -----------------------------------------------------------------------
    print("="*60)
    print("  DASE – Demo Mode")
    print("="*60)
    print("No image path specified. Running a synthetic test…\n")

    # Create a synthetic underwater-looking test image
    h, w = 256, 384
    synthetic = np.zeros((h, w, 3), dtype=np.uint8)
    # Green-blue gradient with red attenuation
    for y in range(h):
        depth_factor = y / h  # deeper (farther) toward bottom
        synthetic[y, :, 0] = int(40 * (1 - depth_factor))   # R attenuated with depth
        synthetic[y, :, 1] = int(100 + 60 * (1 - depth_factor))  # G moderate
        synthetic[y, :, 2] = int(120 + 80 * depth_factor)   # B increases with depth
    # Add some spatial noise to make it interesting
    noise = np.random.randint(0, 20, synthetic.shape, dtype=np.uint8)
    synthetic = np.clip(synthetic.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print("Input image shape:", synthetic.shape)

    # --- Baseline ---
    print("\n[Demo] Running baseline enhancement …")
    baseline_result = baseline_enhance(synthetic, contrast_method='CLAHE')
    baseline_metrics = evaluate_enhancement(synthetic, baseline_result)
    print(f"  UIQM  : {baseline_metrics['UIQM']:.4f}")
    print(f"  UCIQE : {baseline_metrics['UCIQE']:.4f}")
    print(f"  Entropy: {baseline_metrics['Entropy']:.4f}")

    # --- DASE (uses gradient depth fallback when MiDaS unavailable) ---
    print("\n[Demo] Running DASE enhancement (depth fallback if MiDaS unavailable) …")
    dase_result = dase_enhance(synthetic)
    dase_metrics = evaluate_enhancement(synthetic, dase_result)
    print(f"  UIQM  : {dase_metrics['UIQM']:.4f}")
    print(f"  UCIQE : {dase_metrics['UCIQE']:.4f}")
    print(f"  Entropy: {dase_metrics['Entropy']:.4f}")

    print("\n[Demo] Generating comparison figure …")
    create_comparison_figure(
        images={
            'Original (Synthetic)': synthetic,
            'Baseline (CLAHE)':     baseline_result,
            'DASE':                 dase_result,
        },
        metrics={
            'Baseline (CLAHE)': baseline_metrics,
            'DASE':             dase_metrics,
        },
        save_path='demo_comparison.png'
    )

    print("\n[Demo] Generating depth zone visualization …")
    depth_map = dummy_depth_map(synthetic)
    visualize_depth_zones(synthetic, depth_map, save_path='demo_depth_zones.png')

    print("\n[Demo] Complete! Saved: demo_comparison.png, demo_depth_zones.png")
    print("\nTo use with a real image:")
    print("  python underwater_enhancement.py --image your_image.jpg --method dase")
    print("  python underwater_enhancement.py --compare your_image.jpg --output results/")
    print("  python underwater_enhancement.py --batch images/ --output results/")


if __name__ == "__main__":
    main()