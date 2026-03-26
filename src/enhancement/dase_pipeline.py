import cv2
import numpy as np
from typing import Tuple, Dict, Union

from src.config import Config
from src.depth.midas import segment_depth_zones, load_midas_model, estimate_depth, dummy_depth_map
from src.enhancement.baseline import apply_contrast_LAB

def detect_yellow_content(
    image: np.ndarray,
    threshold: float = Config.YELLOW_THRESHOLD
) -> bool:
    """Detect whether the image contains significant yellow-toned regions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, _ = cv2.split(hsv)

    h_min = Config.YELLOW_HUE_MIN // 2
    h_max = Config.YELLOW_HUE_MAX // 2

    yellow_mask = (H >= h_min) & (H <= h_max) & (S > Config.YELLOW_SAT_MIN)
    yellow_ratio = yellow_mask.sum() / yellow_mask.size

    return bool(yellow_ratio > threshold)

def compute_zone_scaling_factors(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    zone: str,
    yellow_content: bool
) -> Tuple[float, float, float]:
    """Compute per-channel scaling factors for a specific depth zone."""
    if R.size == 0 or G.size == 0 or B.size == 0:
        return 1.0, 1.0, 1.0

    ravg = float(np.mean(R))
    Rmax = float(np.max(R)) + 1e-6
    Gmax = float(np.max(G)) + 1e-6
    Bmax = float(np.max(B)) + 1e-6

    rsf_base = np.mean([Gmax, Bmax]) / Rmax
    gsf_base = np.mean([ravg, Bmax]) / Gmax
    bsf_base = np.mean([ravg, Gmax]) / Bmax

    if zone == 'near':
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

    if yellow_content and gsf < 1.0:
        gsf = 1.0 + Config.YELLOW_GREEN_MULTIPLIER * (gsf - 1.0)

    return float(rsf), float(gsf), float(bsf)

def depth_aware_color_correction(
    image: np.ndarray,
    depth_map: np.ndarray
) -> np.ndarray:
    """Apply depth-stratified, spatially-varying color correction."""
    img_f = image.astype(np.float32)
    R_full = img_f[:, :, 0]
    G_full = img_f[:, :, 1]
    B_full = img_f[:, :, 2]

    near_mask, mid_mask, far_mask = segment_depth_zones(depth_map)
    yellow_content = detect_yellow_content(image)

    if yellow_content:
        print("[ColorCorr] Yellow content detected → adapting green suppression.")

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

    rsf_map = np.ones_like(R_full)
    gsf_map = np.ones_like(G_full)
    bsf_map = np.ones_like(B_full)

    for zone_name, mask in zones.items():
        rsf, gsf, bsf = zone_factors[zone_name]
        rsf_map[mask] = rsf
        gsf_map[mask] = gsf
        bsf_map[mask] = bsf

    blur_k = 51
    rsf_map = cv2.GaussianBlur(rsf_map, (blur_k, blur_k), 0)
    gsf_map = cv2.GaussianBlur(gsf_map, (blur_k, blur_k), 0)
    bsf_map = cv2.GaussianBlur(bsf_map, (blur_k, blur_k), 0)

    R_out = np.clip(R_full * rsf_map, 0, 255).astype(np.uint8)
    G_out = np.clip(G_full * gsf_map, 0, 255).astype(np.uint8)
    B_out = np.clip(B_full * bsf_map, 0, 255).astype(np.uint8)

    return np.stack([R_out, G_out, B_out], axis=2)

def compute_fusion_weights(
    depth_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel fusion weights for the three contrast methods."""
    d = depth_map.astype(np.float32)

    w_la    = 1.0 - d
    w_clahe = 1.0 - np.abs(d - 0.5) * 2.0
    w_he    = d

    w_la    = np.clip(w_la, 0, 1)
    w_clahe = np.clip(w_clahe, 0, 1)
    w_he    = np.clip(w_he, 0, 1)

    w_sum = w_la + w_clahe + w_he + 1e-8
    w_la    /= w_sum
    w_clahe /= w_sum
    w_he    /= w_sum

    return w_la, w_clahe, w_he

def depth_weighted_contrast_fusion(
    image: np.ndarray,
    depth_map: np.ndarray
) -> np.ndarray:
    """Fuse HE, CLAHE, and LA contrast-enhanced images weighted by depth."""
    he_out    = apply_contrast_LAB(image, method='HE').astype(np.float32)
    clahe_out = apply_contrast_LAB(image, method='CLAHE').astype(np.float32)
    la_out    = apply_contrast_LAB(image, method='LA').astype(np.float32)

    w_la, w_clahe, w_he = compute_fusion_weights(depth_map)
    w_la    = w_la[:, :, np.newaxis]
    w_clahe = w_clahe[:, :, np.newaxis]
    w_he    = w_he[:, :, np.newaxis]

    fused = w_la * la_out + w_clahe * clahe_out + w_he * he_out
    return np.clip(fused, 0, 255).astype(np.uint8)

def dase_enhance(
    image: np.ndarray,
    depth_model_bundle=None,
    return_intermediate: bool = False
) -> Union[np.ndarray, Dict]:
    """Complete DASE pipeline."""
    print("[DASE] Starting depth-aware enhancement …")

    try:
        if depth_model_bundle is None:
            depth_model_bundle = load_midas_model()
        depth_map = estimate_depth(image, depth_model_bundle)
        print("[DASE] Depth estimation complete.")
    except Exception as e:
        print(f"[DASE] Depth estimation failed ({e}). Using gradient fallback.")
        depth_map = dummy_depth_map(image)

    color_corrected = depth_aware_color_correction(image, depth_map)
    print("[DASE] Depth-aware color correction applied.")

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
