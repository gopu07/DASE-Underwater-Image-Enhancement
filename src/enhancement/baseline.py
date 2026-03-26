import cv2
import numpy as np

from src.config import Config

def hybrid_mean_max_correction(image: np.ndarray) -> np.ndarray:
    """Baseline color correction using the hybrid mean-max method."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    img = image.astype(np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    ravg = float(np.mean(R))
    Rmax = float(np.max(R))
    Gmax = float(np.max(G))
    Bmax = float(np.max(B))

    gsf = np.mean([ravg, Bmax]) / (Gmax + 1e-6)
    bsf = np.mean([ravg, Gmax]) / (Bmax + 1e-6)
    rsf = np.mean([Gmax, Bmax]) / (Rmax + 1e-6)

    R_corr = np.clip(R * rsf, 0, 255).astype(np.uint8)
    G_corr = np.clip(G * gsf, 0, 255).astype(np.uint8)
    B_corr = np.clip(B * bsf, 0, 255).astype(np.uint8)

    return np.stack([R_corr, G_corr, B_corr], axis=2)

def apply_contrast_LAB(image: np.ndarray, method: str = 'CLAHE') -> np.ndarray:
    """Apply contrast enhancement in CIE LAB color space on the L channel."""
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

    else:  # 'LA'
        L_float = L.astype(np.float32)
        L_min, L_max = L_float.min(), L_float.max()
        if L_max - L_min > 1e-6:
            L_enhanced = ((L_float - L_min) / (L_max - L_min) * 255).astype(np.uint8)
        else:
            L_enhanced = L

    lab_enhanced = cv2.merge([L_enhanced, A, B_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

def baseline_enhance(image: np.ndarray, contrast_method: str = 'CLAHE') -> np.ndarray:
    """Complete baseline enhancement pipeline."""
    color_corrected = hybrid_mean_max_correction(image)
    enhanced = apply_contrast_LAB(color_corrected, method=contrast_method)
    return enhanced
