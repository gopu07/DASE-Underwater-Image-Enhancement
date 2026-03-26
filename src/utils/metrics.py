import cv2
import numpy as np
from typing import Dict, Optional

try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Warning] scikit-image not found. SSIM/PSNR will use fallback implementations.")

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for PSNR.")

    if SKIMAGE_AVAILABLE:
        return float(psnr_func(img1, img2, data_range=255))

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for SSIM.")

    if SKIMAGE_AVAILABLE:
        return float(ssim_func(img1, img2, channel_axis=2, data_range=255))

    def _ssim_channel(c1, c2):
        c1, c2 = c1.astype(np.float64), c2.astype(np.float64)
        mu1, mu2 = c1.mean(), c2.mean()
        sigma1 = c1.std()
        sigma2 = c2.std()
        sigma12 = np.mean((c1 - mu1) * (c2 - mu2))
        C1, C2 = 6.5025, 58.5225
        return ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))

    scores = [_ssim_channel(img1[:,:,i], img2[:,:,i]) for i in range(3)]
    return float(np.mean(scores))

def _uicm(image: np.ndarray) -> float:
    img_f = image.astype(np.float32) / 255.0
    R, G, B = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    RG = R - G
    YB = 0.5*(R + G) - B
    mu_rg, sigma_rg = RG.mean(), RG.std()
    mu_yb, sigma_yb = YB.mean(), YB.std()
    uicm = -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sigma_rg**2 + sigma_yb**2)
    return float(uicm)

def _uism(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.mean(edges))

def _uiconm(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(gray))

def compute_uiqm(image: np.ndarray) -> float:
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm   = _uicm(image)
    uism   = _uism(image)
    uiconm = _uiconm(image)
    uiqm   = c1 * uicm + c2 * uism + c3 * uiconm
    return float(uiqm)

def compute_uciqe(image: np.ndarray) -> float:
    lab  = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L    = lab[:,:,0] / 255.0 * 100.0
    A    = lab[:,:,1] - 128.0
    B_ch = lab[:,:,2] - 128.0

    chroma  = np.sqrt(A**2 + B_ch**2)
    sigma_c = float(chroma.std())

    l_sorted = np.sort(L.ravel())
    n = len(l_sorted)
    con_l = float(l_sorted[int(0.99*n)] - l_sorted[int(0.01*n)])

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:,:,1] / 255.0
    mu_s = float(S.mean())

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1*sigma_c + c2*con_l + c3*mu_s)

def compute_entropy(image: np.ndarray) -> float:
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
    """Compute a comprehensive set of image quality metrics."""
    metrics: Dict[str, float] = {}

    metrics['UIQM']    = compute_uiqm(enhanced)
    metrics['UCIQE']   = compute_uciqe(enhanced)
    metrics['Entropy'] = compute_entropy(enhanced)

    if ground_truth is not None:
        gt = ground_truth
        if enhanced.shape != gt.shape:
            gt = cv2.resize(gt, (enhanced.shape[1], enhanced.shape[0]))
        try:
            metrics['PSNR'] = compute_psnr(enhanced, gt)
            metrics['SSIM'] = compute_ssim(enhanced, gt)
        except Exception as e:
            print(f"[Eval] Full-reference metric error: {e}")

    metrics['UIQM_orig']  = compute_uiqm(original)
    metrics['UCIQE_orig'] = compute_uciqe(original)
    metrics['Delta_UIQM'] = metrics['UIQM'] - metrics['UIQM_orig']

    return metrics
