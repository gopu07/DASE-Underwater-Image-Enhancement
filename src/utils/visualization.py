import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Dict

def load_image(path: str) -> np.ndarray:
    """Load an image from disk and convert to RGB uint8."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

def save_image(image: np.ndarray, path: str) -> None:
    """Save an RGB image to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a depth map to the [0, 1] range."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)

def visualize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Convert a depth map to a colormapped RGB visualization."""
    depth_norm = normalize_depth(depth_map)
    colored = (cm.plasma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored

def visualize_depth_zones(
    image: np.ndarray,
    depth_map: np.ndarray,
    segment_depth_zones_fn,
    save_path: Optional[str] = None
) -> None:
    """Plot the input image alongside the depth map and zone segmentation."""
    near_mask, mid_mask, far_mask = segment_depth_zones_fn(depth_map)

    zone_vis = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    zone_vis[near_mask] = [0, 200, 50]
    zone_vis[mid_mask]  = [200, 200, 0]
    zone_vis[far_mask]  = [200, 0, 50]

    overlay = cv2.addWeighted(image, 0.6, zone_vis, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(depth_map, cmap='plasma'); axes[1].set_title("Depth Map"); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title("Depth Zones (G=Near, Y=Mid, R=Far)"); axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Zone visualization saved: {save_path}")
    # plt.show()
    plt.close(fig)

def create_comparison_figure(
    images: Dict[str, np.ndarray],
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """Create a side-by-side comparison figure of multiple enhancement results."""
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
    # plt.show()
    plt.close(fig)
