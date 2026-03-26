import torch
import numpy as np
from typing import Tuple, Dict

from src.config import Config
from src.utils.visualization import normalize_depth

_midas_model_cache: Dict = {}

def load_midas_model(model_type: str = Config.MIDAS_MODEL):
    """Load a pretrained MiDaS depth estimation model from torch.hub."""
    global _midas_model_cache

    if model_type in _midas_model_cache:
        return _midas_model_cache[model_type]

    print(f"[Depth] Loading MiDaS model: {model_type} …")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    except Exception as e:
        print(f"[Depth] torch.hub failed ({e}). Attempting timm fallback …")
        try:
            import timm
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
    """Estimate a per-pixel depth map using MiDaS."""
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

    depth_inverted = -depth_raw
    depth_norm = normalize_depth(depth_inverted)
    return depth_norm

def dummy_depth_map(image: np.ndarray) -> np.ndarray:
    """Generate a simple synthetic depth map as a fallback when MiDaS is unavailable."""
    h, w = image.shape[:2]
    depth = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
    return depth

def segment_depth_zones(
    depth_map: np.ndarray,
    num_zones: int = Config.NUM_ZONES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition the depth map into three spatial zones using percentiles."""
    near_thresh = float(np.percentile(depth_map, Config.NEAR_PERCENTILE))
    far_thresh  = float(np.percentile(depth_map, Config.FAR_PERCENTILE))

    near_mask = depth_map < near_thresh
    mid_mask  = (depth_map >= near_thresh) & (depth_map < far_thresh)
    far_mask  = depth_map >= far_thresh

    return near_mask, mid_mask, far_mask
