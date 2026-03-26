from typing import List, Tuple

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
