# Methodology: Depth-Aware Scene-Adaptive Enhancement (DASE)

This document outlines the core algorithmic contributions implemented in the `src/enhancement/dase_pipeline.py` pipeline.

## 1. Depth-Stratified Color Correction

Traditional underwater image enhancement methods apply a global color correction factor derived from mean and max pixel statistics. However, light attenuation in water is highly non-linear and depth-dependent. DASE improves upon this by spatially stratifying color correction.

### Mechanism:
1. **Depth Estimation**: The MiDaS neural network computes a dense depth map of the scene.
2. **Zone Segmentation**: The depth map is dynamically thresholded at the 33rd and 66th percentiles to create three masks: `near`, `mid`, and `far`.
3. **Adaptive Scaling**:
   - **Near Field**: Requires minimal compensation. Scaling factors are suppressed (blended closer to 1.0) using an alpha parameter (`NEAR_SCALE_FACTOR`).
   - **Mid Field**: Utilizes standard baseline mean-max scaling factors.
   - **Far Field**: Suffers maximum red-channel absorption. The red scaling factor is boosted aggressively using a multiplier (`FAR_SCALE_MULTIPLIER`), while green and blue adjustments are maintained.
4. **Smooth Transition**: The distinct zone scaling maps are fused using a large kernel Gaussian blur to prevent artifacting at boundaries.

## 2. Scene-Adaptive Yellow Preservation

A common artifact in standard correction pipelines is the over-suppression of the green channel, especially troubling when scenes contain substantial yellowish content (e.g., specific coral reefs, artificial lights, sand bottoms). DASE intelligently detects these conditions.

### Mechanism:
1. **HSV Space Thresholding**: The image is mapped to HSV space. Pixels falling into specific hue (20°-40°) and saturation ranges are flagged.
2. **Global Detection**: If the ratio of these yellow pixels exceeds `YELLOW_THRESHOLD`, the scene is categorized as heavy-yellow.
3. **Adaptive Alteration**: The green scaling factor (GSF) across all active zones is relaxed using a multiplier (`YELLOW_GREEN_MULTIPLIER` < 1.0) to preserve yellow/green fidelity without sacrificing red recovery.

## 3. Depth-Weighted Contrast Fusion

Global contrast techniques such as Histogram Equalization (HE) often over-expose the near-field, while softer techniques like Linear Adjustment (LA) fail to retrieve details in the heavily scattered far-field. DASE solves this by fusing multiple methods based on spatial depth.

### Mechanism:
1. **Independent Enhancements**: The scene is processed simultaneously via three methods in the CIE-LAB space:
   - **Linear Stretching (LA)**: Gentle adaptation.
   - **CLAHE**: Locally-adaptive balanced enhancement.
   - **Histogram Equalization (HE)**: Aggressive contrast pull.
2. **Tri-weight Generation**: Pixel-wise depth determines fusion confidence:
   - `W_LA = 1.0 - Depth` (High for near field)
   - `W_CLAHE = 1.0 - 2 * |Depth - 0.5|` (High for mid field)
   - `W_HE = Depth` (High for far field)
3. **Normalized Fusion**: The three outputs are weighted, ensuring `W_LA + W_CLAHE + W_HE = 1`, resulting in a seamlessly layered final output optimized for distance-specific artifacts.
